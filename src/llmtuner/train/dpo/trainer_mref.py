from collections import defaultdict
from contextlib import nullcontext
from typing import TYPE_CHECKING, Dict, Literal, Optional, Tuple, Union
import os
import numpy as np
import torch
import gc         # garbage collect library

from transformers import BatchEncoding, Trainer
from trl import DPOTrainer
from trl.trainer.utils import disable_dropout_in_model
from trl.models import create_reference_model
from ...extras.constants import IGNORE_INDEX
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalLoopOutput,
    EvalPrediction,
    HPSearchBackend,
    HubStrategy,
    IntervalStrategy,
    PredictionOutput,
    RemoveColumnsCollator,
    TrainerMemoryTracker,
    TrainOutput,
    default_compute_objective,
    denumpify_detensorize,
    enable_full_determinism,
    find_executable_batch_size,
    get_last_checkpoint,
    has_length,
    neftune_post_forward_hook,
    number_of_arguments,
    seed_worker,
    set_seed,
    speed_metrics,
)
from transformers.utils import (
    ADAPTER_CONFIG_NAME,
    ADAPTER_SAFE_WEIGHTS_NAME,
    ADAPTER_WEIGHTS_NAME,
    CONFIG_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    PushInProgress,
    can_return_loss,
    find_labels,
    is_accelerate_available,
    is_apex_available,
    is_bitsandbytes_available,
    is_datasets_available,
    is_in_notebook,
    is_ipex_available,
    is_peft_available,
    is_safetensors_available,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_compile_available,
    is_torch_neuroncore_available,
    is_torch_npu_available,
    is_torch_tpu_available,
    logging,
    strtobool,
)

# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
OPTIMIZER_NAME_BIN = "optimizer.bin"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"
FSDP_MODEL_NAME = "pytorch_model_fsdp"


if TYPE_CHECKING:
    from transformers import PreTrainedModel

logger = logging.get_logger(__name__)
class MRefDPOTrainer(DPOTrainer):
    def __init__(
        self,
        beta: float,
        loss_type: Literal["sigmoid", "hinge", "ipo", "kto"],
        ftx_gamma: float,
        mref_naive: bool,
        eta:float,
        alpha:float,
        model: Union["PreTrainedModel", torch.nn.Module],
        ref_model: Optional[Union["PreTrainedModel", torch.nn.Module]] = None,
        disable_dropout: Optional[bool] = True,
        **kwargs,
    ):
        if disable_dropout:
            disable_dropout_in_model(model)
            if ref_model is not None:
                disable_dropout_in_model(ref_model)

        self.use_dpo_data_collator = True  # hack to avoid warning
        self.generate_during_eval = False  # disable at evaluation
        self.label_pad_token_id = IGNORE_INDEX
        self.padding_value = 0
        self.is_encoder_decoder = model.config.is_encoder_decoder
        self.precompute_ref_log_probs = False
        self._precomputed_train_ref_log_probs = False
        self._precomputed_eval_ref_log_probs = False
        self._peft_has_been_casted_to_bf16 = False

        self.ref_model = ref_model
        self.beta = beta
        self.alpha = alpha
        self.label_smoothing = 0
        self.loss_type = loss_type
        self.ftx_gamma = ftx_gamma
        self.mref_naive = mref_naive
        self.eta = eta
        self._stored_metrics = defaultdict(lambda: defaultdict(list))
        self.step_train = 0
        self.ref_model2 = None
        self.policy_chosen_avg = None
        self.policy_rejected_avg = None
        self.ref_chosen_avg = None
        self.ref_rejected_avg = None
        self.ref_chosen_avg2 = None
        self.ref_rejected_avg2 = None
        self.reference_free = False
        

        Trainer.__init__(self, model=model, **kwargs)
        if not hasattr(self, "accelerator"):
            raise AttributeError("Please update `transformers`.")
        

        if ref_model is not None:
            if self.is_deepspeed_enabled:
                if not (
                    getattr(ref_model, "is_loaded_in_8bit", False) or getattr(ref_model, "is_loaded_in_4bit", False)
                ):  # quantized models are already set on the correct device
                    self.ref_model = self._prepare_deepspeed(self.ref_model)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
        
    
    def sft_loss(self, chosen_logits: torch.FloatTensor, chosen_labels: torch.LongTensor) -> torch.Tensor:
        r"""
        Computes supervised cross-entropy loss of given labels under the given logits.

        Returns:
            A tensor of shape (batch_size,) containing the cross-entropy loss of each samples.
        """
        all_logps = self.get_batch_logps(chosen_logits, chosen_labels, average_log_prob=True)
        return -all_logps

    def concatenated_forward(
        self, model: "PreTrainedModel", batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        batch_copied = BatchEncoding({k: v.detach().clone() for k, v in batch.items()})  # avoid error

        all_logits = model(
            input_ids=batch_copied["input_ids"], attention_mask=batch_copied["attention_mask"], return_dict=True
        ).logits.to(torch.float32)

        all_logps = self.get_batch_logps(
            all_logits,
            batch["labels"],
            average_log_prob=False,
            label_pad_token_id=self.label_pad_token_id,
        )
        batch_size = batch["input_ids"].size(0) // 2
        chosen_logps, rejected_logps = all_logps.split(batch_size, dim=0)
        chosen_logits, rejected_logits = all_logits.split(batch_size, dim=0)
        return chosen_logps, rejected_logps, chosen_logits, rejected_logits

    def get_batch_loss_metrics(
        self,
        model: "PreTrainedModel",
        batch: Dict[str, torch.Tensor],
        train_eval: Optional[Literal["train", "eval"]] = "train",
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        r"""
        Computes the DPO loss and other metrics for the given batch of inputs for train or test.
        """
        metrics = {}
        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
        ) = self.concatenated_forward(model, batch)


        with torch.no_grad():
            if self.ref_model is None:
                ref_model = model
                ref_context = self.accelerator.unwrap_model(model).disable_adapter()
            else:
                ref_model = self.ref_model
                ref_context = nullcontext()
            ref_model.eval()

            with ref_context:
                (
                    reference_chosen_logps,
                    reference_rejected_logps,
                    reference_chosen_logits,
                    reference_rejected_logits
                ) = self.concatenated_forward(ref_model, batch)

            if self.ref_model2 is None:
                ref_context = self.accelerator.unwrap_model(model).disable_adapter()
                with ref_context:
                    (
                        reference_chosen_logps2,
                        reference_rejected_logps2,
                        reference_chosen_logits2,
                        reference_rejected_logits2
                    ) = self.concatenated_forward(model, batch)
            else:
                ref_context = nullcontext()
                with ref_context:
                    (
                        reference_chosen_logps2,
                        reference_rejected_logps2,
                        reference_chosen_logits2,
                        reference_rejected_logits2
                    ) = self.concatenated_forward(self.ref_model2, batch)
        
        self.step_train+=1
       
        eta1 = self.eta
        eta2 = self.eta
       
        a2 = reference_chosen_logps2-reference_rejected_logps2
        a1 = reference_chosen_logps-reference_rejected_logps
        
       
        if self.alpha<0: #adapive alpha
            alpha = torch.abs(a1)/(torch.abs(a1)+torch.abs(a2)+1e-6)
        else: #fix alpha
            alpha = self.alpha
        
        # CTRO adaptive clipping
        b1 = (reference_chosen_logps2+reference_chosen_logps)/2
        b2 = (reference_rejected_logps2+reference_rejected_logps)/2
        scale = torch.abs(b1)/(torch.abs(b1)+torch.abs(b2))
        eta1*=scale
        eta2*=(1-scale)
        if self.eta<0:# option using fixed clipping rate
            eta1 = -self.eta
            eta2 = -self.eta
        
        # clip
        reference_chosen_logps = torch.max(reference_chosen_logps, reference_chosen_logps2.detach()*(1+eta1))  
        reference_chosen_logps = torch.min(reference_chosen_logps, reference_chosen_logps2.detach()*(1-eta1))  
        #log sum trick
        reference_chosen_logps0 = reference_chosen_logps+reference_chosen_logps2
        c = torch.max(reference_chosen_logps,reference_chosen_logps2)
        p12 = c + torch.log(alpha*torch.exp(reference_chosen_logps-c)+(1-alpha)*torch.exp(reference_chosen_logps2-c))
        reference_chosen_logps0 -= p12

        # clip
        reference_rejected_logps = torch.max(reference_rejected_logps, reference_rejected_logps2.detach()*(1+eta2))
        reference_rejected_logps = torch.min(reference_rejected_logps, reference_rejected_logps2.detach()*(1-eta2))    
        #log sum trick
        reference_rejected_logps0 = reference_rejected_logps + reference_rejected_logps2
        c = torch.max(reference_rejected_logps,reference_rejected_logps2)
        p12 = c+torch.log(alpha*torch.exp(reference_rejected_logps-c)+(1-alpha)*torch.exp(reference_rejected_logps2-c))
        reference_rejected_logps0 -= p12
       

        if not self.mref_naive:
            #MRPO
            losses, chosen_rewards, rejected_rewards = self.dpo_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                reference_chosen_logps0,
                reference_rejected_logps0,
            )
        else:
            #Multi-DPO
            losses, chosen_rewards, rejected_rewards = self.dpo_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                reference_chosen_logps2,
                reference_rejected_logps2,
            )
            reference_chosen_logps = torch.max(reference_chosen_logps, reference_chosen_logps2.detach()*(1+eta1))  
            reference_chosen_logps = torch.min(reference_chosen_logps, reference_chosen_logps2.detach()*(1-eta1))  
            reference_rejected_logps = torch.max(reference_rejected_logps, reference_rejected_logps2.detach()*(1+eta2))
            reference_rejected_logps = torch.min(reference_rejected_logps, reference_rejected_logps2.detach()*(1-eta2))    
        
            losses2, _, _ = self.dpo_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                reference_chosen_logps,
                reference_rejected_logps,
            )
            losses = losses2*alpha+losses*(1-alpha)
      
        if self.ftx_gamma > 1e-6:
            batch_size = batch["input_ids"].size(0) // 2
            chosen_labels, _ = batch["labels"].split(batch_size, dim=0)
            losses += self.ftx_gamma * self.sft_loss(policy_chosen_logits, chosen_labels)

        reward_accuracies = (chosen_rewards > rejected_rewards).float()
      
        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.cpu().mean()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.cpu().mean()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.cpu().mean()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).cpu().mean()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().cpu().mean()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().cpu().mean()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.detach().cpu().mean()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().cpu().mean()

        return losses.mean(), metrics

    def update_ref_model2(self):
        if self.ref_model2 is not None:
            del self.ref_model2
            gc.collect()
            torch.cuda.empty_cache() 
        self.ref_model2 = create_reference_model(self.model)
        print(f"set new ref_model 2 {self._get_output_dir(trial=1)}")


    def _save_checkpoint(self, model, trial, metrics=None):
        # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
        # want to save except FullyShardedDDP.
        # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"

    
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        if self.hp_search_backend is None and trial is None:
            self.store_flos()

        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0:
            logger.warning(
                f"Checkpoint destination directory {output_dir} already exists and is non-empty."
                "Saving will proceed but saved results may be invalid."
            )
            staging_output_dir = output_dir
        else:
            staging_output_dir = os.path.join(run_dir, f"tmp-{checkpoint_folder}")
        self.save_model(staging_output_dir, _internal_call=True)

        if not self.args.save_only_model:
            # Save optimizer and scheduler
            self._save_optimizer_and_scheduler(staging_output_dir)
            # Save RNG state
            self._save_rng_state(staging_output_dir)

        # Determine the new best metric / best model checkpoint
        if metrics is not None and self.args.metric_for_best_model is not None:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            metric_value = metrics[metric_to_check]

            operator = np.greater if self.args.greater_is_better else np.less
            if (
                self.state.best_metric is None
                or self.state.best_model_checkpoint is None
                or operator(metric_value, self.state.best_metric)
            ):
                self.state.best_metric = metric_value
                self.state.best_model_checkpoint = output_dir
                self.update_ref_model2()
        # Save the Trainer state
        if self.args.should_save:
            self.state.save_to_json(os.path.join(staging_output_dir, TRAINER_STATE_NAME))

        if self.args.push_to_hub:
            self._push_from_checkpoint(staging_output_dir)

        # Place checkpoint in final location after all saving is finished.
        # First wait for everyone to finish writing
        self.args.distributed_state.wait_for_everyone()

        # Then go through the rewriting process, only renaming and rotating from main process(es)
        if self.is_local_process_zero() if self.args.save_on_each_node else self.is_world_process_zero():
            if staging_output_dir != output_dir:
                if os.path.exists(staging_output_dir):
                    os.rename(staging_output_dir, output_dir)

                    # Ensure rename completed in cases where os.rename is not atomic
                    fd = os.open(output_dir, os.O_RDONLY)
                    os.fsync(fd)
                    os.close(fd)

            # Maybe delete some older checkpoints.
            if self.args.should_save:
                self._rotate_checkpoints(use_mtime=True, output_dir=run_dir)

        self.args.distributed_state.wait_for_everyone()