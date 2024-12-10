import os 
import sys

# default param 1-llama (base1), 2-gpt4 (data) 3-no (ft1) 4-0.1 (eval) 5-1 (epoch) 6-0 (seed)
# python run_dpo.py tiny ultrafeedback no 0.1 1 0

var1 = sys.argv[1]
var2 = sys.argv[2]
var3 = sys.argv[3]
var4 = sys.argv[4]
var5 = sys.argv[5]
var6 = sys.argv[6]

bases = {}
bases["llama"] =  "Llama-2-7b-chat-hf" #path to llama model
bases["mistral"] = "mistralai/Mistral-7B-v0.1"
bases["hermes"] = "teknium/OpenHermes-2.5-Mistral-7B"
bases["tiny"] = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

data = {}
data["gpt4"] = "comparison_gpt4_en"
data["ultrafeedback"] = "ultrafeedback_binarized"
data["math"] = "distilabel_math_preference"
data["helpsteer"] = "helpsteer_binarized"
data["nn"] = "nn_binarized"
data["nectar"] = "nectar_binarized"


val_steps = {}
val_steps["comparison_gpt4_en"] = 20
val_steps["ultrafeedback_binarized"] = 500
val_steps["distilabel_math_preference"] = 100
val_steps["helpsteer_binarized"] = 30
val_steps["nn_binarized"] = 30
val_steps["nectar_binarized"] = 500


ft_adaptors = {}




if var1 == "all":
    nbases = list(bases.values())
else:
    nbases = [bases[var1]]

if var2 == "all":
    ndata = list(data.values())
else:
    ndata = [data[var2]]

if not var3 or var3 == "no":
    nft_adaptors = ["no"]
elif "all" in var3:
    nft_adaptors = list(ft_adaptors[var1].values())
else:
    nft_adaptors = [ft_adaptors[var1][var3]]

def run_exp(seed):
    for i in nbases:
        print(i)
        tr_bs = 2
        if var1 == "tiny":
            tr_bs=4
            val_bs=4

        lorat = "q_proj,v_proj"

        if "mistral" in i or "hermes" in i:
            print("REDUCE BATCH SIZE")
            tr_bs = 1
        else:
            lorat = "q_proj,v_proj"
        if var1 == "tiny":
            lorat = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
        for j in ndata:
            print(j)
            for k in nft_adaptors:
                print(k)
                model = i.split("/")[-1]+ "-ft-" + k.split("/")[-1]
                command = f"CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
                --stage dpo \
                --do_train \
                --model_name_or_path {i} \
                "
                if k!="no":
                    command += f"--adapter_name_or_path {k} "
                command += f"--create_new_adapter \
                --dataset {j} \
                --template default \
                --finetuning_type lora \
                --lora_target {lorat} \
                --output_dir ./save/{j}-{var4}/dpo/path_to_dpo-{model}-{j}-seed{seed}_checkpoint \
                --dpo_loss sigmoid \
                --dpo_num_ref 1 \
                --per_device_train_batch_size {tr_bs} \
                --per_device_eval_batch_size 8 \
                --gradient_accumulation_steps 4 \
                --lr_scheduler_type cosine \
                --logging_steps 10 \
                --evaluation_strategy steps \
                --save_total_limit 1 \
                --val_size {var4} \
                --eval_steps {val_steps[j]} \
                --save_strategy steps \
                --save_steps 400 \
                --learning_rate 1e-5 \
                --num_train_epochs {var5} \
                --ref_model_quantization_bit 4 \
                --seed {seed} \
                --overwrite_output_dir \
                --do_eval \
                --plot_loss \
                --fp16 "
                os.system(command)

if "-" not in var6:
    run_exp(var6)
else:
    seeds = var6.split("-")
    for s in seeds:
        run_exp(s.strip())