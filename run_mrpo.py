import os 
import sys

# default param 1-llama (base 1), 2-gpt4 (data) 3-no (ft 1) 4-hermes (base 2) 5-no (ft2) 6-0.1 (eval split) 7-1 (epoch) 8-0 (seed) 9-0.1 (epsilon) 10--1 (alpha)
# python run_mrpo.py llama ultrafeedback no hermes no 0.1 1 0  0.1 -1

var1 = sys.argv[1]
var2 = sys.argv[2]
var3 = sys.argv[3]
var4 = sys.argv[4]
var5 = sys.argv[5]
var6 = sys.argv[6]
var7 = sys.argv[7]
var8 = sys.argv[8]
var9 = sys.argv[9]
var10 = sys.argv[10]

print(var10)

try:
    var11 = sys.argv[11]
    print("MREF NAIVE")
except:
    var11=None

bases = {}
bases["llama"] =  "Llama-2-7b-chat-hf" #path to llama model
bases["mistral"] = "mistralai/Mistral-7B-v0.1"
bases["hermes"] = "teknium/OpenHermes-2.5-Mistral-7B"



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


ref_bases = {}
ref_bases["llama"] =  "Llama-2-7b-chat-hf"  #path to llama model
ref_bases["mistral"] = "mistralai/Mistral-7B-v0.1"
ref_bases["hermes"] = "teknium/OpenHermes-2.5-Mistral-7B"


ref_adaptors = {}


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

if var4 == "all":
    nref_bases = list(ref_bases.values())
else:
    nref_bases = [ref_bases[var4]]

if not var5 or var5 == "no":
    nref_adaptors = ["no"]
elif "all" in var3:
    nref_adaptors = list(ref_adaptors[var4].values())
else:
    nref_adaptors = [ref_adaptors[var4][var5]]



def run_exp(seed):
    val_bs = 2
    if var2 == "helpsteer":
        val_bs = 1
    for i in nbases:
        print(i)
        tr_bs = 2
            
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
                for l in nref_bases:
                    print(l)
                    command += f"--ref_model {l} "
                    for m in nref_adaptors:
                        if m!="no":
                            command += f"--ref_model_adapters {m} "
                        ref_model = l.split("/")[-1]+ "-ft-" + m.split("/")[-1]
                        command += f"--create_new_adapter \
                        --dataset {j} \
                        --template default \
                        --finetuning_type lora \
                        --lora_target {lorat} \
                        --output_dir ./save/{j}-{var6}/mrpo/path_to-mrpo-{model}-{ref_model}-{j}-eta{var9}seed{seed}alpha{var10}_checkpoint \
                        --dpo_loss sigmoid \
                        --dpo_num_ref 2 \
                        --eta {var9} \
                        --alpha {var10} \
                        --per_device_train_batch_size {tr_bs} \
                        --per_device_eval_batch_size {val_bs} \
                        --gradient_accumulation_steps 4 \
                        --lr_scheduler_type cosine \
                        --logging_steps 10 \
                        --evaluation_strategy steps \
                        --save_total_limit 1 \
                        --val_size {var6} \
                        --eval_steps {val_steps[j]} \
                        --save_strategy steps \
                        --save_steps 400 \
                        --learning_rate 1e-5 \
                        --num_train_epochs {var7} \
                        --ref_model_quantization_bit 4 \
                        --seed {seed} \
                        --overwrite_output_dir \
                        --do_eval \
                        --plot_loss \
                        --fp16 "
                        if var11 is not None:
                            command += "--mref_naive "
                        os.system(command)

if "-" not in var8:
    run_exp(var8)
else:
    seeds = var8.split("-")
    for s in seeds:
        run_exp(s.strip())