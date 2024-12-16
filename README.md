# Source code for the paper Multi-Reference Preference Optimization (MRPO) for Large Language Models

- Paper arXiv version: https://arxiv.org/abs/2405.16388  
- Blog: https://hungleai.substack.com/p/many-hands-make-light-work-leveraging
- Code reference: https://github.com/hiyouga/LLaMA-Factory 

# Setup  
python>=3.8
```
pip install -r requirements.txt   
```

# Preference Learning Task
- Prepare datasets: Please follow https://github.com/hiyouga/LLaMA-Factory/blob/main/data/README.md
- Demo with a prepared dataset (comparison_gpt4_en):
Example using MRPO with L<-M on gpt4 dataset
```
python run_mrpo.py llama gpt4 no hermes no 0.99 1 0  0.1 -1
```
Example using DPO with Llama on gpt4 dataset
```
python run_dpo.py llama gpt4 no 0.99 1 0
```

# Note
- This implementation is for demo purpose. Currently, it only supports K=2 and does not compute the reference logprob in advance
- If you want to know more about the hyperparameters, please refer to ./run_mrpo.py
- If you want to refer to MRPO implementation, please refer to ./src/llmtuner/train/dpo/trainer_mref.py 