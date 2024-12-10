# Source code for the paper Multi-Reference Preference Optimization (MRPO) for Large Language Models

- Paper arXiv version: https://arxiv.org/abs/2405.16388  
- Blog: https://hungleai.substack.com/p/many-hands-make-light-work-leveraging
- Code reference: https://github.com/hiyouga/LLaMA-Factory 

# Setup  
python>=3.7
```
pip install -r requirements.txt   
```

# Preference Learning Task
Example using L<-M on ultrafeedback dataset
```
python run_mrpo.py llama ultrafeedback no hermes no 0.1 1 0  0.1 -1
```

# Note
- This implementation is for demo purpose. Currentl, it only supports K=2 and does not compute the reference logprob in advance
- If you want to know more about the hyperparameters, please refer to run_mrpo.py
- If you want to refer to MRPO implementation, please refer to ./src/llmtuner/train/dpo/trainer_mref.py 