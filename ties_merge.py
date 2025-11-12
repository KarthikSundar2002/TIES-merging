import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from pathlib import Path
from transformers import BitsAndBytesConfig
from collections import OrderedDict

import random

qwen_models_path = Path('/scratch/ks02450/').joinpath('qwen_models', '4B-Instruct-2507')


if not qwen_models_path.exists():
    qwen_models_path.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="Qwen/Qwen3-4B-Instruct-2507",
        allow_patterns=[
            "params.json",
            "consolidated.safetensors",
            "tokenizer.model",
            "tokenizer.json",
            "tokenizer_config.json",
        ],
        local_dir=qwen_models_path,
    )
else:
    print(f"Model already downloaded to {qwen_models_path}")

bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-4B-Instruct-2507",
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

#tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
#tokenizer.pad_token = tokenizer.eos_token  # Set pad token
#tokenizer.padding_side = "right"

gsm_adaptor_path = Path('/scratch/ks02450/').joinpath('lora-finetuned-gsm8k')
code_adaptor_path = Path('/scratch/ks02450/').joinpath('lora-finetuned-code-adaptor')

gsm_adaptor = PeftModel.from_pretrained(model, gsm_adaptor_path)
code_adaptor = PeftModel.from_pretrained(model, code_adaptor_path)

lora_modules = ['q_proj', 'v_proj', 'k_proj', 'o_proj']

ties_merged_model_dict = OrderedDict()

k_density = 0.5
pruning_percent = 1 - k_density

#Creating Task Vectors
gsm_task_vectors = []
for name, tensor in gsm_adaptor.named_parameters():
    for lora_name in lora_modules:
        if lora_name in name:
            gsm_task_vectors.append(tensor.abs().flatten())

gsm_task_vectors = torch.cat(gsm_task_vectors).to(torch.uint8)
print(gsm_task_vectors.dtype)

# Approximating Median
num_elements = len(gsm_task_vectors)
sample_size = 10000

indices = random.sample(range(num_elements), sample_size)
sample = gsm_task_vectors[indices]

# Calculate the median of the sample
median_gsm = torch.median(sample)

mask_gsm_task_vectors = gsm_task_vectors < median_gsm
gsm_task_vectors[mask_gsm_task_vectors] = 0

code_task_vectors = []
for name, tensor in code_adaptor.named_parameters():
    for lora_name in lora_modules:
        if lora_name in name:
            code_task_vectors.append(tensor.abs().flatten())

code_task_vectors = torch.cat(code_task_vectors).to(torch.uint8)
num_elements = len(code_task_vectors)
sample_size = 10000

indices = random.sample(range(num_elements), sample_size)
sample = code_task_vectors[indices]

# Calculate the median of the sample
median_code = torch.median(sample)

mask_code_task_vectors = code_task_vectors < median_code
code_task_vectors[mask_code_task_vectors] = 0
print(code_task_vectors)


for name, param in gsm_adaptor.named_parameters():
    for lora_name in lora_modules:
        if lora_name in name:
            gsm_tensor = param.data
            code_tensor = code_adaptor.get_parameter(name).data
            gsm_mask = gsm_tensor.abs() > median_gsm
            code_mask = code_tensor.abs() > median_code
            masked_gsm_tensor = gsm_tensor * gsm_mask
            masked_code_tensor = code_tensor * code_mask
            
            sign_gsm = torch.sign(masked_gsm_tensor)
            sign_code = torch.sign(masked_code_tensor)
            
            elected_sign_tensor = torch.sign(sign_gsm * gsm_mask + sign_code * code_mask)
            a = torch.sum(elected_sign_tensor)
            if a > 0:
                elected_sign = 1
            elif a < 0:
                elected_sign = -1
            else:
                elected_sign = 0
            
            disjoint_mask_gsm = sign_gsm == elected_sign
            disjoint_mask_code = sign_code == elected_sign
            
            merged_tensor = (masked_gsm_tensor * disjoint_mask_gsm + masked_code_tensor * disjoint_mask_code) / 2
            ties_merged_model_dict[name] = merged_tensor    
        else:
            ties_merged_model_dict[name] = param.data.clone()

torch.save(ties_merged_model_dict, qwen_models_path.joinpath('ties_merged_model.pth'))