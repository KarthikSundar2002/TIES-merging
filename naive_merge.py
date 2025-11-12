import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from pathlib import Path
from transformers import BitsAndBytesConfig
from collections import OrderedDict


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

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
tokenizer.pad_token = tokenizer.eos_token  # Set pad token
tokenizer.padding_side = "right"

gsm_adaptor_path = Path('/scratch/ks02450/').joinpath('lora-finetuned-gsm8k')
code_adaptor_path = Path('/scratch/ks02450/').joinpath('lora-finetuned-code-adaptor')

gsm_adaptor = PeftModel.from_pretrained(model, gsm_adaptor_path)
code_adaptor = PeftModel.from_pretrained(model, code_adaptor_path)

naive_merged_model_dict = OrderedDict()

lora_modules = ['q_proj', 'v_proj', 'k_proj', 'o_proj']

for name, param in gsm_adaptor.named_parameters():
    for lora_name in lora_modules:
        if lora_name in name:
            naive_merged_model_dict[name] = (param.data + code_adaptor.get_parameter(name).data) / 2
            break
        else:
            naive_merged_model_dict[name] = param.data.clone()

torch.save(naive_merged_model_dict, qwen_models_path.joinpath('naive_merged_model.pth'))
