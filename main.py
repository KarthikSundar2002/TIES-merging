from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from pathlib import Path


mistral_models_path = Path('/scratch/ks02450/').joinpath('mistral_models', '7B-Instruct-v0.2')


if not mistral_models_path.exists():
    mistral_models_path.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id="mistralai/Mistral-7B-Instruct-v0.2", allow_patterns=["params.json", "consolidated.safetensors", "tokenizer.model", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"], local_dir=mistral_models_path)
else:
    print(f"Model already downloaded to {mistral_models_path}")

from transformers import AutoModelForCausalLM
 
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", load_in_4bit=True)
model.to("cuda")

messages = [
    {"role": "user", "content": "What is your favourite condiment?"},
    {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
    {"role": "user", "content": "Do you have mayonnaise recipes?"}
]

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
model_inputs = encodeds.to("cuda")
 
generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)

# decode with mistral tokenizer
result = tokenizer.decode(generated_ids[0].tolist())
print(result)