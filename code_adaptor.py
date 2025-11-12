from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

from pathlib import Path


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

from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training    

from datasets import load_dataset, DatasetDict

dataset = load_dataset("TokenBender/code_instructions_122k_alpaca_style")
# Access the train split if dataset is a DatasetDict
if isinstance(dataset, DatasetDict):
    dataset = dataset["train"]

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

device = torch.device("cuda")
model.to(device)

model.gradient_checkpointing_enable()


lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    task_type="CAUSAL_LM",
    bias="none",
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)



tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
tokenizer.pad_token = tokenizer.eos_token  # Set pad token
tokenizer.padding_side = "right"

def format_dataset(examples):
    q = examples["instruction"]
    a = examples["output"]
    t = examples["text"]
    prompt = f"[INST] {q} [/INST] {t} {a}"
    res = {
        "text": prompt,
    }
    return res

dataset = dataset.map(format_dataset)
train_dataset = dataset.select(range(len(dataset) - 90000))


from trl import SFTTrainer, SFTConfig

training_config = SFTConfig(
    output_dir="/scratch/ks02450/code-adaptor-finetuned",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    optim="paged_adamw_32bit",
    logging_steps=25,
    learning_rate=2e-4,
    bf16=True,                   # Use bfloat16 for speed (if supported)
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="constant",
    num_train_epochs=1,
    save_steps=500,
    logging_dir="/scratch/ks02450/logs",
    dataset_text_field="text",
)

trainer = SFTTrainer(
    model=model,
    args=training_config,
    train_dataset=train_dataset,
    peft_config=lora_config,
)

trainer.train()

trainer.save_model("/scratch/ks02450/lora-finetuned-code-adaptor/")