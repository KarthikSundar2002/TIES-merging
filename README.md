# Analysis of LoRA Adaptors Merging

Base Modal - Qwen3-4b

Test Set - GSM-8k Test Set
GSM-Specialist (LoRA trained on GSM-8k Training Set) - 79.76%
Code-Specialist ( LoRA trained on a subset on TokenBender Alpaca Dataset) - 38.13%
Naive Merge (Average of the specialist weights) - 46.78% 
Ties Merge
With 50% Pruning - 26.23%
With 30% Pruning - 29.34%
With 20% Pruning - 31.77%
with 10% Pruning - 28.51%

I don't understand why TIES Merge doesn't perform as well as Naive Merge. The authors of TIES merge tried it on fully finetuned models, and it worked great, but yeah, for LoRA adaptors it doesn't really work..