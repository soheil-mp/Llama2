# model_config.py

from transformers import BitsAndBytesConfig

# 4-bit Quantization Configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,
)

# Model and Dataset Names
model_name = "meta-llama/Llama-2-7b-hf"
dataset_name = "vicgalle/alpaca-gpt4"
new_model = "soheill/Llama-2-7b-hf"
