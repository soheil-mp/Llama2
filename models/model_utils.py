# model_utils.py

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import prepare_model_for_kbit_training, LoraConfig, PeftModel
import torch

def load_and_prepare_model(model_name, bnb_config):
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config)
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    return model

def load_base_and_peft_model(model_name, new_model):
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        low_cpu_mem_usage=True, 
        return_dict=True, 
        torch_dtype=torch.float16, 
        device_map= {"": 0})
    model = PeftModel.from_pretrained(base_model, new_model)
    model = model.merge_and_unload()
    return model

def push_to_hub(model, new_model, tokenizer):
    model.push_to_hub(new_model)
    tokenizer.push_to_hub(new_model)
