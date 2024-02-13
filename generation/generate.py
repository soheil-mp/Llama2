# generate.py

from model_utils import load_and_prepare_model
from tokenizers.tokenizer_utils import load_tokenizer
from transformers import TextStreamer
from config.model_config import model_name

def generate_response(user_prompt):
    model = load_and_prepare_model(model_name)
    tokenizer = load_tokenizer(model_name)

    prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{user_prompt.strip()}\n\n### Response:\n"
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda:0")
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    _ = model.generate(**inputs, streamer=streamer, max_new_tokens=500)

if __name__ == "__main__":
    generate_response("What are the three primary colors?")
