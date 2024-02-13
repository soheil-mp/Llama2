# train.py

from model_utils import load_and_prepare_model
from transformers import TrainingArguments
from trl import SFTTrainer
from data.load_data import load_and_display_dataset
from config.model_config import model_name, dataset_name, new_model, bnb_config
import wandb

# Weights & Biases Integration
wandb.login(key="your_wandb_api_key")
run = wandb.init(project='Fine tuning llama-2-7B', job_type="training", anonymous="allow")

def train_model():
    model = load_and_prepare_model(model_name, bnb_config)
    tokenizer = load_tokenizer(model_name)
    dataset = load_and_display_dataset(dataset_name, "train[:10000]")

    peft_config = LoraConfig(lora_alpha= 8, lora_dropout= 0.1, r= 16, bias="none", task_type="CAUSAL_LM", target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj"])
    
    training_arguments = TrainingArguments(output_dir="./results", num_train_epochs=1, per_device_train_batch_size=4, gradient_accumulation_steps=2, optim="paged_adamw_8bit", save_steps=1000, logging_steps=100, learning_rate=2e-4, weight_decay=0.001, fp16=True, max_grad_norm=0.3, warmup_ratio=0.1, group_by_length=True, lr_scheduler_type="linear", report_to="wandb", load_best_model_at_end=True, evaluation_strategy="steps", eval_steps=500)

    trainer = SFTTrainer(model=model, train_dataset=dataset, peft_config=peft_config, max_seq_length=512, dataset_text_field="text", tokenizer=tokenizer, args=training_arguments, packing=True)
    trainer.train()

    # Save and finish W&B run
    trainer.model.save_pretrained(new_model)
    wandb.finish()

if __name__ == "__main__":
    train_model()
