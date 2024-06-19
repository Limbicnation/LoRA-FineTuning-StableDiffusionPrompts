import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer
import wandb
from tqdm import tqdm
import yaml

# Load configuration from config.yaml
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Load tokens from environment variables
hf_token = os.getenv('HF_TOKEN')
wb_token = os.getenv('WANDB_API_KEY')
wandb.login(key=wb_token)

# Initialize a W&B run
try:
    wandb.init(project=config['wandb_project'], entity=config['wandb_entity'])
except wandb.errors.CommError as e:
    print(f"Error initializing W&B run: {e}")
    exit(1)

# Load dataset with error handling
try:
    dataset = load_dataset(config['dataset_name'], split="train")
    dataset = dataset.select(range(200))  # Use a subset for testing
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)

# Tokenizer with error handling
try:
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'], use_fast=True)
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    exit(1)

# Model to fine-tune
model_name = config['model_name']
new_model = config['new_model']

# LoRA configuration
peft_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['k_proj', 'gate_proj', 'v_proj', 'up_proj', 'q_proj', 'o_proj', 'down_proj']
)

# Load model with error handling
try:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        load_in_4bit=True
    )
    model.config.use_cache = False
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Prepare model for LoRA
try:
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
except Exception as e:
    print(f"Error preparing model for LoRA: {e}")
    exit(1)

# Training arguments using SFTConfig
training_arguments = SFTConfig(
    per_device_train_batch_size=config['training_args']['per_device_train_batch_size'],
    gradient_accumulation_steps=config['training_args']['gradient_accumulation_steps'],
    gradient_checkpointing=config['training_args']['gradient_checkpointing'],
    learning_rate=config['training_args']['learning_rate'],
    lr_scheduler_type=config['training_args']['lr_scheduler_type'],
    max_steps=config['training_args']['max_steps'],
    save_strategy=config['training_args']['save_strategy'],
    logging_steps=config['training_args']['logging_steps'],
    output_dir=config['training_args']['output_dir'],
    optim=config['training_args']['optim'],
    warmup_steps=config['training_args']['warmup_steps'],
    bf16=config['training_args']['bf16'],
    report_to=config['training_args']['report_to'],
    max_seq_length=1024  # Added max_seq_length to suppress the warning
)

# Initialize SFTTrainer with the corrected arguments
try:
    trainer = SFTTrainer(
        model=model,
        args=training_arguments,  # SFTConfig instance
        train_dataset=dataset,
        tokenizer=tokenizer,
    )
except Exception as e:
    print(f"Error initializing SFTTrainer: {e}")
    exit(1)

# Fine-tune model with SFT and a progress bar
try:
    progress_bar = tqdm(total=training_arguments.max_steps, desc="Training Progress", leave=True)
    for step, _ in enumerate(trainer.get_train_dataloader()):
        if step >= training_arguments.max_steps:
            break
        trainer.train_step()
        progress_bar.update(1)
    progress_bar.close()
except Exception as e:
    print(f"Error during training: {e}")
    exit(1)

# Save the fine-tuned model
try:
    trainer.save_model()
    tokenizer.save_pretrained(new_model)
except Exception as e:
    print(f"Error saving the model: {e}")
    exit(1)

# Log model as artifact
try:
    artifact = wandb.Artifact(new_model, type="model")
    artifact.add_dir(new_model)  # Assuming 'new_model' is the directory where your model is saved
    wandb.log_artifact(artifact)
except Exception as e:
    print(f"Error logging artifact: {e}")

# Finish the run
wandb.finish()

# Run evaluation if needed
# Note: Implement the evaluation phase if necessary
# evaluation_results = trainer.evaluate()
# print("Evaluation results:", evaluation_results)
