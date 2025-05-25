import os
os.environ["TRITON_ALLOW_MMA"] = "0"
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
import re
import json
import time
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from trl import SFTTrainer
from vllm import LLM, SamplingParams
from datasets import load_dataset, Dataset, DatasetDict, load_from_disk
from sal.config import Config
from sal.models.reward_models import PRM
from sal.models.reward_models import RLHFFlow
from sal.utils.score import aggregate_scores
from peft import LoraConfig, get_peft_model, PeftModel
#from unsloth.chat_template import train_on_response_only
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AdamW, TrainingArguments, DataCollatorForLanguageModeling


#############################################################################################################
################################################ Parameters #################################################
#############################################################################################################
parser = argparse.ArgumentParser(description="Define hyperparameters for the model.")
parser.add_argument("--DEVICE",                 default="cuda:4",   help="Device Type")
parser.add_argument("--n_repetitive_sampleing", default=4,          help="Number of repetitive sampling")
parser.add_argument("--BATCH_SIZE",             default=4,          help="batch size per batch")
parser.add_argument("--gradient_accumu",        default=2,          help="Gradient Accumulation Step")
parser.add_argument("--MAX_LENGTH",             default=1024,       help="Maximum length in Tokenizer")
parser.add_argument("--LR",                     default=1e-7,       help="Learning Rate")
parser.add_argument("--EPOCHs",                 default=1,          help="Number of epochs")
parser.add_argument("--WARM_UP_RATIO",          default=0.1,        help="Ratio of total training steps used for a linear warmup")
parser.add_argument("--n_train_samples",        default=500,        help="Number of testing data")
parser.add_argument("--Score_Filter",           default=0.99,       help="score below which data is filtered")
parser.add_argument("--Lora_Rank",              default=128,        help="Lora Rank PArameter")
parser.add_argument("--Lora_alpha",             default=256,        help="Lora alpha parameter")
parser.add_argument("--Lora_flag",              default=True,       help="Whether apply the Lora or not")

args = parser.parse_args()

config = Config()
#############################################################################################################
########################################## Model and Data Loading ###########################################
#############################################################################################################
repo_name = "meta-llama/Llama-3.2-1B-Instruct"
base_model = AutoModelForCausalLM.from_pretrained(repo_name,
                                             torch_dtype=torch.float32,
                                             device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(repo_name)
tokenizer.pad_token = tokenizer.eos_token
print(base_model)
# Define LoRA configuration
lora_config = LoraConfig(
    r=args.Lora_Rank,  # Rank of the low-rank matrices
    lora_alpha=args.Lora_alpha,  # Scaling factor for LoRA weights
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"],  # Target modules to apply LoRA
    lora_dropout=0.1,  # Dropout rate for LoRA layers
    bias="none",  # Whether to add bias to LoRA layers
    task_type="CAUSAL_LM"  # Task type (causal language modeling)
)

if args.Lora_flag:
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
else:
    model = base_model
    model.requires_grad_(True)
    for name, param in model.named_parameters():
        if name.split(".")[2] != "weight":
            if (int(name.split(".")[2]) >= 15):
                param.requires_grad = True
            else:
                param.requires_grad = False


dataset = pd.read_json("./TTT_data/Best_of_" + str(args.n_repetitive_sampleing) + "_LLaMA-1B.json", orient='records')
# dataset = dataset[dataset['score'] > args.Score_Filter]
# args.n_train_samples = len(dataset['score'] > args.Score_Filter)

hf_dataset = Dataset.from_pandas(dataset)
hf_dataset.save_to_disk("./TTT_data/HF_Best_of_" + str(args.n_repetitive_sampleing) + "_LLaMA-1B")
dataset = load_from_disk("./TTT_data/HF_Best_of_" + str(args.n_repetitive_sampleing) + "_LLaMA-1B")


#############################################################################################################
################################################## Trainer ##################################################
#############################################################################################################
# Tokenize the data
def tokenize_function(example):
    tokens = tokenizer(example['solution'], padding="max_length", truncation=True, max_length=args.MAX_LENGTH)
    # Set padding token labels to -100 to ignore them in loss calculation
    tokens['labels'] = [
        -100 if token == tokenizer.pad_token_id else token for token in tokens['input_ids']
    ]
    return tokens


model.train()
train_args = TrainingArguments(
    output_dir="./training_result",  # Directory to save outputs
    per_device_train_batch_size=args.BATCH_SIZE,  # Batch size for training
    num_train_epochs=args.EPOCHs,  # Number of training epochs
    logging_dir="./logs",  # Directory for logs
    logging_strategy="steps",  # Log at regular intervals
    logging_steps=1,  # Log every 10 steps
    gradient_accumulation_steps=args.gradient_accumu,
    evaluation_strategy="no",  # No evaluation
    save_strategy="no",  # No checkpoint saving (optional)
    report_to="tensorboard",  # Log to TensorBoard (or "none" for console)
    disable_tqdm=False,  # Show progress bar
    remove_unused_columns=True,  # Remove unused dataset columns
    learning_rate=args.LR,  # Learning rate
    weight_decay=0.01,  # Weight decay
    warmup_ratio=args.WARM_UP_RATIO,  # Ratio of total training steps used for a linear warmup from 0 to `learning_rate`.
    adam_beta1=0.9,  # Set beta1
    adam_beta2=0.95,
)

tokenized_dataset = dataset.map(tokenize_function)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=tokenized_dataset,
    args=train_args,
)


trainer_stats = trainer.train()

if args.Lora_flag:
    trainer.save_model("./saved_models/Lora_with_Model/lora_adapter_instruct_" + str(args.n_repetitive_sampleing) + "/epoch_" + str(args.EPOCHs))
    tokenizer.save_pretrained("./saved_models/Lora_with_Model/lora_adapter_instruct_" + str(args.n_repetitive_sampleing) + "/epoch_" + str(args.EPOCHs))

    llama_peft_model = PeftModel.from_pretrained(base_model, "./saved_models/Lora_with_Model/lora_adapter_instruct_" + str(args.n_repetitive_sampleing) + "/epoch_" + str(args.EPOCHs))
    merged_model = llama_peft_model.merge_and_unload()
    merged_model.save_pretrained("./saved_models/Lora_with_Model/merged_lora_model_instruct_" + str(args.n_repetitive_sampleing) + "/epoch_" + str(args.EPOCHs))
    tokenizer.save_pretrained("./saved_models/Lora_with_Model/merged_lora_model_instruct_" + str(args.n_repetitive_sampleing) + "/epoch_" + str(args.EPOCHs))
else:
    model.save_pretrained("./saved_models/Model_Without_Lora/" + str(args.n_repetitive_sampleing) + "_Samples/epoch_" + str(args.EPOCHs) + "/")
    tokenizer.save_pretrained("./saved_models/Model_Without_Lora/" + str(args.n_repetitive_sampleing) + "_Samples/epoch_" + str(args.EPOCHs) + "/")



print("--------------------------------")
print("---------- Finished! -----------")
print("--------------------------------")
for arg, value in vars(args).items():
    print(f"{arg} ===> {value}")

