import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from contextlib import nullcontext
from peft import LoraConfig, get_peft_model, TaskType



MODEL = "meta-llama/Llama-3.2-3B-Instruct"

# Check GPU availability
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# load the dataset

# TODO

# set up chat

tokenizer = AutoTokenizer.from_pretrained(MODEL)
# Some tokenisers don't have a pad token, so we set it to eos_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load model
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    device_map="auto"
    # load_in_4bit=True
)

# Wrap the base model with a LoRA adapter (PEFT)
lora_cfg = LoraConfig(
    task_type=TaskType.CAUSAL_LM,  # causal LM objective
    r=16,                          # LoRA rank (start small; bump based on our GPU Headroom)
    lora_alpha=32,                 # scaling
    lora_dropout=0.05,             # regularization
    target_modules=[
        "q_proj","k_proj","v_proj","o_proj",
        "gate_proj","up_proj","down_proj"
    ]  # common Llama MLP/attn proj layers
)
#return a trainable LoRa-adapted model while freezing the base model weights
model = get_peft_model(model, lora_cfg)


def gen_prompt(tokenizer, sentence):
    converted_sample = [{"role": "user", "content": sentence}]
    prompt = tokenizer.apply_chat_template(
        converted_sample, tokenize=False, add_generation_prompt=True
    )
    return prompt

print("chat > ", end="")
sentence = input()

prompt = gen_prompt(tokenizer, sentence)
print(prompt)


def generate(model, tokenizer, prompt, max_new_tokens=64, skip_special_tokens=False):
    tokenized_input = tokenizer(
        prompt, add_special_tokens=False, return_tensors="pt"
    ).to(model.device)

    model.eval()
    # if it was trained using mixed precision, uses autocast context
    ctx = torch.autocast(device_type=model.device.type, dtype=model.dtype) \
          if model.dtype in [torch.float16, torch.bfloat16] else nullcontext()
    with ctx:  
        gen_output = model.generate(**tokenized_input,
                                    eos_token_id=tokenizer.eos_token_id,
                                    max_new_tokens=max_new_tokens)
    
    output = tokenizer.batch_decode(gen_output, skip_special_tokens=skip_special_tokens)
    return output[0]

print(generate(model, tokenizer, prompt))


# #preprocess the data
# def preprocess_data(example):
#     """An example I stole"""
    
#     return tokenizer(
#         example["text"],  # Assumes your dataset has a 'text' column
#         truncation=True,
#         padding="max_length",
#         max_length=512
#     )

# # more theft
# def format_data(example):
#     # Concatenate text for language modeling
#     example["input_ids"] = tokenizer.encode(example["text"], truncation=True, max_length=512)
#     example["labels"] = example["input_ids"].copy()
#     return example

# formatted_dataset = tokenized_dataset.map(format_data, remove_columns=["text"])
# print(formatted_dataset["train"][0])


# train_test_split = tokenized_dataset["train"].train_test_split(test_size=0.1)
# train_data = train_test_split["train"]
# test_data = train_test_split["test"]
# print(f"Train samples: {len(train_data)}, Test samples: {len(test_data)}")


# # these should be typical LoRA-friendly hyperparameters (but adjust once we start training)
# training_args = TrainingArguments(
#     output_dir="./llama3_finetuned_lora_out", # Where to save the model
#     evaluation_strategy="steps",     # Evaluate during training
#     save_strategy="steps",           # Save checkpoints
#     learning_rate=2e-5,              # A good starting point for fine-tuning (Note: LoRa adapters often require higher LR than FT)
#     per_device_train_batch_size=4,   # Adjust based on GPU memory
#     gradient_accumulation_steps=8,   # Simulates a larger batch size
#     num_train_epochs=3,              # Experiment with more epochs for small datasets
#     logging_steps=100,               # Log training progress
#     save_steps=500,                  # Save model every 500 steps
#     fp16=True,                       # Mixed precision for faster training
#     push_to_hub=False,                # Don't push to Hugging Face Hub
#     bf16=torch.cuda.is_available(),   # bfloat16 if GPU supports it
# )

# # ensure batches are built correctly for causal LM
# data_collator = DataCollatorForLanguageModeling(
#     tokenizer=tokenizer,
#     mlm=False  # causal LM, not masked LM
# )

# trainer = Trainer(
#     model=model,                          # Pretrained LLaMA 3 model, !!! but LoRa-wrapped model !!!
#     args=training_args,                   # Training configurations
#     train_dataset=tokenized_dataset["train"],  # Your tokenized training data
#     eval_dataset=tokenized_dataset["test"]    # Your tokenized validation data
#     data_collator=data_collator           # Ensures proper labels/padding for cuasal LM
# )

# # Start training
# trainer.train()

# model.save_pretrained("./llama32_lora_adapter")  # Save only the LoRA adapter weights
# tokenizer.save_pretrained("./llama32_lora_adapter") # Save tokenizer
# print("Saved LoRA adapter to ./llama32_lora_adapter")


