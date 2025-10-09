import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from contextlib import nullcontext



MODEL = "meta-llama/Llama-3.2-3B-Instruct"

# Check GPU availability
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# load the dataset

# TODO

# set up chat

tokenizer = AutoTokenizer.from_pretrained(MODEL)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    device_map="auto"
    # load_in_4bit=True
)

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


# preprocess the data
# def preprocess_data(example):
#     """An example I stole"""
#     
#     return tokenizer(
#         example["text"],  # Assumes your dataset has a 'text' column
#         truncation=True,
#         padding="max_length",
#         max_length=512
#     )
# 
# # more theft
# def format_data(example):
#     # Concatenate text for language modeling
#     example["input_ids"] = tokenizer.encode(example["text"], truncation=True, max_length=512)
#     example["labels"] = example["input_ids"].copy()
#     return example
# 
# formatted_dataset = tokenized_dataset.map(format_data, remove_columns=["text"])
# print(formatted_dataset["train"][0])
# 
# 
# train_test_split = tokenized_dataset["train"].train_test_split(test_size=0.1)
# train_data = train_test_split["train"]
# test_data = train_test_split["test"]
# print(f"Train samples: {len(train_data)}, Test samples: {len(test_data)}")
# 
# 
# 
# training_args = TrainingArguments(
#     output_dir="./llama3_finetuned", # Where to save the model
#     evaluation_strategy="steps",     # Evaluate during training
#     save_strategy="steps",           # Save checkpoints
#     learning_rate=2e-5,              # A good starting point for fine-tuning
#     per_device_train_batch_size=4,   # Adjust based on GPU memory
#     gradient_accumulation_steps=8,   # Simulates a larger batch size
#     num_train_epochs=3,              # Experiment with more epochs for small datasets
#     logging_steps=100,               # Log training progress
#     save_steps=500,                  # Save model every 500 steps
#     fp16=True,                       # Mixed precision for faster training
#     push_to_hub=False                # Don't push to Hugging Face Hub
# )
# 
# 
# trainer = Trainer(
#     model=model,                          # Pretrained LLaMA 3 model
#     args=training_args,                   # Training configurations
#     train_dataset=tokenized_dataset["train"],  # Your tokenized training data
#     eval_dataset=tokenized_dataset["test"]    # Your tokenized validation data
# )
# 
# # Start training
# trainer.train()


