from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
model = PeftModel.from_pretrained(base_model, "/opt/shared/lora/Qwen/Qwen2.5-7B-Instruct-Finetuned/")
model = model.merge_and_unload()

model.save_pretrained("/opt/shared/lora/merged_model")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
tokenizer.save_pretrained("/opt/shared/lora/merged_model")
