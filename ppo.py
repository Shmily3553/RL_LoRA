from peft import LoraConfig
from trl import AutoModelForCausalLMWithValueHead

model_id = "edbeeching/gpt-neo-125M-imdb"
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = AutoModelForCausalLMWithValueHead.from_pretrained(
    model_id, 
    peft_config=lora_config,
)