import torch
from transformers import AutoTokenizer, pipeline
from peft import LoraConfig
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model

# --- 1. Configuration and Setup ---

# PPO Configuration
ppo_config = PPOConfig(
    batch_size=1,            # Batch size for one PPO optimization step
    mini_batch_size=1,       # Mini-batch size within the PPO step
    learning_rate=1.41e-5,   # Learning rate
)

# Base model to fine-tune
model_name = "gpt2"
# PEFT (LoRA) Configuration
# This is where you define the LoRA parameters
lora_config = LoraConfig(
    r=16,                     # LoRA rank
    lora_alpha=32,            # LoRA alpha
    lora_dropout=0.1,         # Dropout
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["c_attn"], # Target GPT-2's attention layers
)

# --- 2. Load Model, Tokenizer, and Apply LoRA ---

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Set pad token to EOS token
tokenizer.pad_token = tokenizer.eos_token

# Load the model with TRL's AutoModelForCausalLMWithValueHead
# This class automatically adds a value head (for the PPO critic)
# and can directly accept a peft_config to apply LoRA.
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    model_name,
    device_map="auto",             # Automatically uses GPU if available
    peft_config=lora_config,       # Apply LoRA configuration here!
)

# Create a reference model (frozen) for PPOKL divergence
# TRL's `create_reference_model` handles PEFT models correctly
model_ref = create_reference_model(model)

print("Model loaded and LoRA applied.")
model.pretrained_model.print_trainable_parameters()
# Output will be something like:
# trainable params: 786,432 || all params: 125,228,800 || trainable%: 0.6279

# --- 3. Setup PPO Trainer ---

# Initialize the PPO trainer
# It will automatically handle the PEFT model and only train the adapters
ppo_trainer = PPOTrainer(
    ppo_config,
    model,
    model_ref,
    tokenizer
)

# --- 4. Setup Reward Model (Mockup) ---
# For a real task, you'd have a more sophisticated reward model.
# Here, we use a simple sentiment classifier to reward "positive" text.
device = ppo_trainer.device
sentiment_pipe = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=device
)

def get_reward(text):
    """Calculates a reward score for a piece of text using the sentiment model."""
    # The pipeline returns a list of dicts.
    # [{'label': 'POSITIVE', 'score': 0.9998}]
    result = sentiment_pipe(text, top_k=1, truncation=True)
    
    # We reward positive sentiment
    if result[0]['label'] == 'POSITIVE':
        return result[0]['score']
    else:
        # Penalize negative or neutral sentiment
        return 1.0 - result[0]['score']

print("Reward model (sentiment classifier) loaded.")

# --- 5. The PPO Training Loop ---

# This is a dummy "dataset" of prompts
prompts = [
    "I'm feeling really happy today, let's",
    "The best movie I've ever seen was",
    "I think the most beautiful place in the world is",
]

# Generation settings for the model
generation_kwargs = {
    "min_length": -1, # Avoids warnings
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 50, # Generate 50 new tokens
}

print("Starting PPO training...")
for epoch in range(2): # Run for 2 optimization steps
    for query_text in prompts:
        # Tokenize the prompt
        query_tensor = tokenizer(query_text, return_tensors="pt").input_ids.to(device)
        
        # 1. Generate a response from the model
        #    The PPO trainer's `generate` method handles tokenizing and generation
        #    It returns the full text (prompt + response)
        response_tensors = ppo_trainer.generate(
            query_tensor.squeeze(0), # Remove batch dim
            return_prompt=False,     # We only want the response
            **generation_kwargs,
        )
        
        # The response is just the new tokens
        response_text = tokenizer.decode(response_tensors[0])
        full_text = query_text + response_text
        print(f"Epoch {epoch} | Prompt: {query_text}")
        print(f"Epoch {epoch} | Response: {response_text}")

        # 2. Get a reward for the generated text
        #    The reward should be a scalar tensor
        reward_score = get_reward(full_text)
        rewards = [torch.tensor(reward_score, device=device)]
        print(f"Epoch {epoch} | Reward: {reward_score:.4f}")
        
        # 3. Perform the PPO optimization step
        #    The trainer calculates advantages and updates the model (LoRA weights)
        #    We pass the prompt, the response, and the reward
        stats = ppo_trainer.step(
            [query_tensor[0]],  # list of prompt tensors
            [response_tensors[0]], # list of response tensors
            rewards                # list of reward tensors
        )
        
        # Log stats
        print(f"Epoch {epoch} | PPO Stats: {stats.get('ppo/val/clip_reward')}")
        print("-" * 30)

print("Training finished.")

# --- 6. Save the trained LoRA adapters ---

# You only need to save the adapters, not the whole model
adapter_save_path = "./ppo_lora_adapters"
model.save_pretrained(adapter_save_path)
tokenizer.save_pretrained(adapter_save_path)

print(f"LoRA adapters saved to {adapter_save_path}")

# To load your trained model later:
# from peft import PeftModel, PeftConfig
# from transformers import AutoModelForCausalLM
#
# config = PeftConfig.from_pretrained(adapter_save_path)
# base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
# peft_model = PeftModel.from_pretrained(base_model, adapter_save_path)