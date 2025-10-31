# -*- coding: utf-8 -*-

from datasets import load_dataset
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
import torch
import re
from math_verify import LatexExtractionConfig, parse, verify
from latex2sympy2_extended import NormalizationConfig
from trl import GRPOConfig, GRPOTrainer
from huggingface_hub import notebook_login

notebook_login()


dataset_id = 'lmms-lab/multimodal-open-r1-8k-verified'
train_dataset = load_dataset(dataset_id, split='train[:5%]')

model_name = "Qwen/Qwen3-VL-4B-Instruct" # "Qwen/Qwen3-VL-8B-Instruct"
processor = AutoProcessor.from_pretrained(model_name, padding_side="left")

SYSTEM_PROMPT = (
    "You are a helpful AI Assistant that provides well-reasoned and detailed responses. "
    "You first think about the reasoning process as an internal monologue and then provide the user with the answer. "
    "Respond in the following format: <think>\n...\n</think>\n<answer>\n...\n</answer>"
)

def make_conversation(example):
    conversation = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": example["image"]},
                {"type": "text", "text": example["problem"]},
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    return {
        "prompt": prompt,
        "image": example["image"],
    }

train_dataset = train_dataset.map(make_conversation)
train_dataset = train_dataset.remove_columns(['problem', 'original_question', 'original_answer'])

model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16, # MODIFICATION: Use float16 for full fine-tuning
    device_map="auto",
)

# MODIFICATION: Explicitly set peft_config to None to signal full fine-tuning
peft_config = None

def format_reward(completions, **kwargs):
    """Reward function that checks if the reasoning process is enclosed within <think> and </think> tags, while the final answer is enclosed within <answer> and </answer> tags."""
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
    return [1.0 if match else 0.0 for match in matches]

def len_reward(completions, solution, **kwargs) -> float:
    """Compute length-based rewards to discourage overthinking and promote token efficiency.

    Taken from the Kimi 1.5 tech report: https://huggingface.co/papers/2501.12599

    Args:
        completions: List of model completions
        solution: List of ground truth solutions

    Returns:
        List of rewards where:
        - For correct answers: reward = 0.5 - (len - min_len)/(max_len - min_len)
        - For incorrect answers: reward = min(0, 0.5 - (len - min_len)/(max_len - min_len))
    """
    contents = completions

    # First check correctness of answers
    correctness = []
    for content, sol in zip(contents, solution):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        if len(gold_parsed) == 0:
            # Skip unparseable examples
            correctness.append(True)  # Treat as correct to avoid penalizing
            print("Failed to parse gold solution: ", sol)
            continue

        answer_parsed = parse(
            content,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed=True,
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
        correctness.append(verify(answer_parsed, gold_parsed))

    # Calculate lengths
    lengths = [len(content) for content in contents]
    min_len = min(lengths)
    max_len = max(lengths)

    # If all responses have the same length, return zero rewards
    if max_len == min_len:
        return [0.0] * len(completions)

    rewards = []
    for length, is_correct in zip(lengths, correctness):
        lambda_val = 0.5 - (length - min_len) / (max_len - min_len)

        if is_correct:
            reward = lambda_val
        else:
            reward = min(0, lambda_val)

        rewards.append(float(reward))

    return rewards

output_dir = "/scratch/yaruic/RL/Qwen3-VL-4B-Instruct-trl-grpo-nolora"

# Configure training arguments using GRPOConfig
training_args = GRPOConfig(
    learning_rate=2e-5,
    #num_train_epochs=1,
    max_steps=100,                                        # Number of dataset passes. For full trainings, use `num_train_epochs` instead

    # Parameters that control the data preprocessing
    # MODIFICATION: Full fine-tuning requires much more VRAM.
    # Reduced batch size to 1 and added gradient accumulation to prevent OOM errors.
    per_device_train_batch_size=1,                        # MODIFICATION: Reduced from 2
    gradient_accumulation_steps=2,                        # MODIFICATION: Added
    max_completion_length=1024, # default: 256            # Max completion length produced during training
    num_generations=2, # 2, # default: 8                  # Number of generations produced during trainig for comparison
    max_prompt_length=2048, # default: 512                # Max prompt lenght of the input prompt used for generation during training

    fp16=False, # Disable fp16
    bf16=True,  # Enable bf16

    # Parameters related to reporting and saving
    output_dir=output_dir,                                # Where to save model checkpoints and logs
    logging_steps=1,                                      # Log training metrics every N steps
    report_to="none",                                  # Experiment tracking tool

    # Hub integration
    push_to_hub=False,
    log_completions=False
)

"""Configure the GRPO Trainer. We pass the previously configured `training_args`. We don't use eval dataset to maintain memory usage low but you can configure it."""

trainer = GRPOTrainer(
    model=model,
    reward_funcs=[format_reward, len_reward],
    args=training_args,
    train_dataset=train_dataset,
    peft_config=peft_config, # MODIFICATION: This is now None, enabling full fine-tuning
)

"""Show memory stats before training"""

gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)

print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()

"""Show memory stats after training"""

used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)

print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

trainer.save_model(output_dir)
# trainer.push_to_hub(dataset_name=dataset_id)

## Load the fine-tuned model and run inference

base_model = model_name
# MODIFICATION: This path now points to the full fine-tuned model
finetuned_model_path = f"{output_dir}" # Replace with your HF username or organization

# MODIFICATION: Load the fully fine-tuned model directly from the output directory
model = Qwen3VLForConditionalGeneration.from_pretrained(
    finetuned_model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# MODIFICATION: Load the processor from the fine-tuned model directory
processor = AutoProcessor.from_pretrained(finetuned_model_path)

dataset_id = 'lmms-lab/multimodal-open-r1-8k-verified'
train_dataset = load_dataset(dataset_id, split='train[:5%]')

problem = train_dataset[0]['problem']
image = train_dataset[0]['image']

messages = [
    {
        "role": "system", "content": [
            {"type": "text", "text": SYSTEM_PROMPT}
        ]
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": problem},
        ],
    },
]

messages

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device)

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=500)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)