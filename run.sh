#!/bin/bash

#SBATCH --job-name grpo_lora
#SBATCH --nodes 1
#SBATCH --tasks-per-node 1
#SBATCH --cpus-per-task 16
#SBATCH --mem 100gb
#SBATCH --time 10:00:00
#SBATCH --gpus-per-node h100:1
#SBATCH --mail-type ALL

module load anaconda3
source activate rl
module load cuda/12.3.0

srun python grpo_qwen3_vl.py
# nohup python grpo_qwen3_nolora.py > out_nolora.txt 2>&1 &
# nohup python grpo_qwen3_vl.py > out_lora.txt 2>&1 &