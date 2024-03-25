#!/bin/bash
#SBATCH --job-name=soft-neg-ft
#SBATCH --output=slurm_logs/soft-neg-ftt.out
#SBATCH --error=slurm_logs/soft-neg-ft.err
#SBATCH --ntasks=1
#SBATCH -G a40:8
#SBATCH -p rl2-lab
#SBATCH --qos short
#SBATCH --cpus-per-task=8


accelerate --config_file configs/8_gpu.yaml launch soft_neg_finetuning.py \
 --pretrained_model_name_or_path "runwayml/stable-diffusion-v1-5" \
 --train_batch_size 4 \
 --gradient_accumulation_steps 4 \
 --output_dir 'sample_training_logs' \
 --checkpointing_steps 500 \
 --num_train_epochs 10 \
 --report_to 'wandb' \
 --enable_xformers_memory_efficient_attention \
 --checkpoints_total_limit 3 \