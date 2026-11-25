#!/bin/bash
#SBATCH --job-name=mig_normal
#SBATCH --output=logs/normal_%j.out
#SBATCH --error=logs/normal_%j.err
#SBATCH --account=uit
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:2g.35gb:1

mkdir -p logs
nvidia-smi
