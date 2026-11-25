#!/bin/bash
#SBATCH --job-name=mig_small
#SBATCH --output=logs/small_%j.out
#SBATCH --error=logs/small_%j.err
#SBATCH --account=uit
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1g.18gb:1

mkdir -p logs
nvidia-smi
# source /datastore/$USER/venv/bin/activate
# python3 /datastore/$USER/projects/train.py
