#!/bin/bash
#SBATCH --nodes=1 # Get one node
#SBATCH --cpus-per-task=4 # Two cores per task
#SBATCH --ntasks=1 # But only one task
#SBATCH --gres=gpu:2 # And two GPUs
#SBATCH --gres-flags=enforce-binding # Insist on good CPU/GPU alignment
#SBATCH --time=23:59:59 # Run for 1 day, at most
#SBATCH --job-name=PACL  # Name the job so I can see it in squeue
#SBATCH --mail-type=BEGIN,END,FAIL # Send me email for various states
#SBATCH --mail-user ma649596@ucf.edu # Use this address

# Load modules
module load anaconda/anaconda3

source /apps/anaconda/anaconda3/etc/profile.d/conda.sh

conda activate dinov2

export CUDA_VISIBLE_DEVICES=0,1
export MASTER_PORT=12802

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

echo "include-system-site-packages=false" >> $CONDA_PREFIX/pyvenv.cfg
export PYTHONPATH=${PYTHONPATH}:${HOME}/.local/lib/python3.9/site-packages

python dinov2/eval/linear.py \
    --config-file dinov2/configs/eval/vitg14_pretrain.yaml \
    --pretrained-weights https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_pretrain.pth \
    --train-dataset ImageNet:split=TRAIN:root=/datasets/ImageNet2012nonpub/train:extra=../imagenet_extras/ \
    --val-dataset ImageNet:split=VAL:root=/datasets/ImageNet2012nonpub/validation:extra=../imagenet_extras/