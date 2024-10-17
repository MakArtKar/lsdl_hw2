#!/bin/bash

python -m src.train trainer=gpu trainer.max_epochs=100 logger=wandb experiment=rotation_pretrain run_name=$1_pretrain +logger.wandb.name=$1_pretrain logger.wandb.group=$1 num_angles=$2 && \
python -m src.train trainer=gpu trainer.max_epochs=100 logger=wandb experiment=fine_tune ckpt_path=logs/train/runs/$1_pretrain/checkpoints/best.ckpt run_name=$1_fine_tune +logger.wandb.name=$1_fine_tune logger.wandb.group=$1