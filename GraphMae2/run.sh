#!/bin/bash

conda activate graphmae

# for training
nohup python main_large.py \
--dataset social \
--encoder gat \
--decoder gat \
--max_epoch 10 \
--batch_size 4096 \
--seed 0 \
--device 1 \
--config configs/social.yaml \
--data_dir /workspace/data/pretrain  > output.log 2>&1 &
# ps aux | grep "python main_large.py" | awk '{print $2}' | xargs kill -9

# for finetuning
python main_finetune.py \
--dataset rumours_eval \
--max_epoch_f 10 \
--batch_size_f 8 \
--seed 0 \
--device 0 \
--lr 0.01 \
--data_dir /workspace/data/finetune \
--pretrain_path checkpoints/gat_gat_512_2_social_0.5_512_checkpoint.pt

# for finetuning
python main_finetune.py \
--dataset mt_csd \
--max_epoch_f 10 \
--batch_size_f 8 \
--seed 0 \
--device 0 \
--lr 0.01 \
--data_dir /workspace/data/finetune \
--pretrain_path checkpoints/gat_gat_512_2_social_0.5_512_checkpoint.pt


# for infer
python infer.py
