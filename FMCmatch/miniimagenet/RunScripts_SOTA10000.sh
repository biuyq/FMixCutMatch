#!/bin/bash

# Warm up of  10 epochs
python3 train.py --labeled_samples 10000 --epoch 10 --dataset_type "ssl_warmUp" \
--DA "jitter" --experiment_name "WuP_model" --download "True"

# SSL training
python3 train.py --labeled_samples 10000 --labeled_batch_size 64 --swa "False" --lr 0.1 --epoch 20  --load_epoch 10 \
--DA "jitter" --experiment_name "M_SOTA_MINIIMAGENET" --download "True"

# SSL training
python3 train.py --labeled_samples 10000 --labeled_batch_size 32 --swa "False" --lr 0.1 --epoch 30  --load_epoch 20 \
 --DA "jitter" --experiment_name "M_SOTA_MINIIMAGENET" --download "True"

# SSL training
python3 train.py --labeled_samples 10000 --labeled_batch_size 16 --swa "False" --lr 0.1 --epoch 280 --M 250 --load_epoch 30 \
 --DA "jitter" --experiment_name "M_SOTA_MINIIMAGENET" --download "True"

# SSL training
python3 train.py --labeled_samples 10000 --labeled_batch_size 16 --swa "True" --lr 0.001 --epoch 50 --load_epoch 280 \
 --DA "jitter" --experiment_name "M_SOTA_MINIIMAGENET" --download "True"

