#!/bin/bash
# Running things


# Warm up of  10 epochs
python3 train.py --labeled_samples 4000  --lr 0.1 --epoch 10 --swa "False" --dataset_type "sym_noise_warmUp" \
--dropout 0.0 --DA "jitter" --experiment_name "WuP_model" --download "True" --network "WRN28_2_wn"

# SSL training
python3 train.py --labeled_samples 4000 --labeled_batch_size 64 --swa "False" --lr 0.1 --epoch 20  --initial_epoch 10 \
--dropout 0.0 --DA "jitter" --experiment_name "W_SOTA_CIFAR100" --download "True" --network "WRN28_2_wn"

# SSL training
python3 train.py --labeled_samples 4000 --labeled_batch_size 32 --swa "False" --lr 0.1 --epoch 30  --initial_epoch 20 \
--dropout 0.0 --DA "jitter" --experiment_name "W_SOTA_CIFAR100" --download "True" --network "WRN28_2_wn"

# SSL training
python3 train.py --labeled_samples 4000 --labeled_batch_size 16 --swa "False" --lr 0.1 --epoch 250 --initial_epoch 30 \
--dropout 0.0 --DA "jitter" --experiment_name "W_SOTA_CIFAR100" --download "True" --network "WRN28_2_wn"

# SSL training
python3 train.py --labeled_samples 4000 --labeled_batch_size 16 --swa "False" --lr 0.01 --epoch 31 --initial_epoch 250 \
--dropout 0.0 --DA "jitter" --experiment_name "W_SOTA_CIFAR100" --download "True" --network "WRN28_2_wn"

# SSL training
python3 train.py --labeled_samples 4000 --labeled_batch_size 16 --swa "True" --lr 0.001 --epoch 50 --initial_epoch 31 \
--dropout 0.0 --DA "jitter" --experiment_name "W_SOTA_CIFAR100" --download "True" --network "WRN28_2_wn"

