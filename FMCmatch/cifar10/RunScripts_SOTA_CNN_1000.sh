#!/bin/bash
# Running things


# Warm up of  10 epochs
python3 train.py --labeled_samples 1000  --lr 0.1 --epoch 10 --swa "False" --dataset_type "sym_noise_warmUp" \
--dropout 0.2 --DA "jitter" --experiment_name "WuP_model" --download "True" --network "MT_Net"

# SSL training
python3 train.py --labeled_samples 1000 --labeled_batch_size 64 --swa "False" --lr 0.1 --epoch 20  --initial_epoch 10 \
--dropout 0.2 --DA "jitter" --experiment_name "M_SOTA_CIFAR10" --download "True" --network "MT_Net"

# SSL training
python3 train.py --labeled_samples 1000 --labeled_batch_size 32 --swa "False" --lr 0.1 --epoch 30  --initial_epoch 20 \
--dropout 0.2 --DA "jitter" --experiment_name "M_SOTA_CIFAR10" --download "True" --network "MT_Net"

# SSL training
python3 train.py --labeled_samples 1000 --labeled_batch_size 16 --swa "False" --lr 0.1 --epoch 280 --M 250 --initial_epoch 30 \
--dropout 0.2 --DA "jitter" --experiment_name "M_SOTA_CIFAR10" --download "True" --network "MT_Net"

# SSL training
python3 train.py --labeled_samples 1000 --labeled_batch_size 16 --swa "True" --lr 0.001 --epoch 50 --initial_epoch 280 \
--dropout 0.2 --DA "jitter" --experiment_name "M_SOTA_CIFAR10" --download "True" --network "MT_Net"

