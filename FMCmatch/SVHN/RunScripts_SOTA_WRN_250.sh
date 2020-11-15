#!/bin/bash
# Running things


# Warm up of  10 epochs
python3 train.py --labeled_samples 250  --labeled_batch_size 100 --lr 0.1 --epoch 150 --swa "False" --dataset_type "sym_noise_warmUp" \
--dropout 0.0 --DA "jitter" --experiment_name "WuP_model" --download "True" --network "WRN28_2_wn"

# SSL training
python3 train.py --labeled_samples 250 --labeled_batch_size 32 --swa "False" --lr 0.1 --epoch 30 --initial_epoch 150 \
--dropout 0.0 --DA "jitter" --experiment_name "W_SOTA_SVHN" --download "True"  --network "WRN28_2_wn"

# SSL training
python3 train.py --labeled_samples 250 --labeled_batch_size 16 --swa "False" --lr 0.1 --epoch 150 --initial_epoch 30 \
--dropout 0.0 --DA "jitter" --experiment_name "W_SOTA_SVHN" --download "True"  --network "WRN28_2_wn"

# SSL training
python3 train.py --labeled_samples 250 --labeled_batch_size 16 --swa "False" --lr 0.01 --epoch 30 --initial_epoch 150 \
--dropout 0.0 --DA "jitter" --experiment_name "W_SOTA_SVHN" --download "True"  --network "WRN28_2_wn"

# SSL training
python3 train.py --labeled_samples 250 --labeled_batch_size 16 --swa "False" --lr 0.001 --epoch 50 --initial_epoch 30 \
--dropout 0.0 --DA "jitter" --experiment_name "W_SOTA_SVHN" --download "True" --network "WRN28_2_wn"
