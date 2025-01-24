#!/bin/bash

cd ../..

echo "=> generate data"

cd data/mnist || exit 1
rm -rf all_data

python generate_data.py \
    --n_tasks 20 \
    --s_frac 0.2 \
    --test_tasks_frac 0.0 \
    --seed 12345

cd ../..

python train.py \
    mnist \
    --n_rounds 1000 \
    --aggregator_type decentralized \
    --dp_mechanism ldp \
    --epsilon 40 \
    --norm_clip 0.1 \
    --connectivity 1.0 \
    --bz 128 \
    --mbz 1 \
    --lr 0.1 \
    --weight_decay 5e-4 \
    --log_freq 1 \
    --device cpu \
    --optimizer sgd \
    --logs_dir logs/mnist/no_dp/n20/p1.0/epsilon10/seed12345 \
    --seed 12345 \
    --verbose 1
