#!/bin/bash

cd ../.. || exit 1

echo "=> generate data"
cd data/mnist || exit 1
rm -rf all_data

python generate_data.py \
    --n_tasks 20 \
    --by_labels_split \
    --n_components -1 \
    --alpha 10 \
    --s_frac 0.2 \
    --test_tasks_frac 0.0 \
    --seed 12345

cd ../..

# Define variables
n_tasks=20
n_rounds=1000
epsilons=(10 20 30 40)
lrs=(0.1 0.05 0.01 0.005 0.001)
connectivities=(0.5)
dp_mechanisms=("ldp" "pairwise" "mixing")
seeds=(12345)

# Run experiments
for connectivity in "${connectivities[@]}"; do
  for epsilon in "${epsilons[@]}"; do
    for dp_mechanism in "${dp_mechanisms[@]}"; do
      for lr in "${lrs[@]}"; do
        for seed in "${seeds[@]}"; do
          logs_dir="logs/mnist/n${n_tasks}/p${connectivity}/e${epsilon}/${dp_mechanism}/lr${lr}/s${seed}"
          echo "epsilon=${epsilon}, connectivity=${connectivity}, ${dp_mechanism}, lr=${lr}, seed=${seed}"
          python train.py \
              mnist \
              --n_rounds "$n_rounds" \
              --aggregator_type decentralized \
              --dp_mechanism "$dp_mechanism" \
              --epsilon "$epsilon" \
              --norm_clip 0.1 \
              --connectivity "$connectivity" \
              --bz 128 \
              --mbz 1 \
              --lr "$lr" \
              --log_freq 1 \
              --device cpu \
              --optimizer sgd \
              --logs_dir "$logs_dir" \
              --seed "$seed" \
              --verbose 0
        done
      done
    done
  done
done