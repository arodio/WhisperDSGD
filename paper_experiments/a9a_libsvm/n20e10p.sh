#!/bin/bash

cd ../.. || exit 1

echo "=> generate data"
cd data/a9a || exit 1
rm -rf all_data

python generate_data.py \
    --n_tasks 20 \
    --by_labels_split \
    --n_components -1 \
    --alpha 10 \
    --s_frac 1.0 \
    --test_tasks_frac 0.0 \
    --seed 12345

cd ../..

# Define variables
n_tasks=20
n_rounds=5000
epsilons=(10)
connectivities=(0.2 0.4 0.6 0.8 1.0)
dp_mechanisms=("ldp" "pairwise" "mixing")
seeds=(12 123 1234 12345 57 1453 1927 1956 2011)

# Learning rate mapping
declare -A lr_dict
lr_dict["0.2_ldp"]=0.005
lr_dict["0.2_pairwise"]=0.005
lr_dict["0.2_mixing"]=0.005
lr_dict["0.4_ldp"]=0.005
lr_dict["0.4_pairwise"]=0.005
lr_dict["0.4_mixing"]=0.005
lr_dict["0.6_ldp"]=0.005
lr_dict["0.6_pairwise"]=0.005
lr_dict["0.6_mixing"]=0.005
lr_dict["0.8_ldp"]=0.005
lr_dict["0.8_pairwise"]=0.005
lr_dict["0.8_mixing"]=0.01
lr_dict["1.0_ldp"]=0.005
lr_dict["1.0_pairwise"]=0.01
lr_dict["1.0_mixing"]=0.01

# Run experiments
for connectivity in "${connectivities[@]}"; do
  for epsilon in "${epsilons[@]}"; do
    for dp_mechanism in "${dp_mechanisms[@]}"; do
      for seed in "${seeds[@]}"; do
        # Get the learning rate from lr_dict
        key="${connectivity}_${dp_mechanism}"
        lr=${lr_dict[$key]}

        # Logs directory
        logs_dir="logs/a9a_libsvm/n${n_tasks}/p${connectivity}/e${epsilon}/${dp_mechanism}/lr${lr}/s${seed}"
        echo "epsilon=${epsilon}, connectivity=${connectivity}, ${dp_mechanism}, lr=${lr}, seed=${seed}"

        # Run the training script
        python train.py \
            a9a \
            --n_rounds "$n_rounds" \
            --aggregator_type decentralized \
            --dp_mechanism "$dp_mechanism" \
            --epsilon "$epsilon" \
            --norm_clip 0.1 \
            --connectivity "$connectivity" \
            --bz 128 \
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