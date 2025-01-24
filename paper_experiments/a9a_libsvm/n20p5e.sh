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
connectivities=(0.5)
epsilons=(3 5 7 10 15 20 30 40)
dp_mechanisms=("ldp" "pairwise" "mixing")
seeds=(12 123 1234 12345 57 1453 1927 1956 2011)

# Learning rate mapping
declare -A lr_dict
lr_dict["3_ldp"]=0.001
lr_dict["3_pairwise"]=0.001
lr_dict["3_mixing"]=0.001
lr_dict["5_ldp"]=0.001
lr_dict["5_pairwise"]=0.005
lr_dict["5_mixing"]=0.001
lr_dict["7_ldp"]=0.001
lr_dict["7_pairwise"]=0.005
lr_dict["7_mixing"]=0.005
lr_dict["10_ldp"]=0.005
lr_dict["10_pairwise"]=0.005
lr_dict["10_mixing"]=0.005
lr_dict["15_ldp"]=0.005
lr_dict["15_pairwise"]=0.01
lr_dict["15_mixing"]=0.01
lr_dict["20_ldp"]=0.005
lr_dict["20_pairwise"]=0.01
lr_dict["20_mixing"]=0.01
lr_dict["30_ldp"]=0.01
lr_dict["30_pairwise"]=0.01
lr_dict["30_mixing"]=0.01
lr_dict["40_ldp"]=0.01
lr_dict["40_pairwise"]=0.01
lr_dict["40_mixing"]=0.05

# Run experiments
for connectivity in "${connectivities[@]}"; do
  for epsilon in "${epsilons[@]}"; do
    for dp_mechanism in "${dp_mechanisms[@]}"; do
      for seed in "${seeds[@]}"; do
        # Get the learning rate from lr_dict
        key="${epsilon}_${dp_mechanism}"
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