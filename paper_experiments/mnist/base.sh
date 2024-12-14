cd ../..


echo "=> generate data"

alpha_intra=0.1
alpha_inter=100000000

cd data/mnist || exit 1
rm -rf all_data
#python generate_clustered_data.py \
#    --n_tasks 10 \
#    --by_labels_split \
#    --n_components -1 \
#    --n_clusters 2 \
#    --alpha_intra ${alpha_intra} \
#    --alpha_inter ${alpha_inter} ${alpha_inter} \
#    --s_frac 1.0 \
#    --clusters_proportions 0.8 0.2 \
#    --test_tasks_frac 0.0 \
#    --seed 12345
python generate_data.py \
    --n_tasks 10 \
    --pathological_split \
    --n_shards 1 \
    --s_frac 1.0 \
    --test_tasks_frac 0.0 \
    --seed 12345

python generate_data.py \
    --n_tasks 2 \
    --by_labels_split \
    --n_components -1 \
    --alpha 0.1 \
    --s_frac 1.0 \
    --test_tasks_frac 0.0 \
    --seed 12345

cd ../..

topologies=("ring" "complete")
downlink_strategies=("broadcast" "sampling")
sampling_rates=(0.2 0.4 0.6 0.8 1.0)
seeds=(12345 123 456 789 6789)
lr=0.001

for topology in "${topologies[@]}"; do
  for downlink_strategy in "${downlink_strategies[@]}"; do
    for sampling_rate in "${sampling_rates[@]}"; do
      for seed in "${seeds[@]}"; do

        echo "Running with topology=${topology}, downlink_strategy=${downlink_strategy}, sampling_rate=${sampling_rate}, seed=${seed}"

        python train.py \
          mnist \
          --n_rounds 50 \
          --downlink_strategy ${downlink_strategy} \
          --sampling_rate ${sampling_rate} \
          --topology ${topology} \
          --server_freq 2 \
          --clusters_proportions 0.8 0.2 \
          --bz 128 \
          --lr ${lr} \
          --log_freq 3 \
          --device cpu \
          --optimizer sgd \
          --logs_dir logs/mnist/semidec/intra_${alpha_intra}_inter_${alpha_inter}/${topology}/${downlink_strategy}/${sampling_rate}/lr_${lr}/seed_${seed} \
          --seed ${seed} \
          --verbose 1

      done
    done
  done
done
