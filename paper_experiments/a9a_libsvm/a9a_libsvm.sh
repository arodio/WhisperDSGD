cd ../..


echo "=> generate data"

cd data/a9a || exit 1
rm -rf all_data

python generate_data.py \
    --n_tasks 3 \
    --s_frac 1.0 \
    --test_tasks_frac 0.0 \
    --seed 12345

cd ../..

python train.py \
    a9a \
    --n_rounds 100 \
    --aggregator_type decentralized \
    --dp_mechanism ldp \
    --epsilon 10 \
    --norm_clip 0.05 \
    --connectivity 1.0 \
    --bz 128 \
    --lr 0.01 \
    --log_freq 1 \
    --device cpu \
    --optimizer sgd \
    --logs_dir logs/a9a_libsvm/ldp/n20/p1.0/epsilon10/seed_12345 \
    --seed 12345 \
    --verbose 1