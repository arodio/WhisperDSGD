#cd ../..
#
#
#echo "=> generate data"
#
#cd data/titanic || exit 1
#rm -rf all_data
#
#python generate_data.py \
#    --n_tasks 3 \
#    --s_frac 1.0 \
#    --test_tasks_frac 0.0 \
#    --seed 12345

cd ../..

python train.py \
    titanic \
    --n_rounds 100 \
    --aggregator_type centralized \
    --epsilon 10 \
    --norm_clip 100 \
    --connectivity 1.0 \
    --bz 128 \
    --lr 0.1 \
    --weight_decay 5e-4 \
    --log_freq 1 \
    --device cpu \
    --optimizer sgd \
    --logs_dir logs/titanic/no_dp/n20/p1.0/epsilon10/seed12345 \
    --seed 12345 \
    --verbose 1