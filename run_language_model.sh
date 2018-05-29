#!/bin/csh

for i in "AdamOptimizer" "RMSPropOptimizer" "MomentumOptimizer" "GradientDescentOptimizer"
do
    echo $i
    echo "training ... "
    python3 language_model.py \
        --data_path="../data/ptb" \
        --save_path="media/root/c5d1565a-6700-4077-be5d-e3c95fba4ae6/language_model/start_RNN/testdir_"$i"/" \
        --run_mode="training" \
        --optimizer=$i

    echo "testing ... "
    python3 language_model.py \
        --data_path="../data/ptb" \
        --save_path="media/root/c5d1565a-6700-4077-be5d-e3c95fba4ae6/language_model/start_RNN/testdir_"$i"/" \
        --run_mode="testing" \
        --optimizer=$i

done
