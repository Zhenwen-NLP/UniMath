#!/bin/bash

#--load ../snap/baseline/BEST
#'google/flan-t5-base'
output=../snap/geo_t5

python -m torch.distributed.launch --nproc_per_node 1 --master_port 2021 ../unified/mwp_no_mapping_cot_train.py \
        --distributed --multiGPU \
        --train proving_train \
        --valid proving_val \
        --test proving_test \
        --optim adamw \
        --warmup_ratio 0.1 \
        --clip_grad_norm 5 \
        --lr 3e-4 \
        --batch_size 12 \
        --epochs 80 \
        --num_workers 8 \
        --backbone 't5-base' \
        --output $output ${@:2} \
        --num_beams 1 \
        --max_text_length 200 \
        --gen_max_length 40 \
        --weight_decay 1e-2 \
        --input_dropout 0.1 \
        --hidden_dropout 0.1 \
        --local_rank 0