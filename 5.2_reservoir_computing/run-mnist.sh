#!/usr/bin/env bash

num_hidden_list=(2000 4000 8000 10000)
num_layer=1
win_scale=0.3
spectral_radius=1.3
leaky_rate=0.6
win_connectivity=0.1
wrec_connectivity=0.1

for num_hidden in ${num_hidden_list[@]};
  do
  python mnist-reservoir-force-training.py  \
        -num_hidden $num_hidden \
        -num_layer $num_layer \
        -win_scale $win_scale \
        -out_layers '-1' \
        -spectral_radius $spectral_radius \
        -leaky_rate $leaky_rate \
        -win_connectivity $win_connectivity \
        -wrec_connectivity $wrec_connectivity \
        -epoch 1 \
        -lr 0.1 \
        -comp_type 'jit-v1' \
        -gpu-id '0' \
        -save
  done

num_hidden_list=(20000 30000 40000 50000)
wrec_connectivity=0.01
for num_hidden in ${num_hidden_list[@]};
  do
  python mnist-reservoir-force-training.py  \
        -num_hidden $num_hidden \
        -num_layer $num_layer \
        -win_scale $win_scale \
        -out_layers '-1' \
        -spectral_radius $spectral_radius \
        -leaky_rate $leaky_rate \
        -win_connectivity $win_connectivity \
        -wrec_connectivity $wrec_connectivity \
        -epoch 1 \
        -lr 0.1 \
        -comp_type 'jit-v1' \
        -gpu-id '0' \
        -save
  done