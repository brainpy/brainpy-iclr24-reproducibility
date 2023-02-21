#!/usr/bin/env bash

num_hidden_list=(30000 40000 50000)
num_hidden=1000
num_layer=1
win_scale=0.3
spectral_radius=1.3
#leaky_start_list=(0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1)
leaky_start=0.6
leaky_end=0.1
win_connectivity=0.1
wrec_connectivity=0.01

#python mnist-reservoir-force-training.py  \
#        -num_hidden $num_hidden \
#        -num_layer $num_layer \
#        -win_scale $win_scale \
#        -out_layers '-1' \
#        -spectral_radius $spectral_radius \
#        -leaky_start $leaky_start \
#        -leaky_end $leaky_end \
#        -win_connectivity $win_connectivity \
#        -wrec_connectivity $wrec_connectivity \
#        -epoch 5 \
#        -lr 0.1 \
#        -comp_type 'jit-v1' \
#        -gpu-id '0' \
#        -save

for num_hidden in ${num_hidden_list[@]};
  do
  python mnist-reservoir-force-training.py  \
          -num_hidden $num_hidden \
          -num_layer $num_layer \
          -win_scale $win_scale \
          -out_layers '-1' \
          -spectral_radius $spectral_radius \
          -leaky_start $leaky_start \
          -leaky_end $leaky_end \
          -win_connectivity $win_connectivity \
          -wrec_connectivity $wrec_connectivity \
          -epoch 5 \
          -lr 0.1 \
          -comp_type 'jit-v1' \
          -gpu-id '0' \
          -save
  done