python TIMIT-reservoir-force-training.py  \
        -num_hidden 1000 \
        -num_layer 3 \
        -win_scale 0.1 \
        -out_layers '-1' \
        -spectral_radius 0.6 \
        -leaky_start 0.9 \
        -leaky_end 0.5 \
        -win_connectivity 0.1 \
        -wrec_connectivity 0.01 \
        -epoch 10 \
        -lr 0.1 \
        -comp_type 'jit-v1' \
        -gpu-id '0' \
        -save

#num_hidden=1000
#num_layer=3
#win_scale=
#
#python TIMIT-reservoir-force-training.py  \
#        -num_hidden $num_hidden \
#        -num_layer $num_layer \
#        -win_scale $win_scale \
#        -out_layers '-1' \
#        -spectral_radius $spectral_radius \
#        -leaky_start $leaky_start \
#        -leaky_end $leaky_end \
#        -win_connectivity $win_connectivity \
#        -wrec_connectivity $wrec_connectivity \
#        -epoch 10 \
#        -lr 0.1 \
#        -comp_type 'jit-v1' \
#        -gpu-id '0' \
#        -save

