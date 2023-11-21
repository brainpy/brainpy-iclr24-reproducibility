#!/usr/bin/env bash

python kth-reservoir-force-training.py -num_hidden 2000 -win_connectivity 0.01 -wrec_connectivity 0.001 -train_start 10

python kth-reservoir-force-training.py -num_hidden 4000 -win_connectivity 0.01 -wrec_connectivity 0.001 -train_start 10

python kth-reservoir-force-training.py -num_hidden 8000 -win_connectivity 0.01 -wrec_connectivity 0.001 -train_start 10

python kth-reservoir-force-training.py -num_hidden 10000 -win_connectivity 0.01 -wrec_connectivity 0.0002 -train_start 10

python kth-reservoir-force-training.py -num_hidden 20000 -win_connectivity 0.01 -wrec_connectivity 0.0001 -train_start 10

python kth-reservoir-force-training.py -num_hidden 30000 -win_connectivity 0.005 -wrec_connectivity 0.0001 -train_start 10
