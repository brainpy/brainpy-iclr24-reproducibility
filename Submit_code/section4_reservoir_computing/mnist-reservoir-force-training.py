# -*- coding: utf-8 -*-

import argparse
import math
import os
import socket
import sys

import brainpy as bp
import brainpy.math as bm

import brainpy_datasets as bd
import numpy as np
from tqdm import tqdm

from reservoir import JITDeepReservoir, DeepReservoir, SparseReservoir, \
                      JITDeepReservoir_scaling_SR, JITDeepReservoir_scaling_dual

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

root = r"./data"
train_dataset = bd.vision.FashionMNIST(root, split='train', download=True)
test_dataset = bd.vision.FashionMNIST(root, split='test', download=True)

# Standardize data
x_train = np.array(train_dataset.data, dtype=bm.float_)
x_train = x_train.reshape((-1, 28, 28)) / 255
y_train = np.array(train_dataset.targets, dtype=bm.int_)
x_test = np.array(test_dataset.data, dtype=bm.float_)
x_test = x_test.reshape((-1, 28, 28)) / 255
y_test = np.array(test_dataset.targets, dtype=bm.int_)


parser = argparse.ArgumentParser(description='Classify CIFAR')
parser.add_argument('-num_hidden', default=2000, type=int, help='simulating time-steps')
parser.add_argument('-num_layer', default=1, type=int, help='number of layers')
parser.add_argument('-win_scale', default=0.6, type=float)
parser.add_argument('-out_layers', default='-1', type=str)
parser.add_argument('-spectral_radius', default=1.3, type=float)
parser.add_argument('-spectral_radius_start', default=0.1, type=float)
parser.add_argument('-spectral_radius_end', default=1.0, type=float)
parser.add_argument('-leaky_start', default=0.9, type=float)
parser.add_argument('-leaky_end', default=0.1, type=float)
parser.add_argument('-leaky_rate', default=0.6, type=float)
parser.add_argument('-win_connectivity', default=0.1, type=float)
parser.add_argument('-wrec_connectivity', default=0.1, type=float)
parser.add_argument('-lr', default=0.1, type=float)
parser.add_argument('-train_stage', default='final_step', type=str)
parser.add_argument('-comp_type', default='dualjit-v1', type=str)
parser.add_argument('-epoch', default=5, type=int)
parser.add_argument('-gpu-id', default='0', type=str, help='gpu id')
parser.add_argument('-save', action='store_true')

args = parser.parse_args()
print(args.__dict__)

out_layers = [int(l) for l in args.out_layers.split(',')]

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

num_in = 28
num_out = 10

assert args.train_stage in ['final_step', 'all_steps']

if args.save:
  out_path = f'logs/force-mnist/{args.comp_type}/'
  out_path += f'leaky_start={args.leaky_start}-leaky_end={args.leaky_end}-'
  out_path += f'Iscale={args.win_scale}-SR_start={args.spectral_radius_start}-SR_end={args.spectral_radius_end}'
  out_path += f'-Cin={args.win_connectivity}-Crec={args.wrec_connectivity}-lr={args.lr}'
  out_path += f'-out_layers={args.out_layers}-layer={args.num_layer}-hidden-{args.num_hidden}.bp'

if args.comp_type.startswith('jit'):
  reservoir = JITDeepReservoir(
    num_in,
    args.num_hidden,
    args.num_layer,
    leaky_start=args.leaky_start,
    leaky_end=args.leaky_end,
    win_connectivity=args.win_connectivity,
    wrec_connectivity=args.wrec_connectivity,
    win_scale=args.win_scale,
    wrec_sigma=args.spectral_radius / math.sqrt(args.num_hidden * args.wrec_connectivity),
    mode=bm.batching_mode,
    jit_version=args.comp_type.split('-')[1]
  )
elif args.comp_type == 'sparse':
  reservoir = SparseReservoir(
    num_in,
    args.num_hidden,
    args.num_layer,
    leaky_start=args.leaky_start,
    leaky_end=args.leaky_end,
    win_connectivity=args.win_connectivity,
    wrec_connectivity=args.wrec_connectivity,
    win_scale=args.win_scale,
    wrec_sigma=args.spectral_radius / math.sqrt(args.num_hidden * args.wrec_connectivity),
    mode=bm.batching_mode,
  )
elif args.comp_type == 'dense':
  reservoir = DeepReservoir(
    num_in,
    args.num_hidden,
    args.num_layer,
    leaky_start=args.leaky_start,
    leaky_end=args.leaky_end,
    win_connectivity=args.win_connectivity,
    wrec_connectivity=args.wrec_connectivity,
    win_scale=args.win_scale,
    wrec_sigma=args.spectral_radius / math.sqrt(args.num_hidden * args.wrec_connectivity),
    mode=bm.batching_mode,
  )
else:
  raise ValueError

readout = bp.layers.Dense(args.num_hidden * len(out_layers), num_out, b_initializer=None, mode=bm.training_mode)
rls = bp.algorithms.RLS(alpha=args.lr)
rls.register_target(readout.num_in)


@bm.jit
@bm.to_object(child_objs=(reservoir, readout, rls))
def train_step(xs, y):
  reservoir.reset_state(1)
  y = bm.expand_dims(bm.one_hot(y, num_out, dtype=bm.float_), 0)
  if args.train_stage == 'final_step':
    for x in xs:
      o = reservoir(x)
    o = bm.concatenate([o[i] for i in out_layers], axis=1)
    pred = readout(o)
    dw = rls(y, o, pred)
    readout.W += dw
  elif args.train_stage == 'all_steps':
    for x in xs:
      o = reservoir(x)
      o = bm.concatenate([o[i] for i in out_layers], axis=1)
      pred = readout(o)
      dw = rls(y, o, pred)
      readout.W += dw
  else:
    raise ValueError


@bm.jit
@bm.to_object(child_objs=(reservoir, readout))
def predict(xs):
  reservoir.reset_state(xs.shape[0])
  for x in xs.transpose(1, 0, 2):
    o = reservoir(x)
  o = bm.concatenate([o[i] for i in out_layers], axis=1)
  y = readout(o)
  return bm.argmax(y, axis=1)


if args.save and os.path.exists(out_path):
  states = bp.checkpoints.load(out_path)
  acc_max = states['acc']
  print('Old accuracy ', acc_max)
else:
  acc_max = 0.


def train_one_epoch(epoch):
  global acc_max

  # # training
  for i in tqdm(range(x_train.shape[0]), desc='Training'):
    train_step(bm.asarray(x_train[i]), bm.asarray(y_train[i]))

  # verifying
  preds = []
  batch_size = 5
  for i in tqdm(range(0, x_train.shape[0], batch_size), desc='Verifying'):
    preds.append(predict(bm.asarray(x_train[i: i + batch_size])))
  preds = bm.concatenate(preds)
  train_acc = bm.mean(preds == bm.asarray(y_train, dtype=bm.int_))

  # prediction
  preds = []
  for i in tqdm(range(0, x_test.shape[0], batch_size), desc='Predicting'):
    preds.append(predict(bm.asarray(x_test[i: i + batch_size])))
  preds = bm.concatenate(preds)
  test_acc = bm.mean(preds == bm.asarray(y_test, dtype=bm.int_))
  print(f'Train accuracy {train_acc}, Test accuracy {test_acc}')

  if args.save and acc_max < test_acc:
    acc_max = test_acc
    bp.checkpoints.save(out_path,
                       {'reservoir': reservoir.state_dict(),
                        'readout': readout.state_dict(),
                        'args': args.__dict__,
                        'acc': float(test_acc)},
                        step=epoch,
                        overwrite=True)


if __name__ == '__main__':
  for ei in range(args.epoch):
    print(f'Epoch {ei} ...')
    train_one_epoch(ei)
