# -*- coding: utf-8 -*-

"""
"""

import sys
import argparse
import functools
import math
import os
import socket
import pickle

import brainpy as bp
import brainpy.math as bm
import hdf5storage
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

# dataset
# import deeplake

# preprocessing
# import torchaudio.transforms as T

from reservoir import JITDeepReservoir, DeepReservoir

parser = argparse.ArgumentParser(description='Classify TIMIT')
parser.add_argument('-num_hidden', default=10000, type=int, help='simulating time-steps')
parser.add_argument('-num_layer', default=3, type=int, help='number of layer')
parser.add_argument('-win_scale', default=0.6, type=float)
parser.add_argument('-out_layers', default='-1', type=str)
parser.add_argument('-spectral_radius', default=0.4, type=float)
parser.add_argument('-leaky_start', default=0.9, type=float)
parser.add_argument('-leaky_end', default=0.1, type=float)
parser.add_argument('-win_connectivity', default=0.1, type=float)
parser.add_argument('-wrec_connectivity', default=0.01, type=float)
parser.add_argument('-epoch', default=10, type=int)
parser.add_argument('-lr', default=0.1, type=float)
parser.add_argument('-comp_type', default='jit-v1', type=str)
parser.add_argument('-gpu-id', default='0', type=str, help='gpu id')
parser.add_argument('-save', action='store_true')

args = parser.parse_args()
print(args)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

# DIR route
TFRECORD_DIR = './data/mfcc'

# Data prepare
train_data = pickle.load(open(os.path.join(TFRECORD_DIR, 'train_data.pkl'), 'rb'))
train_labels = pickle.load(open(os.path.join(TFRECORD_DIR, 'train_labels.pkl'), 'rb'))
dev_data = pickle.load(open(os.path.join(TFRECORD_DIR, 'dev_data.pkl'), 'rb'))
dev_labels = pickle.load(open(os.path.join(TFRECORD_DIR, 'dev_labels.pkl'), 'rb'))
test_data = pickle.load(open(os.path.join(TFRECORD_DIR, 'test_data.pkl'), 'rb'))
test_labels = pickle.load(open(os.path.join(TFRECORD_DIR, 'test_labels.pkl'), 'rb'))


def format_data(T, shuffle: bool = False):
  xs, ys = T
  resets = []
  for x in xs:
    reset = np.zeros((x.shape[0]), dtype=bool)
    reset[0] = True
    resets.append(reset)

  if shuffle:
    seed = np.random.randint(0, 10000)
    xs = np.random.RandomState(seed).permutation(np.asarray(xs, dtype=object))
    ys = np.random.RandomState(seed).permutation(np.asarray(ys, dtype=object))
    resets = np.random.RandomState(seed).permutation(np.asarray(resets, dtype=object))

  xs = np.concatenate([x for x in xs], axis=0)
  ys = np.concatenate([y for y in ys])
  resets = np.concatenate(list(resets))
  return xs, ys, resets


# Model building
out_layers = [int(l) for l in args.out_layers.split(',')]
num_in = 39
num_out = 61

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
    activation=bm.relu,
    jit_version=args.comp_type.split('-')[1]
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
    activation=bm.relu,
  )
else:
  raise ValueError
readout = bp.layers.Dense(args.num_hidden * len(out_layers), num_out,
                          b_initializer=None, mode=bm.training_mode)
rls = bp.algorithms.RLS(alpha=args.lr)
rls.register_target(readout.num_in)

# training and testing
@functools.partial(bm.jit, static_argnums=2)
@bm.to_object(child_objs=(reservoir, readout, rls))
def train_fun(x, y, reset):
  if reset:
    reservoir.reset_state(1)
  o = reservoir1(x)
  o = bm.concatenate([o[i] for i in out_layers], axis=1)
  pred = readout1(o)
  dw = rls(y, o, pred)
  readout.W += dw

  o = reservoir2(pred)
  o = bm.concatenate([o[i] for i in out_layers], axis=1)
  pred = readout2(o)
  dw = rls(y, o, pred)
  readout.W += dw

  o = reservoir3(pred)
  o = bm.concatenate([o[i] for i in out_layers], axis=1)
  pred = readout3(o)
  dw = rls(y, o, pred)
  readout.W += dw
  return bm.argmax(pred, axis=1)


@functools.partial(bm.jit, static_argnums=1)
@bm.to_object(child_objs=(reservoir, readout))
def predict(x, reset):
  if reset:
    reservoir.reset_state(1)
  o = reservoir(x)
  o = bm.concatenate([o[i] for i in out_layers], axis=1)
  y = readout(o)
  return jnp.argmax(y, axis=1)


def train_all_steps(xs, targets, resets):
  preds_at_frame = []
  targets_at_frame = []
  num_data = xs.shape[0]
  for i in tqdm(range(num_data), desc='Training'):
    pi = train_fun(xs[i], targets[i], resets[i])
    pi = pi.item()
    preds_at_frame.append(pi)
    targets_at_frame.append(np.argwhere(targets[i] == 1)[0][0])
  preds_at_frame = np.asarray(preds_at_frame)
  targets_at_frame = np.asarray(targets_at_frame)
  frame_acc = np.mean(preds_at_frame == targets_at_frame)
  return frame_acc


def predict_all_steps(xs, targets, resets):
  preds_at_frame = []
  targets_at_frame = []
  num_data = xs.shape[0]
  for i in tqdm(range(num_data), desc='Predicting'):
    pi = predict(xs[i], resets[i])
    pi = pi.item()
    preds_at_frame.append(pi)
    targets_at_frame.append(np.argwhere(targets[i] == 1)[0][0])
  preds_at_frame = np.asarray(preds_at_frame)
  targets_at_frame = np.asarray(targets_at_frame)
  frame_acc = np.mean(preds_at_frame == targets_at_frame)
  return frame_acc


if __name__ == '__main__':
  if args.save:
    out_path = f'logs/force-TIMIT/{args.comp_type}/'
    out_path += f'leaky_start={args.leaky_start}-leaky_end={args.leaky_end}-'
    out_path += f'Iscale={args.win_scale}-SR={args.spectral_radius}'
    out_path += f'-Cin={args.win_connectivity}-Crec={args.wrec_connectivity}-lr={args.lr}'
    out_path += f'-out_layers={args.out_layers}-layer_num={args.num_layer}-hidden-{args.num_hidden}.bp'

  if args.save and os.path.exists(out_path):
    states = bp.checkpoints.load(out_path)
    wav_test_acc_max = states['wav_test_acc']
    print('Old wav_test_acc ', wav_test_acc_max)
  else:
    wav_test_acc_max = 0.

  for ii in range(args.epoch):
    print(f'Train {ii}')
    xs_train, ys_train, resets_train = format_data((train_data, train_labels), shuffle=True)
    wav_train_acc = train_all_steps(xs_train, ys_train, resets_train)
    xs_test, ys_test, resets_test = format_data((test_data, test_labels), shuffle=False)
    wav_test_acc = predict_all_steps(xs_test, ys_test, resets_test)
    print(f'Train, wav accuracy {wav_train_acc}')
    print(f'Test, wav accuracy {wav_test_acc}')

    if args.save and wav_test_acc_max < wav_test_acc:
      bp.checkpoints.save(out_path,
                         {'reservoir': reservoir.state_dict(),
                          'readout': readout.state_dict(),
                          'args': args.__dict__,
                          'wav_test_acc': float(wav_test_acc)
                          },
                          step=ii,
                          overwrite=True)
      wav_test_acc_max = wav_test_acc



