# -*- coding: utf-8 -*-

import sys
import argparse
import functools
import math
import os
import socket

import brainpy as bp
import brainpy.math as bm
import hdf5storage
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from reservoir import JITDeepReservoir, DeepReservoir

parser = argparse.ArgumentParser(description='KTH dataset')
parser.add_argument('-num_hidden', default=10000, type=int, help='simulating time-steps')
parser.add_argument('-num_layer', default=1, type=int, help='number of layer')
parser.add_argument('-win_scale', default=0.1, type=float)
parser.add_argument('-out_layers', default='-1', type=str)
parser.add_argument('-spectral_radius', default=1.0, type=float)
parser.add_argument('-leaky_start', default=0.9, type=float)
parser.add_argument('-leaky_end', default=0.1, type=float)
parser.add_argument('-win_connectivity', default=0.1, type=float)
parser.add_argument('-wrec_connectivity', default=0.01, type=float)
parser.add_argument('-train_start', default=10, type=int)
parser.add_argument('-epoch', default=10, type=int)
parser.add_argument('-lr', default=0.1, type=float)
parser.add_argument('-comp_type', default='jit-v1', type=str)
parser.add_argument('-gpu-id', default='0', type=str, help='gpu id')
parser.add_argument('-save', action='store_true')

args = parser.parse_args()
print(args)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

# Please download the KTH file from the internet
path = './data/kth_hog8x8_pca2k_labels.mat'
data = hdf5storage.loadmat(path)['kth_hog_pca2k_labels']
data, labels = [x for x in data[0]], [x for x in data[1]]
data = data[:506] + [data[506]] + data[506:]
labels = labels[:506] + [labels[506]] + labels[506:]


def get_data_v2():
  xs_t1, xs_t2 = [], []
  for i in range(0, len(data), 4):
    xs_t1.append(data[i])
    xs_t1.append(data[i + 1])
    xs_t1.append(data[i + 2])
    xs_t2.append(data[i + 3])

  resets_t1, resets_t2 = [], []
  trains_t1, trains_t2 = [], []
  ys_t1, ys_t2 = [], []
  for i in range(0, len(labels), 4):
    for j in range(4):
      x = labels[i + j][0]
      if j < 3:
        ys_t1.append(x)
      else:
        ys_t2.append(x)

      a = np.zeros(x.shape[0], dtype=bool)
      a[0] = True
      if j < 3:
        resets_t1.append(a)
      else:
        resets_t2.append(a)

      a = np.zeros(x.shape[0], dtype=bool)
      a[args.train_start:] = True
      if j < 3:
        trains_t1.append(a)
      else:
        trains_t2.append(a)

  xs_t1 = [x.transpose() for x in xs_t1]
  xs_t2 = [x.transpose() for x in xs_t2]
  return (xs_t1, ys_t1, resets_t1, trains_t1), (xs_t2, ys_t2, resets_t2, trains_t2)


def format_data(T, shuffle: bool = False):
  xs, ys, resets, states = T
  if shuffle:
    seed = np.random.randint(0, 10000)
    xs = np.random.RandomState(seed).permutation(np.asarray(xs, dtype=object))
    ys = np.random.RandomState(seed).permutation(np.asarray(ys, dtype=object))
    resets = np.random.RandomState(seed).permutation(np.asarray(resets, dtype=object))
    states = np.random.RandomState(seed).permutation(np.asarray(states, dtype=object))

  xs = np.concatenate([x for x in xs], axis=0)
  ys = np.concatenate([y for y in ys]) - 1
  resets = np.concatenate(list(resets))
  trains = np.concatenate(list(states))
  return xs, ys, resets, trains


T1, T2 = get_data_v2()

out_layers = [int(l) for l in args.out_layers.split(',')]
num_in = 2000
num_out = 6

if args.save:
  out_path = f'logs/force-kth/{args.comp_type}/train_start={args.train_start}-'
  out_path += f'leaky_start={args.leaky_start}-leaky_end={args.leaky_end}-'
  out_path += f'Iscale={args.win_scale}-SR={args.spectral_radius}'
  out_path += f'-Cin={args.win_connectivity}-Crec={args.wrec_connectivity}-lr={args.lr}'
  out_path += f'-out_layers={args.out_layers}-layer_num={args.num_layer}-hidden-{args.num_hidden}.bp'

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


@functools.partial(bm.jit, static_argnums=(2, 3))
@bm.to_object(child_objs=(reservoir, readout, rls))
def train_fun(x, y, reset, train):
  if reset:
    reservoir.reset_state(1)
  o = reservoir(x)
  if train:
    o = bm.concatenate([o[i] for i in out_layers], axis=1)
    pred = readout(o)
    dw = rls(bm.one_hot(y, num_out, dtype=bm.float_), o, pred)
    readout.W += dw
    return bm.argmax(pred, axis=1)
  else:
    return None


@functools.partial(bm.jit, static_argnums=(1, 2))
@bm.to_object(child_objs=(reservoir, readout))
def predict(x, reset, state):
  if reset:
    reservoir.reset_state(1)
  o = reservoir(x)
  if state:
    o = bm.concatenate([o[i] for i in out_layers], axis=1)
    y = readout(o)
    return jnp.argmax(y, axis=1)
  else:
    return None


def train_all_steps(xs, resets, states, targets):
  preds_at_frame = []
  targets_at_frame = []
  preds_at_stream = []
  targets_at_steam = []
  stream = np.zeros(num_out)
  num_data = xs.shape[0]
  for i in tqdm(range(num_data), desc='Training'):
    pi = train_fun(xs[i], targets[i], resets[i], states[i])
    if states[i]:
      pi = pi.item()
      preds_at_frame.append(pi)
      targets_at_frame.append(targets[i])
      stream[pi] += 1.
      if (i + 1) == num_data or resets[i + 1]:
        preds_at_stream.append(np.argmax(stream))
        targets_at_steam.append(targets[i])
        stream[:] = 0.
  preds_at_frame = np.asarray(preds_at_frame)
  targets_at_frame = np.asarray(targets_at_frame)
  preds_at_stream = np.asarray(preds_at_stream)
  targets_at_steam = np.asarray(targets_at_steam)
  frame_acc = np.mean(preds_at_frame == targets_at_frame)
  stream_acc = np.mean(preds_at_stream == targets_at_steam)
  return frame_acc, stream_acc



def predict_all_steps(data, resets, states, targets):
  preds_at_frame = []
  targets_at_frame = []
  preds_at_stream = []
  targets_at_steam = []
  stream = np.zeros(num_out)
  num_data = data.shape[0]
  for i in tqdm(range(num_data), desc='Predicting'):
    pi = predict(data[i], resets[i], states[i])
    if states[i]:
      pi = pi.item()
      preds_at_frame.append(pi)
      targets_at_frame.append(targets[i])
      stream[pi] += 1.
      if (i + 1) == num_data or resets[i + 1]:
        preds_at_stream.append(np.argmax(stream))
        targets_at_steam.append(targets[i])
        stream[:] = 0.
  preds_at_frame = np.asarray(preds_at_frame)
  targets_at_frame = np.asarray(targets_at_frame)
  preds_at_stream = np.asarray(preds_at_stream)
  targets_at_steam = np.asarray(targets_at_steam)
  frame_acc = np.mean(preds_at_frame == targets_at_frame)
  stream_acc = np.mean(preds_at_stream == targets_at_steam)
  return frame_acc, stream_acc


if args.save and os.path.exists(out_path):
  states = bp.checkpoints.load_pytree(out_path)
  stream_test_acc_max = states['stream_test_acc']
  print('Old stream_test_acc ', stream_test_acc_max)
else:
  stream_test_acc_max = 0.

for ii in range(args.epoch):
  print(f'Train {ii}')
  xs_Train, ys_Train, resets_Train, states_Train = format_data(T1, shuffle=True)
  frame_train_acc, stream_train_acc = train_all_steps(xs_Train, resets_Train, states_Train, ys_Train)
  xs_Test, ys_Test, resets_Test, states_Test = format_data(T2, shuffle=False)
  frame_test_acc, stream_test_acc = predict_all_steps(xs_Test, resets_Test, states_Test, ys_Test)
  print(f'Train, frame accuracy {frame_train_acc}, stream accuracy {stream_train_acc}')
  print(f'Test, frame accuracy {frame_test_acc}, stream accuracy {stream_test_acc}')

  if args.save and stream_test_acc_max < stream_test_acc:
    bp.checkpoints.save_pytree(out_path,
                               {'reservoir': reservoir.state_dict(),
                                'readout': readout.state_dict(),
                                'args': args.__dict__,
                                'frame_test_acc': float(frame_test_acc),
                                'stream_test_acc': float(stream_test_acc)},
                               overwrite=True)
    stream_test_acc_max = stream_test_acc

