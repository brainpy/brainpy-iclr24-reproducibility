# -*- coding: utf-8 -*-


"""
python mnist-reservoir-force-training.py -num_hidden 500 -num_layer 10 -spectral_radius 1.3 -win_scale 0.1 -comp_type dense -gpu-id 0 -leaky_start 0.2 -leaky_end 0.6

python mnist-reservoir-force-training.py -num_hidden 500 -num_layer 10 -spectral_radius 0.9 -win_scale 0.1 -comp_type dense -gpu-id 1

MNIST
2000, 0.9602
4000, 0.9724
8000, 0.9809
10000, 0.9817
20000, 0.9863
30000, 0.9881
40000, 0.9887
50000, 0.9888


num_layer=1 win_scale=0.3 spectral_radius=1.3 leaky_start=0.6 leaky_end=0.1 win_connectivity=0.1 wrec_connectivity=0.1
fashion MNIST
1000,   0.8518999814987183
2000,   0.8676999807357788
4000,   0.8776999711990356
8000,   0.8924999833106995
10000,  0.8955999612808228
20000,  0.9000999927520752
30000,  0.902899980545044
40000,  0.9021999835968018
50000,  0.9025999903678894

"""

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

from reservoir import JITDeepReservoir, DeepReservoir
from data.fashion_mnist.utils import mnist_reader
X_train, y_train = mnist_reader.load_mnist('data/fashion_mnist/data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('data/fashion_mnist/data/fashion', kind='t10k')
X_train = X_train.reshape(-1, 28, 28)
X_test = X_test.reshape(-1, 28, 28)
# if socket.gethostname() in ['ai01', 'login01']:
#   path = '/home/wusi_lab/wangchaoming/DATA/data'
# elif sys.platform == 'win32':
#   path = 'D:/data'
# else:
#   path = '/mnt/d/data'
# path = './data'
# traindata = bd.vision.MNIST(root=path, split='train', download=True)
# testdata = bd.vision.MNIST(root=path, split='test', download=True)

parser = argparse.ArgumentParser(description='Classify CIFAR')
parser.add_argument('-num_hidden', default=2000, type=int, help='simulating time-steps')
parser.add_argument('-num_layer', default=1, type=int, help='number of layers')
parser.add_argument('-win_scale', default=0.6, type=float)
parser.add_argument('-out_layers', default='-1', type=str)
parser.add_argument('-spectral_radius', default=1.3, type=float)
parser.add_argument('-leaky_start', default=0.9, type=float)
parser.add_argument('-leaky_end', default=0.1, type=float)
parser.add_argument('-win_connectivity', default=0.1, type=float)
parser.add_argument('-wrec_connectivity', default=0.1, type=float)
parser.add_argument('-lr', default=0.1, type=float)
parser.add_argument('-train_stage', default='final_step', type=str)
parser.add_argument('-comp_type', default='jit-v1', type=str)
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

x_train = np.asarray(X_train / 255, dtype=bm.float_)
x_test = np.asarray(X_test / 255, dtype=bm.float_)

if args.save:
  out_path = f'logs/force-mnist/{args.comp_type}/'
  out_path += f'leaky_start={args.leaky_start}-leaky_end={args.leaky_end}-'
  out_path += f'Iscale={args.win_scale}-SR={args.spectral_radius}'
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

  # training
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
