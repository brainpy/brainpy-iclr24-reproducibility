# -*- coding: utf-8 -*-

"""

python kth-reservoir-backpropagation.py -num_hidden 50000 -win_scale 0.6 -spectral_radius 1.3 -win_connectivity 0.1 -wrec_connectivity 0.001 -lr 0.001 -optimizer adam -scheduler step -steps 10 -gamma 0.95 -comp_type jit-v1 -gpu-id 3 -weight_decay 0.0001
python kth-reservoir-backpropagation.py -num_hidden 50000 -win_scale 0.6 -spectral_radius 1.3 -win_connectivity 0.1 -wrec_connectivity 0.001 -lr 0.001 -optimizer adam -scheduler none -comp_type dense -gpu-id 0

"""

import argparse
import math
import os
import socket

import brainpy as bp
import brainpy.math as bm
import hdf5storage
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from reservoir import JITReservoir

"""

num_hidden = 2000
train_stage = 'final_step'

  Epoch 93, train acc 0.92817, test acc 0.93048

"""

# os.environ["XLA_FLAGS"] = '--xla_gpu_force_compilation_parallelism=1'

parser = argparse.ArgumentParser(description='Classify CIFAR')
parser.add_argument('-data', default='kth', type=str)
parser.add_argument('-num_hidden', default=2000, type=int, help='simulating time-steps')
parser.add_argument('-win_scale', default=0.2, type=float)
parser.add_argument('-spectral_radius', default=1.3, type=float)
parser.add_argument('-leaky_rate', default=0.3, type=float)
parser.add_argument('-win_connectivity', default=0.01, type=float)
parser.add_argument('-wrec_connectivity', default=0.01, type=float)
parser.add_argument('-lr', default=1e-2, type=float)
parser.add_argument('-train_stage', default='all_steps', type=str)
parser.add_argument('-optimizer', default='sgd', type=str)
parser.add_argument('-steps', default='20,80,150', type=str)
parser.add_argument('-scheduler', default='multisteps', type=str)
parser.add_argument('-weight_decay', default=0., type=float)
parser.add_argument('-gamma', default=0.2, type=float)
parser.add_argument("-resume", default='', type=str)
parser.add_argument('-comp_type', default='jit-v1', type=str)
parser.add_argument('-gpu-id', default='0', type=str, help='gpu id')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

if socket.gethostname() in ['ai01', 'login01']:
  path = '/GPFS/wusi_lab_permanent/wangchaoming/data/kth_hog8x8_pca2k_labels.mat'
else:
  path = 'D:/data/kth_hog8x8_pca2k_labels.mat'
data = hdf5storage.loadmat(path)['kth_hog_pca2k_labels']
x_train = data[0][:450]
y_train = data[1][:450]
x_test = data[0][450:]
y_test = data[1][450:]
num_in = 2000
num_out = 6

assert args.train_stage in ['final_step', 'all_steps']
out_path = f'logs/bp-{args.data}/'
out_path += f'{args.comp_type}-leaky={args.leaky_rate}-Iscale={args.win_scale}'
out_path += f'-SR={args.spectral_radius}-Iconn={args.win_connectivity}-Rconn={args.wrec_connectivity}/'
out_path += f'{args.optimizer}-{args.scheduler}-lr={args.lr}-gamma={args.gamma}-steps={args.steps}-wd={args.weight_decay}/'
out_path += f'hidden={args.num_hidden}'
os.makedirs(out_path, exist_ok=True)

if args.comp_type == 'jit-v1':
  reservoir = JITReservoir(
    num_in,
    args.num_hidden,
    leaky_rate=args.leaky_rate,
    win_connectivity=args.win_connectivity,
    wrec_connectivity=args.wrec_connectivity,
    win_scale=args.win_scale,
    wrec_sigma=args.spectral_radius / math.sqrt(args.num_hidden * args.wrec_connectivity),
    mode=bm.batching_mode,
    jit_version='v1'
  )
elif args.comp_type == 'jit-v2':
  reservoir = JITReservoir(
    num_in,
    args.num_hidden,
    leaky_rate=args.leaky_rate,
    win_connectivity=args.win_connectivity,
    wrec_connectivity=args.wrec_connectivity,
    win_scale=args.win_scale,
    wrec_sigma=args.spectral_radius / math.sqrt(args.num_hidden * args.wrec_connectivity),
    mode=bm.batching_mode,
    jit_version='v2',
  )
elif args.comp_type == 'dense':
  reservoir = bp.layers.Reservoir(
    num_in,
    args.num_hidden,
    Win_initializer=bp.init.Uniform(-args.win_scale, args.win_scale),
    Wrec_initializer=bp.init.Normal(scale=args.spectral_radius / math.sqrt(args.num_hidden * args.wrec_connectivity)),
    in_connectivity=args.win_connectivity,
    rec_connectivity=args.wrec_connectivity,
    comp_type='dense',
    mode=bm.batching_mode
  )
else:
  raise ValueError
readout = bp.layers.Dense(args.num_hidden, num_out, mode=bm.training_mode)


@bm.jit
@bm.to_object(child_objs=(reservoir, readout))
def loss_fun(xs, ys):
  ys_onehot = bm.one_hot(ys, num_out)
  if args.train_stage == 'final_step':
    for x in xs.transpose(1, 0):
      o = reservoir(x)
    pred = readout(o)
    acc = jnp.argmax(pred[0]) == ys[-1]
    l = bp.losses.mean_squared_error(pred, ys_onehot[-1:])
  elif args.train_stage == 'all_steps':
    l = 0.
    preds = jnp.zeros(num_out)
    for x in xs.transpose(1, 0):
      o = reservoir(x)
      pred = readout(o)
      l += bp.losses.mean_squared_error(pred, ys_onehot)
      preds = preds.at[jnp.argmax(pred[0])].add(1.)
    acc = jnp.argmax(preds) == ys[0]
  else:
    raise ValueError
  return l, acc


grad_fun = bm.grad(loss_fun, grad_vars=readout.train_vars(), return_value=True, has_aux=True)

if args.scheduler == 'multisteps':
  steps = [int(s.strip()) for s in args.steps.split(',')]
  assert len(steps) >= 1
  lr = bp.optim.MultiStepLR(args.lr, steps, gamma=args.gamma)
elif args.scheduler == 'exp':
  lr = bp.optim.ExponentialLR(args.lr, gamma=args.gamma)
elif args.scheduler == 'step':
  lr = bp.optim.StepLR(args.lr, gamma=args.gamma, step_size=int(args.steps))
elif args.scheduler == 'none':
  lr = args.lr
else:
  raise ValueError

if args.optimizer == 'sgd':
  optimizer = bp.optim.SGD(lr=lr, train_vars=readout.train_vars(), weight_decay=args.weight_decay)
elif args.optimizer == 'adam':
  optimizer = bp.optim.Adam(lr=lr, train_vars=readout.train_vars(), weight_decay=args.weight_decay)
else:
  raise ValueError


@bm.jit
@bm.to_object(child_objs=(grad_fun, optimizer))
def train_step(xs, ys):
  grads, l, n = grad_fun(xs, ys)
  optimizer.update(grads)
  return l, n


# training
last_epoch = -1
max_test_acc = 0.
if args.resume:
  states = bp.checkpoints.load(args.resume)
  readout.load_state_dict(states['readout'])
  optimizer.load_state_dict(states['optimizer'])
  last_epoch = states['last_epoch']
  max_test_acc = states['max_test_acc']

train_indices = np.arange(x_train.shape[0])
for epoch_i in range(last_epoch + 1, 400):
  train_indices = np.random.permutation(train_indices)
  pbar = tqdm(total=len(x_train))
  train_acc = 0.
  for i in train_indices:
    X = jnp.asarray(x_train[i], dtype=bm.float_)
    Y = jnp.asarray(y_train[i][0], dtype=bm.int_)
    reservoir.reset_state(1)
    l, n = train_step(X, Y)
    pbar.set_description(f'Training, loss {l:.5f}, acc {n}')
    pbar.update()
    train_acc += n
  pbar.close()

  pbar = tqdm(total=len(x_test))
  test_acc = 0.
  for i in range(x_test.shape[0]):
    X = jnp.asarray(x_test[i], dtype=bm.float_)
    Y = jnp.asarray(y_test[i][0], dtype=bm.int_)
    reservoir.reset_state(1)
    l, n = loss_fun(X, Y)
    pbar.set_description(f'Testing, loss {l:.5f}, acc {n}')
    pbar.update()
    test_acc += n
  pbar.close()

  optimizer.lr.step_epoch()

  t_acc = test_acc / len(x_test)
  print(f'Epoch {epoch_i}, '
        f'lr {optimizer.lr()}, '
        f'train acc {(train_acc / len(x_train)):.5f}, '
        f'test acc {t_acc:.5f}')
  print()
  if max_test_acc < t_acc:
    max_test_acc = t_acc
    states = {
      'readout': readout.state_dict(),
      'optimizer': optimizer.state_dict(),
      'max_test_acc': max_test_acc,
      'last_epoch': epoch_i,
    }
    bp.checkpoints.save(out_path, states, round(float(max_test_acc), 5), keep=4)
