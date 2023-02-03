# -*- coding: utf-8 -*-

import os
import sys
import socket
import math

import brainpy as bp
import brainpy.math as bm
import brainpy_datasets as bd
import brainpylib as bl
import jax
import jax.numpy as jnp
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

# os.environ["XLA_FLAGS"] = '--xla_gpu_force_compilation_parallelism=1'

if socket.gethostname() in ['ai01', 'login01']:
  path = '/home/wusi_lab/wangchaoming/DATA/data'
elif sys.platform == 'win32':
  path = 'D:/data'
else:
  path = '/mnt/d/data'
traindata = bd.vision.MNIST(root=path, split='train', download=True)
testdata = bd.vision.MNIST(root=path, split='test', download=True)


class JITReservoir(bp.DynamicalSystem):
  """Reservoir node with just-in-time connectivity."""

  def __init__(
      self,
      features_in,
      features_out,
      leaky_rate: float = 0.3,
      win_connectivity: float = 0.1,
      wrec_connectivity: float = 0.1,
      win_scale: float = 0.1,
      wrec_sigma: float = 0.1,
      activation: callable = bm.tanh,
      name: str = None,
      mode: bm.Mode = None
  ):
    super().__init__(name=name, mode=mode)

    bp.check.is_subclass(self.mode, bm.BatchingMode)

    self.features_in = bp.check.is_integer(features_in, min_bound=1)
    self.features_out = bp.check.is_integer(features_out, min_bound=1)
    self.leaky_rate = bp.check.is_float(leaky_rate, min_bound=0.)
    self.win_connectivity = bp.check.is_float(win_connectivity, min_bound=0.)
    self.wrec_connectivity = bp.check.is_float(wrec_connectivity, min_bound=0.)
    self.win_scale = bp.check.is_float(win_scale, min_bound=0.)
    self.wrec_sigma = bp.check.is_float(wrec_sigma, min_bound=0.)
    self.activation = bp.check.is_callable(activation)
    self.win_seed = bm.random.randint(0, 100000).item()
    self.wrec_seed = bm.random.randint(0, 100000).item()
    self.jit_version = 'v2'

    self.reset_state(self.mode)

  def reset_state(self, batch_size=None):
    self.state = bp.init.variable_(jnp.zeros, self.features_out, batch_size)

  def update(self, *args):
    x = args[0] if len(args) == 1 else args[1]
    if x.ndim == 1:
      hidden = bl.jitconn_ops.matvec_prob_conn_uniform_weight(x,
                                                              w_low=-self.win_scale,
                                                              w_high=self.win_scale,
                                                              conn_prob=self.win_connectivity,
                                                              shape=(self.features_out, self.features_in),
                                                              seed=self.win_seed)
      hidden += bl.jitconn_ops.matvec_prob_conn_normal_weight(self.state.value,
                                                              w_mu=0.,
                                                              w_sigma=self.wrec_sigma,
                                                              conn_prob=self.wrec_connectivity,
                                                              shape=(self.features_out, self.features_out),
                                                              seed=self.wrec_seed)
    elif x.ndim == 2:
      if self.jit_version == 'v1':
        hidden = jax.vmap(
          lambda a: bl.jitconn_ops.matvec_prob_conn_uniform_weight(a,
                                                                   w_low=-self.win_scale,
                                                                   w_high=self.win_scale,
                                                                   conn_prob=self.win_connectivity,
                                                                   shape=(self.features_out, self.features_in),
                                                                   seed=self.win_seed)
        )(x)
        hidden += jax.vmap(
          lambda a: bl.jitconn_ops.matvec_prob_conn_normal_weight(a,
                                                                  w_mu=0.,
                                                                  w_sigma=self.wrec_sigma,
                                                                  conn_prob=self.wrec_connectivity,
                                                                  shape=(self.features_out, self.features_out),
                                                                  seed=self.wrec_seed)
        )(self.state.value)
      elif self.jit_version == 'v2':
        hidden = bl.jitconn_ops.matmat_prob_conn_uniform_weight(x,
                                                                w_low=-self.win_scale,
                                                                w_high=self.win_scale,
                                                                conn_prob=self.win_connectivity,
                                                                shape=(self.features_in, self.features_out),
                                                                seed=self.win_seed)
        hidden += bl.jitconn_ops.matmat_prob_conn_normal_weight(self.state.value,
                                                                w_mu=0.,
                                                                w_sigma=self.wrec_sigma,
                                                                conn_prob=self.wrec_connectivity,
                                                                shape=(self.features_out, self.features_out),
                                                                seed=self.wrec_seed)
      else:
        raise ValueError
    else:
      raise ValueError
    state = (1 - self.leaky_rate) * self.state + self.leaky_rate * self.activation(hidden)
    self.state.value = state
    return state


def offline_train(num_hidden=2000, num_in=28, num_out=10):
  # training
  x_train = jnp.asarray(traindata.data / 255, dtype=bm.float_)
  x_train = x_train.reshape(-1, x_train.shape[-1])
  y_train = bm.one_hot(jnp.repeat(traindata.targets, x_train.shape[1]), 10, dtype=bm.float_)

  reservoir = bp.layers.Reservoir(
    num_in,
    num_hidden,
    Win_initializer=bp.init.Uniform(-0.6, 0.6),
    Wrec_initializer=bp.init.Normal(scale=0.1),
    in_connectivity=0.1,
    rec_connectivity=0.9,
    spectral_radius=1.3,
    leaky_rate=0.2,
    comp_type='dense',
    mode=bm.batching_mode
  )
  reservoir.reset_state(1)
  # outs = jnp.asarray([reservoir(dict(), x) for x in x_train])
  outs = bm.for_loop(bm.Partial(reservoir, {}), x_train)
  weight = bp.algorithms.RidgeRegression(alpha=1e-8)(y_train, outs)

  # predicting
  reservoir.reset_state(1)
  esn = bp.Sequential(
    reservoir,
    bp.layers.Dense(num_hidden,
                    num_out,
                    W_initializer=weight,
                    b_initializer=None,
                    mode=bm.training_mode)
  )

  preds = bm.for_loop(lambda x: jnp.argmax(esn({}, x), axis=-1),
                      x_train,
                      child_objs=esn)
  accuracy = jnp.mean(preds == jnp.repeat(traindata.targets, x_train.shape[1]))
  print(accuracy)


def force_online_train(num_hidden=2000, num_in=28, num_out=10, train_stage='final_step'):
  """

  num_hidden = 2000
  train_stage = 'final_step'

    Train accuracy 0.95923334
    Test accuracy 0.95669997

  num_hidden = 4000
  train_stage = 'final_step'

    Train accuracy 0.97256666
    Test accuracy 0.96959996

  num_hidden = 8000
  train_stage = 'final_step'

    Train accuracy 0.98260003
    Test accuracy 0.97679996


  num_hidden = 50000


  """
  assert train_stage in ['final_step', 'all_steps']

  x_train = jnp.asarray(traindata.data / 255, dtype=bm.float_)
  x_test = jnp.asarray(testdata.data / 255, dtype=bm.float_)
  y_train = bm.one_hot(traindata.targets, 10, dtype=bm.float_)

  # reservoir = bp.layers.Reservoir(
  #   num_in,
  #   num_hidden,
  #   Win_initializer=bp.init.Uniform(-0.6, 0.6),
  #   Wrec_initializer=bp.init.Normal(scale=0.1),
  #   in_connectivity=0.1,
  #   rec_connectivity=0.9,
  #   spectral_radius=1.3,
  #   leaky_rate=0.2,
  #   comp_type='dense',
  #   mode=bm.batching_mode
  # )
  reservoir = JITReservoir(
    num_in,
    num_hidden,
    leaky_rate=0.3,
    win_connectivity=0.1,
    wrec_connectivity=0.1,
    win_scale=0.6,
    wrec_sigma=1.5 / math.sqrt(num_hidden * 0.1),
    mode=bm.batching_mode
  )
  readout = bp.layers.Dense(num_hidden, num_out, b_initializer=None, mode=bm.training_mode)
  rls = bp.algorithms.RLS()
  rls.register_target(num_hidden)

  @bm.jit
  @bm.to_object(child_objs=(reservoir, readout, rls))
  def train_step(xs, y):
    reservoir.reset_state(xs.shape[0])
    if train_stage == 'final_step':
      for x in xs.transpose(1, 0, 2):
        o = reservoir(x)
      pred = readout(o)
      dw = rls(y, o, pred)
      readout.W += dw
    elif train_stage == 'all_steps':
      for x in xs.transpose(1, 0, 2):
        o = reservoir(x)
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
    y = readout(o)
    return jnp.argmax(y, axis=1)

  # training
  batch_size = 1
  for i in tqdm(range(0, x_train.shape[0], batch_size), desc='Training'):
    train_step(x_train[i: i + batch_size], y_train[i: i + batch_size])

  # verifying
  preds = []
  batch_size = 500
  for i in tqdm(range(0, x_train.shape[0], batch_size), desc='Verifying'):
    preds.append(predict(x_train[i: i + batch_size]))
  preds = jnp.concatenate(preds)
  acc = jnp.mean(preds == jnp.asarray(traindata.targets, dtype=bm.int_))
  print('Train accuracy', acc)

  # prediction
  preds = []
  for i in tqdm(range(0, x_test.shape[0], batch_size), desc='Predicting'):
    preds.append(predict(x_test[i: i + batch_size]))
  preds = jnp.concatenate(preds)
  acc = jnp.mean(preds == jnp.asarray(testdata.targets, dtype=bm.int_))
  print('Test accuracy', acc)


def backpropagation_training(num_hidden=2000, num_in=28, num_out=10, train_stage='final_step'):
  """

  num_hidden = 2000
  train_stage = 'final_step'

    Epoch 91, train acc 0.95157, test acc 0.95342

  num_hidden = 4000
  train_stage = 'final_step'

    Epoch 89, train acc 0.96447, test acc 0.96559

  num_hidden = 8000
  train_stage = 'final_step'

    Epoch 97, train acc 0.97519, test acc 0.97340



  num_hidden = 4000
  train_stage = 'all_steps'

    Epoch 98, train acc 0.91246, test acc 0.91960
  """
  assert train_stage in ['final_step', 'all_steps']

  x_train = jnp.asarray(traindata.data / 255, dtype=bm.float_)
  y_train = jnp.asarray(traindata.targets, dtype=bm.int_)
  x_test = jnp.asarray(testdata.data / 255, dtype=bm.float_)
  y_test = jnp.asarray(testdata.targets, dtype=bm.int_)

  reservoir = bp.layers.Reservoir(
    num_in,
    num_hidden,
    Win_initializer=bp.init.Uniform(-0.6, 0.6),
    Wrec_initializer=bp.init.Normal(scale=1.3 / jnp.sqrt(num_hidden * 0.9)),
    in_connectivity=0.1,
    rec_connectivity=0.9,
    comp_type='dense',
    mode=bm.batching_mode
  )
  readout = bp.layers.Dense(num_hidden, num_out, mode=bm.training_mode)

  @bm.jit
  @bm.to_object(child_objs=(reservoir, readout))
  def loss_fun(xs, ys):
    ys_onehot = bm.one_hot(ys, num_out)
    if train_stage == 'final_step':
      for x in xs.transpose(1, 0, 2):
        o = reservoir(x)
      pred = readout(o)
      acc = jnp.mean(jnp.argmax(pred, axis=1) == ys)
      l = bp.losses.mean_squared_error(pred, ys_onehot)
    elif train_stage == 'all_steps':
      l = 0.
      preds = 0.
      for x in xs.transpose(1, 0, 2):
        o = reservoir(x)
        pred = readout(o)
        l += bp.losses.mean_squared_error(pred, ys_onehot)
        p = jnp.zeros((x.shape[0], num_out)).at[jnp.arange(x.shape[0]), jnp.argmax(pred, axis=1)].set(1.)
        preds += p
      acc = jnp.mean(jnp.argmax(preds, axis=1) == ys)
    else:
      raise ValueError
    return l, acc

  grad_fun = bm.grad(loss_fun, grad_vars=readout.train_vars(), return_value=True, has_aux=True)
  optimizer = bp.optim.Adam(lr=1e-3, train_vars=readout.train_vars())

  @bm.jit
  @bm.to_object(child_objs=(grad_fun, optimizer))
  def train_step(xs, ys):
    grads, l, n = grad_fun(xs, ys)
    optimizer.update(grads)
    return l, n

  # training
  batch_size = 128
  for epoch_i in range(100):
    x_train = bm.random.permutation(x_train, key=123)
    y_train = bm.random.permutation(y_train, key=123)
    num_batch = x_train.shape[0] // batch_size
    pbar = tqdm(total=num_batch)
    train_acc = []
    for i in range(0, x_train.shape[0], batch_size):
      X = x_train[i: i + batch_size]
      reservoir.reset_state(X.shape[0])
      l, n = train_step(X, y_train[i: i + batch_size])
      pbar.set_description(f'Training, loss {l:.5f}, acc {n:.5f}')
      pbar.update()
      train_acc.append(n)
    pbar.close()

    num_batch = x_test.shape[0] // batch_size
    pbar = tqdm(total=num_batch)
    test_acc = []
    for i in range(0, x_test.shape[0], batch_size):
      X = x_test[i: i + batch_size]
      reservoir.reset_state(X.shape[0])
      l, n = loss_fun(X, y_test[i: i + batch_size])
      pbar.set_description(f'Testing, loss {l:.5f}, acc {n:.5f}')
      pbar.update()
      test_acc.append(n)
    pbar.close()

    print(f'Epoch {epoch_i}, '
          f'train acc {jnp.asarray(train_acc).mean():.5f}, '
          f'test acc {jnp.asarray(test_acc).mean():.5f}')
    print()
    optimizer.lr.step_epoch()


def largescale_bp_training(num_hidden=2000,
                           win_scale=0.6,
                           spectral_radius=1.3,
                           leaky_rate=0.3,
                           win_connectivity=0.01,
                           wrec_connectivity=0.01,
                           train_stage='final_step',
                           num_in=28,
                           num_out=10,
                           lr=1e-3,
                           resume=False):
  """

  num_hidden = 2000
  train_stage = 'final_step'

    Epoch 93, train acc 0.92817, test acc 0.93048

  """
  assert train_stage in ['final_step', 'all_steps']

  out_path = f'logs/Iscale={win_scale}-SR={spectral_radius}-Iconn={win_connectivity}'
  out_path += f'-Rconn={wrec_connectivity}/hidden={num_hidden}-leaky={leaky_rate}-lr={lr}'
  os.makedirs(out_path, exist_ok=True)

  x_train = jnp.asarray(traindata.data / 255, dtype=bm.float_)
  y_train = jnp.asarray(traindata.targets, dtype=bm.int_)
  x_test = jnp.asarray(testdata.data / 255, dtype=bm.float_)
  y_test = jnp.asarray(testdata.targets, dtype=bm.int_)

  reservoir = JITReservoir(
    num_in,
    num_hidden,
    leaky_rate=leaky_rate,
    win_connectivity=win_connectivity,
    wrec_connectivity=wrec_connectivity,
    win_scale=win_scale,
    wrec_sigma=spectral_radius / (num_hidden * 0.01) ** 0.5,
    mode=bm.batching_mode
  )
  readout = bp.layers.Dense(num_hidden, num_out, mode=bm.training_mode)

  @bm.jit
  @bm.to_object(child_objs=(reservoir, readout))
  def loss_fun(xs, ys):
    ys_onehot = bm.one_hot(ys, num_out)
    if train_stage == 'final_step':
      for x in xs.transpose(1, 0, 2):
        o = reservoir(x)
      pred = readout(o)
      acc = jnp.mean(jnp.argmax(pred, axis=1) == ys)
      l = bp.losses.mean_squared_error(pred, ys_onehot)
    elif train_stage == 'all_steps':
      l = 0.
      preds = 0.
      for x in xs.transpose(1, 0, 2):
        o = reservoir(x)
        pred = readout(o)
        l += bp.losses.mean_squared_error(pred, ys_onehot)
        p = jnp.zeros((x.shape[0], num_out)).at[jnp.arange(x.shape[0]), jnp.argmax(pred, axis=1)].set(1.)
        preds += p
      acc = jnp.mean(jnp.argmax(preds, axis=1) == ys)
    else:
      raise ValueError
    return l, acc

  grad_fun = bm.grad(loss_fun, grad_vars=readout.train_vars(), return_value=True, has_aux=True)
  optimizer = bp.optim.Adam(lr=bp.optim.MultiStepLR(lr, [20, 50, 80, 120], gamma=0.2),
                            train_vars=readout.train_vars())

  @bm.jit
  @bm.to_object(child_objs=(grad_fun, optimizer))
  def train_step(xs, ys):
    grads, l, n = grad_fun(xs, ys)
    optimizer.update(grads)
    return l, n

  # training
  last_epoch = -1
  max_test_acc = 0.
  if resume:
    states = bp.checkpoints.load(out_path)
    readout.load_state_dict(states['readout'])
    optimizer.load_state_dict(states['optimizer'])
    last_epoch = states['last_epoch']
    max_test_acc = states['max_test_acc']

  batch_size = 128
  for epoch_i in range(last_epoch + 1, 200):
    x_train = bm.random.permutation(x_train, key=123)
    y_train = bm.random.permutation(y_train, key=123)
    num_batch = x_train.shape[0] // batch_size
    pbar = tqdm(total=num_batch)
    train_acc = []
    for i in range(0, x_train.shape[0], batch_size):
      X = x_train[i: i + batch_size]
      reservoir.reset_state(X.shape[0])
      l, n = train_step(X, y_train[i: i + batch_size])
      pbar.set_description(f'Training, loss {l:.5f}, acc {n:.5f}')
      pbar.update()
      train_acc.append(n)
    pbar.close()

    num_batch = x_test.shape[0] // batch_size
    pbar = tqdm(total=num_batch)
    test_acc = []
    for i in range(0, x_test.shape[0], batch_size):
      X = x_test[i: i + batch_size]
      reservoir.reset_state(X.shape[0])
      l, n = loss_fun(X, y_test[i: i + batch_size])
      pbar.set_description(f'Testing, loss {l:.5f}, acc {n:.5f}')
      pbar.update()
      test_acc.append(n)
    pbar.close()

    optimizer.lr.step_epoch()

    t_acc = jnp.asarray(test_acc).mean()
    print(f'Epoch {epoch_i}, '
          f'train acc {jnp.asarray(train_acc).mean():.5f}, '
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


if __name__ == '__main__':
  pass
  # offline_train()
  force_online_train(num_hidden=10000)
  # backpropagation_training(num_hidden=4000)
  # backpropagation_training(num_hidden=4000, train_stage='all_steps')
  # largescale_bp_training(num_hidden=4000, lr=0.01)
