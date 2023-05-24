# -*- coding: utf-8 -*-

import time

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import brainpy as bp
import brainpy.math as bm
import brainpy_datasets as bd
from utils import DMS

bp.math.set(dt=100., mode=bm.training_mode)

# data
ds = DMS(dt=bm.dt, mode='rate', num_trial=64 * 100, bg_fr=1.)
_loader = bd.cognitive.TaskLoader(ds, batch_size=64, data_first_axis='T')


# EI RNN model
class EI_RNN(bp.DynamicalSystemNS):
  def __init__(
      self, num_input, num_hidden, num_output, dt,
      e_ratio=0.8, sigma_rec=0., seed=None,
      w_ir=bp.init.KaimingUniform(scale=1.),
      w_rr=bp.init.KaimingUniform(scale=1.),
      w_ro=bp.init.KaimingUniform(scale=1.)
  ):
    super(EI_RNN, self).__init__()
    self.mode.is_parent_of(bm.TrainingMode)

    # parameters
    self.tau = 100
    self.num_input = num_input
    self.num_hidden = num_hidden
    self.num_output = num_output
    self.e_size = int(num_hidden * e_ratio)
    self.i_size = num_hidden - self.e_size
    self.alpha = dt / self.tau
    self.sigma_rec = (2 * self.alpha) ** 0.5 * sigma_rec  # Recurrent noise
    self.rng = bm.random.RandomState(seed=seed)

    # hidden mask
    mask = np.tile([1] * self.e_size + [-1] * self.i_size, (num_hidden, 1))
    np.fill_diagonal(mask, 0)
    self.mask = bm.asarray(mask, dtype=bm.float_)

    # input weight
    self.w_ir = bm.TrainVar(bp.init.parameter(w_ir, (num_input, num_hidden)))

    # recurrent weight
    bound = 1 / num_hidden ** 0.5
    self.w_rr = bm.TrainVar(bp.init.parameter(w_rr, (num_hidden, num_hidden)))
    self.w_rr[:, :self.e_size] /= (self.e_size / self.i_size)
    self.b_rr = bm.TrainVar(self.rng.uniform(-bound, bound, num_hidden))

    # readout weight
    bound = 1 / self.e_size ** 0.5
    self.w_ro = bm.TrainVar(bp.init.parameter(w_ro, (self.e_size, num_output)))
    self.b_ro = bm.TrainVar(self.rng.uniform(-bound, bound, num_output))

    # variables
    self.reset_state(1)

  def reset_state(self, batch_size):
    self.h = bm.Variable(bm.zeros((batch_size, self.num_hidden)), batch_axis=0)
    self.o = bm.Variable(bm.zeros((batch_size, self.num_output)), batch_axis=0)

  def cell(self, x, h):
    ins = x @ self.w_ir + h @ (bm.abs(self.w_rr) * self.mask) + self.b_rr
    state = h * (1 - self.alpha) + ins * self.alpha
    state += self.sigma_rec * self.rng.randn(self.num_hidden)
    return bm.relu(state)

  def readout(self, h):
    return h @ self.w_ro + self.b_ro

  def update(self, x):
    self.h.value = self.cell(x, self.h)
    self.o.value = self.readout(self.h[:, :self.e_size])
    return self.h.value, self.o.value

  def predict(self, xs):
    self.h[:] = 0.
    return bm.for_loop(self.update, xs)

  def loss(self, xs, ys):
    hs, os = self.predict(xs)
    outs = os[-ds.t_test:]
    # Define the accuracy
    acc = bm.mean(bm.equal(ys, bm.argmax(bm.mean(outs, axis=0), axis=1)))
    # loss function
    tiled_targets = bm.tile(bm.expand_dims(ys, 0), (ds.t_test, 1))
    # loss function
    loss = bp.losses.cross_entropy_loss(outs, tiled_targets)
    return loss, acc


def loss(xs, ys):
  hs, os = bp.LoopOverTime(net)(xs)
  outs = os[-ds.t_test:]
  # Define the accuracy
  acc = bm.mean(bm.equal(ys, bm.argmax(bm.mean(outs, axis=0), axis=1)))
  # loss function
  tiled_targets = bm.tile(bm.expand_dims(ys, 0), (ds.t_test, 1))
  # loss function
  loss = bp.losses.cross_entropy_loss(outs, tiled_targets)
  return loss, acc


if __name__ == '__main__':
  hidden_size = 100
  net = EI_RNN(num_input=ds.num_inputs,
               num_hidden=hidden_size,
               num_output=ds.num_outputs,
               dt=ds.dt,
               sigma_rec=0.15)

  # Adam optimizer
  opt = bp.optim.Adam(lr=1e-3, train_vars=net.train_vars().unique())

  # gradient function
  grad_f = bm.grad(loss,
                   grad_vars=net.train_vars().unique(),
                   return_value=True,
                   has_aux=True)

  # training function
  @bm.jit
  def train(xs, ys):
    grads, loss, acc = grad_f(xs, ys)
    opt.update(grads)
    return loss, acc

  # training
  final_acc = []
  final_loss = []
  for epoch_i in range(200):
    losses = []
    accs = []
    t0 = time.time()
    for x, y in _loader:
      net.reset_state(x.shape[1])
      l, a = train(x, y)
      losses.append(l)
      accs.append(a)
    l = np.mean(losses)
    a = np.mean(accs)
    print(f'Epoch {epoch_i}, time {time.time() - t0:5f} s, '
          f'loss {l:5f}, acc {a:5f}')
    final_acc.append(a)
    final_loss.append(l)

  x, y = zip(*[ds[i] for i in range(50)])  # get 50 trials
  x = np.asarray(x).transpose((1, 0, 2))
  y = np.asarray(y)
  looper = bp.LoopOverTime(net)
  net.reset_state(x.shape[1])
  rnn_activity, action_pred = looper(x)
  rnn_activity = bm.as_numpy(rnn_activity)
  output_activity = bm.as_numpy(action_pred)

  # visualize the recurrent activity
  sns.set(font_scale=2)
  sns.set_style("darkgrid")
  fig, gs = bp.visualize.get_figure(1, 1, 4., 12.)
  ax1 = fig.add_subplot(gs[0, 0])
  ts = np.arange(0, x.shape[0]) * ds.dt
  trial = 2
  _ = plt.plot(ts, rnn_activity[:, trial, :net.e_size], color='blue', label='Excitatory', linewidth=2)
  _ = plt.plot(ts, rnn_activity[:, trial, net.e_size:], color='red', label='Inhibitory')
  plt.xlabel('Time (ms)')
  plt.ylabel('Recurrent Activity')
  plt.savefig('./EI_RNN_recurrent_potential.png', dpi=300)

  # visualize the output activity
  trial = 2
  fig, gs = bp.visualize.get_figure(1, 1, 4., 12.)
  ax2 = fig.add_subplot(gs[0, 0])
  for i in range(output_activity.shape[-1]):
    _ = plt.plot(ts, output_activity[:, trial, i], label=f'Readout {i}', alpha=1, linewidth=4)
  plt.xlabel('Time (ms)')
  plt.ylabel('Output Activity')
  plt.legend()
  plt.savefig('./EI_RNN_output_act.png', dpi=300)