# -*- coding: utf-8 -*-

"""
Reproduce the results of the``spytorch`` tutorial 2 & 3:

- https://github.com/surrogate-gradient-learning/spytorch/blob/master/notebooks/SpyTorchTutorial2.ipynb
- https://github.com/surrogate-gradient-learning/spytorch/blob/master/notebooks/SpyTorchTutorial3.ipynb

CPU m1 pro:
- Train acc: 89.6%
- Test acc: 86.0%
- Each epoch: 24-25 s

CPU Intel:
- Each epoch: 46-48 s

RTX A6000
- Test acc: 86.2%
- Each epoch: 16-17 s

"""
import time

import brainpy_datasets as bd
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

import brainpy as bp
import brainpy.math as bm

from jax.lax import stop_gradient

# bm.set_platform('cpu')
bm.set_environment(bm.training_mode)

alpha = 0.9
beta = 0.85

class SynapticLIF(bp.DynamicalSystemNS):
  def __init__(self, size,
               alpha, beta, V_th=1.0, mode=None,
               spike_fun=bm.surrogate.arctan, spike_grad=None,
               init_hidden=False, inhibition=False,
               learn_beta=False, learn_alpha=False, learn_threshold=False, reset_mechanism='subtract'):
    super().__init__(mode=mode)

    self.size = size
    self.V_th = V_th
    self.reset_mechanism = reset_mechanism
    if learn_alpha:
      self.alpha = bm.TrainVar(alpha)
    else:
      self.alpha = alpha
    if learn_beta:
      self.beta = bm.TrainVar(beta)
    else:
      self.beta = beta

    self.spike_fun = spike_fun

    assert self.mode.is_parent_of(bm.TrainingMode)
    self.reset_state(self.mode)

  def update(self, x):
    self.I_syn.value = self.alpha * self.I_syn + x
    self.V.value = self.beta * self.V + self.I_syn

    if self.reset_mechanism == 'subtract':
      spike = self.spike_fun(self.V - self.V_th)
      spike_no_grad = stop_gradient(spike)
      self.V.value -= spike_no_grad * self.V_th
      self.spike.value = spike

    elif self.reset_mechanism == 'zero':
      spike = self.spike_fun(self.V - self.V_th)
      spike_no_grad = stop_gradient(spike)
      self.V.value *= spike_no_grad
      self.spike.value = spike

    else:
      spike = self.spike_fun(self.V - self.V_th)
      self.spike.value = spike

    return spike

  def reset_state(self, batch_size=None):
    self.V = bp.init.variable_(bm.zeros, self.size, batch_size)
    self.I_syn = bp.init.variable_(bm.zeros, self.size, batch_size)
    sp_type = bm.float_ if isinstance(self.mode, bm.TrainingMode) else bool  # the gradient of spike is a float
    self.spike = bp.init.variable_(lambda s: bm.zeros(s, dtype=sp_type), self.size, batch_size)


class SNN(bp.DynamicalSystemNS):
  """
  This class implements a spiking neural network model with three layers:

     i >> r >> o

  Each two layers are connected through the exponential synapse model.
  """

  def __init__(self, num_in, num_rec, num_out):
    super(SNN, self).__init__()

    # parameters
    self.num_in = num_in
    self.num_rec = num_rec
    self.num_out = num_out

    # neuron groups
    # self.r = SynapticLIF(num_rec, V_th=1., spike_fun=bm.surrogate.arctan,
    #                      alpha=alpha, beta=beta, learn_beta=True, learn_alpha=True)
    # self.o = SynapticLIF(num_out, V_th=1., spike_fun=bm.surrogate.arctan,
    #                      alpha=alpha, beta=beta, learn_beta=True, learn_alpha=True, reset_mechanism='none')

    # self.i2r = bp.layers.Dense(self.num_in, self.num_rec)
    # self.r2o = bp.layers.Dense(self.num_rec, self.num_out, W_initializer=bp.init.KaimingNormal(8.))

    self.r = bp.neurons.LIF(num_rec, tau=10, V_reset=0, V_rest=0, V_th=1.,
                            input_var=False)
    self.o = bp.neurons.LeakyIntegrator(num_out, tau=5, input_var=False)
    self.i2r = bp.experimental.Exponential(bp.conn.All2All(pre=num_in, post=num_rec), tau=10.,
                                       g_max=bp.init.KaimingNormal(scale=2.))
    # synapse: r->o
    self.r2o = bp.experimental.Exponential(bp.conn.All2All(pre=num_rec, post=num_out), tau=10.,
                                  g_max=bp.init.KaimingNormal(scale=2.))

  def update(self, spike):
    return self.o(self.r2o(self.r(self.i2r(spike))))


def plot_voltage_traces(mem, spk=None, dim=(3, 5), spike_height=5):
  gs = GridSpec(*dim)
  mem = 1. * mem
  if spk is not None:
    mem[spk > 0.0] = spike_height
  mem = bm.as_numpy(mem)
  for i in range(np.prod(dim)):
    if i == 0:
      a0 = ax = plt.subplot(gs[i])
    else:
      ax = plt.subplot(gs[i], sharey=a0)
    ax.plot(mem[i, :, :])
    ax.axis("off")
  plt.tight_layout()
  plt.show()


def print_classification_accuracy(output, target):
  """ Dirty little helper function to compute classification accuracy. """
  m = jnp.max(output, axis=1)  # max over time
  am = jnp.argmax(m, axis=1)  # argmax over output units
  acc = jnp.mean(target == am)  # compare to labels
  print("Accuracy %.3f" % acc)


def current2firing_time(x, tau=20., thr=0.2, tmax=1.0, epsilon=1e-7):
  """Computes first firing time latency for a current input x
  assuming the charge time of a current based LIF neuron.

  Args:
  x -- The "current" values

  Keyword args:
  tau -- The membrane time constant of the LIF neuron to be charged
  thr -- The firing threshold value
  tmax -- The maximum time returned
  epsilon -- A generic (small) epsilon > 0

  Returns:
  Time to first spike for each "current" x
  """
  x = np.clip(x, thr + epsilon, 1e9)
  T = tau * np.log(x / (x - thr))
  T = np.where(x < thr, tmax, T)
  return T


def sparse_data_generator(X, y, batch_size, nb_steps, nb_units, shuffle=True):
  """ This generator takes datasets in analog format and
  generates spiking network input as sparse tensors.

  Args:
  X: The data ( sample x event x 2 ) the last dim holds (time,neuron) tuples
  y: The labels
  """

  labels_ = np.array(y, dtype=bm.int_)
  sample_index = np.arange(len(X))

  # compute discrete firing times
  tau_eff = 2. / bm.get_dt()
  unit_numbers = np.arange(nb_units)
  firing_times = np.array(current2firing_time(X, tau=tau_eff, tmax=nb_steps), dtype=bm.int_)

  if shuffle:
    np.random.shuffle(sample_index)

  counter = 0
  number_of_batches = len(X) // batch_size
  while counter < number_of_batches:
    batch_index = sample_index[batch_size * counter:batch_size * (counter + 1)]
    all_batch, all_times, all_units = [], [], []
    for bc, idx in enumerate(batch_index):
      c = firing_times[idx] < nb_steps
      times, units = firing_times[idx][c], unit_numbers[c]
      batch = bc * np.ones(len(times), dtype=bm.int_)
      all_batch.append(batch)
      all_times.append(times)
      all_units.append(units)
    all_batch = np.concatenate(all_batch).flatten()
    all_times = np.concatenate(all_times).flatten()
    all_units = np.concatenate(all_units).flatten()
    x_batch = np.zeros((nb_steps, batch_size, nb_units))
    x_batch[all_times, all_batch, all_units] = 1.
    y_batch = labels_[batch_index]
    yield x_batch, y_batch
    counter += 1



class Trainer:
  def __init__(self, net, lr):
    self.model = net
    self.looper = bp.LoopOverTime(net, out_vars=net.r.spike)
    self.f_grad = bm.grad(self.loss, grad_vars=net.train_vars().unique(), child_objs=net, return_value=True)
    self.f_opt = bp.optim.Adam(lr=bp.optim.CosineAnnealingLR(lr, T_max=300), train_vars=net.train_vars().unique())
    self.fit = bm.jit(self.train, child_objs=(self.f_grad, self.f_opt))

  def loss(self, xs, ys):
    predicts, spikes = self.looper(xs)
    # Here we set up our regularizer loss
    # The strength paramters here are merely a guess and
    # there should be ample room for improvement by
    # tuning these paramters.
    l1_loss = 1e-5 * bm.sum(spikes)  # L1 loss on total number of spikes
    l2_loss = 1e-5 * bm.mean(bm.sum(bm.sum(spikes, axis=0), axis=0) ** 2)  # L2 loss on spikes per neuron
    # l2_loss = 1e-5 * bm.mean(bm.sum(spikes, axis=(0, 1)) ** 2)  # L2 loss on spikes per neuron
    # predictions
    predicts = bm.max(predicts, axis=0)
    loss = bp.losses.cross_entropy_loss(predicts, ys)
    return loss + l2_loss + l1_loss

  def train(self, xs, ys):
    grads, loss = self.f_grad(xs, ys)
    self.f_opt.update(grads)
    return loss


def train(model, x_data, y_data, lr=1e-3, nb_epochs=10, batch_size=128, nb_steps=128, nb_inputs=28 * 28):
  loss = []
  trainer = Trainer(model, lr)
  for i in range(nb_epochs):
    t0 = time.time()
    for xs, ys in sparse_data_generator(x_data, y_data, batch_size, nb_steps, nb_inputs):
      trainer.looper.reset_state(xs.shape[1])
      l = trainer.fit(xs, ys)
      loss.append(l)
    print(f'Epoch {i}, {time.time() - t0}, loss: {l}')
  return loss


def compute_classification_accuracy(model, x_data, y_data, batch_size=128, nb_steps=100, nb_inputs=28 * 28):
  """ Computes classification accuracy on supplied data in batches. """
  accs = []
  trainer = Trainer(model, lr=1e-3)
  for xs, ys in sparse_data_generator(x_data, y_data, batch_size, nb_steps, nb_inputs, shuffle=False):
    trainer.looper.reset_state(xs.shape[1])
    predicts, spikes = trainer.looper(xs)
    m = bm.max(predicts, 0)  # max over time
    am = bm.argmax(m, 1)  # argmax over output units
    tmp = bm.mean(ys == am)  # compare to labels
    accs.append(tmp)
  return bm.mean(bm.asarray(accs))


def get_mini_batch_results(model, x_data, y_data, batch_size=128, nb_steps=100, nb_inputs=28 * 28):
  # runner = bp.DSRunner(model,
  #                      monitors={'r.spike': model.r.spike},
  #                      progress_bar=False)
  trainer = Trainer(model, lr=1e-3)
  data = sparse_data_generator(x_data, y_data, batch_size, nb_steps, nb_inputs, shuffle=False)
  xs, ys = next(data)
  trainer.looper.reset_state(xs.shape[1])
  predicts, spikes = trainer.looper(xs)
  # output = runner.predict(inputs=x_local, reset_state=True)
  return predicts, spikes


num_input = 28 * 28
net = SNN(num_in=num_input, num_rec=100, num_out=10)

# load the dataset
root = r"./data"
train_dataset = bd.vision.FashionMNIST(root, split='train', download=True)
test_dataset = bd.vision.FashionMNIST(root, split='test', download=True)

# Standardize data
x_train = np.array(train_dataset.data, dtype=bm.float_)
x_train = x_train.reshape(x_train.shape[0], -1) / 255
y_train = np.array(train_dataset.targets, dtype=bm.int_)
x_test = np.array(test_dataset.data, dtype=bm.float_)
x_test = x_test.reshape(x_test.shape[0], -1) / 255
y_test = np.array(test_dataset.targets, dtype=bm.int_)

print('Begin training ...')

# training
train_losses = train(net, x_train, y_train, lr=1e-3, nb_epochs=30, batch_size=256, nb_steps=100, nb_inputs=28 * 28)

plt.figure(figsize=(3.3, 2), dpi=150)
plt.plot(train_losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

print("Training accuracy: %.3f" % (compute_classification_accuracy(net, x_train, y_train, batch_size=512)))
print("Test accuracy: %.3f" % (compute_classification_accuracy(net, x_test, y_test, batch_size=512)))

outs, spikes = get_mini_batch_results(net, x_train, y_train)
# Let's plot the hidden layer spiking activity for some input stimuli
fig = plt.figure(dpi=100)
plot_voltage_traces(outs)
plt.show()

nb_plt = 4
gs = GridSpec(1, nb_plt)
plt.figure(figsize=(7, 3), dpi=150)
for i in range(nb_plt):
  plt.subplot(gs[i])
  plt.imshow(bm.as_numpy(spikes[i]).T, cmap=plt.cm.gray_r, origin="lower")
  if i == 0:
    plt.xlabel("Time")
    plt.ylabel("Units")
plt.tight_layout()
plt.show()


