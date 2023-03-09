# -*- coding: utf-8 -*-

'''
Reproduce the results of the``spytorch`` tutorial 2 & 3:

- https://github.com/surrogate-gradient-learning/spytorch/blob/master/notebooks/SpyTorchTutorial2.ipynb
- https://github.com/surrogate-gradient-learning/spytorch/blob/master/notebooks/SpyTorchTutorial3.ipynb


Apple m1 cpu:
- Training acc: 89.4%
- Test acc: 84.0%
- Time: 26-30 s

CPU Intel:
- Time: 38-40 s

RTX A6000:
- Time: 44-46 s
'''

import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import utils

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import numpy as np
import itertools
import time
from matplotlib.gridspec import GridSpec

alpha = 0.9
beta = 0.85

time_step = 1e-3
nb_steps = 100
batch_size = 256
data_path = r'./data'
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device = torch.device("cpu")
torch.set_num_threads(1)

num_inputs = 28*28
num_hidden = 100
num_outputs = 10

num_steps = 100

# Define a transform
transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])

train_dataset = datasets.FashionMNIST(data_path, train=True, download=True)
test_dataset = datasets.FashionMNIST(data_path, train=False, download=True)

# Create DataLoaders
# Standardize data
# x_train = torch.tensor(train_dataset.train_data, device=device, dtype=dtype)
x_train = np.array(train_dataset.data, dtype=np.float_)
x_train = x_train.reshape(x_train.shape[0], -1)/255
# x_test = torch.tensor(test_dataset.test_data, device=device, dtype=dtype)
x_test = np.array(test_dataset.data, dtype=np.float_)
x_test = x_test.reshape(x_test.shape[0], -1)/255

# y_train = torch.tensor(train_dataset.train_labels, device=device, dtype=dtype)
# y_test  = torch.tensor(test_dataset.test_labels, device=device, dtype=dtype)
y_train = np.array(train_dataset.targets, dtype=np.int_)
y_test  = np.array(test_dataset.targets, dtype=np.int_)


# Define Network
class Net(nn.Module):
   def __init__(self, num_inputs, num_hidden, num_outputs):
      super().__init__()

      self.num_inputs = num_inputs  # number of inputs
      self.num_hidden = num_hidden  # number of hidden neurons
      self.num_outputs = num_outputs  # number of classes (i.e., output neurons)

      # initialize layers
      self.i2r = nn.Linear(self.num_inputs, self.num_hidden)
      self.r = snn.Synaptic(alpha=alpha, beta=beta, learn_beta=True, learn_alpha=True,
                            spike_grad=surrogate.atan())
      # self.r = snn.Leaky(beta=beta, learn_beta=True, spike_grad=surrogate.atan())
      # self.r = snn.LIF(beta=beta, learn_beta=True,
      #                  spike_grad=surrogate.fast_sigmoid(slope=25))
      self.r2o = nn.Linear(self.num_hidden, self.num_outputs)
      self.o = snn.Synaptic(alpha=alpha, beta=beta, learn_beta=True, learn_alpha=True,
                            spike_grad=surrogate.atan(),
                            reset_mechanism="none")
      # self.o = snn.Leaky(beta=beta, learn_beta=True, spike_grad=surrogate.atan(), reset_mechanism="none")
      # self.o = snn.LIF(beta=beta, learn_beta=True,
      #                  spike_grad=surrogate.fast_sigmoid(slope=25))

   def forward(self, x):
      syn1, mem1 = self.r.init_synaptic()
      # mem1 = self.r.init_leaky()
      syn2, mem2 = self.o.init_synaptic()
      # mem2 = self.o.init_leaky()

      spk1_rec = []  # Record the output trace of spikes
      mem2_rec = []  # Record the output trace of membrane potential

      for step in range(num_steps):
        x_timestep = x[:, step, :]
        cur1 = self.i2r(x_timestep)
        spk1, syn1, mem1 = self.r(cur1, syn1, mem1)
        # spk1, mem1 = self.r(cur1, mem1)
        cur2 = self.r2o(spk1)
        _, syn2, mem2 = self.o(cur2, syn2, mem2)
        # _, mem2 = self.o(cur2, mem2)

        spk1_rec.append(spk1)
        mem2_rec.append(mem2)

      return torch.stack(spk1_rec, dim=0), torch.stack(mem2_rec, dim=0)


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
  """ This generator takes datasets in analog format and generates spiking network input as sparse tensors.

  Args:
      X: The data ( sample x event x 2 ) the last dim holds (time,neuron) tuples
      y: The labels
  """

  labels_ = np.array(y, dtype=np.int_)
  number_of_batches = len(X) // batch_size
  sample_index = np.arange(len(X))

  # compute discrete firing times
  tau_eff = 20e-3 / time_step
  firing_times = np.array(current2firing_time(X, tau=tau_eff, tmax=nb_steps), dtype=np.int_)
  unit_numbers = np.arange(nb_units)

  if shuffle:
    np.random.shuffle(sample_index)

  total_batch_count = 0
  counter = 0
  while counter < number_of_batches:
    batch_index = sample_index[batch_size * counter:batch_size * (counter + 1)]
    all_batch, all_times, all_units = [], [], []
    for bc, idx in enumerate(batch_index):
      c = firing_times[idx] < nb_steps
      times, units = firing_times[idx][c], unit_numbers[c]
      batch = bc * np.ones(len(times), dtype=np.int_)
      all_batch.append(batch)
      all_times.append(times)
      all_units.append(units)
    all_batch = np.concatenate(all_batch).flatten()
    all_times = np.concatenate(all_times).flatten()
    all_units = np.concatenate(all_units).flatten()
    x_batch = np.zeros((batch_size, nb_steps, nb_units))
    x_batch[all_batch, all_times, all_units] = 1.
    y_batch = np.asarray(labels_[batch_index])
    yield x_batch, y_batch
    counter += 1


# pass data into the network, sum the spikes over time
# and compare the neuron with the highest number of spikes
# with the target

def print_batch_accuracy(data, targets, train=False):
    output, _ = net(data.view(batch_size, -1))
    _, idx = output.sum(dim=0).max(1)
    acc = np.mean((targets == idx).detach().cpu().numpy())

    if train:
        print(f"Train set accuracy for a single minibatch: {acc*100:.2f}%")
    else:
        print(f"Test set accuracy for a single minibatch: {acc*100:.2f}%")


def compute_classification_accuracy(net, x_data, y_data):
  """ Computes classification accuracy on supplied data in batches. """
  accs = []
  for x_local, y_local in sparse_data_generator(x_data, y_data, batch_size, nb_steps, num_inputs, shuffle=False):
    x_local = torch.tensor(x_local).float()
    y_local = torch.tensor(y_local)
    x_local = x_local.to(device)
    y_local = y_local.to(device)
    net.eval()
    spk_rec, mem_rec = net(x_local)
    m, _ = torch.max(mem_rec, 0)  # max over time
    _, am = torch.max(m, 1)  # argmax over output units
    tmp = np.mean((y_local == am).detach().cpu().numpy())  # compare to labels
    accs.append(tmp)
  return np.mean(accs)

def plot_voltage_traces(mem, spk=None, dim=(3, 5), spike_height=5):
  gs = GridSpec(*dim)
  if spk is not None:
    dat = 1.0 * mem
    dat[spk > 0.0] = spike_height
    dat = dat.detach().cpu().numpy()
  else:
    dat = mem.detach().cpu().numpy()
  for i in range(np.prod(dim)):
    if i == 0:
      a0 = ax = plt.subplot(gs[i])
    else:
      ax = plt.subplot(gs[i], sharey=a0)
    ax.plot(dat[:, i, :])
    ax.axis("off")


if __name__ == '__main__':
  num_epochs = 5
  loss_hist = []
  net = Net(num_inputs, num_hidden, num_outputs).to(device)

  optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, betas=(0.9, 0.999))

  # loss = nn.CrossEntropyLoss()
  log_softmax_fn = nn.LogSoftmax(dim=1)
  loss_fn = nn.NLLLoss()

  for epoch in range(num_epochs):
    local_loss = []
    iter_cnt = 1
    # Minibatch training loop
    t0 = time.time()
    for data, targets in sparse_data_generator(x_train, y_train, batch_size=batch_size, nb_steps=nb_steps,
                                               nb_units=num_inputs):
      data = torch.tensor(data).float()
      targets = torch.tensor(targets)
      data = data.to(device)
      targets = targets.to(device)

      # forward pass
      net.train()
      spk_rec, mem_rec = net(data)

      m, _ = torch.max(mem_rec, 0)
      log_p_y = log_softmax_fn(m)

      # Here we set up our regularizer loss
      # The strength paramters here are merely a guess and there should be ample room for improvement by
      # tuning these paramters.
      reg_loss = 1e-5 * torch.sum(spk_rec)  # L1 loss on total number of spikes
      reg_loss += 1e-5 * torch.mean(torch.sum(torch.sum(spk_rec, dim=0), dim=0) ** 2)  # L2 loss on spikes per neuron

      # Here we combine supervised loss and the regularizer
      loss_val = loss_fn(log_p_y, targets) + reg_loss

      # initialize the loss & sum over time
      # loss_val = torch.zeros((1), dtype=torch.float, device=device)
      # for step in range(num_steps):
      #   loss_val += loss(mem_rec[step], targets)

      # Gradient calculation + weight update
      optimizer.zero_grad()
      loss_val.backward()
      optimizer.step()

      # Store loss history for future plotting
      local_loss.append(loss_val.item())
      if iter_cnt % 50 == 0:
        print(f"Epoch {epoch}, Iteration {iter_cnt}, Train Set Loss: {loss_val.item():.2f}")
      iter_cnt += 1

    t1 = time.time()
    mean_loss = np.mean(local_loss)
    print("Epoch %i: loss=%.5f  time: %f" % (epoch + 1, mean_loss, t1 - t0))
    loss_hist.append(mean_loss)
      # Test set
      # with torch.no_grad():
      #   net.eval()
      #   # test_data, test_targets = next(iter(test_loader))
      #   # test_data = test_data.to(device)
      #   # test_targets = test_targets.to(device)
      #
      #
      #   # Test set forward pass
      #   test_spk, test_mem = net(test_data.view(batch_size, -1))
      #
      #   # Test set loss
      #   test_loss = torch.zeros((1), dtype=torch.float, device=device)
      #   for step in range(num_steps):
      #     test_loss += loss(test_mem[step], test_targets)
      #   test_loss_hist.append(test_loss.item())
      #
      #   # Print train/test loss/accuracy
      #   if iter_counter % 50 == 0:
      #     print(f"Epoch {epoch}, Iteration {iter_counter}")
      #     print(f"Train Set Loss: {loss_hist[counter]:.2f}")
      #     print(f"Test Set Loss: {test_loss_hist[counter]:.2f}")
      #     print_batch_accuracy(data, targets, train=True)
      #     print_batch_accuracy(test_data, test_targets, train=False)
      #     print("\n")
      #   counter += 1
      #   iter_counter += 1

  print("Training accuracy: %.3f" % (compute_classification_accuracy(net, x_train, y_train)))
  print("Test accuracy: %.3f" % (compute_classification_accuracy(net, x_test, y_test)))


  def get_mini_batch_results(net, x_data, y_data, batch_size=128, nb_steps=100, nb_inputs=28 * 28):
    for x_local, y_local in sparse_data_generator(x_data, y_data, batch_size, nb_steps, num_inputs, shuffle=False):
      x_local = torch.tensor(x_local).float()
      y_local = torch.tensor(y_local)
      x_local = x_local.to(device)
      y_local = y_local.to(device)
      net.eval()
      spk_rec, mem_rec = net(x_local)

      return mem_rec, spk_rec

  outs, spikes = get_mini_batch_results(net, x_train, y_train)
  # Let's plot the hidden layer spiking activity for some input stimuli
  fig = plt.figure(dpi=100)
  plot_voltage_traces(outs)
  plt.show()