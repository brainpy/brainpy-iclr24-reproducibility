# spikingjelly.activation_based.examples.conv_fashion_mnist
'''
Reproduce the results of the``spytorch`` tutorial 2 & 3:

- https://github.com/surrogate-gradient-learning/spytorch/blob/master/notebooks/SpyTorchTutorial2.ipynb
- https://github.com/surrogate-gradient-learning/spytorch/blob/master/notebooks/SpyTorchTutorial3.ipynb

CPU m1 pro:
- Each epoch: 60-61 s

CPU Intel:
- Each epoch: 45-47 s

RTX A6000
- Each epoch: 55-57 s
'''

import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.gridspec import GridSpec
from spikingjelly.activation_based import neuron, functional, surrogate, layer, monitor
from torchvision import datasets, transforms

num_inputs = 28 * 28
num_hidden = 100
num_outputs = 10

lr = 1e-3

alpha = 0.9
beta = 0.85

time_step = 1e-3
nb_steps = 100
batch_size = 256
data_path = r'./data'
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device = torch.device("cpu")

num_steps = 100

# Define a transform
transform = transforms.Compose([
  transforms.Resize((28, 28)),
  transforms.Grayscale(),
  transforms.ToTensor(),
  transforms.Normalize((0,), (1,))])

train_dataset = datasets.FashionMNIST(data_path, train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(data_path, train=False, download=True, transform=transform)

# Create DataLoaders
# Standardize data
# x_train = torch.tensor(train_dataset.train_data, device=device, dtype=dtype)
x_train = np.array(train_dataset.data, dtype=np.float_)
x_train = x_train.reshape(x_train.shape[0], -1) / 255
# x_test = torch.tensor(test_dataset.test_data, device=device, dtype=dtype)
x_test = np.array(test_dataset.data, dtype=np.float_)
x_test = x_test.reshape(x_test.shape[0], -1) / 255

# y_train = torch.tensor(train_dataset.train_labels, device=device, dtype=dtype)
# y_test  = torch.tensor(test_dataset.test_labels, device=device, dtype=dtype)
y_train = np.array(train_dataset.targets, dtype=np.int_)
y_test = np.array(test_dataset.targets, dtype=np.int_)


class NonSpikingLIFNode(neuron.LIFNode):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def single_step_forward(self, x: torch.Tensor):
    self.v_float_to_tensor(x)

    if self.training:
      self.neuronal_charge(x)
    else:
      if self.v_reset is None:
        if self.decay_input:
          self.v = self.neuronal_charge_decay_input_reset0(x, self.v, self.tau)
        else:
          self.v = self.neuronal_charge_no_decay_input_reset0(x, self.v, self.tau)

      else:
        if self.decay_input:
          self.v = self.neuronal_charge_decay_input(x, self.v, self.v_reset, self.tau)
        else:
          self.v = self.neuronal_charge_no_decay_input(x, self.v, self.v_reset, self.tau)


class SynapticLIFNode(neuron.IFNode):
  def __init__(self, alpha, beta, reset_mechanism='subtract'):
    super().__init__()
    self.reset_mechanism = reset_mechanism
    self.alpha = alpha
    self.beta = beta

  def single_step_forward(self, x: torch.Tensor):
    if self.training:
      self.neuronal_charge(x)

  def neuronal_charge(self, x: torch.Tensor):
    self.I_syn = self.alpha * self.I_syn + x
    self.v = self.beta * self.v + self.I_syn


class Net(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc = nn.Sequential(
      layer.SynapseFilter(tau=2., learnable=True),
      layer.Linear(num_inputs, num_hidden),
      neuron.LIFNode(surrogate_function=surrogate.ATan()),
      layer.SynapseFilter(tau=2., learnable=True),
      layer.Linear(num_hidden, num_outputs),
      neuron.LIFNode(surrogate_function=surrogate.ATan(), v_reset=None, v_threshold=0., decay_input=True,
                     store_v_seq=True)
    )

  def forward(self, x):
    for t in range(nb_steps):
      self.fc(x[:, t, :])

    return self.fc[-1].v


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


def print_batch_accuracy(data, targets, train=False):
  output, _ = net(data.view(batch_size, -1))
  _, idx = output.sum(dim=0).max(1)
  acc = np.mean((targets == idx).detach().cpu().numpy())

  if train:
    print(f"Train set accuracy for a single minibatch: {acc * 100:.2f}%")
  else:
    print(f"Test set accuracy for a single minibatch: {acc * 100:.2f}%")


def compute_classification_accuracy(net, xs, ys):
  """ Computes classification accuracy on supplied data in batches. """
  accs = []
  for x_local, y_local in sparse_data_generator(xs, ys, batch_size=batch_size, nb_steps=nb_steps,
                                                nb_units=num_inputs):
    x_local = torch.tensor(x_local).float()
    y_local = torch.tensor(y_local)
    x_local = x_local.to(device)
    y_local = y_local.to(device)

    v_seq_monitor = monitor.AttributeMonitor('v', pre_forward=False, net=net, instance=neuron.LIFNode)

    net.eval()
    net(x_local)

    mem_rec = v_seq_monitor['fc.5']
    mem_rec = torch.stack(mem_rec, dim=0)

    m, _ = torch.max(mem_rec, 0)  # max over time
    _, am = torch.max(m, 1)  # argmax over output units
    tmp = np.mean((y_local == am).detach().cpu().numpy())  # compare to labels
    accs.append(tmp)

    v_seq_monitor.remove_hooks()
    del v_seq_monitor
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

  net = Net().to(device)

  # 使用Adam优化器
  optimizer = torch.optim.Adam(net.parameters(), lr=lr)

  log_softmax_fn = nn.LogSoftmax(dim=1)
  loss_fn = nn.NLLLoss()

  for epoch in range(num_epochs):
    local_loss = []
    iter_cnt = 1
    t0 = time.time()
    for data, targets in sparse_data_generator(x_train, y_train, batch_size=batch_size, nb_steps=nb_steps,
                                               nb_units=num_inputs):
      functional.reset_net(net)
      data = torch.tensor(data).float()
      targets = torch.tensor(targets).long()
      data = data.to(device)
      targets = targets.to(device)

      spike_seq_monitor = monitor.OutputMonitor(net, neuron.LIFNode)
      v_seq_monitor = monitor.AttributeMonitor('v', pre_forward=False, net=net, instance=neuron.LIFNode)

      net.train()
      net(data)
      mem_rec = v_seq_monitor['fc.5']
      spk_rec = spike_seq_monitor['fc.2']

      mem_rec = torch.stack(mem_rec, dim=0)
      spk_rec = torch.stack(spk_rec, dim=0)

      m, _ = torch.max(mem_rec, 0)
      log_p_y = log_softmax_fn(m)
      reg_loss_1 = 1e-5 * torch.sum(spk_rec)  # L1 loss on total number of spikes
      reg_loss_2 = 1e-5 * torch.mean(torch.sum(spk_rec, dim=(0, 1)) ** 2)  # L2 loss on spikes per neuron
      loss_val = loss_fn(log_p_y, targets) + reg_loss_1 + reg_loss_2

      optimizer.zero_grad()
      loss_val.backward()
      optimizer.step()

      # Store loss history for future plotting
      local_loss.append(loss_val.item())
      if iter_cnt % 50 == 0:
        print(f"Epoch {epoch}, Iteration {iter_cnt}, Train Set Loss: {loss_val.item():.2f}")
      iter_cnt += 1

      spike_seq_monitor.remove_hooks()
      v_seq_monitor.remove_hooks()
      del spike_seq_monitor
      del v_seq_monitor

    t1 = time.time()
    mean_loss = np.mean(local_loss)
    print("Epoch %i: loss=%.5f  time: %f" % (epoch + 1, mean_loss, t1 - t0))
    loss_hist.append(mean_loss)

    print("Training accuracy: %.3f" % (compute_classification_accuracy(net, x_train, y_train)))
    print("Test accuracy: %.3f" % (compute_classification_accuracy(net, x_test, y_test)))


  def get_mini_batch_results(net, xs, ys):
    for x_local, y_local in sparse_data_generator(xs, ys, batch_size=batch_size, nb_steps=nb_steps,
                                                  nb_units=num_inputs):
      img = torch.tensor(x_local).float()
      label = torch.tensor(y_local)
      spike_seq_monitor = monitor.OutputMonitor(net, neuron.LIFNode)
      v_seq_monitor = monitor.AttributeMonitor('v', pre_forward=False, net=net, instance=neuron.LIFNode)

      net.eval()
      net(img)
      mem_rec = v_seq_monitor['fc.5']
      spk_rec = spike_seq_monitor['fc.2']

      mem_rec = torch.stack(mem_rec, dim=0)
      spk_rec = torch.stack(spk_rec, dim=0)

      return mem_rec, spk_rec


  outs, spikes = get_mini_batch_results(net, x_train, y_train)
  # Let's plot the hidden layer spiking activity for some input stimuli
  fig = plt.figure(dpi=100)
  plot_voltage_traces(outs)
  plt.show()
