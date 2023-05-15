# -*- coding: utf-8 -*-

'''
Reproduce the results of the``spytorch`` tutorial 2 & 3:

- https://github.com/surrogate-gradient-learning/spytorch/blob/master/notebooks/SpyTorchTutorial2.ipynb
- https://github.com/surrogate-gradient-learning/spytorch/blob/master/notebooks/SpyTorchTutorial3.ipynb

CPU Intel:
- Test acc:
- Each epoch: 48-50 s

RTX A6000
- Test acc: 86.2%
- Each epoch: 48-50 s
'''

import time
from typing import NamedTuple

import numpy as np
import torch
import torchvision
# Notice the difference between "LIF" (leaky integrate-and-fire) and "LI" (leaky integrator)
from norse.torch import LICell, LIState
from norse.torch import LIFState
from norse.torch.module.coba_lif import CobaLIFCell, CobaLIFParameters
from torchvision import datasets

T = 100
LR = 1e-3
alpha = 0.9
beta = 0.85

time_step = 1e-3
nb_steps = 100
batch_size = 256
data_path = r'./data'
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# device = torch.device("cpu")

num_inputs = 28 * 28
num_hidden = 100
num_outputs = 10

loss_fn = torch.nn.NLLLoss()


class SNNState(NamedTuple):
  lif0: LIFState
  readout: LIState


class SNN(torch.nn.Module):
  def __init__(
      self, input_features, hidden_features, output_features, record=False, dt=0.1
  ):
    super(SNN, self).__init__()
    self.l1 = CobaLIFCell(
      input_features,
      hidden_features,
      p=CobaLIFParameters(alpha=100, v_thresh=torch.as_tensor(1.), v_reset=torch.as_tensor(0.),
                          v_rest=torch.as_tensor(0.), e_rev_I=torch.as_tensor(0.)),
      dt=dt,
    )
    self.input_features = input_features
    self.fc_out = torch.nn.Linear(hidden_features, output_features, bias=False)
    self.out = LICell(dt=dt)

    self.hidden_features = hidden_features
    self.output_features = output_features
    self.record = record

  def forward(self, x):
    batch_size, seq_length, _ = x.shape
    s1 = so = None
    voltages = []
    spikes = []

    if self.record:
      self.recording = SNNState(
        LIFState(
          z=torch.zeros(seq_length, batch_size, self.hidden_features),
          v=torch.zeros(seq_length, batch_size, self.hidden_features),
          i=torch.zeros(seq_length, batch_size, self.hidden_features),
        ),
        LIState(
          v=torch.zeros(seq_length, batch_size, self.output_features),
          i=torch.zeros(seq_length, batch_size, self.output_features),
        ),
      )

    for ts in range(seq_length):
      z = x[:, ts, :].view(-1, self.input_features)
      z, s1 = self.l1(z, s1)
      z = self.fc_out(z)
      vo, so = self.out(z, so)
      if self.record:
        self.recording.lif0.z[ts, :] = s1.z
        self.recording.lif0.v[ts, :] = s1.v
        self.recording.lif0.i[ts, :] = s1.i
        self.recording.readout.v[ts, :] = so.v
        self.recording.readout.i[ts, :] = so.i
      voltages += [vo]
      spikes += [s1.z]

    return torch.stack(voltages), torch.stack(spikes)


def decode(x):
  x, _ = torch.max(x, 0)

  log_p_y = torch.nn.functional.log_softmax(x, dim=1)
  return log_p_y


class Model(torch.nn.Module):
  def __init__(self, snn, decoder):
    super(Model, self).__init__()
    self.snn = snn
    self.decoder = decode

  def forward(self, x):
    mem, spk = self.snn(x)
    log_p_y = self.decoder(mem)
    return log_p_y, spk, mem


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


# if torch.cuda.is_available():
#     DEVICE = torch.device("cuda")
# else:
#     DEVICE = torch.device("cpu")
DEVICE = torch.device("cpu")
model = Model(
  snn=SNN(
    input_features=num_inputs,
    hidden_features=num_hidden,
    output_features=num_outputs,
  ),
  decoder=decode,
).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

from tqdm.notebook import trange

EPOCHS = 30  # Increase this number for better performance


def train(model, device, x_train, y_train, optimizer, epoch, max_epochs):
  t0 = time.time()
  model.train()
  losses = []
  iter_cnt = 0
  for data, target in sparse_data_generator(x_train, y_train, batch_size=batch_size, nb_steps=nb_steps,
                                            nb_units=num_inputs):
    data = torch.tensor(data).float()
    target = torch.tensor(target).long()
    data = data.to(device)
    target = target.to(device)

    optimizer.zero_grad()

    output, spk, mem = model(data)

    reg_loss = 1e-5 * torch.sum(spk)  # L1 loss on total number of spikes
    reg_loss += 1e-5 * torch.mean(torch.sum(spk, dim=(0, 1)) ** 2)  # L2 loss on spikes per neuron
    loss = loss_fn(output, target) + reg_loss
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if iter_cnt % 50 == 0:
      print(f"Epoch {epoch}, Iteration {iter_cnt}, Train Set Loss: {loss.item():.2f}")
    iter_cnt += 1

  t1 = time.time()
  mean_loss = np.mean(losses)
  print("Epoch %i: loss=%.5f  time: %f" % (epoch, mean_loss, t1 - t0))
  return losses, mean_loss


def run_model(model, device, x_test, y_test, epoch):
  model.eval()
  accs = []
  losses = []
  with torch.no_grad():
    for data, target in sparse_data_generator(x_test, y_test, batch_size=batch_size, nb_steps=nb_steps,
                                              nb_units=num_inputs):
      data = torch.tensor(data).float()
      target = torch.tensor(target).long()
      data = data.to(device)
      target = target.to(device)

      output, spk, mem = model(data)

      reg_loss = 1e-5 * torch.sum(spk)  # L1 loss on total number of spikes
      reg_loss += 1e-5 * torch.mean(torch.sum(spk, dim=(0, 1)) ** 2)  # L2 loss on spikes per neuron
      loss = loss_fn(output, target) + reg_loss
      losses.append(loss.detach().cpu().numpy())

      m, _ = torch.max(mem, 0)  # max over time
      _, am = torch.max(m, 1)  # argmax over output units
      tmp = np.mean((target == am).detach().cpu().numpy())  # compare to labels
      accs.append(tmp)

  return np.mean(losses), np.mean(accs)


transform = torchvision.transforms.Compose(
  [
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
  ]
)

train_dataset = datasets.FashionMNIST(data_path, train=True, download=True)
test_dataset = datasets.FashionMNIST(data_path, train=True, download=True)

x_train = np.array(train_dataset.data, dtype=np.float_)
x_train = x_train.reshape(x_train.shape[0], -1) / 255
x_test = np.array(test_dataset.data, dtype=np.float_)
x_test = x_test.reshape(x_test.shape[0], -1) / 255
y_train = np.array(train_dataset.targets, dtype=np.int_)
y_test = np.array(test_dataset.targets, dtype=np.int_)

training_losses = []
mean_losses = []
test_losses = []
accuracies = []

for epoch in trange(EPOCHS):
  training_loss, mean_loss = train(
    model, DEVICE, x_train, y_train, optimizer, epoch, max_epochs=EPOCHS
  )
  test_loss, accuracy = run_model(model, DEVICE, x_test, y_test, epoch)
  training_losses += training_loss
  mean_losses.append(mean_loss)
  test_losses.append(test_loss)
  accuracies.append(accuracy)

print(f"final accuracy: {accuracies[-1]}")
