# -*- coding: utf-8 -*-

import torch
import numpy as np
import matplotlib.pyplot as plt

from norse.torch import LIFParameters, LIFState
from norse.torch.module.lif import LIFCell, LIFRecurrentCell

# Notice the difference between "LIF" (leaky integrate-and-fire) and "LI" (leaky integrator)
from norse.torch import LICell, LIState

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from typing import NamedTuple


T = 100
LR = 1e-3
alpha = 0.9
beta = 0.85

time_step = 1e-3
nb_steps = 100
batch_size = 256
data_path = r'./data'
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

num_inputs = 28*28
num_hidden = 100
num_outputs = 10


class SNNState(NamedTuple):
    lif0: LIFState
    readout: LIState


class SNN(torch.nn.Module):
    def __init__(
        self, input_features, hidden_features, output_features, record=False, dt=0.001
    ):
        super(SNN, self).__init__()
        self.l1 = LIFRecurrentCell(
            input_features,
            hidden_features,
            p=LIFParameters(alpha=100, v_th=torch.tensor(0.5)),
            dt=dt,
        )
        self.input_features = input_features
        self.fc_out = torch.nn.Linear(hidden_features, output_features, bias=False)
        self.out = LICell(dt=dt)

        self.hidden_features = hidden_features
        self.output_features = output_features
        self.record = record

    def forward(self, x):
      seq_length, batch_size, _, _, _ = x.shape
      s1 = so = None
      voltages = []

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
        z = x[ts, :, :, :].view(-1, self.input_features)
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

      return torch.stack(voltages)


def decode(x):
  x, _ = torch.max(x, 0)
  log_p_y = torch.nn.functional.log_softmax(x, dim=1)
  return log_p_y


class Model(torch.nn.Module):
  def __init__(self, encoder, snn, decoder):
    super(Model, self).__init__()
    self.encoder = encoder
    self.snn = snn
    self.decoder = decoder

  def forward(self, x):
    x = self.encoder(x)
    x = self.snn(x)
    log_p_y = self.decoder(x)
    return log_p_y



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


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

model = Model(
    encoder=SpikeLatencyLIFEncoder(
        seq_length=T,
    ),
    snn=SNN(
        input_features=num_inputs,
        hidden_features=num_hidden,
        output_features=num_outputs,
    ),
    decoder=decode,
).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

from tqdm.notebook import tqdm, trange

EPOCHS = 30  # Increase this number for better performance


def train(model, device, train_loader, optimizer, epoch, max_epochs):
    model.train()
    losses = []

    for (data, target) in tqdm(train_loader, leave=False):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = torch.nn.functional.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    mean_loss = np.mean(losses)
    return losses, mean_loss


def test(model, device, test_loader, epoch):
  model.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      data, target = data.to(device), target.to(device)
      output = model(data)
      test_loss += torch.nn.functional.nll_loss(
        output, target, reduction="sum"
      ).item()  # sum up batch loss
      pred = output.argmax(
        dim=1, keepdim=True
      )  # get the index of the max log-probability
      correct += pred.eq(target.view_as(pred)).sum().item()

  test_loss /= len(test_loader.dataset)

  accuracy = 100.0 * correct / len(test_loader.dataset)

  return test_loss, accuracy


train_dataset = datasets.FashionMNIST(data_path, train=True, download=True)
test_dataset = datasets.FashionMNIST(data_path, train=True, download=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

training_losses = []
mean_losses = []
test_losses = []
accuracies = []

for epoch in trange(EPOCHS):
    training_loss, mean_loss = train(
        model, DEVICE, train_loader, optimizer, epoch, max_epochs=EPOCHS
    )
    test_loss, accuracy = test(model, DEVICE, test_loader, epoch)
    training_losses += training_loss
    mean_losses.append(mean_loss)
    test_losses.append(test_loss)
    accuracies.append(accuracy)

print(f"final accuracy: {accuracies[-1]}")