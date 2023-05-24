# -*- coding: utf-8 -*-

'''
simulation time: 10000.
machine: Intel CPU
bindsnet = 
'''

import argparse
import os
from time import time as t

from bindsnet.encoding import poisson
from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes, CurrentLIFNodes
from bindsnet.network.topology import Connection, SparseConnection

import torch
import matplotlib.pyplot as plt

# "Warm up" the GPU.
torch.set_default_tensor_type("torch.cuda.FloatTensor")
x = torch.rand(1000)
del x


def BindsNET_cpu(scale, time):
  t0 = t()

  torch.set_default_tensor_type("torch.FloatTensor")

  t1 = t()

  # Create network object.
  network = Network(dt=0.1)

  # Create input layer of Poisson spike generators.
  source_exc_layer = Input(n=3200 * scale)
  source_inh_layer = Input(n=800 * scale)

  # Create excitatory and inhibitory populations.
  # exc_layer = LIFNodes(n=100)
  # inh_layer = LIFNodes(n=100)
  exc_layer = CurrentLIFNodes(n=3200 * scale)
  inh_layer = CurrentLIFNodes(n=800 * scale)

  # Create connections between layers.
  source_exc_conn = Connection(source_exc_layer, exc_layer)
  source_inh_conn = Connection(source_inh_layer, inh_layer)
  exc_exc_conn = Connection(exc_layer, exc_layer, sparsity=0.02)
  inh_inh_conn = Connection(inh_layer, inh_layer, sparsity=0.02)
  exc_inh_conn = Connection(exc_layer, inh_layer, sparsity=0.02)
  inh_exc_conn = Connection(inh_layer, exc_layer, sparsity=0.02)

  # Add layers and connections to the network object.
  network.add_layer(source_exc_layer, 'XE')
  network.add_layer(source_inh_layer, 'XI')
  network.add_layer(exc_layer, 'E')
  network.add_layer(inh_layer, 'I')
  network.add_connection(source_exc_conn, 'XE', 'E')
  network.add_connection(source_inh_conn, 'XI', 'I')
  network.add_connection(exc_exc_conn, 'E', 'E')
  network.add_connection(inh_inh_conn, 'I', 'I')
  network.add_connection(exc_inh_conn, 'E', 'I')
  network.add_connection(inh_exc_conn, 'I', 'E')

  data = {"E": poisson(datum=torch.rand(3200 * scale), time=time, dt=0.1),
          "I": poisson(datum=torch.rand(800 * scale), time=time, dt=0.1)}
  # data = {"E": torch.ones(3200 * scale) * 20.,
  #         "I": torch.ones(800 * scale) * 20.}
  # data = {"E": torch.bernoulli(20. * torch.ones(3200 * scale)).byte(),
  #         "I": torch.bernoulli(20. * torch.ones(800 * scale)).byte()}
  network.run(inputs=data, time=time)

  return t() - t0, t() - t1


def BindsNET_gpu(scale, time):
  if torch.cuda.is_available():
    t0 = t()

    torch.set_default_tensor_type("torch.FloatTensor")

    t1 = t()

    # Create network object.
    network = Network(dt=0.1)

    # Create input layer of Poisson spike generators.
    source_exc_layer = Input(n=3200 * scale)
    source_inh_layer = Input(n=800 * scale)

    # Create excitatory and inhibitory populations.
    # exc_layer = LIFNodes(n=100)
    # inh_layer = LIFNodes(n=100)
    exc_layer = CurrentLIFNodes(n=3200 * scale)
    inh_layer = CurrentLIFNodes(n=800 * scale)

    # Create connections between layers.
    source_exc_conn = Connection(source_exc_layer, exc_layer)
    source_inh_conn = Connection(source_inh_layer, inh_layer)
    exc_exc_conn = Connection(exc_layer, exc_layer, sparsity=0.02)
    inh_inh_conn = Connection(inh_layer, inh_layer, sparsity=0.02)
    exc_inh_conn = Connection(exc_layer, inh_layer, sparsity=0.02)
    inh_exc_conn = Connection(inh_layer, exc_layer, sparsity=0.02)

    # Add layers and connections to the network object.
    network.add_layer(source_exc_layer, 'XE')
    network.add_layer(source_inh_layer, 'XI')
    network.add_layer(exc_layer, 'E')
    network.add_layer(inh_layer, 'I')
    network.add_connection(source_exc_conn, 'XE', 'E')
    network.add_connection(source_inh_conn, 'XI', 'I')
    network.add_connection(exc_exc_conn, 'E', 'E')
    network.add_connection(inh_inh_conn, 'I', 'I')
    network.add_connection(exc_inh_conn, 'E', 'I')
    network.add_connection(inh_exc_conn, 'I', 'E')

    data = {"E": poisson(datum=torch.rand(3200 * scale), time=time, dt=0.1).to("cuda"),
            "I": poisson(datum=torch.rand(800 * scale), time=time, dt=0.1).to("cuda")}
    # data = {"E": (torch.ones(3200 * scale) * 20.).to("cuda"),
    #         "I": (torch.ones(800 * scale) * 20.).to("cuda")}
    # data = {"E": torch.bernoulli(20. * torch.ones(3200 * scale)).byte(),
    #         "I": torch.bernoulli(20. * torch.ones(800 * scale)).byte()}

    network.to("cuda")

    network.run(inputs=data, time=time)

    return t() - t0, t() - t1


t_total, t_cost = BindsNET_cpu(scale=1, time=10000)
print(f"Total time: {t_total}, Running time: {t_cost}")

# t_total, t_cost = BindsNET_gpu(scale=1, time=10000)
# print(f"Total time: {t_total}, Running time: {t_cost}")



# Simulate network for 500 milliseconds.
# spikes = {}
# voltages = {}
# for l in network.layers:
#   spikes[l] = torch.zeros(500, network.layers[l].n)
#   voltages[l] = torch.zeros(500, network.layers[l].n)
#
# for i in range(500):
#   network.run(inputs={'X': torch.bernoulli(0.1 * torch.ones(source_layer.n))}, time=1)
#   for l in spikes:
#     spikes[l][i] = network.monitors[l].get('s').squeeze()
#     voltages[l][i] = network.monitors[l].get('v').squeeze()


# # Plot spikes and voltages of excitatory and inhibitory populations.
# plt.figure(figsize=(12, 6))
# plt.subplot(221)
# plt.plot(spikes['E'].t(), torch.arange(spikes['E'].size(1)).repeat(spikes['E'].size(0), 1), '|k')
# plt.title('Excitatory spikes')
# plt.xlabel('Time (ms)')
# plt.ylabel('Neuron index')
#
# plt.subplot(222)
# plt.plot(spikes['I'].t(), torch.arange(spikes['I'].size(1)).repeat(spikes['I'].size(0), 1), '|k')
# plt.title('Inhibitory spikes')
# plt.xlabel('Time (ms)')
# plt.ylabel('Neuron index')
#
# plt.subplot(223)
# plt.plot(voltages['E'])
# plt.title('Excitatory voltages')
# plt.xlabel('Time (ms)')
# plt.ylabel('Voltage (mV)')
#
# plt.subplot(224)
# plt.plot(voltages['I'])
# plt.title('Inhibitory voltages')
# plt.xlabel('Time (ms)')
# plt.ylabel('Voltage (mV)')
#
# # Show plots
# plt.show()
