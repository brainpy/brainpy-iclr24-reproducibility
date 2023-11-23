# -*- coding: utf-8 -*-

import seaborn as sns
import json

import brainpy as bp
import os

import matplotlib.pyplot as plt
import numpy as np


def read_fn(fn, xs, filter=None):
  with open(os.path.join('speed_results/', fn), 'r') as fin:
    rs = json.load(fin)
  if filter is None:
    times = [(a['num_neuron'], a['sim_time']) for a in tuple(rs.values())[0]]
  else:
    times = [(a['num_neuron'], a['sim_time']) for a in tuple(rs.values())[0] if filter(a)]
  times = dict(times)
  return [(times[x] if x in times else np.nan) for x in xs]


xs = [4000 * i for i in [1, 2, 4, 6, 8, 10]]

annarchy_lif = read_fn('annarchy-1.json', xs=xs)
brainpy_cpu = read_fn('brainpy-jax3.json', xs=xs)
brainpy_gpu = read_fn('brainpy-gpu.json', xs=xs)
neuron_lif = read_fn('neuron.json', xs=xs)
brainpy_np = read_fn('brainpy-np.json', xs=xs)
brian2 = read_fn('brian2-2.json', xs=xs)
# nest_lif = read_fn('nest.json', filter=lambda a: a['num_thread'] == 1, xs=xs)
nest_lif = read_fn('nest-COBALIF.json', xs=xs)
bindsnet = read_fn('bindsnet.json', xs=xs)
bindsnet_gpu = read_fn('bindsnet_gpu.json', xs=xs)


sns.set(font_scale=2.7)
sns.set_style("darkgrid")
fig, gs = bp.visualize.get_figure(1, 1, 6., 9.)
ax1 = fig.add_subplot(gs[0, 0])
linestyle = "--"
linewidth = 4
markersize = 15
cmap=[plt.cm.tab10(1),plt.cm.tab10(6),plt.cm.tab10(2),plt.cm.tab10(3),plt.cm.tab10(4),plt.cm.tab10(5),plt.cm.tab10(0),plt.cm.tab10(7),plt.cm.tab10(8),plt.cm.tab10(9),plt.cm.tab10(10)]
plt.semilogy(xs, neuron_lif, linestyle=linestyle, marker='o', color=cmap[3], label='NEURON', linewidth=linewidth, markersize=markersize)
plt.semilogy(xs, nest_lif, linestyle=linestyle, marker='s', color=cmap[7], label='NEST', linewidth=linewidth, markersize=markersize)
plt.semilogy(xs, annarchy_lif, linestyle=linestyle, marker='*', color=cmap[5], label='ANNarchy', linewidth=linewidth, markersize=markersize)
plt.semilogy(xs, bindsnet, linestyle=linestyle, marker='^', color=cmap[2], label='BindsNet CPU', linewidth=linewidth, markersize=markersize)
plt.semilogy(xs, bindsnet_gpu, linestyle=linestyle, marker='X', color=cmap[4], label='BindsNet GPU', linewidth=linewidth, markersize=markersize)
plt.semilogy(xs, brian2, linestyle=linestyle, marker='P', color=cmap[0], label='Brian2', linewidth=linewidth, markersize=markersize)
plt.semilogy(xs, brainpy_cpu, linestyle=linestyle, marker='D', color=cmap[6], label='BrainPy CPU', linewidth=linewidth, markersize=markersize)
plt.semilogy(xs, brainpy_gpu, linestyle=linestyle, marker='v', color=cmap[1], label='BrainPy GPU', linewidth=linewidth, markersize=markersize)
plt.xticks(xs)

ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
plt.xlabel('Number of neurons')
plt.ylabel('Simulation time [s]')
# lg = plt.legend(fontsize=25, loc='upper left')
# lg = plt.legend(loc=2, bbox_to_anchor=(1.05, 1.0), borderaxespad=0., fontsize=25)
# lg.get_frame().set_alpha(0.3)
plt.title('COBA LIF Network')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

plt.show()