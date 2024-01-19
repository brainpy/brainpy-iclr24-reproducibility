# -*- coding: utf-8 -*-

import seaborn as sns
import json

import brainpy as bp
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def read_fn(fn, xs, filter=None):
  with open(os.path.join('speed_result_HH/', fn), 'r') as fin:
    rs = json.load(fin)
  if filter is None:
    times = [(a['num_neuron'], a['sim_time']) for a in tuple(rs.values())[0]]
  else:
    times = [(a['num_neuron'], a['sim_time']) for a in tuple(rs.values())[0] if filter(a)]
  times = dict(times)
  return [(times[x] if x in times else np.nan) for x in xs]


xs = [4000 * i for i in [1, 2, 4, 6, 8, 10, 15, 20, 30]]

brainpy_cpu = read_fn('brainpy-2-cpu.json', xs=xs)
brainpy_gpu = read_fn('brainpy-2-gpu.json', xs=xs)
brian2 = read_fn('brian2.json', xs=xs)
nest = read_fn('NEST.json', xs=xs)
neuron = read_fn('NEURON.json', xs=xs)

sns.set(font_scale=2.5)
sns.set_style("darkgrid")
fig, gs = bp.visualize.get_figure(1, 1, 6., 9.)
ax = fig.add_subplot(gs[0, 0])
linestyle = "--"
linewidth = 4
markersize = 15
cmap=[plt.cm.tab10(1),plt.cm.tab10(6),plt.cm.tab10(2),plt.cm.tab10(3),plt.cm.tab10(4),plt.cm.tab10(5),plt.cm.tab10(0),plt.cm.tab10(7),plt.cm.tab10(8),plt.cm.tab10(9),plt.cm.tab10(10)]
plt.semilogy(xs, neuron, linestyle="--", marker='o', color=cmap[3], label='NEURON', linewidth=linewidth, markersize=markersize)
plt.semilogy(xs, nest, linestyle="--", marker='s', color=cmap[7], label='NEST', linewidth=linewidth, markersize=markersize)
plt.semilogy(xs, brian2, linestyle="--", marker='P', color=cmap[0], label='Brian2', linewidth=linewidth, markersize=markersize)
plt.semilogy(xs, brainpy_cpu, linestyle="--", marker='D', color=cmap[6], label='BrainPy-CPU', linewidth=linewidth, markersize=markersize)
plt.semilogy(xs, brainpy_gpu, linestyle="--", marker='v', color=cmap[1], label='BrainPy-GPU', linewidth=linewidth, markersize=markersize)
plt.xticks([0, 20000, 40000, 60000, 80000, 100000, 120000])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlabel('Number of neurons')
# plt.ylabel('Simulation time [s]')
# lg = plt.legend(fontsize=15, loc='upper right')
# lg = plt.legend(loc=2, bbox_to_anchor=(1.05, 1.0), borderaxespad=0., fontsize=25)
# lg.get_frame().set_alpha(0.3)
plt.title('COBA HH Network')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
# plt.tight_layout()
plt.show()