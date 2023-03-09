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

custom_operator = np.asarray([2, 3, 6, 8, 11, 14])
sparse_operator = np.asarray([103, 385, 1470, 3230, 5775, 9190])
dense_operator = np.asarray([49, 216, 925, 2154, 3923, 6100])
pytorch = np.asarray([148, 547, 2102, 4601, 7968, 12404])

sns.set(font_scale=1.5)
sns.set_style("white")
fig, gs = bp.visualize.get_figure(1, 1, 6., 9.)
ax = fig.add_subplot(gs[0, 0])
plt.semilogy(xs, pytorch, linestyle="--", marker='x', label='PyTorch', linewidth=2)
plt.semilogy(xs, sparse_operator, linestyle="--", marker='P', label='JAX sparse operator', linewidth=3, markersize=10)
plt.semilogy(xs, dense_operator, linestyle="--", marker='s', label='JAX dense operator', linewidth=3, markersize=10)
plt.semilogy(xs, custom_operator, linestyle="--", marker='o', label='custom operator')
plt.xticks(xs)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlabel('Number of neurons')
plt.ylabel('Simulation time [s]')
lg = plt.legend(fontsize=12, loc='upper right')
lg.get_frame().set_alpha(0.3)
plt.title('COBA Network')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
plt.show()