# -*- coding: utf-8 -*-
'''
This script should be run on GPU.
'''

import time

import brainpy as bp
import brainpy.math as bm
import jax
import pynvml

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)

bm.set_platform('gpu')
bm.disable_gpu_memory_preallocation()

sizes = [2000, 4000, 8000, 10000, 20000, 30000, 40000, 50000]
probs = [0.01, 0.1, 0.2, 0.4]


def dense(size, prob):
  bm.random.seed()
  vector = bm.random.randn(size)
  dense_A = bm.random.randn(size, size)
  t0 = time.time()
  try:
    for _ in range(100):
      hidden = jax.block_until_ready(vector @ dense_A)
    used = pynvml.nvmlDeviceGetMemoryInfo(handle).used / 1024 / 1024 / 1024
    cost_t = time.time() - t0
    print(f'Dense: size {size}, prob {prob}, cost_t {cost_t} s, space {used} GB')
  except:
    pass
  bm.clear_buffer_memory()


def sparse(size, prob):
  bm.random.seed()
  vector = bm.random.randn(size)
  sparse_A = bp.conn.FixedProb(prob=prob, allow_multi_conn=True)(size, size).require('pre2post')
  t0 = time.time()
  try:
    for _ in range(100):
      hidden = jax.block_until_ready(bm.sparse.csrmv(1., sparse_A[0], sparse_A[1], vector, shape=(size, size)))
    cost_t = time.time() - t0
    used = pynvml.nvmlDeviceGetMemoryInfo(handle).used / 1024 / 1024 / 1024
    print(f'Sparse: size {size}, prob {prob}, cost_t {cost_t} s, space {used} GB')
  except:
    pass
  bm.clear_buffer_memory()


def jit(size, prob):
  bm.random.seed()
  vector = bm.random.randn(size)
  jit_sigma = 0.1
  jit_seed = 2023
  t0 = time.time()
  try:
    for _ in range(100):
      hidden = jax.block_until_ready(bm.jitconn.mv_prob_uniform(vector,
                                                                w_low=0.,
                                                                w_high=jit_sigma,
                                                                conn_prob=prob,
                                                                shape=(size, size),
                                                                seed=jit_seed))
    cost_t = time.time() - t0
    used = pynvml.nvmlDeviceGetMemoryInfo(handle).used / 1024 / 1024 / 1024
    print(f'JIT: size {size}, prob {prob}, cost_t {cost_t} s, space {used} GB')
  except:
    pass
  bm.clear_buffer_memory()


def visualize():
  import seaborn as sns
  import matplotlib.pyplot as plt
  import numpy as np

  sizes = [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 100000, 300000, 500000, 1000000]

  time_dense = np.asarray(
    [0.5889952182769775, 1.47330904006958, 3.754756450653076, 5.139796018600464, 11.846440076828003])
  time_sparse = np.asarray(
    [0.9121279716491699, 1.025848627090454, 1.4970710277557373, 2.142857551574707, 2.9624083042144775,
     4.003606081008911, 4.168002367019653, 5.100186347961426, 8.344945430755615, 79.97696185112])
  time_jit = np.asarray([0.12275505065917969, 0.10699820518493652, 0.18382978439331055, 0.18160152435302734,
                         0.2651815414428711, 0.30461549758911133, 0.4543952941894531, 0.47950029373168945,
                         0.7580647468566895, 5.468803405761719, 15.045670509338379, 62.35301637649536])

  memory_dense = np.asarray([1.068, 2.191, 4.06, 6.672, 10.031]) * 1024
  memory_sparse = np.asarray([0.791, 0.808, 0.885, 0.961, 0.967, 1.183, 1.327, 1.5, 1.898, 3.879]) * 1024
  memory_jit = np.asarray([0.693, 0.693, 0.693, 0.693, 0.693, 0.693, 0.693, 0.693, 0.693, 0.693, 0.699, 0.705]) * 1024

  times_jit_test = np.asarray([48, 58, 77, 88, 166, 306, 405, 709])
  times_sparse_test = np.asarray([60, 69, 87, 101, 389, 420, 687, 1039])
  times_dense_test = np.asarray([13, 36, 131, 202, 773, 1896])

  fig, gs = bp.visualize.get_figure(1, 1, 7., 10.5)
  sns.set(font_scale=3)
  sns.set_style("darkgrid")
  ax1 = fig.add_subplot(gs[0, 0])
  ax1.semilogy(sizes, time_jit, marker='o', label='JIT conn OP', linewidth=4, markersize=15)
  ax1.semilogy(sizes[:5], time_dense, marker='s', label='Dense OP', linewidth=4, markersize=15)
  ax1.semilogy(sizes[:-2], time_sparse, marker='d', label='Sparse OP', linewidth=4, markersize=15)
  ax1.set_xlabel('Matrix size')
  ax1.set_ylabel('Simulation time [s]')
  ax1.set_xscale("log")
  ax1.spines['top'].set_visible(False)
  ax1.spines['right'].set_visible(False)
  plt.legend(fontsize=20, loc='lower right')
  plt.title('Speed comparison')
  handles, labels = plt.gca().get_legend_handles_labels()
  order = [1, 2, 0]
  plt.legend([handles[i] for i in order], [labels[i] for i in order], fontsize=25, loc='lower right')
  plt.savefig('speed_comparison.pdf', bbox_inches='tight')

  ax2 = fig.add_subplot(gs[0, 0])
  ax2.semilogy(sizes, memory_jit, marker='o', label='JIT conn op', linewidth=4, markersize=15, linestyle='-')
  ax2.semilogy(sizes[:5], memory_dense, marker='s', label='Dense op', linewidth=4, markersize=15, linestyle='-')
  ax2.semilogy(sizes[:-2], memory_sparse, marker='d', label='Sparse op', linewidth=4, markersize=15, linestyle='-')
  ax2.set_ylabel('Memory Usage [MB]')
  ax2.set_xlabel('Matrix size')
  ax2.set_xscale("log")
  ax2.spines['top'].set_visible(False)
  ax2.spines['right'].set_visible(False)
  plt.title('Memory comparison')
  handles, labels = plt.gca().get_legend_handles_labels()
  order = [1, 2, 0]
  plt.legend([handles[i] for i in order], [labels[i] for i in order], fontsize=25, loc='upper right')
  plt.savefig('memory_comparison.pdf', bbox_inches='tight')


if __name__ == '__main__':
  sizes = [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 100000, 300000, 500000, 1000000]
  probs = [0.01, 0.1, 0.2, 0.4]
  for size in sizes:
    for prob in probs:
      dense(size, prob)   # stop at size=60000
      sparse(size, prob)  # stop at size=300000
      jit(size, prob)
