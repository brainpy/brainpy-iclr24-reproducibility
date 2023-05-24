# -*- coding: utf-8 -*-

import time

import brainpy as bp
import brainpy.math as bm
import jax
import pynvml
from brainpylib.jitconn_ops import matvec_prob_conn_normal_weight
from brainpylib.sparse_ops import cusparse_csr_matvec

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
      hidden = jax.block_until_ready(cusparse_csr_matvec(1., sparse_A[0], sparse_A[1], vector, shape=(size, size)))
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
      hidden = jax.block_until_ready(matvec_prob_conn_normal_weight(vector,
                                                                    w_mu=0.,
                                                                    w_sigma=jit_sigma,
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
  import matplotlib.pyplot as plt

  dense_res = [(10000, 20000, 30000, ),  # size
               (0.26956772804260254, 0.5185866355895996, 1.3511295318603516, ),  # time
               (1.2187995910644531, 2.335987091064453, 4.199268341064453, )]  # memory

  sparse_res = [(10000, 20000, 30000, 60000, 80000, ),
                (0.762305736541748, 1.6999387741088867, 3.432894706726074, 11.964096546173096, 20.59521222114563, ),
                (0.9668464660644531, 1.0781745910644531, 1.2637214660644531, 2.271533966064453, 3.314502716064453, )]

  jit_res = [(10000, 20000, 30000, 60000, 80000,
              100000, 500000, 10000000),
             (0.07238912582397461, 0.15148401260375977, 0.2659947872161865, 0.7660796642303467, 1.2067103385925293,
              1.7565438747406006, 41.09631705284119, 166.66332340240479, ),
             (0.9297370910644531, 0.9297370910644531, 0.9297370910644531, 0.9297370910644531, 0.9297370910644531,
              0.9297370910644531, 0.9297370910644531, 0.9297370910644531,)]

  plt.figure(figsize=(5, 4))
  plt.plot(dense_res[0], dense_res[1], label='dense op')
  plt.plot(sparse_res[0], sparse_res[1], label='sparse op')
  plt.plot(jit_res[0], jit_res[1], label='jit op')
  plt.xscale('log')
  plt.yscale('log')
  plt.ylabel('Time [ms]')
  plt.xlabel('Matrix size')
  plt.title('Speed Comparison')
  plt.legend()

  plt.figure(figsize=(5, 4))
  plt.plot(dense_res[0], dense_res[2], label='dense op')
  plt.plot(sparse_res[0], sparse_res[2], label='sparse op')
  plt.plot(jit_res[0], jit_res[2], label='jit op')
  plt.xscale('log')
  plt.ylabel('Memory [GB]')
  plt.xlabel('Matrix size')
  plt.title('Memory Comparison')
  plt.legend()

  plt.show()


if __name__ == '__main__':
  visualize()

  # # sizes = [2000, 4000, 8000, 10000, 20000, 30000, 40000, 50000]
  # # probs = [0.01, 0.1, 0.2, 0.4]
  # sizes = [10000, 30000, 60000, 80000, 100000]
  # sizes = [10000, 30000, ]
  # # sizes = [int(5e5)]
  # probs = [0.1]
  #
  # for size in sizes:
  #   for prob in probs:
  #     dense(size, prob)
  #     sparse(size, prob)
  #     jit(size, prob)
