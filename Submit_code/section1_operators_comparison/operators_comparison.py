# -*- coding: utf-8 -*-

import time

import brainpy as bp
import brainpy.math as bm
import jax
import torch
from jax.experimental import sparse

# If cpu
bm.set_platform('cpu')
device = torch.device('cpu')


# If gpu
# bm.set_platform('gpu')
# bm.disable_gpu_memory_preallocation()
# device = torch.device('cuda')


def jax_dense(size, prob, spikes):
  bm.random.seed()
  vector = bm.random.binomial(1, spikes, size)
  dense_A = bm.random.randn(size, size)
  t0 = time.time()
  try:
    for _ in range(100):
      hidden = jax.block_until_ready(vector @ dense_A)
    cost_t = time.time() - t0
    print(f'JAX Dense: size {size}, prob {prob}, spike {spikes}, cost_t {cost_t} s')
  except:
    raise Exception('JAX dense failed')
  del vector, dense_A
  bm.clear_buffer_memory()


def jax_sparse(size, prob, spikes):
  bm.random.seed()
  vector = bm.random.binomial(1, spikes, size).astype(float).to_jax()
  sparse_A = sparse.CSR.fromdense(bm.random.randn(size, size).to_jax())

  t0 = time.time()
  try:
    for _ in range(100):
      hidden = jax.block_until_ready(sparse.csr_matvec(sparse_A, vector))
    cost_t = time.time() - t0
    print(f'JAX sparse: size {size}, prob {prob}, spike {spikes}, cost_t {cost_t} s')
  except:
    raise Exception('JAX sparse failed')
  del vector, sparse_A
  bm.clear_buffer_memory()


def torch_dense(size, prob, spikes):
  torch.manual_seed(0)
  torch.set_num_threads(1)
  vector = torch.distributions.Binomial(1, spikes).sample((size,)).to(device)
  dense_A = torch.randn(size, size).to(device)
  if device == torch.device('cuda'):
    torch.cuda.synchronize(device="cuda:0")
  t0 = time.time()
  try:
    for _ in range(100):
      hidden = vector @ dense_A
    if device == torch.device('cuda'):
      torch.cuda.synchronize(device="cuda:0")
    cost_t = time.time() - t0
    print(f'Torch Dense: size {size}, prob {prob}, spike {spikes}, cost_t {cost_t} s')
  except:
    raise Exception('Torch dense failed')
  del vector, dense_A
  torch.cuda.empty_cache()


def torch_sparse(size, prob, spikes):
  torch.manual_seed(0)
  torch.set_num_threads(1)
  vector = torch.distributions.Binomial(1, spikes).sample((size,)).to(device)
  sparse_A = torch.randn(size, size).to(device).to_sparse_csr()
  if device == torch.device('cuda'):
    torch.cuda.synchronize(device="cuda:0")
  t0 = time.time()
  try:
    for _ in range(100):
      hidden = sparse_A.matmul(vector)
    if device == torch.device('cuda'):
      torch.cuda.synchronize(device="cuda:0")
    cost_t = time.time() - t0
    print(f'Torch sparse: size {size}, prob {prob}, spike {spikes}, cost_t {cost_t} s')
  except:
    raise Exception('Torch sparse failed')
  del vector, sparse_A
  torch.cuda.empty_cache()


def brainpy_event_op(size, prob, spikes):
  bm.random.seed()
  vector = bm.random.random(size) < spikes
  sparse_A = bp.conn.FixedProb(prob=prob, allow_multi_conn=True)(size, size).require('pre2post')
  out = bm.random.random(size).value

  def func(i, value):
    return bm.event.csrmv(1., sparse_A[0], sparse_A[1], events=vector, shape=(size, size),
                          transpose=True)

  jax.block_until_ready(jax.lax.fori_loop(0, 100, func, out))

  t0 = time.time()
  try:
    jax.block_until_ready(jax.lax.fori_loop(0, 100, func, out))
    cost_t = time.time() - t0
    print(f'BrainPy: size {size}, prob {prob}, spike {spikes}, cost_t {cost_t} s')
  except:
    raise Exception('BrainPy failed')
  bm.clear_buffer_memory()


if __name__ == '__main__':
  sizes = [2000, 4000, 6000, 8000, 10000, 20000, 30000, 40000, 50000, 60000]
  probs = [0.01, 0.1]
  spikes = [0.001, 0.01, 0.1]
  for spike in spikes:
    for size in sizes:
      for prob in probs:
        jax_dense(size, prob, spike)
        jax_sparse(size, prob, spike)
        torch_dense(size, prob, spike)
        torch_sparse(size, prob, spike)
        brainpy_event_op(size, prob, spike)
