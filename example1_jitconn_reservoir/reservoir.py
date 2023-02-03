# -*- coding: utf-8 -*-


import brainpy as bp
import brainpy.math as bm
import jax
import jax.numpy as jnp
import numpy as np
from brainpylib.jitconn_ops import (
  matvec_prob_conn_uniform_weight,
  matvec_prob_conn_normal_weight,
  matmat_prob_conn_uniform_weight,
  matmat_prob_conn_normal_weight,
)

__all__ = [
  'JITReservoir',
  'JITDeepReservoir',
  'DeepReservoir',
]


def jit_evolve_layer1(x, state,
                      win_scale, win_connectivity, win_seed,
                      wrec_sigma, wrec_connectivity, wrec_seed,
                      jit_version='v1'):
  features_in = x.shape[-1]
  features_out = state.shape[-1]
  if x.ndim == 2 and x.shape[0] == 1:
    x = x[0]
  if state.ndim == 2 and state.shape[0] == 1:
    state = state[0]
  if x.ndim == 1:
    assert jit_version == 'v1'
    hidden = matvec_prob_conn_uniform_weight(x,
                                             w_low=-win_scale,
                                             w_high=win_scale,
                                             conn_prob=win_connectivity,
                                             shape=(features_out, features_in),
                                             seed=win_seed)
    hidden += matvec_prob_conn_normal_weight(state,
                                             w_mu=0.,
                                             w_sigma=wrec_sigma,
                                             conn_prob=wrec_connectivity,
                                             shape=(features_out, features_out),
                                             seed=wrec_seed)
  elif x.ndim == 2:
    if jit_version == 'v1':
      hidden = jax.vmap(
        lambda a: matvec_prob_conn_uniform_weight(a,
                                                  w_low=-win_scale,
                                                  w_high=win_scale,
                                                  conn_prob=win_connectivity,
                                                  shape=(features_out, features_in),
                                                  seed=win_seed)
      )(x)
      hidden += jax.vmap(
        lambda a: matvec_prob_conn_normal_weight(a,
                                                 w_mu=0.,
                                                 w_sigma=wrec_sigma,
                                                 conn_prob=wrec_connectivity,
                                                 shape=(features_out, features_out),
                                                 seed=wrec_seed)
      )(state)
    elif jit_version == 'v2':
      hidden = matmat_prob_conn_uniform_weight(x,
                                               w_low=-win_scale,
                                               w_high=win_scale,
                                               conn_prob=win_connectivity,
                                               shape=(features_in, features_out),
                                               seed=win_seed)
      hidden += matmat_prob_conn_normal_weight(state,
                                               w_mu=0.,
                                               w_sigma=wrec_sigma,
                                               conn_prob=wrec_connectivity,
                                               shape=(features_out, features_out),
                                               seed=wrec_seed)
    else:
      raise ValueError
  else:
    raise ValueError
  return hidden


def jit_evolve_layer_other(x, state, wrec_sigma, wrec_connectivity, win_seed, wrec_seed,
                           jit_version='v1'):
  assert x.shape[-1] == state.shape[-1]
  features_out = state.shape[-1]
  if x.ndim == 2 and x.shape[0] == 1:
    x = x[0]
  if state.ndim == 2 and state.shape[0] == 1:
    state = state[0]
  if x.ndim == 1:
    assert jit_version == 'v1'
    hidden = matvec_prob_conn_normal_weight(x,
                                            w_mu=0.,
                                            w_sigma=wrec_sigma,
                                            conn_prob=wrec_connectivity,
                                            shape=(features_out, features_out),
                                            seed=win_seed)
    hidden += matvec_prob_conn_normal_weight(state,
                                             w_mu=0.,
                                             w_sigma=wrec_sigma,
                                             conn_prob=wrec_connectivity,
                                             shape=(features_out, features_out),
                                             seed=wrec_seed)
  elif x.ndim == 2:
    if jit_version == 'v1':
      hidden = jax.vmap(
        lambda a: matvec_prob_conn_normal_weight(a,
                                                 w_mu=0.,
                                                 w_sigma=wrec_sigma,
                                                 conn_prob=wrec_connectivity,
                                                 shape=(features_out, features_out),
                                                 seed=win_seed)
      )(x)
      hidden += jax.vmap(
        lambda a: matvec_prob_conn_normal_weight(a,
                                                 w_mu=0.,
                                                 w_sigma=wrec_sigma,
                                                 conn_prob=wrec_connectivity,
                                                 shape=(features_out, features_out),
                                                 seed=wrec_seed)
      )(state)
    elif jit_version == 'v2':
      hidden = matmat_prob_conn_normal_weight(x,
                                              w_mu=0.,
                                              w_sigma=wrec_sigma,
                                              conn_prob=wrec_connectivity,
                                              shape=(features_out, features_out),
                                              seed=win_seed)
      hidden += matmat_prob_conn_normal_weight(state,
                                               w_mu=0.,
                                               w_sigma=wrec_sigma,
                                               conn_prob=wrec_connectivity,
                                               shape=(features_out, features_out),
                                               seed=wrec_seed)
    else:
      raise ValueError
  else:
    raise ValueError
  return hidden


class JITReservoir(bp.DynamicalSystem):
  """Reservoir node with just-in-time connectivity."""

  def __init__(
      self,
      features_in,
      features_out,
      leaky_rate: float = 0.3,
      win_connectivity: float = 0.1,
      wrec_connectivity: float = 0.1,
      win_scale: float = 0.1,
      wrec_sigma: float = 0.1,
      activation: callable = bm.tanh,
      name: str = None,
      jit_version='v1',
      mode: bm.Mode = None
  ):
    super().__init__(name=name, mode=mode)

    assert self.mode.is_a(bm.BatchingMode)

    self.features_in = bp.check.is_integer(features_in, min_bound=1)
    self.features_out = bp.check.is_integer(features_out, min_bound=1)
    self.leaky_rate = bp.check.is_float(leaky_rate, min_bound=0.)
    self.win_connectivity = bp.check.is_float(win_connectivity, min_bound=0.)
    self.wrec_connectivity = bp.check.is_float(wrec_connectivity, min_bound=0.)
    self.win_scale = bp.check.is_float(win_scale, min_bound=0.)
    self.wrec_sigma = bp.check.is_float(wrec_sigma, min_bound=0.)
    self.activation = bp.check.is_callable(activation)
    self.win_seed = bm.random.randint(0, 100000).item()
    self.wrec_seed = bm.random.randint(0, 100000).item()

    print(f'win_seed: {self.win_seed}, wrec_seed: {self.wrec_seed}')
    self.jit_version = jit_version
    assert jit_version in ['v1', 'v2']

    self.reset_state()

  def reset_state(self, batch_size=1):
    self.state = bp.init.variable_(jnp.zeros, self.features_out, batch_size)

  def state_dict(self):
    return {'win_seed': self.win_seed, 'wrec_seed': self.wrec_seed}

  def load_state_dict(self, state_dict, warn: bool = True):
    self.win_seed = state_dict['win_seed']
    self.wrec_seed = state_dict['wrec_seed']

  def update(self, *args):
    x = args[0] if len(args) == 1 else args[1]
    hidden = jit_evolve_layer1(x, self.state.value,
                               win_scale=self.win_scale,
                               win_connectivity=self.win_connectivity,
                               win_seed=self.win_seed,
                               wrec_sigma=self.wrec_sigma,
                               wrec_connectivity=self.wrec_connectivity,
                               wrec_seed=self.wrec_seed,
                               jit_version=self.jit_version)
    state = (1 - self.leaky_rate) * self.state + self.leaky_rate * self.activation(hidden)
    self.state.value = state
    return state


class JITDeepReservoir(bp.DynamicalSystem):
  """Deep Reservoir model with just-in-time connectivity."""

  def __init__(
      self,
      features_in,
      features_out,
      num_layer,
      leaky_start: float = 0.9,
      leaky_end: float = 0.1,
      win_connectivity: float = 0.1,
      wrec_connectivity: float = 0.1,
      win_scale: float = 0.1,
      wrec_sigma: float = 0.1,
      activation: callable = bm.tanh,
      name: str = None,
      jit_version='v1',
      mode: bm.Mode = None
  ):
    super().__init__(name=name, mode=mode)

    assert self.mode.is_a(bm.BatchingMode)
    self.num_layer = bp.check.is_integer(num_layer, min_bound=1)
    self.features_in = bp.check.is_integer(features_in, min_bound=1)
    self.features_out = bp.check.is_integer(features_out, min_bound=1)
    bp.check.is_float(leaky_start)
    bp.check.is_float(leaky_end)
    self.leaky_rates = np.linspace(leaky_start, leaky_end, num_layer)
    self.win_connectivity = bp.check.is_float(win_connectivity, min_bound=0.)
    self.wrec_connectivity = bp.check.is_float(wrec_connectivity, min_bound=0.)
    self.win_scale = bp.check.is_float(win_scale, min_bound=0.)
    self.wrec_sigma = bp.check.is_float(wrec_sigma, min_bound=0.)
    self.activation = bp.check.is_callable(activation)
    self.win_seed = bm.random.randint(0, 100000, num_layer).to_numpy()
    self.wrec_seed = bm.random.randint(0, 100000, num_layer).to_numpy()
    self.jit_version = jit_version
    assert jit_version in ['v1', 'v2']

    print('win_seed', self.win_seed)
    print('wrec_seed', self.wrec_seed)

    self.reset_state()

  def reset_state(self, batch_size=1):
    self.state = bp.init.variable_(jnp.zeros, (self.num_layer, self.features_out), batch_size, batch_axis=1)

  def state_dict(self):
    return {'win_seed': self.win_seed, 'wrec_seed': self.wrec_seed}

  def load_state_dict(self, state_dict, warn: bool = True):
    self.win_seed = state_dict['win_seed']
    self.wrec_seed = state_dict['wrec_seed']

  def update(self, *args):
    x = args[0] if len(args) == 1 else args[1]
    for i in range(self.num_layer):
      if i == 0:
        hidden = jit_evolve_layer1(x,
                                   self.state[i],
                                   win_scale=self.win_scale,
                                   win_connectivity=self.win_connectivity,
                                   win_seed=self.win_seed[i],
                                   wrec_sigma=self.wrec_sigma,
                                   wrec_connectivity=self.wrec_connectivity,
                                   wrec_seed=self.wrec_seed[i],
                                   jit_version=self.jit_version)
      else:
        hidden = jit_evolve_layer_other(x,
                                        self.state[i],
                                        win_seed=self.win_seed[i],
                                        wrec_sigma=self.wrec_sigma,
                                        wrec_connectivity=self.wrec_connectivity,
                                        wrec_seed=self.wrec_seed[i],
                                        jit_version=self.jit_version)
      state = (1 - self.leaky_rates[i]) * self.state[i] + self.leaky_rates[i] * self.activation(hidden)
      self.state[i] = state
      x = state
    return self.state.value


class DeepReservoir(bp.DynamicalSystem):
  """Deep Reservoir model with just-in-time connectivity."""

  def __init__(
      self,
      features_in,
      features_out,
      num_layer,
      leaky_start: float = 0.9,
      leaky_end: float = 0.1,
      win_connectivity: float = 0.1,
      wrec_connectivity: float = 0.1,
      win_scale: float = 0.1,
      wrec_sigma: float = 0.1,
      activation: callable = bm.tanh,
      name: str = None,
      mode: bm.Mode = None
  ):
    super().__init__(name=name, mode=mode)

    assert self.mode.is_a(bm.BatchingMode)
    self.num_layer = bp.check.is_integer(num_layer, min_bound=1)
    self.features_in = bp.check.is_integer(features_in, min_bound=1)
    self.features_out = bp.check.is_integer(features_out, min_bound=1)
    bp.check.is_float(leaky_start)
    bp.check.is_float(leaky_end)
    self.leaky_rates = np.linspace(leaky_start, leaky_end, num_layer)
    self.win_connectivity = bp.check.is_float(win_connectivity, min_bound=0.)
    self.wrec_connectivity = bp.check.is_float(wrec_connectivity, min_bound=0.)
    self.win_scale = bp.check.is_float(win_scale, min_bound=0.)
    self.wrec_sigma = bp.check.is_float(wrec_sigma, min_bound=0.)
    self.activation = bp.check.is_callable(activation)
    self.win = [bm.random.uniform(-win_scale, win_scale, (features_in, features_out)) *
                (bm.random.rand(features_in, features_out) <= self.win_connectivity)]
    self.win += [bm.random.normal(0., wrec_sigma, (features_out, features_out)) *
                 (bm.random.rand(features_out, features_out) <= self.wrec_connectivity)
                 for _ in range(self.num_layer - 1)]
    self.wrec = [bm.random.normal(0., wrec_sigma, (features_out, features_out))
                 * (bm.random.rand(features_out, features_out) <= self.wrec_connectivity)
                 for _ in range(self.num_layer)]

    self.reset_state()

  def reset_state(self, batch_size=1):
    self.state = bp.init.variable_(jnp.zeros, (self.num_layer, self.features_out), batch_size, batch_axis=1)

  def state_dict(self):
    return {'win': {str(i): w for i, w in enumerate(self.win)},
            'wrec': {str(i): w for i, w in enumerate(self.wrec)}}

  def load_state_dict(self, state_dict, warn: bool = True):
    for i in range(self.num_layer):
      self.win[i].value = state_dict['win'][str(i)]
      self.wrec[i].value = state_dict['wrec'][str(i)]

  def update(self, *args):
    x = args[0] if len(args) == 1 else args[1]
    for i in range(self.num_layer):
      hidden = x @ self.win[i] + self.state[i] @ self.wrec[i]
      state = (1 - self.leaky_rates[i]) * self.state[i] + self.leaky_rates[i] * self.activation(hidden)
      self.state[i] = state
      x = state
    return self.state.value
