# -*- coding: utf-8 -*-
import time

import brainpy as bp
import brainpy.math as bm

taum = 20
taue = 5
taui = 10
Vt = -50
Vr = -60
El = -60
Ee = 0.
Ei = -80.
Ib = 20.
ref = 5.0
we = 0.6  # excitatory synaptic conductance [nS]
wi = 6.7  # inhibitory synaptic conductance [nS]


class LIF(bp.dyn.NeuDyn):
  def __init__(self, size, **kwargs):
    super(LIF, self).__init__(size=size, **kwargs)

    # parameters
    self.V_rest = Vr
    self.V_reset = El
    self.V_th = Vt
    self.tau = taum
    self.tau_ref = ref

    # variables
    self.V = bm.Variable(bm.zeros(self.num))
    self.input = bm.Variable(bm.zeros(self.num))
    self.spike = bm.Variable(bm.zeros(self.num, dtype=bool))
    self.t_last_spike = bm.Variable(bm.ones(self.num) * -1e7)

  def update(self):
    t = bp.share['t']
    dt = bp.share['dt']
    refractory = (t - self.t_last_spike) <= self.tau_ref
    V = self.V + (-self.V + self.V_rest + self.input) / self.tau * dt
    V = bm.where(refractory, self.V, V)
    spike = self.V_th <= V
    self.t_last_spike.value = bm.where(spike, t, self.t_last_spike)
    self.V.value = bm.where(spike, self.V_reset, V)
    self.spike.value = spike
    self.input[:] = Ib


class EINet(bp.Network):
  def __init__(self, scale=1.0, method='exp_auto'):
    # network size
    num_exc = int(3200 * scale)
    num_inh = int(800 * scale)

    self.num = num_exc + num_inh

    pars = dict(V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5.,
                V_initializer=bp.init.Normal(-55., 2.))
    N = bp.neurons.LIF(num_exc + num_inh, **pars, method=method)
    E = bp.synapses.Exponential(N[:num_exc], N, bp.conn.FixedProb(prob=0.02, allow_multi_conn=True),
                                g_max=0.6 / scale, output=bp.synouts.COBA(E=0.),
                                tau=5., method=method)
    I = bp.synapses.Exponential(N[num_exc:], N, bp.conn.FixedProb(prob=0.02, allow_multi_conn=True),
                                g_max=6.7 / scale, output=bp.synouts.COBA(E=-80.),
                                tau=10., method=method)

    super(EINet, self).__init__(E, I, N=N)


def run(scale, platform='cpu'):
  bm.set_platform(platform)

  net = EINet(scale=scale)
  update = bm.jit(net.step_run)

  t0 = time.time()
  update(0)
  t1 = time.time()

  t2 = time.time()
  update(1)
  t3 = time.time()

  compilation_time = (t1 - t0) - (t3 - t2)
  print(f'Network size {net.num}, compilation spend {compilation_time} s')


if __name__ == '__main__':
  for scale in [1, 1, 2, 4, 6, 8, 10]:
      run(scale=scale)

