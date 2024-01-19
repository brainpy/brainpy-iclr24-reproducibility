# -*- coding: utf-8 -*-

import brainpy as bp
import brainpy.math as bm
import json

# If cpu
bm.set_platform('cpu')
# If gpu
# bm.set_platform('gpu')

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


class LIF(bp.NeuGroup):
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

  def update(self, tdi):
    refractory = (tdi.t - self.t_last_spike) <= self.tau_ref
    V = self.V + (-self.V + self.V_rest + self.input) / self.tau * tdi.dt
    V = bm.where(refractory, self.V, V)
    spike = self.V_th <= V
    self.t_last_spike.value = bm.where(spike, tdi.t, self.t_last_spike)
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
    E = bp.synapses.Exponential(N[:num_exc], N, bp.conn.FixedProb(prob=0.02),
                                g_max=0.6 / scale, output=bp.synouts.COBA(E=0.),
                                tau=5., method=method)
    I = bp.synapses.Exponential(N[num_exc:], N, bp.conn.FixedProb(prob=0.02),
                                g_max=6.7 / scale, output=bp.synouts.COBA(E=-80.),
                                tau=10., method=method)

    super(EINet, self).__init__(E, I, N=N)


def run(scale, duration, res_dict=None):
  runner = bp.DSRunner(EINet(scale=scale))
  t, _ = runner.predict(duration, eval_time=True)
  if res_dict is not None:
    res_dict['brainpy'].append({'num_neuron': runner.target.num,
                                'sim_len': duration,
                                'num_thread': 1,
                                'sim_time': t,
                                'dt': runner.dt})


if __name__ == '__main__':
  speed_res = {'brainpy': []}
  for scale in [1, 2, 4, 6, 8, 10]:
    for stim in [5. * 1e3]:
      run(scale=scale, res_dict=speed_res, duration=stim)

  with open('speed_results/brainpy-gpu.json', 'w') as f:
    json.dump(speed_res, f, indent=2)