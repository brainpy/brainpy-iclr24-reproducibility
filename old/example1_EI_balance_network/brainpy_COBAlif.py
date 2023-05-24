# -*- coding: utf-8 -*-

import brainpy as bp
import brainpy.math as bm

bm.set_platform('cpu')

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


class COBALIF(bp.Network):
  def __init__(self, scale=1., method='exp_auto'):
    super(COBALIF, self).__init__()
    num_exc = int(3200 * scale)
    num_inh = int(800 * scale)
    self.num = num_exc + num_inh

    self.N = LIF(num_exc+num_inh)
    self.E2E = bp.synapses.Exponential(pre=self.N[:num_exc], post=self.N[:num_exc],
                                       conn=bp.conn.FixedProb(prob=0.02),
                                       g_max=we / scale, tau=taue, method=method,
                                       output=bp.synouts.COBA(E=Ee))
    self.E2I = bp.synapses.Exponential(pre=self.N[:num_exc], post=self.N[num_exc:],
                                       conn=bp.conn.FixedProb(prob=0.02),
                                       g_max=we / scale, tau=taue, method=method,
                                       output=bp.synouts.COBA(E=Ee))
    self.I2E = bp.synapses.Exponential(pre=self.N[num_exc:], post=self.N[:num_exc],
                                       conn=bp.conn.FixedProb(prob=0.02),
                                       g_max=wi / scale, tau=taui, method=method,
                                       output=bp.synouts.COBA(E=Ei))
    self.I2I = bp.synapses.Exponential(pre=self.N[num_exc:], post=self.N[num_exc:],
                                       conn=bp.conn.FixedProb(prob=0.02),
                                       g_max=wi / scale, tau=taui, method=method,
                                       output=bp.synouts.COBA(E=Ei))


def run(scale, duration, res_dict=None):
  runner = bp.DSRunner(COBALIF(scale=scale))
  t, _ = runner.predict(duration, eval_time=True)
  if res_dict is not None:
    res_dict['brainpy'].append({'num_neuron': runner.target.num,
                                'sim_len': duration,
                                'num_thread': 1,
                                'sim_time': t,
                                'dt': runner.dt})


if __name__ == '__main__':
  import json

  # run(scale=4, res_dict=None, duration=1e4)

  speed_res = {'brainpy': []}
  for scale in [1, 2, 4, 6, 8, 10]:
  # for scale in [15, 20, 30]:
    for stim in [5. * 1e3]:
      run(scale=scale, res_dict=speed_res, duration=stim)

  with open('speed_results/brainpy-cpu.json', 'w') as f:
    json.dump(speed_res, f, indent=2)