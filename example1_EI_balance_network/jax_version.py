# -*- coding: utf-8 -*-

'''
simulation time: 10000.
0.4 0.8 1.6 2.4 3.2 4.0
jax_version = [49, 3:36, 15:25, 35:54, 1:05:23, 1:41:40]
jax_version = [49, 216, 925, 2154, 3923, 6100]
'''

import brainpy as bp

taum = 20
taue = 5
taui = 10
Vt = -50
Vr = -60
El = -60
Erev_exc = 0.
Erev_inh = -80.
Ib = 20.
ref = 5.0


class LIF(bp.NeuGroup):
  def __init__(self, size, **kwargs):
    super(LIF, self).__init__(size=size, **kwargs)

    self.V = bp.math.Variable(bp.math.ones(size) * Vr)
    self.Isyn = bp.math.Variable(bp.math.zeros(size))
    self.t_spike = bp.math.Variable(-1e7 * bp.math.ones(size))
    self.spike = bp.math.Variable(bp.math.zeros(size, dtype=bool))

    self.integral = bp.odeint(self.derivative)

  def derivative(self, V, t, Isyn):
    return (Isyn + (El - V) + Ib) / taum

  def update(self, tdi):
    _t, _dt = tdi.t, tdi.dt
    for i in range(self.num):
      self.spike[i] = 0.
      if (_t - self.t_spike[i]) > ref:
        V = self.integral(self.V[i], _t, self.Isyn[i])
        self.spike[i] = 0.
        if V >= Vt:
          self.V[i] = Vr
          self.spike[i] = 1.
          self.t_spike[i] = _t
        else:
          self.V[i] = V
    self.Isyn[:] = 0.


class ExpCOBA(bp.TwoEndConn):
  def __init__(self, pre, post, conn, E, w, tau, method='exp_auto', **kwargs):
    super(ExpCOBA, self).__init__(pre, post, conn=conn, **kwargs)

    # parameters
    self.E = E
    self.w = w
    self.tau = tau

    self.conn_mat = self.conn.requires('conn_mat')  # connections
    self.g = bp.math.Variable(bp.math.zeros(post.num))  # variables

    self.integral = bp.odeint(self.derivative)

  def derivative(self, g, t):
    dg = - g / self.tau
    return dg
  def update(self, tdi):
    _t, _dt = tdi.t, tdi.dt
    post_vs = (self.pre.spike @ self.conn_mat) * self.w
    self.g = self.g - self.g / self.tau * _dt + post_vs
    self.post.input += self.g * (self.E - self.post.V)


class EINet(bp.Network):
  def __init__(self, scale=1.0, method='exp_auto'):
    # network size
    num_exc = int(3200 * scale)
    num_inh = int(800 * scale)

    # neurons
    pars = dict(V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5.,
                V_initializer=bp.init.Normal(-55., 2.))
    E = bp.neurons.LIF(num_exc, **pars, method=method)
    I = bp.neurons.LIF(num_inh, **pars, method=method)

    # synapses
    we = 0.6 / scale  # excitatory synaptic weight (voltage)
    wi = 6.7 / scale  # inhibitory synaptic weight
    E2E = ExpCOBA(E, E, bp.conn.FixedProb(prob=0.02), E=0., w=we, tau=5., method=method)
    E2I = ExpCOBA(E, I, bp.conn.FixedProb(prob=0.02), E=0., w=we, tau=5., method=method)
    I2E = ExpCOBA(I, E, bp.conn.FixedProb(prob=0.02), E=-80., w=wi, tau=10., method=method)
    I2I = ExpCOBA(I, I, bp.conn.FixedProb(prob=0.02), E=-80., w=wi, tau=10., method=method)

    super(EINet, self).__init__(E2E, E2I, I2E, I2I, E=E, I=I)


if __name__ == '__main__':
  net = EINet(scale=10.)
  # simulation
  runner = bp.DSRunner(
    net,
    monitors=['E.spike'],
    inputs=[('E.input', 20.), ('I.input', 20.)]
  )
  runner.run(10000.)

  # bp.visualize.raster_plot(runner.mon.ts, runner.mon['E.spike'], show=True)




