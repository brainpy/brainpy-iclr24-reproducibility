# -*- coding: utf-8 -*-

'''
simulation time: 10000.
machine: M1 pro CPU
0.4 0.8 1.6 2.4 3.2 4.0
custom_operator = [2, 3, 6, 8, 11, 14]
sparse_operator = [103, 385, 24:30, 53:50, 1:36:15, 2:33:10]
sparse_operator = [103, 385, 1470, 3230, 5775, 9190]
'''

import brainpy as bp
import brainpy.math as bm
import brainpylib as bl
import jax.numpy as jnp

bp.math.set_platform('cpu')


class ExpCOBA_custom_operator(bp.TwoEndConn):
  def __init__(self, pre, post, conn, g_max=1., delay=0., tau=8.0, E=0.,
               method='exp_auto'):
    super(ExpCOBA_custom_operator, self).__init__(pre=pre, post=post, conn=conn)
    self.check_pre_attrs('spike')
    self.check_post_attrs('input', 'V')

    # parameters
    self.E = E
    self.tau = tau
    self.delay = delay
    self.g_max = g_max
    self.pre2post = self.conn.require('pre2post')

    # variables
    self.g = bm.Variable(bm.zeros(self.post.num))

    # function
    self.integral = bp.odeint(lambda g, t: -g / self.tau, method=method)

  def update(self, tdi):
    self.g.value = self.integral(self.g, tdi.t, tdi.dt)
    # self.g += bm.pre2post_event_sum(self.pre.spike, self.pre2post, self.post.num, self.g_max)
    self.g += bl.event_ops.event_csr_matvec(self.g_max, self.pre2post[0], self.pre2post[1], self.pre.spike,
                                            shape=(self.pre.num, self.post.num), transpose=True)
    self.post.input += self.g * (self.E - self.post.V)


class ExpCOBA_sparse_operator(bp.TwoEndConn):
  def __init__(self, pre, post, conn, g_max=1., delay=0., tau=8.0, E=0.,
               method='exp_auto'):
    super(ExpCOBA_sparse_operator, self).__init__(pre=pre, post=post, conn=conn)
    self.check_pre_attrs('spike')
    self.check_post_attrs('input', 'V')

    # parameters
    self.E = E
    self.tau = tau
    self.delay = delay
    self.g_max = g_max
    self.pre_ids, self.post_ids = self.conn.require('pre_ids', 'post_ids')

    # variables
    self.g = bm.Variable(bm.zeros(self.post.num))

    # function
    self.integral = bp.odeint(lambda g, t: -g / self.tau, method=method)

  def update(self, tdi):
    self.g.value = self.integral(self.g, tdi.t, tdi.dt)
    self.g += bm.pre2post_sum(self.pre.spike * self.g_max, self.post.num, self.post_ids, self.pre_ids)
    # self.g += bm.syn2post_sum(bm.pre2syn(self.pre.spike, self.pre_ids) * self.g_max, self.post_ids, self.post.num)
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
    E2E = ExpCOBA_custom_operator(E, E, bp.conn.FixedProb(prob=0.02), E=0., g_max=we, tau=5., method=method)
    E2I = ExpCOBA_custom_operator(E, I, bp.conn.FixedProb(prob=0.02), E=0., g_max=we, tau=5., method=method)
    I2E = ExpCOBA_custom_operator(I, E, bp.conn.FixedProb(prob=0.02), E=-80., g_max=wi, tau=10., method=method)
    I2I = ExpCOBA_custom_operator(I, I, bp.conn.FixedProb(prob=0.02), E=-80., g_max=wi, tau=10., method=method)
    # E2E = ExpCOBA_sparse_operator(E, E, bp.conn.FixedProb(prob=0.02), E=0., g_max=we, tau=5., method=method)
    # E2I = ExpCOBA_sparse_operator(E, I, bp.conn.FixedProb(prob=0.02), E=0., g_max=we, tau=5., method=method)
    # I2E = ExpCOBA_sparse_operator(I, E, bp.conn.FixedProb(prob=0.02), E=-80., g_max=wi, tau=10., method=method)
    # I2I = ExpCOBA_sparse_operator(I, I, bp.conn.FixedProb(prob=0.02), E=-80., g_max=wi, tau=10., method=method)

    super(EINet, self).__init__(E2E, E2I, I2E, I2I, E=E, I=I)


if __name__ == '__main__':

  net = EINet(scale=10.)
  # simulation
  runner = bp.DSRunner(
    net,
    # monitors=['E.spike'],
    inputs=[('E.input', 20.), ('I.input', 20.)],
  )
  runner.run(10000.)

  # visualization
  # bp.visualize.raster_plot(runner.mon.ts, runner.mon['E.spike'], show=True)