import brainpy as bp
import brainpy.math as bm
import brainpy_datasets as bd
import jax

import utils
from utils import bptt, DMS

bp.math.set(dt=1., mode=bm.training_mode)


class GIF(bp.NeuGroupNS):
  def __init__(
      self, size, V_rest=0., V_th_inf=1., R=1., tau=20.,
      tau_th=100., Ath=1., tau_I1=5., A1=0., tau_I2=50., A2=0.,
      adaptive_th=False, V_initializer=bp.init.OneInit(0.), I1_initializer=bp.init.ZeroInit(),
      I2_initializer=bp.init.ZeroInit(), Vth_initializer=bp.init.OneInit(1.),
      method='exp_auto', keep_size=False, name=None, mode=None,
      spike_fun=bm.surrogate.ReluGrad(), v_scale_var: bool = False,
  ):
    super().__init__(size=size, keep_size=keep_size, name=name, mode=mode)
    assert self.mode.is_parent_of(bm.TrainingMode, bm.NonBatchingMode)

    # params
    self.V_rest = bp.init.parameter(V_rest, self.varshape, allow_none=False)
    self.V_th_inf = bp.init.parameter(V_th_inf, self.varshape, allow_none=False)
    self.R = bp.init.parameter(R, self.varshape, allow_none=False)
    self.tau = bp.init.parameter(tau, self.varshape, allow_none=False)
    self.tau_th = bp.init.parameter(tau_th, self.varshape, allow_none=False)
    self.tau_I1 = bp.init.parameter(tau_I1, self.varshape, allow_none=False)
    self.tau_I2 = bp.init.parameter(tau_I2, self.varshape, allow_none=False)
    self.Ath = bp.init.parameter(Ath, self.varshape, allow_none=False)
    self.A1 = bp.init.parameter(A1, self.varshape, allow_none=False)
    self.A2 = bp.init.parameter(A2, self.varshape, allow_none=False)
    self.spike_fun = bp.check.is_callable(spike_fun, 'spike_fun')
    self.adaptive_th = adaptive_th
    self.v_scale_var = v_scale_var

    # initializers
    self._V_initializer = bp.check.is_initializer(V_initializer)
    self._I1_initializer = bp.check.is_initializer(I1_initializer)
    self._I2_initializer = bp.check.is_initializer(I2_initializer)
    self._Vth_initializer = bp.check.is_initializer(Vth_initializer)

    # variables
    self.reset_state(self.mode)

    # integral
    self.int_V = bp.odeint(method=method, f=self.dV)
    self.int_I1 = bp.odeint(method=method, f=self.dI1)
    self.int_I2 = bp.odeint(method=method, f=self.dI2)
    self.int_Vth = bp.odeint(method=method, f=self.dVth)

  def reset_state(self, batch_size=None):
    self.V = bp.init.variable_(self._V_initializer, self.varshape, batch_size)
    self.I1 = bp.init.variable_(self._I1_initializer, self.varshape, batch_size)
    self.I2 = bp.init.variable_(self._I2_initializer, self.varshape, batch_size)
    if self.adaptive_th:
      self.V_th = bp.init.variable_(self._Vth_initializer, self.varshape, batch_size)
    if self.v_scale_var:
      self.Vs = bp.init.variable_(bm.zeros, self.varshape, batch_size)
    self.spike = bp.init.variable_(bm.zeros, self.varshape, batch_size)

  def dI1(self, I1, t):
    return - I1 / self.tau_I1

  def dI2(self, I2, t):
    return - I2 / self.tau_I2

  def dVth(self, V_th, t):
    return -(V_th - self.V_th_inf) / self.tau_th

  def dV(self, V, t, I_ext):
    return (- V + self.V_rest + self.R * I_ext) / self.tau

  @property
  def derivative(self):
    return bp.JointEq(self.dI1, self.dI2, self.dVth, self.dV)

  def update(self, x=None):
    t = bp.share.load('t')
    dt = bp.share.load('dt')
    x = 0. if x is None else x
    I1 = jax.lax.stop_gradient(bm.where(self.spike, self.A1, self.int_I1(self.I1.value, t, dt)).value)
    I2 = self.int_I2(self.I2.value, t, dt) + self.A2 * self.spike
    V = self.int_V(self.V.value, t, I_ext=(x + I1 + I2), dt=dt)
    if self.adaptive_th:
      V_th = self.int_Vth(self.V_th.value, t, dt) + self.Ath * self.spike
      V_th_ng = jax.lax.stop_gradient(V_th)
      vs = (V - V_th) / V_th_ng
      if self.v_scale_var:
        self.Vs.value = vs
      spike = self.spike_fun(vs)
      V -= V_th_ng * spike
      self.V_th.value = V_th
    else:
      vs = (V - self.V_th_inf) / self.V_th_inf
      if self.v_scale_var:
        self.Vs.value = vs
      spike = self.spike_fun(vs)
      V -= self.V_th_inf * spike
    self.spike.value = spike
    self.I1.value = I1
    self.I2.value = I2
    self.V.value = V
    return spike


class Trainer:
  def __init__(
      self, net, n_test: int, lr: float = 1e-2,
      fr_reg=False, fr_reg_target=1., fr_reg_factor=1.,
      mem_reg=False, mem_reg_target=(-2., 0.4), mem_reg_factor=1.,
  ):
    self.n_test = n_test
    self.mem_reg = mem_reg
    self.mem_reg_factor = mem_reg_factor
    self.fr_reg = fr_reg
    self.fr_reg_factor = fr_reg_factor  # regularization coefficient for firing rate
    self.fr_reg_target = fr_reg_target / 1000.  # target firing rate for regularization [Hz]
    self.model = net
    self.mem_reg_target = mem_reg_target
    self.f_grad = bm.grad(self.loss, grad_vars=self.model.train_vars().unique(), return_value=True, has_aux=True)
    self.f_opt = bp.optim.Adam(lr=lr, train_vars=self.model.train_vars().unique())
    self.f_train = bm.jit(self.train)
    if mem_reg:
      assert net.r.v_scale_var

  def loss(self, xs, ys):
    out_vars = dict()
    if self.fr_reg:
      out_vars['spikes'] = self.model.r.spike
    if self.mem_reg:
      out_vars['Vs'] = self.model.r.Vs
    outputs, out_vars = bp.LoopOverTime(self.model, out_vars=out_vars)(xs)
    outs = outputs[-self.n_test:]
    # Define the accuracy
    accuracy = bm.mean(bm.equal(ys, bm.argmax(bm.mean(outs, axis=0), axis=1)))

    # loss function
    tiled_targets = bm.tile(bm.expand_dims(ys, 0), (self.n_test, 1))
    # loss function
    loss = bp.losses.cross_entropy_loss(outs, tiled_targets)

    # Firing rate regularization
    if self.fr_reg:
      loss_reg_f = bm.sum(bm.square(bm.mean(out_vars['spikes'], axis=(0, 1)) - self.fr_reg_target)) * self.fr_reg_factor
      loss += loss_reg_f
    else:
      loss_reg_f = 0.
    if self.mem_reg:
      loss_reg_v = self.mem_reg_factor * bm.square(bm.mean(
        bm.relu(out_vars['Vs'] - self.mem_reg_target[1]) ** 2 +
        bm.relu(-(out_vars['Vs'] - self.mem_reg_target[0])) ** 2
      ))
      loss += loss_reg_v
    else:
      loss_reg_v = 0.
    return loss, {'acc': accuracy, 'reg_v': loss_reg_v, 'loss': loss, 'reg_fr': loss_reg_f, }

  def train(self, xs, ys):
    grads, loss, aux = self.f_grad(xs, ys)
    self.f_opt.update(grads)
    return loss, aux


class GLIF_Exp(bp.DynamicalSystemNS):
  def __init__(self, num_in, num_rec, num_out, tau_o=1e1, tau_ext=1e1,
               spike_fun=bm.surrogate.relu_grad, gif_pars=None,
               inits: bp.init.Initializer = bp.init.KaimingNormal()):
    super().__init__()

    # parameters
    gif_pars = dict() if gif_pars is None else gif_pars
    self.gif_pars = gif_pars
    self.num_in = num_in
    self.num_rec = num_rec
    self.num_out = num_out
    self.i2r = bp.layers.Dense(num_in, num_rec, W_initializer=inits)
    self.r2r = bp.layers.Dense(num_rec, num_rec, W_initializer=inits)
    self.r2o = bp.layers.Dense(num_rec, num_out, W_initializer=inits)
    self.ext = bp.neurons.Leaky(num_rec, tau=tau_ext)
    self.r = GIF(num_rec, V_rest=0., V_th_inf=1.,
                 spike_fun=spike_fun,
                 V_initializer=bp.init.ZeroInit(),
                 Vth_initializer=bp.init.OneInit(1.), **gif_pars)
    self.o = bp.neurons.Leaky(num_out, tau=tau_o)

  def update(self, spikes):
    ext = self.ext(self.i2r(spikes) + self.r2r(self.r.spike.value))
    return self.o(self.r2o(self.r(ext)))


if __name__ == '__main__':
  ds = DMS(dt=bm.dt, mode='spiking', num_trial=64 * 100, bg_fr=1.)
  _loader = bd.cognitive.TaskLoader(ds, batch_size=64, data_first_axis='T')

  # Adaptive spiking pattern
  gif_pars = dict(Ath=1, A1=0., A2=-0.1, adaptive_th=False, tau_I2=2e3, tau_I1=10., v_scale_var=True)

  # Tonic bursting pattern
  # gif_pars = dict(Ath=0.1, A1=8., A2=-0.6, adaptive_th=False, tau_I2=1e3, tau_I1=10., v_scale_var=True)

  net = GLIF_Exp(num_in=ds.num_inputs, num_rec=100, num_out=ds.num_outputs, tau_ext=1e2,
                 inits=bp.init.KaimingNormal(distribution='normal', scale=0.2),
                 gif_pars=gif_pars)

  trainer = Trainer(net, ds.t_test, mem_reg=True, lr=1e-3)
  bptt(_loader, trainer, fn=None, n_epoch=5)

  utils.verify_lif(net, ds, fn=None, num_show=1, sps_inc=2.)

