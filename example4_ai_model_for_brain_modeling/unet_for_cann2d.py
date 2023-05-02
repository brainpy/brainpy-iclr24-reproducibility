import functools

import argparse
import time

import numpy as np
import brainpy as bp
import brainpy.math as bm
import jax


class CANN2D(bp.NeuGroup):
  def __init__(self, length, tau=1., k=8.1, a=0.5, A=10., J0=4.,
               z_min=-bm.pi, z_max=bm.pi, name=None):
    super(CANN2D, self).__init__(size=(length, length), name=name)

    # parameters
    self.length = length
    self.tau = tau  # The synaptic time constant
    self.k = k  # Degree of the rescaled inhibition
    self.a = a  # Half-width of the range of excitatory connections
    self.A = A  # Magnitude of the external input
    self.J0 = J0  # maximum connection value

    # feature space
    self.z_min = z_min
    self.z_max = z_max
    self.z_range = z_max - z_min
    self.x = bm.linspace(z_min, z_max, length)  # The encoded feature values
    self.rho = length / self.z_range  # The neural density
    self.dx = self.z_range / length  # The stimulus density

    # The connections
    self.conn_mat = self.make_conn()

    # variables
    self.r = bm.Variable(bm.zeros((length, length)))
    self.u = bm.Variable(bm.zeros((length, length)))
    self.input = bm.Variable(bm.zeros((length, length)))

  def dist(self, d):
    v_size = bm.asarray([self.z_range, self.z_range])
    return bm.where(d > v_size / 2, v_size - d, d)

  def make_conn(self):
    x1, x2 = bm.meshgrid(self.x, self.x)
    value = bm.stack([x1.flatten(), x2.flatten()]).T

    @jax.vmap
    def get_J(v):
      d = self.dist(bm.abs(v - value))
      d = bm.linalg.norm(d, axis=1)
      # d = d.reshape((self.length, self.length))
      Jxx = self.J0 * bm.exp(-0.5 * bm.square(d / self.a)) / (bm.sqrt(2 * bm.pi) * self.a)
      return Jxx

    return get_J(value)

  def get_stimulus_by_pos(self, pos):
    assert bm.size(pos) == 2
    x1, x2 = bm.meshgrid(self.x, self.x)
    value = bm.stack([x1.flatten(), x2.flatten()]).T
    d = self.dist(bm.abs(bm.asarray(pos) - value))
    d = bm.linalg.norm(d, axis=1)
    d = d.reshape((self.length, self.length))
    return self.A * bm.exp(-0.25 * bm.square(d / self.a))

  def update(self, tdi, x=None):
    if x is not None:
      self.input += x
    r1 = bm.square(self.u)
    r2 = 1.0 + self.k * bm.sum(r1)
    self.r.value = r1 / r2
    interaction = (self.r.flatten() @ self.conn_mat).reshape((self.length, self.length))
    self.u.value = self.u + (-self.u + self.input + interaction) / self.tau * tdi.dt
    self.input[:] = 0.


class CANN2D_FFT(bp.NeuGroup):
  def __init__(self, length, batch_size, tau=1., k=8.1, a=0.5, J0=4.,
               z_min=-bm.pi, z_max=bm.pi, name=None):
    super().__init__(size=(length, length), name=name)

    # parameters
    self.batch_size = batch_size
    self.length = length
    self.tau = tau  # The synaptic time constant
    self.k = k  # Degree of the rescaled inhibition
    self.a = a  # Half-width of the range of excitatory connections
    self.J0 = J0  # maximum connection value

    # feature space
    self.z_min = z_min
    self.z_max = z_max
    self.z_range = z_max - z_min
    self.x = bm.linspace(z_min, z_max, length)  # The encoded feature values
    self.rho = length / self.z_range  # The neural density
    self.dx = self.z_range / length  # The stimulus density

    # The connections
    conn_mat = self._make_conn()
    self.conn_fft = bm.fft.fft2(conn_mat)
    self.reset_state(batch_size)

  def reset_state(self, batch_size):
    self.r = bm.Variable(bm.zeros((batch_size, self.length, self.length)))
    self.u = bm.Variable(bm.zeros((batch_size, self.length, self.length)))

  def _dist(self, d):
    v_size = bm.asarray([self.z_range, self.z_range])
    return bm.where(d > v_size / 2, v_size - d, d)

  def _make_conn(self):
    x1, x2 = bm.meshgrid(self.x, self.x)
    value = bm.stack([x1.flatten(), x2.flatten()]).T
    d = self._dist(bm.abs(value[0] - value))
    d = bm.linalg.norm(d, axis=1)
    d = d.reshape((self.length, self.length))
    Jxx = self.J0 * bm.exp(-0.5 * bm.square(d / self.a)) / (bm.sqrt(2 * bm.pi) * self.a)
    return Jxx

  def _stimulus(self, pos, amp, value):
    d = self._dist(bm.abs(pos - value))
    d = bm.linalg.norm(d, axis=1)
    d = d.reshape((self.length, self.length))
    return amp * bm.exp(-0.25 * bm.square(d / self.a))

  def get_stimulus_by_pos(self, pos, amps):
    # A: Magnitude of the external input
    assert len(pos) == self.batch_size
    assert len(amps) == self.batch_size
    x1, x2 = bm.meshgrid(self.x, self.x)
    value = bm.stack([x1.flatten(), x2.flatten()]).T
    return jax.vmap(self._stimulus, in_axes=(0, 0, None))(pos, amps, value)

  def update(self, *args):
    x = args[0] if len(args) == 1 else args[1]
    assert x.ndim == 3
    r1 = bm.square(self.u)
    r2 = 1.0 + self.k * bm.sum(r1)
    self.r.value = r1 / r2
    r = jax.vmap(bm.fft.fft2)(self.r)
    interaction = bm.real(jax.vmap(bm.fft.ifft2)(r * self.conn_fft))
    self.u.value = self.u + (-self.u + x + interaction) / self.tau * bm.get_dt()


class double_conv2d_bn(bp.layers.Layer):
  def __init__(self, in_channels, out_channels, kernel_size=3, strides=1, padding=1,
               lstm=False, input_size=None):
    super(double_conv2d_bn, self).__init__()
    if lstm:
      assert input_size is not None
      self.conv1 = bp.layers.Conv2dLSTMCell(input_size, in_channels, out_channels,
                                            kernel_size=kernel_size, stride=strides, padding=padding, )
      self.conv2 = bp.layers.Conv2dLSTMCell(input_size, out_channels, out_channels,
                                            kernel_size=kernel_size, stride=strides, padding=padding, )
    else:
      self.conv1 = bp.layers.Conv2d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=strides, padding=padding, )
      self.conv2 = bp.layers.Conv2d(out_channels, out_channels,
                                    kernel_size=kernel_size, stride=strides, padding=padding, )
    self.bn1 = bp.layers.BatchNorm2d(out_channels)
    self.bn2 = bp.layers.BatchNorm2d(out_channels)

  def update(self, s, x):
    out = bm.relu(self.bn1(s, self.conv1(x)))
    out = bm.relu(self.bn2(s, self.conv2(out)))
    return out


class deconv2d_bn(bp.layers.Layer):
  def __init__(self, in_channels, out_channels, kernel_size=2, strides=2):
    super(deconv2d_bn, self).__init__()
    self.conv1 = bp.layers.ConvTranspose2d(in_channels, out_channels,
                                           kernel_size=kernel_size, stride=strides, )
    self.bn1 = bp.layers.BatchNorm2d(out_channels)

  def update(self, s, x):
    out = bm.relu(self.bn1(s, self.conv1(x)))
    return out


class UNet(bp.layers.Layer):
  """

  References
  ----------

  U-Net: Convolutional Networks for Biomedical Image Segmentation

  """

  def __init__(self, in_channels=1, out_channels=1, times=(2, 4, 8, 16, 32), lstm=False):
    super(UNet, self).__init__()

    self.pool1 = bp.layers.MaxPool2d(2)
    self.pool2 = bp.layers.MaxPool2d(2)
    self.pool3 = bp.layers.MaxPool2d(2)
    self.pool4 = bp.layers.MaxPool2d(2)
    self.pool5 = bp.layers.MaxPool2d(2)

    self.layer1_conv = double_conv2d_bn(in_channels, in_channels * times[0])
    self.layer2_conv = double_conv2d_bn(in_channels * times[0], in_channels * times[1])
    self.layer3_conv = double_conv2d_bn(in_channels * times[1], in_channels * times[2])
    self.layer4_conv = double_conv2d_bn(in_channels * times[2], in_channels * times[3])
    pars = {'input_size': (14, 14), 'lstm': True} if lstm else {'lstm': False}
    self.layer5_conv = double_conv2d_bn(in_channels * times[3], in_channels * times[4], **pars)
    self.layer6_conv = double_conv2d_bn(in_channels * times[4], in_channels * times[3])
    self.layer7_conv = double_conv2d_bn(in_channels * times[3], in_channels * times[2])
    self.layer8_conv = double_conv2d_bn(in_channels * times[2], in_channels * times[1])
    self.layer9_conv = double_conv2d_bn(in_channels * times[1], in_channels * times[0])
    self.layer10_conv = bp.layers.Conv2d(in_channels * times[0], out_channels, kernel_size=3, stride=1, padding=1)

    self.deconv1 = deconv2d_bn(in_channels * times[4], in_channels * times[3])
    self.deconv2 = deconv2d_bn(in_channels * times[3], in_channels * times[2])
    self.deconv3 = deconv2d_bn(in_channels * times[2], in_channels * times[1])
    self.deconv4 = deconv2d_bn(in_channels * times[1], in_channels * times[0])

  def update(self, s, x):
    conv1 = self.layer1_conv(s, x)
    conv2 = self.layer2_conv(s, self.pool1(conv1))
    conv3 = self.layer3_conv(s, self.pool2(conv2))
    conv4 = self.layer4_conv(s, self.pool3(conv3))
    conv5 = self.layer5_conv(s, self.pool4(conv4))
    convt1 = self.deconv1(s, conv5)
    conv6 = self.layer6_conv(s, bm.cat([convt1, conv4], dim=-1))
    convt2 = self.deconv2(s, conv6)
    conv7 = self.layer7_conv(s, bm.cat([convt2, conv3], dim=-1))
    convt3 = self.deconv3(s, conv7)
    conv8 = self.layer8_conv(s, bm.cat([convt3, conv2], dim=-1))
    convt4 = self.deconv4(s, conv8)
    conv9 = self.layer9_conv(s, bm.cat([convt4, conv1], dim=-1))
    out = self.layer10_conv(s, conv9)
    return out


parser = argparse.ArgumentParser(description='')
parser.add_argument('-lr', default=1e-3, type=float)
parser.add_argument('-batch_size', default=64, type=int)
parser.add_argument('-epoch', default=1000, type=int)
parser.add_argument('-save', action='store_true')
parser.add_argument('-train_method', type=str, default='step', choices=['step', 'all'])
args = parser.parse_args()
print(args.__dict__)


with bm.training_environment():
  net = UNet(2, 1, lstm=True)
  # y = net({'fit': False}, bm.random.rand(10, 224, 224, 2))

cann = CANN2D_FFT(224, args.batch_size)


@bm.jit(child_objs=net, inline=True)
def fun_step_predict(states, inputs):
  if states.ndim == 3:  # (B, H, W)
    states = bm.expand_dims(states, axis=-1)  # (B, H, W, C)
  if inputs.ndim == 3:  # (B, H, W)
    inputs = bm.expand_dims(inputs, axis=-1)  # (B, H, W, C)
  xs = bm.concatenate([states, inputs], axis=-1)
  return net({'fit': False}, xs)


@bm.jit(child_objs=(cann, net), inline=True, static_argnums=2)
def fun_single_step_loss(position, amp, fit=True):
  inputs = cann.get_stimulus_by_pos(position, amp)
  xs = cann.u.value
  cann.update(inputs)
  ys = cann.u.value
  if xs.ndim == 3:  # (B, H, W)
    xs = bm.expand_dims(xs, axis=-1)  # (B, H, W, C)
  if inputs.ndim == 3:  # (B, H, W)
    inputs = bm.expand_dims(inputs, axis=-1)  # (B, H, W, C)
  if ys.ndim == 3:  # (B, H, W)
    ys = bm.expand_dims(ys, axis=-1)  # (B, H, W, C)
  xs = bm.concatenate([xs, inputs], axis=-1)
  l = bp.losses.mean_squared_error(net({'fit': fit}, xs), ys)
  return l


fun_single_step_grad = bm.grad(fun_single_step_loss, grad_vars=net.train_vars(), return_value=True)
optimizer = bp.optimizers.Adam(lr=args.lr, train_vars=net.train_vars())


@bm.jit(child_objs=(fun_single_step_grad, optimizer, cann))
def step_fun_train(position, amp):
  grads, l = fun_single_step_grad(position, amp)
  optimizer.update(grads)
  return l


length = 200

# train 20 ms
data_fs = [
  # given an input at a fixed position
  lambda: {'pos': np.tile(np.random.uniform(-np.pi, np.pi, (1, 1, 2)), (length, 1, 1)),
           'amp': np.tile(np.random.uniform(2., 20, (1, 1)), (length, 1))},
  # given an input at a fixed position at the first 10 ms, then input disappears
  lambda: {'pos': np.tile(np.random.uniform(-np.pi, np.pi, (1, 1, 2)), (length, 1, 1)),
           'amp': np.concatenate([np.tile(np.random.uniform(2., 20, (1, 1)), (length // 2, 1)),
                                  np.zeros((length // 2, 1))],
                                 axis=0)},
  # given a moving bump
  lambda: {'pos': np.asarray([np.linspace(-np.pi, np.pi, length, ),
                              np.linspace(-np.pi, np.pi, length, )]).T.reshape((length, 1, 2)),
           'amp': np.tile(np.random.uniform(2., 20., (1, 1)), (length, 1))},
  # given a moving bump at the first 10 ms, then input disappears
  lambda: {'pos': np.asarray([np.linspace(-np.pi, np.pi, length),
                              np.linspace(-np.pi, np.pi, length)]).T.reshape((length, 1, 2)),
           'amp': np.concatenate([np.tile(np.random.uniform(2., 20, (1, 1)), (length // 2, 1)),
                                  np.zeros((length // 2, 1))],
                                 axis=0)},
  # given a bump with varying amplitude
  lambda: {'pos': np.tile(np.random.uniform(-np.pi, np.pi, (1, 1, 2)), (length, 1, 1)),
           'amp': np.random.uniform(2., 20, (length, 1))},
]


def generate_data():
  all_pos = []
  all_amp = []
  for di in np.random.randint(0, len(data_fs), args.batch_size):
    pos_and_amp = data_fs[di]()
    all_pos.append(pos_and_amp['pos'])
    all_amp.append(pos_and_amp['amp'])
  all_pos = np.concatenate(all_pos, axis=1)
  all_amp = np.concatenate(all_amp, axis=1)
  for li in range(length):
    yield all_pos[li], all_amp[li]


min_loss = 1e9

for i in range(args.epoch):
  t0 = time.time()
  losses = []
  cann.reset_state(args.batch_size)
  net.reset_state(args.batch_size)
  for data in generate_data():
    loss = step_fun_train(*data)
    losses.append(loss)
  current_loss = np.mean(losses)
  print(f'Epoch {i}, time {time.time() - t0:.5f} s, loss {current_loss:.6f}')

  if current_loss < min_loss:
    bp.checkpoints.save_pytree('results/unet-cann-2d.bp',
                               target={'epoch': i,
                                       'optimizer': optimizer.state_dict(),
                                       'model': net.state_dict(),
                                       'args': args.__dict__},
                               overwrite=True)
    min_loss = current_loss
