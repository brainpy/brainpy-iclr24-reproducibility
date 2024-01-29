import sys

from brian2 import *

if sys.argv[1] == 'cuda_standalone':
  import brian2cuda
  set_device("cuda_standalone", build_on_run=False)

else:
  clear_cache('cython')

defaultclock.dt = 0.1 * ms


def run_(scale=1.):
  start_scope()
  device.reinit()
  device.activate()

  taum = 20 * ms
  taue = 5 * ms
  taui = 10 * ms
  Vt = -50 * mV
  Vr = -60 * mV
  El = -60 * mV
  Erev_exc = 0. * mV
  Erev_inh = -80. * mV
  I = 20. * mvolt
  num_exc = int(3200 * scale)
  num_inh = int(800 * scale)

  eqs = '''
  dv/dt  = (ge*(Erev_exc-v)+gi*(Erev_inh-v)-(v-El) + I)*(1./taum) : volt (unless refractory)
  dge/dt = -ge/taue : 1 
  dgi/dt = -gi/taui : 1 
  '''
  P = NeuronGroup(num_exc + num_inh,
                  eqs,
                  threshold='v>Vt',
                  reset='v = Vr',
                  refractory=5 * ms,
                  method='euler')

  we = 0.6 / scale  # excitatory synaptic weight (voltage)
  wi = 6.7 / scale  # inhibitory synaptic weight
  Ce = Synapses(P[:num_exc], P, on_pre='ge += we')
  Ci = Synapses(P[num_exc:], P, on_pre='gi += wi')

  P.v = (np.random.randn(num_exc + num_inh) * 5. - 55.) * mvolt
  Ce.connect(p=0.02)
  Ci.connect(p=0.02)

  t1 = time.time()
  run(0.1 * ms)
  t2 = time.time()

  # t3 = time.time()
  # run(0.1 * ms)
  # t4 = time.time()
  # compilation_time = (t2 - t1) - (t4 - t3)

  compilation_time = (t2 - t1)
  print(f'Network size = {num_exc + num_inh}, Compilation spend {compilation_time} s')


if __name__ == '__main__':
  run_(scale=float(sys.argv[2]))
