# -*- coding: utf-8 -*-
""" to create figures for spiking network models in joglekar et al neuron 2018


Modified from https://github.com/OpenSourceBrain/JoglekarEtAl18

"""

from brian2 import *
clear_cache('cython')

import scipy.io
import random as pyrand
import numpy as np
import matplotlib.pyplot as plt

import os

rnd_seed = 1
pyrand.seed(324823 + rnd_seed)
numpy.random.seed(324823 + rnd_seed)


# Raster Plot
def rasterPlot(xValues, yValues, duration, figure, N, saveFigure, path):
  ticks = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5,
           8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5,
           16.5, 17.5, 18.5, 19.5, 20.5, 21.5, 22.5, 23.5,
           24.5, 25.5, 26.5, 27.5, 28.5]

  areasName = ['V1', 'V2', 'V4', 'DP', 'MT',
               '8m', '5', '8l', 'TEO', '2', 'F1',
               'STPc', '7A', '46d', '10', '9/46v',
               '9/46d', 'F5', 'TEpd', 'PBr', '7m', '7B',
               'F2', 'STPi', 'PROm', 'F7', '8B', 'STPr', '24c']

  plt.figure()
  plt.plot(xValues, 1.0 * yValues / (4 * 400), '.', markersize=1)
  plt.plot([0, duration], np.arange(N + 1).repeat(2).reshape(-1, 2).T, 'k-')
  plt.ylabel('Area')
  plt.yticks(np.arange(N))
  plt.xlabel('time (ms)')
  plt.ylim(0, N)
  plt.yticks(ticks[:N], areasName[:N])
  plt.xlim(0, duration)

  # Save figure
  if saveFigure == 'yes':
    plt.savefig(path + '/figures/figure' + figure + '_' + str(N) + 'areas.png')

  plt.show()

  return 0


# Plot for Maximum firing rate
def firingRatePlot(maxrategood, maxratebad, figure, N, saveFigure, path):
  areasName = ['V1', 'V2', 'V4', 'DP', 'MT',
               '8m', '5', '8l', 'TEO', '2', 'F1',
               'STPc', '7A', '46d', '10', '9/46v',
               '9/46d', 'F5', 'TEpd', 'PBr', '7m', '7B',
               'F2', 'STPi', 'PROm', 'F7', '8B', 'STPr', '24c']

  plt.semilogy(range(N), maxratebad, '.-', color=(0.4660, 0.6740, 0.1880), linewidth=1, label='weak GBA')
  plt.semilogy(range(N), maxrategood, '.-', color=(0.4940, 0.1840, 0.5560), linewidth=1, label='strong GBA')
  plt.xticks(range(N), areasName)
  plt.xticks(rotation=90)
  plt.xlabel('Area')
  plt.ylabel('Maximum Firing Rate (Hz)')
  plt.legend(('weak GBA', 'strong GBA'))

  # Save figure
  if saveFigure == 'yes':
    plt.savefig(path + '/figures/figure' + figure + '_' + str(N) + 'areas.png')

  plt.show()

  return 0


def firingRate(N, goodprop, duration):
  binsize = 10  # [ms]
  stepsize = 1  # [ms]

  # Store maximum firing rate for each area
  maxrategood = np.empty([N, 1])

  # sort net spikes
  netspikegood = len(goodprop)

  goodpropsorted = goodprop[goodprop[:, 1].argsort(),]

  netbinno = int(1 + (duration) - (binsize))
  poprategood = np.empty([N, netbinno])

  countgood = 0  # for each spike.

  monareaktimeallgood = []

  for u in range(N):
    monareaktimegood = []

    while ((countgood < netspikegood) and (goodpropsorted[countgood, 1] < 1600 * (u + 1))):
      monareaktimegood.append(goodpropsorted[countgood, 0])  # append spike times for each area.
      countgood = countgood + 1

    valsgood = np.histogram(monareaktimegood, bins=int(duration / stepsize))

    valszerogood = valsgood[0]

    astep = binsize

    valsnewgood = np.zeros(netbinno)

    acount = 0
    while acount < netbinno:
      valsnewgood[acount] = sum(valszerogood[int(acount):int(acount + astep)])
      acount = acount + 1

    valsrategood = valsnewgood * ((1000 / binsize) / (1600))
    poprategood[u, :] = valsrategood

    # compute population firing rates.
    maxrategood[u, 0] = max(valsrategood[int(len(valsrategood) / 3):])

  return maxrategood, poprategood


def gen_params(n_areas, regime, gba, duration):
  para = {
    'N': 2000,  # Number of neurons
    'NAreas': n_areas,  # Number of areas
    'Ne': 0.8,  # fraction of excitatory neurons
    'Vr': -70. * mV,  # Membrane potential rest
    'Vreset': -60. * mV,  # Membrane potential reset
    'Vt': -50. * mV,  # Membrane potential threshold
    'taumE': 20.,  # Membrane time constant (excitatory neurons)
    'taumI': 10.,  # Membrane time constant (inhibitory neurons)
    'tref': 2.,  # refractory time
    'probIntra': .1,  # connection probability intra area
    'probInter': .1,  # connection probability inter areas
    'sigma': 3.,  # Noise
    'alpha': 4.,  # gradient
    'dlocal': 2.,  # delays local
    'speed': 3.5,  # axonal conduction velocity
    'lrvar': 0.1,  # standard deviation delay long range
    'path': 'Matlab/'  # path to .mat files
  }

  # hierarchy values file
  hierVals = scipy.io.loadmat(para["path"] + 'hierValspython.mat')
  hierValsnew = hierVals['hierVals'][:]
  hier = hierValsnew / max(hierValsnew)  # hierarchy normalized.
  para['hier'] = hier[:para["NAreas"]]

  # fln values file
  flnMatp = scipy.io.loadmat(para["path"] + 'efelenMatpython.mat')
  conn = flnMatp['flnMatpython'][:][:]  # fln values..Cij is strength from j to i
  para['conn'] = conn[:para["NAreas"], :para["NAreas"]]

  distMatp = scipy.io.loadmat(para["path"] + 'subgraphWiring29.mat')
  distMat = distMatp['wiring'][:][:]  # distances between areas values..
  delayMat = distMat / para['speed']
  para['delayMat'] = delayMat[:para["NAreas"], :para["NAreas"]]

  para['k'] = 400

  # Membrane resitance
  R = 50 * Mohm

  para['duration'] = duration * ms;
  if regime == 'asynchronous':

    # general for assynchronous regime
    para['VextE'] = 14.2
    para['VextI'] = 14.7
    para['muIE'] = .19 / 4
    para['wII'] = .075
    para['wEE'] = .01
    para['wIE'] = .075

    if gba == 'weak':
      para['wEI'] = .0375
      para['muEE'] = .0375
      para['currval'] = (300 * pA * R) / mV

    elif gba == 'strong':
      para['wEI'] = .05
      para['muEE'] = .05
      para['currval'] = (126 * pA * R) / mV

    para['currdur'] = 1500

  elif regime == 'synchronous':

    # general for synchronous regime
    para['muIE'] = .19
    para['wII'] = .3
    para['wEE'] = .04
    para['wIE'] = .3

    if gba == 'weak':
      para['VextI'] = 14.0
      para['VextE'] = 15.4
      para['wEI'] = .56
      para['muEE'] = .16
    elif gba == 'strong':
      para['VextI'] = 14.0
      para['VextE'] = 16.0
      para['wEI'] = .98
      para['muEE'] = .25

    para['currdur'] = 80
    para['currval'] = (202 * pA * R) / mV

  return para


def equations():
  # Equations
  eqsE = Equations('''
    dV/dt=(-(V-Vr) + stimulus(t,i) + Vext )*(1./tau) + 
    (sigma*(1./tau)**0.5)*xi : volt (unless refractory)
    
    Vext : volt    
    tau: second
    sigma : volt
    Vr:volt
    
    ''')

  eqsI = Equations('''
      dV/dt=(-(V-Vr) + Vext )*(1./tau) + 
      (sigma*(1./tau)**0.5)*xi : volt (unless refractory)
     
    Vext : volt    
    tau: second
    sigma : volt
    Vr:volt  
      
      ''')

  return eqsE, eqsI


def setStimulus(para):
  # Stimulus
  netsteps = round(para['duration'] / defaultclock.dt)

  a1 = np.zeros([3000, 1])  # input given to v1 for fixed duration.
  a2 = para['currval'] * np.ones([para['currdur'], 1])
  a3 = np.zeros([int(netsteps - 3000 - para['currdur']), 1])
  aareaone = np.vstack((a1, a2, a3))

  timelen = len(aareaone)
  excotherareas = para['k'] * 4 * (para['NAreas'] - 1)
  aareaonenet = np.tile(aareaone, (1, para['k'] * 4))
  arest = np.zeros([timelen, excotherareas])
  netarr = np.hstack((aareaonenet, arest))

  stimulus = TimedArray(netarr * mV, dt=defaultclock.dt)

  return stimulus


def network(para):
  # Equations
  eqsE, eqsI = equations()

  # Total number of excitatory neurons
  NE = int(para['NAreas'] * para['N'] * para['Ne'])

  # Total number of inhibitory neurons
  NI = int((para['NAreas'] * para['N']) - NE)

  # Parameters
  paraVt = para['Vt']
  paraVreset = para['Vreset']

  # Neuron groups
  E = NeuronGroup(N=NE, method='euler', model=eqsE, threshold='V > paraVt', reset='V=paraVreset',
                  refractory=para['tref'] * ms)
  I = NeuronGroup(N=NI, method='euler', model=eqsI, threshold='V > paraVt', reset='V=paraVreset',
                  refractory=para['tref'] * ms)

  # E I across areas
  Exc, Inh = [], []
  Exc = [E[y * (para['k'] * 4):(y + 1) * (para['k'] * 4)] for y in range(para['NAreas'])]
  Inh = [I[z * (para['k']):(z + 1) * (para['k'])] for z in range(para['NAreas'])]

  # List to store connections
  Exc_C_loc = [None] * para['NAreas']
  Inh_C_loc = [None] * para['NAreas']
  EtoI_C_loc = [None] * para['NAreas']
  ItoE_C_loc = [None] * para['NAreas']

  Exc_C_lr_fromi = []
  EtoI_C_lr_fromi = []

  # set up synaptic connections
  h = 0
  while h < para['NAreas']:
    # print(h)  #local.
    Exc_C_loc[h] = Synapses(Exc[h], Exc[h], 'w:volt', delay=para["dlocal"] * ms, on_pre='V+=w')
    Inh_C_loc[h] = Synapses(Inh[h], Inh[h], 'w:volt', delay=para["dlocal"] * ms, on_pre='V+= w ')
    EtoI_C_loc[h] = Synapses(Exc[h], Inh[h], 'w:volt', delay=para["dlocal"] * ms, on_pre='V+= w ')
    ItoE_C_loc[h] = Synapses(Inh[h], Exc[h], 'w:volt', delay=para["dlocal"] * ms, on_pre='V+= w ')

    Exc_C_loc[h].connect(p=para["probIntra"])
    Inh_C_loc[h].connect(p=para["probIntra"])
    EtoI_C_loc[h].connect(p=para["probIntra"])
    ItoE_C_loc[h].connect(p=para["probIntra"])

    Exc_C_loc[h].w = (1 + para["alpha"] * para["hier"][h]) * para['wEE'] * mV
    Inh_C_loc[h].w = -para['wII'] * mV
    EtoI_C_loc[h].w = (1 + para["alpha"] * para["hier"][h]) * para['wIE'] * mV
    ItoE_C_loc[h].w = -para['wEI'] * mV

    j = 0  # long range to j.
    while j < para['NAreas']:
      if j != h:
        # print j
        exc_lr_itoj, etoi_lr_itoj = None, None

        exc_lr_itoj = Synapses(Exc[h], Exc[j], 'w:volt', on_pre='V+= w ')
        etoi_lr_itoj = Synapses(Exc[h], Inh[j], 'w:volt', on_pre='V+= w ')

        exc_lr_itoj.connect(p=para["probInter"])
        etoi_lr_itoj.connect(p=para["probInter"])

        exc_lr_itoj.w = (1 + para["alpha"] * para["hier"][j]) * para['muEE'] * para["conn"][j, h] * mV
        etoi_lr_itoj.w = (1 + para["alpha"] * para["hier"][j]) * para['muIE'] * para["conn"][j, h] * mV

        # Mean for delay distribution
        meanlr = para["delayMat"][j, h]
        # Standard deviation for delay distribution
        varlr = para['lrvar'] * meanlr

        exc_lr_itoj.delay = np.random.normal(meanlr, varlr, len(exc_lr_itoj.w)) * ms
        etoi_lr_itoj.delay = np.random.normal(meanlr, varlr, len(etoi_lr_itoj.w)) * ms

        Exc_C_lr_fromi.append(exc_lr_itoj)
        EtoI_C_lr_fromi.append(etoi_lr_itoj)

      j = j + 1
    h = h + 1

  # Initial conditions
  E.V = para['Vr'] + rand(len(E)) * (para['Vt'] - para['Vr'])
  E.tau = para['taumE'] * ms
  E.Vext = para['VextE'] * mV
  E.sigma = para['sigma'] * mV
  E.Vr = para['Vr']

  I.V = para['Vr'] + rand(len(I)) * (para['Vt'] - para['Vr'])
  I.tau = para['taumI'] * ms
  I.Vext = para['VextI'] * mV
  I.sigma = para['sigma'] * mV
  I.Vr = para['Vr']

  # Monitors
  monitorsE = SpikeMonitor(E)
  monitorsI = SpikeMonitor(I)

  # Network
  net = Network(E, I, Exc_C_loc, EtoI_C_loc, ItoE_C_loc, Inh_C_loc, Exc_C_lr_fromi, EtoI_C_lr_fromi, monitorsE,
                monitorsI)

  net.store()
  print("net stored")
  net.run(para['duration'], report='text')

  return monitorsE, monitorsI


def run_network(n_areas, regime, gba, duration):
  # Parameters
  para = gen_params(n_areas, regime, gba, duration)
  # Run Network - Brian
  monitor_spike, monitor_v = network(para)

  return monitor_spike


# Arguments
N = 29
figure = '5B'
saveData = 'no'
saveFigure = 'yes'
duration = 0.1

# Path
path = os.path.abspath(os.getcwd())

# Simulations

# Figure 5B
file = path + '/files/spikes_figure' + figure + '_' + str(N) + 'areas.txt'

t0 = time.time()
monitors = run_network(N, 'asynchronous', 'weak', duration)
t1 = time.time()

print(f'Compilation time {(t1 - t0)} s')


# xValues = monitors.t / ms
# yValues = monitors.i
# # Save data
# if saveData == 'yes':
#   np.savetxt(file, np.column_stack([monitors.t / ms, monitors.i]))
#
# rasterPlot(xValues, yValues, duration, figure, N, saveFigure, path)

