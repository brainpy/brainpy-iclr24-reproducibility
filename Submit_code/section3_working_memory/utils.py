import os
import time

import matplotlib.pyplot as plt
import numpy as np
import tqdm

import brainpy as bp
import brainpy.math as bm
import brainpy_datasets as bd


@bp.tools.numba_jit
def _dmc(num_steps, num_inputs, n_motion_choice, motion_tuning,
         is_spiking_mode, sample_time, test_time, fr, bg_fr):
  # data
  X = np.zeros((num_steps, num_inputs))

  # sample
  match = np.random.randint(2)
  sample_dir = np.random.randint(n_motion_choice)

  # Generate the sample and test stimuli based on the rule
  half_num_motion = n_motion_choice // 2
  sample_cat = sample_dir // half_num_motion
  if match == 1:  # match trial
    # do not use sample_dir as a match test stimulus
    dir0 = sample_cat * half_num_motion
    dir1 = dir0 + half_num_motion
    test_dir = np.random.choice(np.arange(dir0, dir1))
  else:
    test_dir = (1 - sample_cat) * half_num_motion + np.random.randint(half_num_motion)

  # SAMPLE stimulus
  X[sample_time] += motion_tuning[sample_dir] * fr
  # TEST stimulus
  X[test_time] += motion_tuning[test_dir] * fr
  X += bg_fr

  # to spiking
  if is_spiking_mode:
    X = np.random.random(X.shape) < X
    X = X.astype(np.float_)

  # can use a greater weight for test period if needed
  return X, match


class DMC(bd.cognitive.CognitiveTask):
  times = ('dead', 'fixation', 'sample', 'delay', 'test')
  output_features = ('non-match', 'match')

  def __init__(
      self,
      dt=100., t_fixation=500., t_sample=500., t_delay=1000., t_test=500.,
      limits=(0., np.pi * 2), kappa=2, bg_fr=1., ft_motion=bd.cognitive.Feature(24, 100, 40.),
      num_trial=1024, mode='rate', seed=None,
  ):
    super().__init__(dt=dt, num_trial=num_trial, seed=seed)
    # time
    self.t_fixation = int(t_fixation / dt)
    self.t_sample = int(t_sample / dt)
    self.t_delay = int(t_delay / dt)
    self.t_test = int(t_test / dt)
    self.num_steps = self.t_fixation + self.t_sample + self.t_delay + self.t_test
    self._times = {
      'fixation': self.t_fixation,
      'sample': self.t_sample,
      'delay': self.t_delay,
      'test': self.t_test,
    }
    test_onset = self.t_fixation + self.t_sample + self.t_delay
    self.test_time = slice(test_onset, test_onset + self.t_test)
    self.fix_time = slice(0, test_onset)
    self.sample_time = slice(self.t_fixation, self.t_fixation + self.t_sample)

    # input shape
    self.features = ft_motion.set_name('motion')
    self.features.set_mode(mode)
    self.bg_fr = bg_fr  # background firing rate
    self.v_min = limits[0]
    self.v_max = limits[1]
    self.v_range = limits[1] - limits[0]

    # Tuning function data
    self.n_motion_choice = 8
    self.kappa = kappa  # concentration scaling factor for von Mises

    # Generate list of preferred directions
    # dividing neurons by 2 since two equal
    # groups representing two modalities
    pref_dirs = np.arange(self.v_min, self.v_max, self.v_range / ft_motion.num)

    # Generate list of possible stimulus directions
    stim_dirs = np.arange(self.v_min, self.v_max, self.v_range / self.n_motion_choice)

    d = np.cos(np.expand_dims(stim_dirs, 1) - pref_dirs)
    self.motion_tuning = np.exp(self.kappa * d) / np.exp(self.kappa)

  @property
  def num_inputs(self) -> int:
    return self.features.num

  @property
  def num_outputs(self) -> int:
    return 2

  def sample_a_trial(self, index):
    fr = self.features.fr(self.dt)
    bg_fr = bd.cognitive.firing_rate(self.bg_fr, self.dt, self.features.mode)
    return _dmc(self.num_steps, self.num_inputs, self.n_motion_choice,
                self.motion_tuning, self.features.is_spiking_mode,
                self.sample_time, self.test_time, fr, bg_fr)


@bp.tools.numba_jit
def _dms(num_steps, num_inputs, n_motion_choice, motion_tuning, is_spiking_mode,
         sample_time, test_time, fr, bg_fr, rotate_dir):
  # data
  X = np.zeros((num_steps, num_inputs))

  # sample
  match = np.random.randint(2)
  sample_dir = np.random.randint(n_motion_choice)

  # Generate the sample and test stimuli based on the rule
  if match == 1:  # match trial
    test_dir = (sample_dir + rotate_dir) % n_motion_choice
  else:
    test_dir = np.random.randint(n_motion_choice)
    while test_dir == ((sample_dir + rotate_dir) % n_motion_choice):
      test_dir = np.random.randint(n_motion_choice)

  # SAMPLE stimulus
  X[sample_time] += motion_tuning[sample_dir] * fr
  # TEST stimulus
  X[test_time] += motion_tuning[test_dir] * fr
  X += bg_fr

  # to spiking
  if is_spiking_mode:
    X = np.random.random(X.shape) < X
    X = X.astype(np.float_)

  # can use a greater weight for test period if needed
  return X, match


_rotate_choice = {
  '0': 0,
  '45': 1,
  '90': 2,
  '135': 3,
  '180': 4,
  '225': 5,
  '270': 6,
  '315': 7,
  '360': 8,
}


class DMS(bd.cognitive.CognitiveTask):
  times = ('dead', 'fixation', 'sample', 'delay', 'test')
  output_features = ('non-match', 'match')

  def __init__(
      self,
      dt=100., t_fixation=500., t_sample=500., t_delay=1000., t_test=500.,
      limits=(0., np.pi * 2), rotation_match='0', kappa=2,
      bg_fr=1., ft_motion=bd.cognitive.Feature(24, 100, 40.),
      num_trial=1024, mode='rate', seed=None,
  ):
    super().__init__(dt=dt, num_trial=num_trial, seed=seed)
    # time
    self.t_fixation = int(t_fixation / dt)
    self.t_sample = int(t_sample / dt)
    self.t_delay = int(t_delay / dt)
    self.t_test = int(t_test / dt)
    self.num_steps = self.t_fixation + self.t_sample + self.t_delay + self.t_test
    self._times = {
      'fixation': self.t_fixation,
      'sample': self.t_sample,
      'delay': self.t_delay,
      'test': self.t_test,
    }
    test_onset = self.t_fixation + self.t_sample + self.t_delay
    self.test_time = slice(test_onset, test_onset + self.t_test)
    self.fix_time = slice(0, test_onset)
    self.sample_time = slice(self.t_fixation, self.t_fixation + self.t_sample)

    # input shape
    self.features = ft_motion.set_name('motion')
    self.features.set_mode(mode)
    self.rotation_match = rotation_match
    self._rotate = _rotate_choice[rotation_match]
    self.bg_fr = bg_fr  # background firing rate
    self.v_min = limits[0]
    self.v_max = limits[1]
    self.v_range = limits[1] - limits[0]

    # Tuning function data
    self.n_motion_choice = 8
    self.kappa = kappa  # concentration scaling factor for von Mises

    # Generate list of preferred directions
    # dividing neurons by 2 since two equal
    # groups representing two modalities
    pref_dirs = np.arange(self.v_min, self.v_max, self.v_range / ft_motion.num)

    # Generate list of possible stimulus directions
    stim_dirs = np.arange(self.v_min, self.v_max, self.v_range / self.n_motion_choice)

    d = np.cos(np.expand_dims(stim_dirs, 1) - pref_dirs)
    self.motion_tuning = np.exp(self.kappa * d) / np.exp(self.kappa)

  @property
  def num_inputs(self) -> int:
    return self.features.num

  @property
  def num_outputs(self) -> int:
    return 2

  def sample_a_trial(self, index):
    fr = self.features.fr(self.dt)
    bg_fr = bd.cognitive.firing_rate(self.bg_fr, self.dt, self.features.mode)
    return _dms(self.num_steps, self.num_inputs, self.n_motion_choice,
                self.motion_tuning, self.features.is_spiking_mode,
                self.sample_time, self.test_time, fr, bg_fr, self._rotate)

@bp.tools.numba_jit
def _dms_dmrs(num_steps, num_inputs, n_motion_choice,
              motion_tuning, rule_tuning, is_spiking_mode,
              sample_time, test_time, rule_time,
              fr, bg_fr, rule_fr, rotate_dir):
  # data
  X = np.zeros((num_steps, num_inputs))

  # sample
  match = np.random.randint(2)
  sample_dir = np.random.randint(n_motion_choice)

  # Generate the sample and test stimuli based on the rule
  if match == 1:  # match trial
    test_dir = (sample_dir + rotate_dir) % n_motion_choice
  else:
    test_dir = np.random.randint(n_motion_choice)
    while test_dir == ((sample_dir + rotate_dir) % n_motion_choice):
      test_dir = np.random.randint(n_motion_choice)

  # SAMPLE stimulus
  X[sample_time] += motion_tuning[sample_dir] * fr
  # TEST stimulus
  X[test_time] += motion_tuning[test_dir] * fr
  X += bg_fr
  # rule
  X[rule_time] += (rule_tuning * rule_fr)

  # to spiking
  if is_spiking_mode:
    X = np.random.random(X.shape) < X
    X = X.astype(np.float_)

  # can use a greater weight for test period if needed
  return X, match


class DMS_DMRS(bd.cognitive.CognitiveTask):
  times = ('dead', 'fixation', 'sample', 'delay', 'test')
  output_features = ('non-match', 'match')

  def __init__(
      self,
      dt=100., t_fixation=500., t_sample=500., t_delay=1000., t_test=500.,
      limits=(0., np.pi * 2), rotation_match='0',
      kappa=2, bg_fr=1., choices=('0', '90'),
      ft_motion=bd.cognitive.Feature(24, 100, 40.),
      ft_rule=bd.cognitive.Feature(6, 20, 40.),
      num_trial=1024, mode='rate', seed=None,
  ):
    super().__init__(dt=dt, num_trial=num_trial, seed=seed)

    # time
    self.t_fixation = int(t_fixation / dt)
    self.t_sample = int(t_sample / dt)
    self.t_delay = int(t_delay / dt)
    self.t_test = int(t_test / dt)
    self.num_steps = self.t_fixation + self.t_sample + self.t_delay + self.t_test
    self._times = {
      'fixation': self.t_fixation,
      'sample': self.t_sample,
      'delay': self.t_delay,
      'test': self.t_test,
    }
    test_onset = self.t_fixation + self.t_sample + self.t_delay
    self.test_time = slice(test_onset, test_onset + self.t_test)
    self.sample_time = slice(self.t_fixation, self.t_fixation + self.t_sample)
    t_rule_start = self.t_delay // 2
    t_rule = self.t_delay // 4
    self.rule_time = slice(self.t_fixation + self.t_sample + t_rule_start,
                           self.t_fixation + self.t_sample + t_rule_start + t_rule)

    # input shape
    for c in choices:
      assert c in _rotate_choice
    self.choices = [_rotate_choice[c] for c in choices]
    self.features = ft_motion.set_name('motion') + ft_rule.set_name('rule')
    self.features.set_mode(mode)
    self.rotation_match = rotation_match
    self._rotate = _rotate_choice[rotation_match]
    self.bg_fr = bg_fr  # background firing rate
    self.v_min = limits[0]
    self.v_max = limits[1]
    self.v_range = limits[1] - limits[0]
    self.num_rule = 2

    # Tuning function data
    self.n_motion_choice = 8
    self.kappa = kappa  # concentration scaling factor for von Mises

    # Generate list of preferred directions
    # dividing neurons by 2 since two equal
    # groups representing two modalities
    pref_dirs = np.arange(self.v_min, self.v_max, self.v_range / ft_motion.num)
    # Generate list of possible stimulus directions
    stim_dirs = np.arange(self.v_min, self.v_max, self.v_range / self.n_motion_choice)
    d = np.cos(np.expand_dims(stim_dirs, 1) - pref_dirs)
    motion_tuning = np.exp(self.kappa * d) / np.exp(self.kappa)
    self.motion_tuning = np.concatenate(
      [motion_tuning, np.zeros((self.n_motion_choice, self.features['rule'].num))],
      axis=1
    )

    # rule tuning
    rule_tuning = np.zeros((self.num_rule, self.features.num))
    per_rule = self.features['rule'].num // self.num_rule
    for i in range(self.num_rule):
      s = self.features['motion'].num + per_rule * i
      e = self.features['motion'].num + per_rule * (i + 1)
      rule_tuning[i, s: e] = 1.
    self.rule_tuning = rule_tuning

  @property
  def num_inputs(self) -> int:
    return self.features.num

  @property
  def num_outputs(self) -> int:
    return 2

  def sample_a_trial(self, index):
    motion_fr = self.features['motion'].fr(self.dt)
    bg_fr = bd.cognitive.firing_rate(self.bg_fr, self.dt, self.features.mode)
    rule_fr = self.features['rule'].fr(self.dt)
    rule = self.rng.choice([0, 1])
    return _dms_dmrs(self.num_steps, self.num_inputs, self.n_motion_choice,
                     self.motion_tuning, self.rule_tuning[rule],
                     self.features.is_spiking_mode,
                     self.sample_time, self.test_time, self.rule_time,
                     motion_fr, bg_fr, rule_fr, self.choices[rule])


@bp.tools.numba_jit
def _dms_dmc(num_steps, num_inputs, n_motion_choice,
             motion_tuning, rule_tuning, is_spiking_mode,
             sample_time, test_time, rule_time,
             fr, bg_fr, rule_fr, rule):
  # data
  X = np.zeros((num_steps, num_inputs))

  # sample
  match = np.random.randint(2)
  sample_dir = np.random.randint(n_motion_choice)

  # Generate the sample and test stimuli based on the rule
  if rule == 0:
    if match == 1:  # match trial
      test_dir = sample_dir
    else:
      test_dir = np.random.randint(n_motion_choice)
      while test_dir == sample_dir:
        test_dir = np.random.randint(n_motion_choice)
  elif rule == 1:
    half_num_motion = n_motion_choice // 2
    sample_cat = sample_dir // half_num_motion
    if match == 1:  # match trial
      # do not use sample_dir as a match test stimulus
      dir0 = sample_cat * half_num_motion
      dir1 = dir0 + half_num_motion
      test_dir = np.random.choice(np.arange(dir0, dir1))
    else:
      test_dir = (1 - sample_cat) * half_num_motion + np.random.randint(half_num_motion)
  else:
    raise ValueError

  # SAMPLE stimulus
  X[sample_time] += motion_tuning[sample_dir] * fr
  # TEST stimulus
  X[test_time] += motion_tuning[test_dir] * fr
  # rule fr
  X[rule_time] += (rule_tuning * rule_fr)
  # background fr
  X += bg_fr

  # to spiking
  if is_spiking_mode:
    X = np.random.random(X.shape) < X
    X = X.astype(np.float_)

  # can use a greater weight for test period if needed
  return X, match


class DMS_DMC(bd.cognitive.CognitiveTask):
  times = ('dead', 'fixation', 'sample', 'delay', 'test')
  output_features = ('non-match', 'match')

  def __init__(
      self,
      dt=100., t_fixation=500., t_sample=500., t_delay=1000., t_test=500.,
      limits=(0., np.pi * 2), rotation_match='0', kappa=2, bg_fr=1.,
      ft_motion=bd.cognitive.Feature(24, 100, 40.),
      ft_rule=bd.cognitive.Feature(6, 20, 40.),
      num_trial=1024, mode='rate', seed=None,
  ):
    super().__init__(dt=dt, num_trial=num_trial, seed=seed)

    # time
    self.t_fixation = int(t_fixation / dt)
    self.t_sample = int(t_sample / dt)
    self.t_delay = int(t_delay / dt)
    self.t_test = int(t_test / dt)
    self.num_steps = self.t_fixation + self.t_sample + self.t_delay + self.t_test
    self._times = {
      'fixation': self.t_fixation,
      'sample': self.t_sample,
      'delay': self.t_delay,
      'test': self.t_test,
    }
    test_onset = self.t_fixation + self.t_sample + self.t_delay
    self.test_time = slice(test_onset, test_onset + self.t_test)
    self.sample_time = slice(self.t_fixation, self.t_fixation + self.t_sample)
    self.rule_time = slice(0, self.t_fixation + self.t_sample + self.t_delay + self.t_test)

    # input shape
    self.features = ft_motion.set_name('motion') + ft_rule.set_name('rule')
    self.features.set_mode(mode)
    self.rotation_match = rotation_match
    self._rotate = _rotate_choice[rotation_match]
    self.bg_fr = bg_fr  # background firing rate
    self.v_min = limits[0]
    self.v_max = limits[1]
    self.v_range = limits[1] - limits[0]

    # Tuning function data
    self.n_motion_choice = 8
    self.kappa = kappa  # concentration scaling factor for von Mises

    # Generate list of preferred directions
    # dividing neurons by 2 since two equal
    # groups representing two modalities
    pref_dirs = np.arange(self.v_min, self.v_max, self.v_range / ft_motion.num)
    # Generate list of possible stimulus directions
    stim_dirs = np.arange(self.v_min, self.v_max, self.v_range / self.n_motion_choice)
    d = np.cos(np.expand_dims(stim_dirs, 1) - pref_dirs)
    motion_tuning = np.exp(self.kappa * d) / np.exp(self.kappa)
    self.motion_tuning = np.concatenate(
      [motion_tuning, np.zeros((self.n_motion_choice, self.features['rule'].num))],
      axis=1
    )

    # rule tuning
    self.num_rule = 2
    rule_tuning = np.zeros((self.num_rule, self.features.num))
    per_rule = self.features['rule'].num // self.num_rule
    for i in range(self.num_rule):
      s = self.features['motion'].num + per_rule * i
      e = self.features['motion'].num + per_rule * (i + 1)
      rule_tuning[i, s: e] = 1.
    self.rule_tuning = rule_tuning

  @property
  def num_inputs(self) -> int:
    return self.features.num

  @property
  def num_outputs(self) -> int:
    return 2

  def sample_a_trial(self, index):
    motion_fr = self.features['motion'].fr(self.dt)
    bg_fr = bd.cognitive.firing_rate(self.bg_fr, self.dt, self.features.mode)
    rule_fr = self.features['rule'].fr(self.dt)
    rule = self.rng.choice([0, 1])
    return _dms_dmc(self.num_steps, self.num_inputs, self.n_motion_choice,
                    self.motion_tuning, self.rule_tuning[rule],
                    self.features.is_spiking_mode,
                    self.sample_time, self.test_time, self.rule_time,
                    motion_fr, bg_fr, rule_fr, rule)


@bp.tools.numba_jit
def _dms_dmrs_dmc(num_steps, num_inputs, n_motion_choice,
                  motion_tuning, rule_tuning, is_spiking_mode,
                  sample_time, test_time, rule_time,
                  fr, bg_fr, rule_fr, rotate_dir, rule):
  # data
  X = np.zeros((num_steps, num_inputs))

  # sample
  match = np.random.randint(2)
  sample_dir = np.random.randint(n_motion_choice)

  # Generate the sample and test stimuli based on the rule
  if rule == 2:
    half_num_motion = n_motion_choice // 2
    sample_cat = sample_dir // half_num_motion
    if match == 1:  # match trial
      # do not use sample_dir as a match test stimulus
      dir0 = sample_cat * half_num_motion
      dir1 = dir0 + half_num_motion
      test_dir = np.random.choice(np.arange(dir0, dir1))
    else:
      test_dir = (1 - sample_cat) * half_num_motion + np.random.randint(half_num_motion)
  else:
    if match == 1:  # match trial
      test_dir = (sample_dir + rotate_dir) % n_motion_choice
    else:
      test_dir = np.random.randint(n_motion_choice)
      while test_dir == ((sample_dir + rotate_dir) % n_motion_choice):
        test_dir = np.random.randint(n_motion_choice)

  # SAMPLE stimulus
  X[sample_time] += motion_tuning[sample_dir] * fr
  # TEST stimulus
  X[test_time] += motion_tuning[test_dir] * fr
  # rule fr
  X[rule_time] += (rule_tuning * rule_fr)
  # background fr
  X += bg_fr

  # to spiking
  if is_spiking_mode:
    X = np.random.random(X.shape) < X
    X = X.astype(np.float_)

  # can use a greater weight for test period if needed
  return X, match


class DMS_DMRS_DMC(bd.cognitive.CognitiveTask):
  times = ('dead', 'fixation', 'sample', 'delay', 'test')
  output_features = ('non-match', 'match')

  def __init__(
      self,
      dt=100., t_fixation=500., t_sample=500., t_delay=1000., t_test=500.,
      limits=(0., np.pi * 2), rotation_match='0',
      kappa=2, bg_fr=1., choices=('0', '90', '0'),
      ft_motion=bd.cognitive.Feature(24, 100, 40.),
      ft_rule=bd.cognitive.Feature(9, 30, 40.),
      num_trial=1024, mode='rate', seed=None,
  ):
    super().__init__(dt=dt, num_trial=num_trial, seed=seed)

    # time
    self.t_fixation = int(t_fixation / dt)
    self.t_sample = int(t_sample / dt)
    self.t_delay = int(t_delay / dt)
    self.t_test = int(t_test / dt)
    self.num_steps = self.t_fixation + self.t_sample + self.t_delay + self.t_test
    self._times = {
      'fixation': self.t_fixation,
      'sample': self.t_sample,
      'delay': self.t_delay,
      'test': self.t_test,
    }
    test_onset = self.t_fixation + self.t_sample + self.t_delay
    self.test_time = slice(test_onset, test_onset + self.t_test)
    self.sample_time = slice(self.t_fixation, self.t_fixation + self.t_sample)
    self.rule_time = slice(0, self.t_fixation + self.t_sample + self.t_delay + self.t_test)

    # input shape
    for c in choices:
      assert c in _rotate_choice
    self.choices = [_rotate_choice[c] for c in choices]
    self.features = ft_motion.set_name('motion') + ft_rule.set_name('rule')
    self.features.set_mode(mode)
    self.rotation_match = rotation_match
    self._rotate = _rotate_choice[rotation_match]
    self.bg_fr = bg_fr  # background firing rate
    self.v_min = limits[0]
    self.v_max = limits[1]
    self.v_range = limits[1] - limits[0]

    # Tuning function data
    self.n_motion_choice = 8
    self.kappa = kappa  # concentration scaling factor for von Mises

    # Generate list of preferred directions
    # dividing neurons by 2 since two equal
    # groups representing two modalities
    pref_dirs = np.arange(self.v_min, self.v_max, self.v_range / ft_motion.num)
    # Generate list of possible stimulus directions
    stim_dirs = np.arange(self.v_min, self.v_max, self.v_range / self.n_motion_choice)
    d = np.cos(np.expand_dims(stim_dirs, 1) - pref_dirs)
    motion_tuning = np.exp(self.kappa * d) / np.exp(self.kappa)
    self.motion_tuning = np.concatenate(
      [motion_tuning, np.zeros((self.n_motion_choice, self.features['rule'].num))],
      axis=1
    )

    # rule tuning
    self.num_rule = 3
    rule_tuning = np.zeros((self.num_rule, self.features.num))
    per_rule = self.features['rule'].num // self.num_rule
    for i in range(self.num_rule):
      s = self.features['motion'].num + per_rule * i
      e = self.features['motion'].num + per_rule * (i + 1)
      rule_tuning[i, s: e] = 1.
    self.rule_tuning = rule_tuning

  @property
  def num_inputs(self) -> int:
    return self.features.num

  @property
  def num_outputs(self) -> int:
    return 2

  def sample_a_trial(self, index):
    motion_fr = self.features['motion'].fr(self.dt)
    bg_fr = bd.cognitive.firing_rate(self.bg_fr, self.dt, self.features.mode)
    rule_fr = self.features['rule'].fr(self.dt)
    rule = self.rng.choice([0, 1, 2])
    return _dms_dmrs_dmc(self.num_steps, self.num_inputs, self.n_motion_choice,
                         self.motion_tuning, self.rule_tuning[rule],
                         self.features.is_spiking_mode,
                         self.sample_time, self.test_time, self.rule_time,
                         motion_fr, bg_fr, rule_fr, self.choices[rule], rule)


class _LocationDMS(bd.cognitive.CognitiveTask):
  times = ('dead', 'fixation', 'sample', 'delay', 'test')
  output_features = ('non-match', 'match')

  def __init__(
      self,
      dt=100., t_fixation=500., t_sample=500., t_delay=1000., t_test=500.,
      limits=(0., np.pi * 2), rotation_match='0', kappa=2,
      bg_fr=1., ft_motion=bd.cognitive.Feature(24, 100, 40.),
      num_trial=1024, mode='rate', seed=None,
  ):
    super().__init__(dt=dt, num_trial=num_trial, seed=seed)
    # time
    self.t_fixation = int(t_fixation / dt)
    self.t_sample = int(t_sample / dt)
    self.t_delay = int(t_delay / dt)
    self.t_test = int(t_test / dt)
    self.num_steps = self.t_fixation + self.t_sample + self.t_delay + self.t_test
    self._times = {
      'fixation': self.t_fixation,
      'sample': self.t_sample,
      'delay': self.t_delay,
      'test': self.t_test,
    }
    test_onset = self.t_fixation + self.t_sample + self.t_delay
    self.test_time = slice(test_onset, test_onset + self.t_test)
    self.fix_time = slice(0, test_onset)
    self.sample_time = slice(self.t_fixation, self.t_fixation + self.t_sample)

    # input shape
    self.features = ft_motion.set_name('motion')
    self.features.set_mode(mode)
    self.rotation_match = rotation_match
    self._rotate = _rotate_choice[rotation_match]
    self.bg_fr = bg_fr  # background firing rate
    self.v_min = limits[0]
    self.v_max = limits[1]
    self.v_range = limits[1] - limits[0]

    # Tuning function data
    self.n_motion_choice = 8
    self.kappa = kappa  # concentration scaling factor for von Mises

    # Generate list of preferred directions
    # dividing neurons by 2 since two equal
    # groups representing two modalities
    pref_dirs = np.arange(self.v_min, self.v_max, self.v_range / ft_motion.num)

    # Generate list of possible stimulus directions
    stim_dirs = np.arange(self.v_min, self.v_max, self.v_range / self.n_motion_choice)

    d = np.cos(np.expand_dims(stim_dirs, 1) - pref_dirs)
    self.motion_tuning = np.exp(self.kappa * d) / np.exp(self.kappa)

  @property
  def num_inputs(self) -> int:
    return self.features.num

  @property
  def num_outputs(self) -> int:
    return 2

  def sample_a_trial(self, index):
    fr = self.features.fr(self.dt)
    bg_fr = bd.cognitive.firing_rate(self.bg_fr, self.dt, self.features.mode)
    return _dms(self.num_steps, self.num_inputs, self.n_motion_choice,
                self.motion_tuning, self.features.is_spiking_mode,
                self.sample_time, self.test_time, fr, bg_fr, self._rotate)


class _DualDMS(bd.cognitive.CognitiveTask):
  times = ('dead', 'fixation', 'sample', 'delay', 'test')
  output_features = ('non-match', 'match')

  def __init__(
      self,
      dt=100., t_fixation=500., t_sample=500., t_delay=1000., t_test=500.,
      limits=(0., np.pi * 2), rotation_match='0', kappa=2,
      bg_fr=1., ft_motion=bd.cognitive.Feature(24, 100, 40.),
      num_trial=1024, mode='rate', seed=None,
  ):
    super().__init__(dt=dt, num_trial=num_trial, seed=seed)
    # time
    self.t_fixation = int(t_fixation / dt)
    self.t_sample = int(t_sample / dt)
    self.t_delay = int(t_delay / dt)
    self.t_test = int(t_test / dt)
    self.num_steps = self.t_fixation + self.t_sample + self.t_delay + self.t_test
    self._times = {
      'fixation': self.t_fixation,
      'sample': self.t_sample,
      'delay': self.t_delay,
      'test': self.t_test,
    }
    test_onset = self.t_fixation + self.t_sample + self.t_delay
    self.test_time = slice(test_onset, test_onset + self.t_test)
    self.fix_time = slice(0, test_onset)
    self.sample_time = slice(self.t_fixation, self.t_fixation + self.t_sample)

    # input shape
    self.features = ft_motion.set_name('motion')
    self.features.set_mode(mode)
    self.rotation_match = rotation_match
    self._rotate = _rotate_choice[rotation_match]
    self.bg_fr = bg_fr  # background firing rate
    self.v_min = limits[0]
    self.v_max = limits[1]
    self.v_range = limits[1] - limits[0]

    # Tuning function data
    self.n_motion_choice = 8
    self.kappa = kappa  # concentration scaling factor for von Mises

    # Generate list of preferred directions
    # dividing neurons by 2 since two equal
    # groups representing two modalities
    pref_dirs = np.arange(self.v_min, self.v_max, self.v_range / ft_motion.num)

    # Generate list of possible stimulus directions
    stim_dirs = np.arange(self.v_min, self.v_max, self.v_range / self.n_motion_choice)

    d = np.cos(np.expand_dims(stim_dirs, 1) - pref_dirs)
    self.motion_tuning = np.exp(self.kappa * d) / np.exp(self.kappa)

  @property
  def num_inputs(self) -> int:
    return self.features.num

  @property
  def num_outputs(self) -> int:
    return 2

  def sample_a_trial(self, index):
    fr = self.features.fr(self.dt)
    bg_fr = bd.cognitive.firing_rate(self.bg_fr, self.dt, self.features.mode)
    return _dms(self.num_steps, self.num_inputs, self.n_motion_choice,
                self.motion_tuning, self.features.is_spiking_mode,
                self.sample_time, self.test_time, fr, bg_fr, self._rotate)


class _ABBA(bd.cognitive.CognitiveTask):
  times = ('dead', 'fixation', 'sample', 'delay', 'test')
  output_features = ('non-match', 'match')

  def __init__(
      self,
      dt=100., t_fixation=500., t_sample=500., t_delay=1000., t_test=500.,
      limits=(0., np.pi * 2), rotation_match='0', kappa=2,
      bg_fr=1., ft_motion=bd.cognitive.Feature(24, 100, 40.),
      num_trial=1024, mode='rate', seed=None,
  ):
    super().__init__(dt=dt, num_trial=num_trial, seed=seed)
    # time
    self.t_fixation = int(t_fixation / dt)
    self.t_sample = int(t_sample / dt)
    self.t_delay = int(t_delay / dt)
    self.t_test = int(t_test / dt)
    self.num_steps = self.t_fixation + self.t_sample + self.t_delay + self.t_test
    self._times = {
      'fixation': self.t_fixation,
      'sample': self.t_sample,
      'delay': self.t_delay,
      'test': self.t_test,
    }
    test_onset = self.t_fixation + self.t_sample + self.t_delay
    self.test_time = slice(test_onset, test_onset + self.t_test)
    self.fix_time = slice(0, test_onset)
    self.sample_time = slice(self.t_fixation, self.t_fixation + self.t_sample)

    # input shape
    self.features = ft_motion.set_name('motion')
    self.features.set_mode(mode)
    self.rotation_match = rotation_match
    self._rotate = _rotate_choice[rotation_match]
    self.bg_fr = bg_fr  # background firing rate
    self.v_min = limits[0]
    self.v_max = limits[1]
    self.v_range = limits[1] - limits[0]

    # Tuning function data
    self.n_motion_choice = 8
    self.kappa = kappa  # concentration scaling factor for von Mises

    # Generate list of preferred directions
    # dividing neurons by 2 since two equal
    # groups representing two modalities
    pref_dirs = np.arange(self.v_min, self.v_max, self.v_range / ft_motion.num)

    # Generate list of possible stimulus directions
    stim_dirs = np.arange(self.v_min, self.v_max, self.v_range / self.n_motion_choice)

    d = np.cos(np.expand_dims(stim_dirs, 1) - pref_dirs)
    self.motion_tuning = np.exp(self.kappa * d) / np.exp(self.kappa)

  @property
  def num_inputs(self) -> int:
    return self.features.num

  @property
  def num_outputs(self) -> int:
    return 2

  def sample_a_trial(self, index):
    fr = self.features.fr(self.dt)
    bg_fr = bd.cognitive.firing_rate(self.bg_fr, self.dt, self.features.mode)
    return _dms(self.num_steps, self.num_inputs, self.n_motion_choice,
                self.motion_tuning, self.features.is_spiking_mode,
                self.sample_time, self.test_time, fr, bg_fr, self._rotate)


class _ABCA(bd.cognitive.CognitiveTask):
  times = ('dead', 'fixation', 'sample', 'delay', 'test')
  output_features = ('non-match', 'match')

  def __init__(
      self,
      dt=100., t_fixation=500., t_sample=500., t_delay=1000., t_test=500.,
      limits=(0., np.pi * 2), rotation_match='0', kappa=2,
      bg_fr=1., ft_motion=bd.cognitive.Feature(24, 100, 40.),
      num_trial=1024, mode='rate', seed=None,
  ):
    super().__init__(dt=dt, num_trial=num_trial, seed=seed)
    # time
    self.t_fixation = int(t_fixation / dt)
    self.t_sample = int(t_sample / dt)
    self.t_delay = int(t_delay / dt)
    self.t_test = int(t_test / dt)
    self.num_steps = self.t_fixation + self.t_sample + self.t_delay + self.t_test
    self._times = {
      'fixation': self.t_fixation,
      'sample': self.t_sample,
      'delay': self.t_delay,
      'test': self.t_test,
    }
    test_onset = self.t_fixation + self.t_sample + self.t_delay
    self.test_time = slice(test_onset, test_onset + self.t_test)
    self.fix_time = slice(0, test_onset)
    self.sample_time = slice(self.t_fixation, self.t_fixation + self.t_sample)

    # input shape
    self.features = ft_motion.set_name('motion')
    self.features.set_mode(mode)
    self.rotation_match = rotation_match
    self._rotate = _rotate_choice[rotation_match]
    self.bg_fr = bg_fr  # background firing rate
    self.v_min = limits[0]
    self.v_max = limits[1]
    self.v_range = limits[1] - limits[0]

    # Tuning function data
    self.n_motion_choice = 8
    self.kappa = kappa  # concentration scaling factor for von Mises

    # Generate list of preferred directions
    # dividing neurons by 2 since two equal
    # groups representing two modalities
    pref_dirs = np.arange(self.v_min, self.v_max, self.v_range / ft_motion.num)

    # Generate list of possible stimulus directions
    stim_dirs = np.arange(self.v_min, self.v_max, self.v_range / self.n_motion_choice)

    d = np.cos(np.expand_dims(stim_dirs, 1) - pref_dirs)
    self.motion_tuning = np.exp(self.kappa * d) / np.exp(self.kappa)

  @property
  def num_inputs(self) -> int:
    return self.features.num

  @property
  def num_outputs(self) -> int:
    return 2

  def sample_a_trial(self, index):
    fr = self.features.fr(self.dt)
    bg_fr = bd.cognitive.firing_rate(self.bg_fr, self.dt, self.features.mode)
    return _dms(self.num_steps, self.num_inputs, self.n_motion_choice,
                self.motion_tuning, self.features.is_spiking_mode,
                self.sample_time, self.test_time, fr, bg_fr, self._rotate)


def bptt(loader, trainer, fn: str = None, n_epoch: int = 30):
  """BPTT training with (loss, regularization, accuracy) output."""
  if fn is not None and os.path.exists(fn):
    states = bp.checkpoints.load_pytree(fn)
    hists = {k: v.tolist() for k, v in states['hists'].items()}
  else:
    hists = {}
  max_acc = max(hists['acc']) if ('acc' in hists) else 0.
  for epoch_i in range(n_epoch):
    metrics = dict()
    bar = tqdm.tqdm(total=len(loader))
    for x, y in loader:
      trainer.model.reset_state(x.shape[1])
      t0 = time.time()
      loss, aux = trainer.f_train(x, y)
      desc = []
      for k, v in aux.items():
        v = float(v)
        if k not in metrics:
          metrics[k] = []
        metrics[k].append(v)
        desc.append(f'{k} = {v:.6f}')
      bar.update()
      desc = ', '.join(desc)
      bar.set_description(f'{desc}, time = {time.time() - t0:.5f} s', refresh=True)
    bar.close()
    desc = []
    for k, v in metrics.items():
      v = np.mean(v).item()
      desc.append(f'{k} = {v:.6f}')
      if k not in hists:
        hists[k] = []
      hists[k].append(v)
    desc = ', '.join(desc)
    print(f'Epoch {epoch_i}, {desc}')
    if fn is not None and max_acc < hists['acc'][-1]:
      max_acc = hists['acc'][-1]
      states = {
        'net': trainer.model.state_dict(),
        'opt': trainer.f_opt.state_dict(),
        'hists': hists
      }
      bp.checkpoints.save_pytree(fn, states, overwrite=True)


def verify_lif(net, ds, fn=None, num_show=5, sps_inc=10.):
  if fn is not None:
    states = bp.checkpoints.load_pytree(fn)
    net.load_state_dict(states['net'])

  # looping the model over time
  outs = (net.r.spike, net.r.V)
  if hasattr(net.r2r, 'g'):
    outs += (net.r2r.g,)
  looper = bp.LoopOverTime(net, out_vars=outs)

  for i in range(num_show):
    fig, gs = bp.visualize.get_figure(4 + int(hasattr(net.r2r, 'g')), 1, 2., 10.)

    x, _ = ds[i]
    ts = np.arange(0, x.shape[0]) * ds.dt
    max_t = x.shape[0] * ds.dt

    # insert empty row
    ax_inp = fig.add_subplot(gs[0, 0])
    indices, times = bp.measure.raster_plot(x, ts)
    ax_inp.plot(times, indices, '.')
    # bp.visualize.raster_plot(ts, x, xlim=(0., max_t), ax=ax_inp)
    ax_inp.set_xlim(0., max_t)
    ax_inp.set_ylabel('Input Activity')

    looper.reset_state(1)
    readout, rrr = looper(np.expand_dims(x, 1))
    spikes, mems = rrr[:2]

    ax = fig.add_subplot(gs[1, 0])
    mems = bm.as_numpy(bm.where(spikes, mems + sps_inc, mems))
    for i in range(0, net.r.num, 10):
      plt.plot(ts, mems[:, 0, i])
    ax.set_xlim(0., max_t)
    ax.set_ylabel('Recurrent Potential')

    # spiking activity
    ax_rec = fig.add_subplot(gs[2, 0])
    indices, times = bp.measure.raster_plot(spikes[:, 0], ts)
    ax_rec.plot(times, indices, '.')
    ax_rec.set_xlim(0., max_t)
    ax_rec.set_ylabel('Recurrent Spiking')

    # decision activity
    ax_out = fig.add_subplot(gs[3, 0])
    for i in range(readout.shape[-1]):
      ax_out.plot(ts, readout[:, 0, i], label=f'Readout {i}', alpha=0.7)
    ax_out.set_ylabel('Output Activity')
    ax_out.set_xlabel('Time [ms]')
    ax_out.set_xlim(0., max_t)
    plt.legend()

    if hasattr(net.r2r, 'g'):
      conds = rrr[2]
      ax = fig.add_subplot(gs[4, 0])
      for i in range(0, net.r.num, 10):
        plt.plot(ts, conds[:, 0, i])
      ax.set_xlim(0., max_t)
      ax.set_ylabel('Synapse Conductance')

    plt.show()
