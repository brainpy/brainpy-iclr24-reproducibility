import tensorflow as tf
import numpy as np
from python_speech_features import mfcc, fbank, delta
from sklearn.preprocessing import StandardScaler
import scipy.io.wavfile as wav
import os
import time
import pickle

phn_61 = ['aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay', 'b', 'bcl', 'ch', 'd', 'dcl', 'dh',
          'dx', 'eh', 'el', 'em', 'en', 'eng', 'epi', 'er', 'ey', 'f', 'g', 'gcl', 'h#', 'hh', 'hv',
          'ih', 'ix', 'iy', 'jh', 'k', 'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow', 'oy', 'p', 'pau', 'pcl',
          'q', 'r', 's', 'sh', 't', 'tcl', 'th', 'uh', 'uw', 'ux', 'v', 'w', 'y', 'z', 'zh']

mapping = {'ah': 'ax', 'ax-h': 'ax', 'ux': 'uw', 'aa': 'ao', 'ih': 'ix',
               'axr': 'er', 'el': 'l', 'em': 'm', 'en': 'n', 'nx': 'n',
               'eng': 'ng', 'sh': 'zh', 'hv': 'hh', 'bcl': 'h#', 'pcl': 'h#',
               'dcl': 'h#', 'tcl': 'h#', 'gcl': 'h#', 'kcl': 'h#',
               'q': 'h#', 'epi': 'h#', 'pau': 'h#'}

phn_39 = ['ae', 'ao', 'aw', 'ax', 'ay', 'b', 'ch', 'd', 'dh', 'dx', 'eh',
             'er', 'ey', 'f', 'g', 'h#', 'hh', 'ix', 'iy', 'jh', 'k', 'l',
             'm', 'n', 'ng', 'ow', 'oy', 'p', 'r', 's', 't', 'th', 'uh', 'uw',
             'v', 'w', 'y', 'z', 'zh']

development_set_lower = ['faks0', 'mmdb1', 'mbdg0', 'fedw0', 'mtdt0', 'fsem0', 'mdvc0', 'mrjm4', 'mjsw0', 'mteb0',
                  'fdac1', 'mmdm2', 'mbwm0', 'mgjf0', 'mthc0', 'mbns0', 'mers0', 'fcal1', 'mreb0', 'mjfc0',
                  'fjem0', 'mpdf0', 'mcsh0', 'mglb0', 'mwjg0', 'mmjr0', 'fmah0', 'mmwh0', 'fgjd0', 'mrjr0',
                  'mgwt0', 'fcmh0', 'fadg0', 'mrtk0', 'fnmr0', 'mdls0', 'fdrw0', 'fjsj0', 'fjmg0', 'fmml0',
                  'mjar0', 'fkms0', 'fdms0', 'mtaa0', 'frew0', 'mdlf0', 'mrcs0', 'majc0', 'mroa0', 'mrws1']

development_set = [name.upper() for name in development_set_lower]

core_test_set_lower = ['mdab0', 'mwbt0', 'felc0', 'mtas1', 'mwew0', 'fpas0', 'mjmp0', 'mlnt0', 'fpkt0',
             'mlll0', 'mtls0', 'fjlm0', 'mbpm0', 'mklt0', 'fnlp0', 'mcmj0', 'mjdh0', 'fmgd0',
            'mgrt0', 'mnjm0', 'fdhc0', 'mjln0', 'mpam0', 'fmld0']

core_test_set = [name.upper() for name in core_test_set_lower]

TIMIT_DIR = '/home/brainpy/ztq/Data' # root directory for timit, it would be joined with timit/train or timit/test
TFRECORD_DIR = './data' # directory for tfrecords files


def prepare_timit_dataset(train_set=True, dev_set=True, test_set=True, feats_type='mfcc'):
  '''
  feats_type:
  - mfcc: 13 mel frequency cepstral coefficients + delta + delta delta, total 39 dimension
  '''

  def create_tfrecords(tfrecord_path, root_dir, fname, filter_fn):
    writer = tf.compat.v1.python_io.TFRecordWriter(os.path.join(tfrecord_path, (fname + '.tfrecords')))
    feats_list = []
    phoneme_list = []
    start = time.time()
    cnt = 0
    for path, dirs, files in os.walk(root_dir):
      for file in files:
        if filter_fn(file, path):
          continue
        if file.endswith('wav'):
          fullFileName = os.path.join(path, file)
          fnameNoSuffix = os.path.splitext(fullFileName)[0]
          rate, sig = wav.read(fullFileName)

          if feats_type == 'mfcc':
            mfcc_feat = mfcc(sig, rate)
            mfcc_feat_delta = delta(mfcc_feat, 2)
            mfcc_feat_delta_delta = delta(mfcc_feat_delta, 2)
            feats = np.concatenate((mfcc_feat, mfcc_feat_delta, mfcc_feat_delta_delta), axis=1)
          else:  # fbank
            filters, energy = fbank(sig, rate, nfilt=40)
            log_filters, log_energy = np.log(filters), np.log(energy)
            logfbank_feat = np.concatenate((log_filters, log_energy.reshape(-1, 1)), axis=1)
            logfbank_feat_delta = delta(logfbank_feat, 2)
            logfbank_feat_delta_delta = delta(logfbank_feat_delta, 2)
            feats = np.concatenate((logfbank_feat, logfbank_feat_delta, logfbank_feat_delta_delta), axis=1)
          feats_list.append(feats)

          # .phn
          phoneme = []
          with open(fnameNoSuffix + '.PHN', 'r') as f:
            for line in f.read().splitlines():
              phn = line.split(' ')[2]
              p_index = phn_61.index(phn)
              phoneme.append(p_index)
          phoneme_list.append(phoneme)

          cnt += 1

    if fname == 'train':
      scaler = StandardScaler()
      scaler.fit(np.concatenate(feats_list, axis=0))
      print('scaler.n_samples_seen_:', scaler.n_samples_seen_)
      pickle.dump(scaler, open(os.path.join(tfrecord_path, 'scaler.pkl'), 'wb'))

    if not os.path.exists(os.path.join(tfrecord_path, 'scaler.pkl')):
      raise Exception('scaler.pkl not exist, call with [train_set=True]')
    else:
      scaler = pickle.load(open(os.path.join(tfrecord_path, 'scaler.pkl'), 'rb'))

    for feats, phoneme in zip(feats_list, phoneme_list):
      seq_exam = tf.train.SequenceExample()
      seq_exam.context.feature['feats_dim'].int64_list.value.append(feats.shape[1])
      seq_exam.context.feature['feats_seq_len'].int64_list.value.append(feats.shape[0])
      seq_exam.context.feature['labels_seq_len'].int64_list.value.append(len(phoneme))

      feats = scaler.transform(feats)
      for feat in feats:
        seq_exam.feature_lists.feature_list['features'].feature.add().float_list.value[:] = feat
      for p in phoneme:
        seq_exam.feature_lists.feature_list['labels'].feature.add().int64_list.value.append(p)
      writer.write(seq_exam.SerializeToString())

    writer.close()
    print('{} created: {} utterances - {:.0f}s'.format(fname + '.tfrecords', cnt, (time.time() - start)))

  # end create_tfrecords() definition

  def create_records(tfrecord_path, root_dir, fname, filter_fn):
    feats_list = []
    phoneme_list = []
    start = time.time()
    cnt = 0
    for path, dirs, files in os.walk(root_dir):
      for file in files:
        if filter_fn(file, path):
          continue
        if file.endswith('wav'):
          fullFileName = os.path.join(path, file)
          fnameNoSuffix = os.path.splitext(fullFileName)[0]
          rate, sig = wav.read(fullFileName)

          if feats_type == 'mfcc':
            mfcc_feat = mfcc(sig, rate)
            mfcc_feat_delta = delta(mfcc_feat, 2)
            mfcc_feat_delta_delta = delta(mfcc_feat_delta, 2)
            feats = np.concatenate((mfcc_feat, mfcc_feat_delta, mfcc_feat_delta_delta), axis=1)
          else:  # fbank
            filters, energy = fbank(sig, rate, nfilt=40)
            log_filters, log_energy = np.log(filters), np.log(energy)
            logfbank_feat = np.concatenate((log_filters, log_energy.reshape(-1, 1)), axis=1)
            logfbank_feat_delta = delta(logfbank_feat, 2)
            logfbank_feat_delta_delta = delta(logfbank_feat_delta, 2)
            feats = np.concatenate((logfbank_feat, logfbank_feat_delta, logfbank_feat_delta_delta), axis=1)
          feats_list.append(feats)

          # frame_list = np.linspace(0, (feats.shape[1] - 2) / 100, feats.shape[1] - 1)
          # .phn
          phoneme = []
          with open(fnameNoSuffix + '.PHN', 'r') as f:
            for line in f.read().splitlines():
              phn = line.split(' ')[2]
              # if float(line.split(' ')[0]) <= frame_list[time_cnt] * 16000.0 <= float(line.split(' ')[1]):
              p_index = phn_61.index(phn)
              phoneme.append([line.split(' ')[0:2], p_index])
          phoneme_list.append(phoneme)

          cnt += 1

    if fname == 'train':
      scaler = StandardScaler()
      scaler.fit(np.concatenate(feats_list, axis=0))
      print('scaler.n_samples_seen_:', scaler.n_samples_seen_)
      pickle.dump(scaler, open(os.path.join(tfrecord_path, 'scaler.pkl'), 'wb'))

    if not os.path.exists(os.path.join(tfrecord_path, 'scaler.pkl')):
      raise Exception('scaler.pkl not exist, call with [train_set=True]')
    else:
      scaler = pickle.load(open(os.path.join(tfrecord_path, 'scaler.pkl'), 'rb'))

    records = {}
    records['feats_dim'] = []
    records['feats_seq_len'] = []
    records['labels_seq_len'] = []
    records['features'] = []
    records['labels'] = []

    for feats, phoneme in zip(feats_list, phoneme_list):
      feats = scaler.transform(feats)
      records['feats_dim'].append(feats.shape[1])
      records['feats_seq_len'].append(feats.shape[0])
      records['labels_seq_len'].append(len(phoneme))
      records['features'].append(feats)
      records['labels'].append(phoneme)

    pickle.dump(records, open(os.path.join(tfrecord_path, (fname + '_records.pkl')), 'wb'))
    print('{} created: {} utterances - {:.0f}s'.format(fname + '.tfrecords', cnt, (time.time() - start)))

  tfrecord_path = os.path.join(TFRECORD_DIR, feats_type)
  if not os.path.isdir(tfrecord_path):
    os.makedirs(tfrecord_path)

  if train_set:
    create_records(tfrecord_path, os.path.join(TIMIT_DIR, 'TIMIT/TRAIN'), 'train',
                     lambda file, _: file.startswith('SA'))
  if dev_set:
    create_records(tfrecord_path, os.path.join(TIMIT_DIR, 'TIMIT/TEST'), 'dev',
                     lambda file, path: file.startswith('SA') or os.path.split(path)[1] not in development_set)
  if test_set:
    create_records(tfrecord_path, os.path.join(TIMIT_DIR, 'TIMIT/TEST'), 'test',
                     lambda file, path: file.startswith('SA') or os.path.split(path)[1] not in core_test_set)


def find_phn_index(start_time, phone_times):
  # given a frame start time, a list of phone times for this audio file, and a list of phones
  # returns the list of the phone corresponding to this frame in the list
  sample_rate = 16000.0

  # phone_times: a list of [[start time, end time], phone index] per wav
  for i in range(len(phone_times)):
     # phone_times[i][0][0]: start time
     # phone_times[i][0][1]: end time
     # phone_times[i][1]: phone index
    # print(phone_times[i][0][0], float(start_time)*sample_rate, phone_times[i][0][1])

    if float(phone_times[i][0][0]) <= float(start_time) * sample_rate <= float(phone_times[i][0][1]):
      return phone_times[i][1]

  # if nothing matches, return silence
  # this happens when the start time is beyond the last frame
  # see timit doc
  return phn_61.index("h#")


def preprocessing_input(records):
  # parameters:
  a_logE = 0.27
  b_logE = 1.75
  a_delta_logE = 1.77
  b_delta_logE = 1.25
  a_acc_logE = 4.97
  b_acc_logE = 1.00
  a_c1_12 = 0.10
  b_c1_12 = 1.25
  a_delta_c1_12 = 0.61
  b_delta_c1_12 = 0.50
  a_acc_c1_12 = 1.75
  b_acc_c1_12 = 0.25

  scaling_data = []
  target_labels = []
  phone = records['labels']
  cnt = 0
  for feature_per_wav in records['features']:
    means = np.mean(feature_per_wav, axis=1, keepdims=True)
    regulate_feature = feature_per_wav - means
    regulate_feature[:, 0] = regulate_feature[:, 0] * a_logE * b_logE
    regulate_feature[:, 1:13] = regulate_feature[:, 1:13] * a_c1_12 * b_c1_12
    regulate_feature[:, 13] = regulate_feature[:, 13] * a_delta_logE * b_delta_logE
    regulate_feature[:, 14:26] = regulate_feature[:, 14:26] * a_delta_c1_12 * b_delta_c1_12
    regulate_feature[:, 26] = regulate_feature[:, 26] * a_acc_logE * b_acc_logE
    regulate_feature[:, 27:] = regulate_feature[:, 27:] * a_acc_c1_12 * b_acc_c1_12
    scaling_data.append(regulate_feature)

    frame_list = np.linspace(0, (regulate_feature.shape[0] - 1) / 100, regulate_feature.shape[0])
    target_label = []
    for i in range(len(frame_list)):
      start_time = frame_list[i]

      # find correspond phoneme and index in the list
      phn_index = find_phn_index(start_time, phone[cnt])

      # create target of right size
      target = [0 for x in range(len(phn_61))]
      target[phn_index] = 1
      target_label.append(target)

    target_labels.append(np.asarray(target_label))
    cnt += 1
  return scaling_data, target_labels


if __name__ == '__main__':
  # first step
  # prepare_timit_dataset(feats_type='mfcc')

  # second step

  # DIR route
  TFRECORD_DIR = './data/mfcc'
  train_records = pickle.load(open(os.path.join(TFRECORD_DIR, 'train_records.pkl'), 'rb'))
  dev_records = pickle.load(open(os.path.join(TFRECORD_DIR, 'dev_records.pkl'), 'rb'))
  test_records = pickle.load(open(os.path.join(TFRECORD_DIR, 'test_records.pkl'), 'rb'))

  train_data, train_labels = preprocessing_input(train_records)
  dev_data, dev_labels = preprocessing_input(dev_records)
  test_data, test_labels = preprocessing_input(test_records)

  pickle.dump(train_data, open(os.path.join(TFRECORD_DIR, 'train_data.pkl'), 'wb'))
  pickle.dump(train_labels, open(os.path.join(TFRECORD_DIR, 'train_labels.pkl'), 'wb'))
  pickle.dump(dev_data, open(os.path.join(TFRECORD_DIR, 'dev_data.pkl'), 'wb'))
  pickle.dump(dev_labels, open(os.path.join(TFRECORD_DIR, 'dev_labels.pkl'), 'wb'))
  pickle.dump(test_data, open(os.path.join(TFRECORD_DIR, 'test_data.pkl'), 'wb'))
  pickle.dump(test_labels, open(os.path.join(TFRECORD_DIR, 'test_labels.pkl'), 'wb'))



# from sphfile import SPHFile
# import glob
#
# if __name__ == "__main__":
#     path = r'/home/brainpy/ztq/Data/TIMIT/TRAIN/*/*/*.WAV'
#     sph_files = glob.glob(path)
#     print(len(sph_files), "train utterences")
#     for i in sph_files:
#         print(i)
#         sph = SPHFile(i)
#         filename = i.replace(".WAV", ".wav")
#         sph.write_wav(filename)
#
#     path = r'/home/brainpy/ztq/Data/TIMIT/TEST/*/*/*.WAV'
#     sph_files_test = glob.glob(path)
#     print(len(sph_files_test), "test utterences")
#     for i in sph_files_test:
#         sph = SPHFile(i)
#         sph.write_wav(filename=i.replace(".WAV", ".wav"))
#
#     print("Completed")

