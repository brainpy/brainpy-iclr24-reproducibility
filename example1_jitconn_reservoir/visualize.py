
import brainpy as bp
import matplotlib.pyplot as plt


def reservoir_mnist():
  sizes = [2000,   4000,   8000,   10000,  20000,  30000,  40000,  50000]
  accs =  [0.9602, 0.9724, 0.9809, 0.9817, 0.9863, 0.9881, 0.9888, 0.9891]

  fig, gs = bp.visualize.get_figure(1, 1, 4.5, 6.0)
  ax = fig.add_subplot(gs[0, 0])
  plt.plot(sizes, accs)
  plt.ylabel('Accuracy')
  plt.xlabel('Reservoir size')
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  plt.show()


def reservoir_kth():
  '''
python kth-reservoir-force-training.py -num_hidden 2000 -win_connectivity 0.01 -wrec_connectivity 0.001 -train_start 10

python kth-reservoir-force-training.py -num_hidden 4000 -win_connectivity 0.01 -wrec_connectivity 0.001 -train_start 10
   0.9333

python kth-reservoir-force-training.py -num_hidden 8000 -win_connectivity 0.01 -wrec_connectivity 0.001 -train_start 10

python kth-reservoir-force-training.py -num_hidden 10000 -win_connectivity 0.01 -wrec_connectivity 0.0002 -train_start 10

python kth-reservoir-force-training.py -num_hidden 20000 -win_connectivity 0.01 -wrec_connectivity 0.0001 -train_start 10

python kth-reservoir-force-training.py -num_hidden 30000 -win_connectivity 0.005 -wrec_connectivity 0.0001 -train_start 10
  0.94

  '''

  sizes = [2000,   4000,   8000,   10000,  20000,  30000,  ]
  accs =  [0.8867, 0.9000, 0.9067, 0.9133, 0.9333, 0.9400, ]

  fig, gs = bp.visualize.get_figure(1, 1, 4.5, 6.0)
  ax = fig.add_subplot(gs[0, 0])
  plt.plot(sizes, accs)
  plt.ylabel('Accuracy')
  plt.xlabel('Reservoir size')
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  plt.show()


if __name__ == '__main__':
  reservoir_mnist()
  # reservoir_kth()



