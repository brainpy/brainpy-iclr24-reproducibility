# Code for the paper "BrainPy: a differentiable brain simulator bridging brain simulation and brain-inspired computing"

## Requirements
The Python version is 3.10, most of the dependencies can be installed by running:
```
pip install -r requirements.txt
```
If using GPU, please install jax[cuda] version in advance according to the JAX official guide.

## Usage
All the files are organized in the following structure:
```
Submit_code
├── README.md
├── requirements.txt
├── EI_balanced_network
│   ├── COBAHH
│   │   ├── brainpy_COBAHH.py
│   │   ├── brian2_COBAHH.py
│   │   ├── COBAHH_nest
│   │   ├── COBAHH_neuron
│   ├── COBAlif
│   │   ├── brainpy_COBAlif.py
│   │   ├── NEURON_COBAlif.py
│   │   ├── nest_COBAlif.py
│   │   ├── bindsnet_COBAlif.py
│   │   ├── brian2_COBAlif.py
│   │   ├── ANNarchy_COBAlif.py
│   │   ├── pytorch_COBAlif.py
├── operators_comparison
│   ├── operators_comparison.py
├── SNN_training
│   ├── BrainPy_fashion_mnist.py
│   ├── Norse_fashion_mnist.py
│   ├── snnTorch_fashion_mnist.py
│   ├── SpikingJelly_fashion_mnist.py
│   ├── brainpy_VGG-SNN.py
├── working_memory
│   ├── utils.py
│   ├── spiking_version.py
│   ├── rate_version.py
├── jitop_comparison
│   ├── jit_conn_comparison.py
├── reservoir_computing
│   ├── reservoir.py
│   ├── kth-reservoir-force-training.py
│   ├── mnist-reservoir-force-training.py
│   ├── run-KTH.sh
│   ├── run-mnist.sh
```

### Section 1.1: EI balanced network
EI balance network with LIF can be simulated by running:
```
python brainpy_COBAlif.py
```
EI balance network with HH can be simulated by running:
```
python brainpy_COBAHH.py
```
Other frameworks can be simulated by running the corresponding files.

### Section 1.2: Operators comparison
The operators comparison can be simulated by running:
```
python operators_comparison.py
```

### Section 2: SNN training
The simple SNN can be trained by running:
```
python BrainPy_fashion_mnist.py
python Norse_fashion_mnist.py
python snnTorch_fashion_mnist.py
python SpikingJelly_fashion_mnist.py
```
Different frameworks have different training speed.

The VGG SNN can be trained by running:
```
python brainpy_VGG-SNN.py
```

### Section 3: Working memory
The spike version of working memory can be simulated by running:
```
python spiking_version.py
```
The rate version of working memory can be simulated by running:
```
python rate_version.py
```

### Section 4.1: JIT operators comparison
The JIT operators comparison can be simulated by running:
```
python jit_conn_comparison.py
```

### Section 4.2: Reservoir computing
The KTH dataset can be run by:
```
bash run-KTH.sh
```
Please notice that the KTH dataset is not included in this repository, please download it from the internet.

The MNIST dataset can be run by:
```
bash run-mnist.sh
```
