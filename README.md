# Parallelization of CNN - Alexnet: CPU and GPU


Make sure that the data is downloaded and is in the /data directory.

For the data to be downloaded
```
bash get_mnist.sh
```

## OpenMP
The following files correspond to our OpenMP implementation of the network.

```
make
./neural_net_in_cpp data/
```

## CUDA
The following files correspond to our CUDA implementation of the network. For running compiling the code the command is as follows:

```
nvcc -x cu src/vgg.cpp src/NetworkModel.cpp src/FullyConnected.cpp src/Sigmoid.cpp src/Dropout.cpp src/SoftmaxClassifier.cpp src/MNISTDataLoader.cpp src/ReLU.cpp src/Tensor.cpp src/Conv2d.cpp src/MaxPool.cpp src/LinearLRScheduler.cpp -I../include -o vgg.x -arch=sm_70 -std=c++11

./vgg.x data/
```

License
----

MIT

