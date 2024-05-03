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
./vgg data
```

## CUDA
The following files correspond to our CUDA implementation of the network.
```
./cuda/src/alexnet.cpp
```
We run the following command to compile 
```nvcc -x cu src/alexnet.cpp src/NetworkModel.cpp src/FullyConnected.cpp src/Sigmoid.cpp src/Dropout.cpp src/SoftmaxClassifier.cpp src/MNISTDataLoader.cpp src/ReLU.cpp src/Tensor.cpp src/Conv2d.cpp src/MaxPool.cpp src/LinearLRScheduler.cpp -I../include -o alexnet.x -arch=sm_70 -std=c++11```

License
----

MIT

