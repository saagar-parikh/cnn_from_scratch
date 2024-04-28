# Neural Network in Pure C++

Simple modular implementation of a neural network in C++ using only the STL. 

### Installation
Get the MNIST data set:

```sh
bash get_mnist.sh
```
sudo apt install make
sudo apt-get update && sudo apt-get install build-essential
Generate your Makefile:
```sh
cmake -DCMAKE_BUILD_TYPE=Release
```
Make the code:
```sh
make
```
Run:
```sh
./neural_net_in_cpp data
```
The training should take about a minute and achieve ~97% accuracy.

### Todos
 - [x] Fully connected;
 - [x] Sigmoid;
 - [x] Dropout;
 - [x] ReLU;
 - [ ] Tanh;
 - [ ] Leaky ReLU;
 - [ ] Batch normalization;
 - [x] Convolutional layers;
 - [x] Max pooling;
 - [ ] Other optimizers (Adam, RMSProp, etc);
 - [x] Learning rate scheduler;
 - [ ] Plots;
 - [ ] Filter visualization
 - [ ] CUDA?

License
----

MIT

nvcc -x cu src/main.cpp src/NetworkModel.cpp src/FullyConnected.cpp src/Sigmoid.cpp src/Dropout.cpp src/SoftmaxClassifier.cpp src/MNISTDataLoader.cpp src/ReLU.cpp src/Tensor.cpp src/Conv2d.cpp src/MaxPool.cpp src/LinearLRScheduler.cpp -I../include -o main.x -arch=sm_70 -std=c++11

TODO:
Check Flatten
Change the input_size, fc_layer_initialization_shape, fc_layers