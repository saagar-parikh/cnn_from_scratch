#include <iostream>
#include "../include/NetworkModel.h"
#include "../include/Module.h"
#include "../include/FullyConnected.h"
#include "../include/Sigmoid.h"
#include "../include/Dropout.h"
#include "../include/SoftmaxClassifier.h"
#include "../include/MNISTDataLoader.h"
#include "../include/ReLU.h"
#include "../include/Tensor.h"
#include "../include/Conv2d.h"
#include "../include/MaxPool.h"
#include "../include/LinearLRScheduler.h"
#include <chrono>
#include <typeinfo>
#include <omp.h>
using namespace std::chrono;
using namespace std;

/*
 * Train a neural network on the MNIST data set and evaluate its performance
 */
__global__ void matmul_kernel(float *mat1, float *mat2, float *output, int dim_1, int dim_2, int dim_3)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < dim_1 && col < dim_3)
    {
        float sum = 0;
        for (int k = 0; k < dim_2; k++)
        {
            sum += mat1[row * dim_2 + k] * mat2[k * dim_3 + col];
        }
        output[row * dim_3 + col] = sum;
    }
    __syncthreads();
}

vector<Tensor<double>> fc_init_weights(int inp_size, int out_size, int seed){
    std::default_random_engine generator(seed);
    std::normal_distribution<double> distribution(0.0, 1.0);
    int weights_dims[] = {inp_size, out_size};
    Tensor<double> weights = Tensor<double>(2, weights_dims);
    weights.randn(generator, distribution, sqrt(2.0 / inp_size));
    int bias_dims[] = {out_size};
    Tensor<double> bias = Tensor<double>(1, bias_dims);
    bias.randn(generator, distribution, 0);
    return {weights, bias};
}

vector<Tensor<double>> conv_init_weights(int in_channels, int out_channels, int kernel_size, int stride, int padding, int seed) {
    std::default_random_engine generator(seed);
    std::normal_distribution<double> distribution(0.0, 1.0);

    int kernel_dims[] = {out_channels, in_channels, kernel_size, kernel_size};
    Tensor<double> kernels = Tensor<double>(4, kernel_dims);
    kernels.randn(generator, distribution, sqrt(2.0 / (kernel_size * kernel_size * out_channels)));

    int bias_dims[] = {out_channels};
    Tensor<double> bias = Tensor<double>(1, bias_dims);
    bias.randn(generator, distribution, 0);

    return {kernels,bias};
}


void fc_forward(float*& mat1, float*& mat2, float*& out, Tensor<double> input, Tensor<double> weights){
        int input_num_dims = input.num_dims;
        int input_dims[4];
        std::copy(input.dims, input.dims + input.num_dims, input_dims);
        if (input.num_dims != 2)
        {
            // flatten tensor
            int flatten_size = 1;
            for (int i = 1; i < input.num_dims; ++i)
            {
                flatten_size *= input.dims[i];
            }
            int dims[] = {input.dims[0], flatten_size};
            input.view(2, dims);
        }
        Tensor<double> input_ = input;

        assert(input.dims[1] == weights.dims[0]);

        cudaError_t err = cudaMalloc(&mat1, input.dims[0] * input.dims[1] * sizeof(float));
        if (err != cudaSuccess) 
        {
            cout << "CUDA malloc failed: " << cudaGetErrorString(err)
                << " [Requested dims: " << input.dims[0] << " x " << input.dims[1]
                << ", Size: " << (input.dims[0] * input.dims[1] * sizeof(float)) << " bytes]" << endl;
            exit(-1);
        }

        err = cudaMalloc(&mat2, weights.dims[0] * weights.dims[1] * sizeof(float));
        if (err != cudaSuccess)
        {
            cout << "Dev Memory not allocated2" << endl;
            exit(-1);
        }

        err = cudaMalloc(&out, input.dims[0] * weights.dims[1] * sizeof(float));
        if (err != cudaSuccess)
        {
            cout << "Dev Memory not allocated3" << endl;
            exit(-1);
        }

        cudaMemcpy(mat1,
                   input.data_,
                   input.dims[0] * input.dims[1] * sizeof(float),
                   cudaMemcpyHostToDevice);

        cudaMemcpy(mat2,
                   weights.data_,
                   weights.dims[0] * weights.dims[1] * sizeof(float),
                   cudaMemcpyHostToDevice);
}

int input_size = 224 * 224;
int output_size = 2;
    
// template<typename T>
int main(int argc, char **argv)
{
    if (argc < 2)
    {
        throw runtime_error("Please provide the data directory path as an argument");
    }
    printf("Data directory: %s\n", argv[1]);
    string data_path = argv[1];

    // in_channels, out_channels, kernel_size, stride, padding, seed

    int seed = 0;
    vector<Module *> modules = {new FullyConnected(224 * 224, 512, seed), new ReLU(), new FullyConnected(512, 2, seed)};
    // , 

    auto lr_sched = new LinearLRScheduler(0.2, -0.000005);
    NetworkModel model = NetworkModel(modules, new SoftmaxClassifier(), lr_sched);
    //    model.load("network.txt");
#if defined(_OPENMP)
    printf("Using OpenMP\n");
#endif

    printf("Loading testing set... ");
    fflush(stdout);
    MNISTDataLoader test_loader(data_path + "/mnist_test_images.ubyte", data_path + "/mnist_test_labels.ubyte", 32);

    // MNISTDataLoader test_loader(data_path + "/t10k-images-idx3-ubyte", data_path + "/t10k-labels-idx1-ubyte", 32);
    // model.load("network.txt");
    printf("Loaded.\n");

    model.eval();

    // Test and measure accuracy
    int hits = 0;
    int total = 0;
    printf("Testing...\n");
    int num_test_batches = test_loader.getNumBatches();

    // initialize fc layer
    vector<Tensor<double>> fc1_weights_and_biases = fc_init_weights(input_size, 512, 0);
    vector<Tensor<double>> fc2_weights_and_biases = fc_init_weights(512, output_size, 0);

    Tensor<double> fc2_weights = fc2_weights_and_biases[0];
    Tensor<double> fc2_bias = fc2_weights_and_biases[1];

    
    for (int i = 0; i < num_test_batches; ++i)
    {
        if ((i + 1) % 10 == 0 || i == (num_test_batches - 1))
        {
            printf("\rIteration %d/%d", i + 1, num_test_batches);
            fflush(stdout);
        }
        pair<Tensor<double>, vector<int>> xy = test_loader.nextBatch();

        ////////////////////////////////////////
        // FC layer 1
        auto &module = model.modules_[0];

        auto input = xy.first;

        Tensor<double> fc1_weights = fc1_weights_and_biases[0];
        Tensor<double> fc1_bias = fc1_weights_and_biases[1];
        float *mat1, *mat2, *out;
        int new_dims[] = {input.dims[0], fc1_weights.dims[1]};
        Tensor<double> product(2, new_dims);

        fc_forward(mat1, mat2, out, input, fc1_weights);

        dim3 dimBlock(16, 16);
        dim3 dimGrid(2, 2);
        cudaEvent_t st2, et2;
        cudaEventCreate(&st2);
        cudaEventCreate(&et2);

        cudaEventRecord(st2);
        matmul_kernel<<<dimGrid, dimBlock>>>(mat1, mat2, out, int(input.dims[0]), int(input.dims[1]), int(fc1_weights.dims[1]));
        cudaEventRecord(et2);

        // host waits until et2 has occured
        cudaEventSynchronize(et2);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, st2, et2);

        cout << "Kernel time: " << milliseconds << "ms" << endl;
        cudaMemcpy(product.data_,
                   out,
                   input.dims[0] * fc1_weights.dims[1] * sizeof(float),
                   cudaMemcpyDeviceToHost);

        cudaFree(mat1);
        cudaFree(mat2);
        cudaFree(out);

        //// END matmul kernel ////////////////////////////////
        Tensor<double> output = product + fc1_bias;

        //// END FC1 //////////////////////////////////////////
        //// RELU
        auto &module1 = model.modules_[1];
        Tensor<double> &output1 = module1->forward(output);        

        ///// END RELU /////////////////////////////////////////
        // FC layer 2
        auto &module2 = model.modules_[2];
        Tensor<double> fc2_weights = fc2_weights_and_biases[0];
        Tensor<double> fc2_bias = fc2_weights_and_biases[1];

        // float *mat1, *mat2, *out;
        int new_dims1[] = {output1.dims[0], fc2_weights.dims[1]};
        Tensor<double> product2(2, new_dims1);

        fc_forward(mat1, mat2, out, output1, fc2_weights);

        dim3 dimBlock2(16, 16);
        dim3 dimGrid2(2, 2);

        cudaEventRecord(st2);
        matmul_kernel<<<dimGrid2, dimBlock2>>>(mat1, mat2, out, int(output1.dims[0]), int(output1.dims[1]), int(fc2_weights.dims[1]));
        cudaEventRecord(et2);

        // host waits until et2 has occured
        cudaEventSynchronize(et2);

        milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, st2, et2);

        cout << "Kernel time: " << milliseconds << "ms" << endl;
        cudaMemcpy(product2.data_,
                   out,
                   output1.dims[0] * fc2_weights.dims[1] * sizeof(float),
                   cudaMemcpyDeviceToHost);

        cudaFree(mat1);
        cudaFree(mat2);
        cudaFree(out);

        //// END matmul kernel ////////////////////////////////
        output = product2 + fc2_bias;        
        //// END FC2 ///////////////////////////////////////////
        
    }
    printf("Testing done\n");

    return 0;
}