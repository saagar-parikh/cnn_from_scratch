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
}
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
    vector<Module *> modules = {new FullyConnected(28 * 28, 2, seed), new ReLU()};

    auto lr_sched = new LinearLRScheduler(0.2, -0.000005);
    NetworkModel model = NetworkModel(modules, new SoftmaxClassifier(), lr_sched);
    //    model.load("network.txt");
#if defined(_OPENMP)
    printf("Using OpenMP\n");
#endif

    printf("Loading testing set... ");
    fflush(stdout);
    MNISTDataLoader test_loader(data_path + "/mnist_test_images.ubyte", data_path + "/mnist_test_labels.ubyte", 1);

    // MNISTDataLoader test_loader(data_path + "/t10k-images-idx3-ubyte", data_path + "/t10k-labels-idx1-ubyte", 32);
    model.load("network.txt");
    printf("Loaded.\n");

    model.eval();

    // Test and measure accuracy
    int hits = 0;
    int total = 0;
    printf("Testing...\n");
    int num_test_batches = test_loader.getNumBatches();

    // initialize fc layer
    int input_size = 224 * 224;
    int output_size = 2;
    std::default_random_engine generator(seed);
    std::normal_distribution<double> distribution(0.0, 1.0);
    int weights_dims[] = {input_size, output_size};
    Tensor<double> fc_weights = Tensor<double>(2, weights_dims);
    fc_weights.randn(generator, distribution, sqrt(2.0 / input_size));
    int bias_dims[] = {output_size};
    Tensor<double> fc_bias = Tensor<double>(1, bias_dims);
    fc_bias.randn(generator, distribution, 0);
    // auto &module = model.modules_[0];
    for (int i = 0; i < num_test_batches; ++i)
    {
        if ((i + 1) % 10 == 0 || i == (num_test_batches - 1))
        {
            printf("\rIteration %d/%d", i + 1, num_test_batches);
            fflush(stdout);
        }
        pair<Tensor<double>, vector<int>> xy = test_loader.nextBatch();

        // vector<Module *> modules = {
        //     new Conv2d(1, 32, 3, 1, 1, seed),
        //     new ReLU(),
        //     new MaxPool(2, 2),
        //     new Conv2d(32, 64, 3, 1, 1, seed),
        //     new ReLU(),
        //     new MaxPool(2, 2),
        //     new FullyConnected(7 * 7 * 64, 128, seed),
        //     new ReLU(),
        //     new FullyConnected(128, 10, seed)
        // };
        auto &module = model.modules_[0];
        // cout << typeid(module);
        // vector<int> predictions = module->forward(xy.first);
        // vector<int> predictions = {0};
        // forward pass of full connected layer
        cout << "module 0 selected" << endl;
        // Tensor<double> &output = module->forward(xy.first);
        auto input = xy.first;
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

        //////////////////////////////////////////
        // func:    matmul
        // inputs:  input, fc_weight
        // outputs: output

        //////////////////////////////////////////
        // assert(int num_dims == 2 && fc_weights.num_dims == 2);
        assert(input.dims[1] == fc_weights.dims[0]);

        int new_dims[] = {input.dims[0], fc_weights.dims[1]};
        Tensor<double> product(2, new_dims);

        float *mat1, *mat2, *out;

        cudaError_t err = cudaMalloc(&mat1, input.dims[0] * input.dims[1] * sizeof(float));
        if (err != cudaSuccess)
        {
            cout << "Dev Memory not allocated1 " << err << " " << input.dims[0] << " " << input.dims[1] << endl;
            exit(-1);
        }

        err = cudaMalloc(&mat2, fc_weights.dims[0] * fc_weights.dims[1] * sizeof(float));
        if (err != cudaSuccess)
        {
            cout << "Dev Memory not allocated2" << endl;
            exit(-1);
        }

        err = cudaMalloc(&out, input.dims[0] * fc_weights.dims[1] * sizeof(float));
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
                   fc_weights.data_,
                   fc_weights.dims[0] * fc_weights.dims[1] * sizeof(float),
                   cudaMemcpyHostToDevice);

        dim3 dimBlock(16, 16);
        dim3 dimGrid(2, 2);
        matmul_kernel<<<dimGrid, dimBlock>>>(mat1, mat2, out, int(input.dims[0]), int(input.dims[1]), int(fc_weights.dims[1]));

        cudaMemcpy(product.data_,
                   out,
                   input.dims[0] * fc_weights.dims[1] * sizeof(float),
                   cudaMemcpyDeviceToHost);

        cudaFree(mat1);
        cudaFree(mat2);
        cudaFree(out);

        ////////////////////////////////////////////////////////////

        Tensor<double> output = product + fc_bias;

        cout << "module 0 done" << endl;

        auto &module1 = model.modules_[1];
        cout << "module 1 selected" << endl;
        Tensor<double> &output1 = module1->forward(output);
        cout << "module 1 done" << endl;

        // vector<int> predictions = model.predict(xy.first);

        //     for (int j = 0; j < predictions.size(); ++j)
        //     {
        //         if (predictions[j] == xy.second[j])
        //         {
        //             hits++;
        //         }
        //     }
        //     total += xy.second.size();
    }
    // free(module);
    printf("Testing done\n");

    // printf("Accuracy: %.2f%% (%d/%d)\n", ((double)hits * 100) / total, hits, total);

    return 0;
}