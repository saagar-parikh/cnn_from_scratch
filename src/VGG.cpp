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
#include<chrono>
using namespace std::chrono;
using namespace std;

/*
 * Train a neural network on the MNIST data set and evaluate its performance
 */
float *cudaMalloc_fn(int size)
{
    float* var;
    cudaError_t err = cudaMalloc(&var,size);
    if (err != cudaSuccess)
    {
        cout << "Dev Memory not allocated2" << endl;
        exit(-1);
    }
    return var;
}
 
__global__ void conv2d_kernel(float *input, float *kernels, float *output, float *bias,
                               int B, int C, int H, int W, 
                               int M, int KH, int KW, 
                               int outH, int outW, 
                               int pad, int stride) {
    int w = blockIdx.x * blockDim.x + threadIdx.x; // Calculate the width index
    int h = blockIdx.y * blockDim.y + threadIdx.y; // Calculate the height index
    int m = blockIdx.z % M; // Calculate the output channel index
    int b = blockIdx.z / M; // Calculate the batch index

    if (b < B && w < outW && h < outH) {
        float total = 0.0;

        for (int c = 0; c < C; ++c) {
            for (int kh = 0; kh < KH; ++kh) {
                for (int kw = 0; kw < KW; ++kw) {
                    int ih = h * stride - pad + kh;
                    int iw = w * stride - pad + kw;

                    if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
              
                        float pixel = input[((b * C + c) * H + ih) * W + iw];
                        float weight = kernels[((m * C + c) * KH + kh) * KW + kw];
                        total += pixel * weight;
                    }
                }
            }
        }
        total += bias[m];

        output[((b * M + m) * outH + h) * outW + w] = total;
    }
}

__global__ void pool_kernel(float *input, float *output, float *indexes,
                               int B, int C, int H, int W,
                               int size_,
                               int outH, int outW,
                               int stride) {
    int w = blockIdx.x * blockDim.x + threadIdx.x; // Calculate the width index
    int h = blockIdx.y * blockDim.y + threadIdx.y; // Calculate the height index
    int c = blockIdx.z % C; // Calculate the output channel index
    int b = blockIdx.z / C; // Calculate the batch index

    if (b < B && w < outW && h < outH) {
        float total = 0.0;

        double max = -999999999;
        int index = 0;
        for (int kh = 0; kh < size_; ++kh) {
            for (int kw = 0; kw < size_; ++kw) {
                int ih = h * stride + kh;
                int iw = w * stride + kw;

                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                    double value = input[((b * C + c) * H + ih) * W + iw];
                    if (value > max) {
                        index = c * size_ + kw;
                        max = value;
                    }
                }
            }
        }
        output[((b * C + c) * outH + h) * outW + w] = max;
        indexes[((b * C + c) * outH + h) * outW + w] = index;
    }
}

__global__ void matmul_kernel(float *mat1, float *mat2, 
            float *output, int dim_1, int dim_2, int dim_3)
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

Tensor<double> fc_forward(Tensor<double> input, FullyConnected *module_fc){

        Tensor<double> fc_weights = module_fc->weights;
        Tensor<double> fc_bias = module_fc->bias;

        int new_dims[] = {input.dims[0], fc_weights.dims[1]};
        Tensor<double> product(2, new_dims);

        int input_num_dims = input.num_dims;
        int input_dims[4];
        std::copy(input.dims, input.dims + input.num_dims, input_dims);
        if (input.num_dims != 2)
        {
            int flatten_size = 1;
            for (int i = 1; i < input.num_dims; ++i)
                flatten_size *= input.dims[i];
            int dims[] = {input.dims[0], flatten_size};
            input.view(2, dims);
        }

        float *mat1 = cudaMalloc_fn(input.dims[0] * input.dims[1] * sizeof(float));
        float *mat2 = cudaMalloc_fn(fc_weights.dims[0] * fc_weights.dims[1] * sizeof(float));
        float *out = cudaMalloc_fn(input.dims[0] * fc_weights.dims[1] * sizeof(float));

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
        cudaEvent_t st2, et2;
        cudaEventCreate(&st2);
        cudaEventCreate(&et2);

        cudaEventRecord(st2);
        matmul_kernel<<<dimGrid, dimBlock>>>(mat1, mat2, out, int(input.dims[0]), int(input.dims[1]), int(fc_weights.dims[1]));
        cudaEventRecord(et2);

        // host waits until et2 has occured
        cudaEventSynchronize(et2);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, st2, et2);

        cout << "FC_Kernel time: " << milliseconds << "ms" << endl;
        cudaMemcpy(product.data_,
                   out,
                   input.dims[0] * fc_weights.dims[1] * sizeof(float),
                   cudaMemcpyDeviceToHost);

        cudaFree(mat1);
        cudaFree(mat2);
        cudaFree(out);

        return product + fc_bias;
}

Tensor<double> conv_forward(Tensor<double> input, Conv2d *module_cnn)
{
    Tensor<double> cnn_kernels = module_cnn->kernels;
    Tensor<double> cnn_bias = module_cnn->bias;

    int padding = module_cnn->padding;
    int stride = module_cnn->stride;

    int w = ((input.dims[3] + 2 * padding - (cnn_kernels.dims[3] - 1) - 1) / stride) + 1;
    int h = ((input.dims[2] + 2 * padding - (cnn_kernels.dims[2] - 1) - 1) / stride) + 1;
    int result_dims[] = {input.dims[0], cnn_kernels.dims[0], h, w};
    
    Tensor<double> output(4, result_dims);   

    float *inp = cudaMalloc_fn(input.dims[0] * input.dims[1] * input.dims[2] * input.dims[3] * sizeof(float));
    float *ker = cudaMalloc_fn(cnn_kernels.dims[2] * cnn_kernels.dims[3] * sizeof(float));
    float *bi = cudaMalloc_fn(cnn_bias.dims[0] * sizeof(float));
    float *out_c = cudaMalloc_fn(w * h * input.dims[0] * cnn_kernels.dims[0] * sizeof(float));

    cudaMemcpy(inp,
                input.data_,
                input.dims[0] * input.dims[1] * input.dims[2] * input.dims[3] * sizeof(float),
                cudaMemcpyHostToDevice);

    cudaMemcpy(ker,
                cnn_kernels.data_,
                cnn_kernels.dims[2] * cnn_kernels.dims[3] * sizeof(float),
                cudaMemcpyHostToDevice);
    
    cudaMemcpy(bi,
                cnn_bias.data_,
                cnn_bias.dims[0] * sizeof(float),
                cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((w + 15) / 16, (h + 15) / 16, 
                cnn_kernels.dims[0] * input.dims[0]);

    cudaEvent_t st2, et2;
    cudaEventCreate(&st2);
    cudaEventCreate(&et2);

    cudaEventRecord(st2);

    conv2d_kernel<<<numBlocks, threadsPerBlock>>>(inp, ker, out_c, bi,
                                                input.dims[0], input.dims[1], input.dims[2], input.dims[3], cnn_kernels.dims[0], cnn_kernels.dims[2], cnn_kernels.dims[3], h, w, padding, stride);
    cudaEventRecord(et2);

    // host waits until et2 has occured
    cudaEventSynchronize(et2);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, st2, et2);

    cout << "Conv_Kernel time: " << milliseconds << "ms" << endl;
    
    cudaMemcpy(output.data_,
                out_c,
                w * h * input.dims[0] * cnn_kernels.dims[0] * sizeof(float),
                cudaMemcpyDeviceToHost);

    cudaFree(inp);
    cudaFree(ker);
    cudaFree(out_c);
    cudaFree(bi);

    return output;
}


Tensor<double> pool_forward(Tensor<double> input, MaxPool *module_pool)
{

    int stride = module_pool->stride_;
    int size_ = module_pool->size_;
    int w = ((input.dims[3] - (size_ - 1) - 1) / stride) + 1;
    int h = ((input.dims[2] - (size_ - 1) - 1) / stride) + 1;
    int result_dims[] = {input.dims[0], input.dims[1], h, w};
    
    Tensor<double> output(4, result_dims);   
    Tensor<double> indexes(4, result_dims);   

    float *inp = cudaMalloc_fn(input.dims[0] * input.dims[1] * input.dims[2] * input.dims[3] * sizeof(float));
    float *ind = cudaMalloc_fn(w * h * input.dims[0] * input.dims[1] * sizeof(float));
    float *out_c = cudaMalloc_fn(w * h * input.dims[0] * input.dims[1] * sizeof(float));

    cudaMemcpy(inp,
                input.data_,
                input.dims[0] * input.dims[1] * input.dims[2] * input.dims[3] * sizeof(float),
                cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((w + 15) / 16, (h + 15) / 16, 
                input.dims[1] * input.dims[0]); 
    
    cudaEvent_t st2, et2;
    cudaEventCreate(&st2);
    cudaEventCreate(&et2);

    cudaEventRecord(st2);
    pool_kernel<<<numBlocks, threadsPerBlock>>>(inp, out_c, ind,
                                                input.dims[0], input.dims[1], input.dims[2], input.dims[3],size_, h, w, stride);
    cudaEventRecord(et2);

    // host waits until et2 has occured
    cudaEventSynchronize(et2);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, st2, et2);

    cout << "Pool_Kernel time: " << milliseconds << "ms" << endl;
    
    cudaMemcpy(indexes.data_,
                ind,
                w * h * input.dims[0] * input.dims[1] * sizeof(float),
                cudaMemcpyDeviceToHost);
    cudaMemcpy(output.data_,
            out_c,
            w * h * input.dims[0] * input.dims[1]* sizeof(float),
            cudaMemcpyDeviceToHost);

    cudaFree(inp);
    cudaFree(ind);
    cudaFree(out_c);

    return output;
}


Tensor<double> flatten(Tensor<double> input)
{
    int new_dims_c[] = {input.dims[0], input.dims[1]*input.dims[2]*input.dims[3]};
    Tensor<double> output_flat(2,new_dims_c);
    
    const int batch_size = input.dims[0];
    const int channels = input.dims[1];
    const int height = input.dims[2];
    const int width = input.dims[3];
    const int feature_size = channels * height * width;

    for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < channels; ++c) {
            for (int h = 0; h < height; ++h) {
                for (int w = 0; w < width; ++w) {
                    int flat_index = b * feature_size + c * (height * width) + h * width + w;
                    int original_index = b * (channels * height * width) + c * (height * width) + h * width + w;
                    output_flat.data_[flat_index] = input.data_[original_index];
                }
            }
        }
    }
    return output_flat;
}

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
    // vector<Module *> modules = {
    //     new Conv2d(1, 32, 11, 4, 0, seed), //54x54x32
    //     new MaxPool(3,2), //26x26
    //     new Conv2d(32, 64, 5, 1, 2, seed), //26x26
    //     new MaxPool(3,2), //12x12
    //     new Conv2d(64, 128, 3, 1, 2, seed), //14x14
    //     new Conv2d(128, 256, 3, 1, 2, seed), //16x16
    //     new Conv2d(256, 32, 3, 1, 2, seed), //18x18
    //     new MaxPool(3,2), //8x8
    //     new FullyConnected(8*8*32, 512, seed), 
    //     new FullyConnected(512, 2, seed)
    vector<Module *> modules = {
    new Conv2d(1, 64, 3, 1, 1, seed), //  224x224
    new Conv2d(64, 64, 3, 1, 1, seed), // 224x224
    new MaxPool(2,2), // 112x112


    new Conv2d(64, 128, 3, 1, 1, seed), //112x112
    new Conv2d(128, 128, 3, 1, 1, seed), //112x112
    new MaxPool(2,2), //56x56


    new Conv2d(128, 256, 3, 1, 1, seed), //56x56
    new Conv2d(256, 256, 3, 1, 1, seed), //56x56
    new Conv2d(256, 256, 3, 1, 1, seed), //56x56
    new MaxPool(2,2), // 28x28

    new Conv2d(256, 512, 3, 1, 1, seed), //28x28
    new Conv2d(512, 512, 3, 1, 1, seed), //28x28
    new Conv2d(512, 512, 3, 1, 1, seed),  //28x28
    new MaxPool(2,2), //14x14

    new Conv2d(512, 512, 3, 1, 1, seed), //14x14
    new Conv2d(512, 512, 3, 1, 1, seed), //14x14
    new Conv2d(512, 512, 3, 1, 1, seed), //14x14
    new MaxPool(2,2), // 7x7

    
    new FullyConnected(7*7*512, 4096, seed), //7x7x512
    new FullyConnected(4096, 4096, seed),
    new FullyConnected(4096, 2, seed)
    };
    // , 

    auto lr_sched = new LinearLRScheduler(0.2, -0.000005);
    NetworkModel model = NetworkModel(modules, new SoftmaxClassifier(), lr_sched);
    //    model.load("network.txt");


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
    
    auto total_time = 0.0;
    # pragma omp parallel for num_threads(num_test_batches)
    for (int i = 0; i < num_test_batches; ++i)
    {   auto start_time = high_resolution_clock::now();
        if ((i + 1) % 10 == 0 || i == (num_test_batches - 1))
        {
            printf("\rIteration %d/%d", i + 1, num_test_batches);
            fflush(stdout);
        }
        pair<Tensor<double>, vector<int>> xy = test_loader.nextBatch();
        Tensor<double> output;
        Conv2d *conv_module;
        FullyConnected *fc_module;
        MaxPool *p_module;

        ///////////////////////////////////////
        conv_module = (Conv2d *) modules[0];
        output = conv_forward(xy.first,conv_module);

        conv_module = (Conv2d *) modules[1];
        output = conv_forward(xy.first,conv_module);

        p_module = (MaxPool *) modules[2];
        output = pool_forward(output, p_module);

        //////////////////////////////////////////

        conv_module = (Conv2d *) modules[3];
        output = conv_forward(xy.first,conv_module);

        conv_module = (Conv2d *) modules[4];
        output = conv_forward(xy.first,conv_module);

        p_module = (MaxPool *) modules[5];
        output = pool_forward(output, p_module);

        //////////////////////////////////////////

        conv_module = (Conv2d *) modules[6];
        output = conv_forward(xy.first,conv_module);

        conv_module = (Conv2d *) modules[7];
        output = conv_forward(xy.first,conv_module);

        conv_module = (Conv2d *) modules[8];
        output = conv_forward(xy.first,conv_module);

        p_module = (MaxPool *) modules[9];
        output = pool_forward(output, p_module);

        //////////////////////////////////////////

        conv_module = (Conv2d *) modules[10];
        output = conv_forward(xy.first,conv_module);

        conv_module = (Conv2d *) modules[11];
        output = conv_forward(xy.first,conv_module);

        conv_module = (Conv2d *) modules[12];
        output = conv_forward(xy.first,conv_module);

        p_module = (MaxPool *) modules[13];
        output = pool_forward(output, p_module);

        //////////////////////////////////////////

        conv_module = (Conv2d *) modules[14];
        output = conv_forward(xy.first,conv_module);

        conv_module = (Conv2d *) modules[15];
        output = conv_forward(xy.first,conv_module);

        conv_module = (Conv2d *) modules[16];
        output = conv_forward(xy.first,conv_module);

        p_module = (MaxPool *) modules[17];
        output = pool_forward(output, p_module);
  
        ///// Handling ouput to flatten
        output = flatten(output);

        //cout << " flat shape" << output_flat.dims[0] << " " << output_flat.dims[1] << endl;
        ////////////////////////////////////////

        fc_module = (FullyConnected *) modules[18];
        output = fc_forward(output,fc_module);

        //// END FC1 //////////////////////////////////////////
        //// RELU
        fc_module = (FullyConnected *) modules[19];
        output = fc_forward(output,fc_module);        

        ///// END RELU /////////////////////////////////////////
        // FC layer 2
        auto end_time = high_resolution_clock::now();

        auto duration = duration_cast<microseconds>(end_time - start_time).count();

        cout << "Per Iteration Time: " << duration <<endl;
        total_time += duration;
    }

    cout << "Average Time: " << total_time/num_test_batches/1e6 << " s" << endl;
    cout << "Total Time: " << total_time/1e6 <<  " s" << endl;
    
    cout << num_test_batches <<endl;
    printf("Testing done\n");

    return 0;
}