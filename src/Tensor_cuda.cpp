//
// Created by lucas on 12/04/19.
//

#include "../include/Tensor.h"
#include <cstring> // memset
#include <chrono>
#include <iostream>

template class Tensor<int>;

template class Tensor<float>;

template class Tensor<double>;

template <typename T>
void Tensor<T>::zero()
{
    memset(data_, 0, sizeof(T) * size_);
}

template <typename T>
T Tensor<T>::get(int i, int j)
{
    assert(num_dims == 2);
    return data_[j + i * dims[1]];
}

template <typename T>
T Tensor<T>::get(int i)
{
    assert(num_dims == 1);
    return data_[i];
}

template <typename T>
void Tensor<T>::set(int i, int j, T value)
{
    assert(num_dims == 2);
    data_[j + i * dims[1]] = value;
}

template <typename T>
void Tensor<T>::set(int i, T value)
{
    data_[i] = value;
}

template <typename T>
void Tensor<T>::add(int i, T value)
{
    data_[i] += value;
}

template <typename T>
void Tensor<T>::add(int i, int j, T value)
{
    assert(num_dims == 2);
    data_[j + i * dims[1]] += value;
}

template <typename T>
void Tensor<T>::view(int new_num_dims, int *new_dims)
{
    assert(new_num_dims > 0 && new_num_dims <= 4);
    this->num_dims = new_num_dims;
    std::copy(new_dims, new_dims + 4, this->dims);
}

template <typename T>
Tensor<T>::Tensor(int num_dims, int const *dims)
{
    assert(num_dims > 0 && num_dims <= 4);
    int size = 1;
    for (int i = 0; i < num_dims; ++i)
    {
        size *= dims[i];
        this->dims[i] = dims[i];
    }
    size_ = size;
    //    std::shared_ptr<T[]> data_sp(new T[size_]);
    T *data_sp = new T[size_];
    data_ = data_sp;
    this->num_dims = num_dims;
}

template <typename T>
T Tensor<T>::get(int i, int j, int k)
{
    assert(num_dims == 3);
    return data_[k + j * dims[2] + i * dims[1] * dims[2]];
}

template <typename T>
T Tensor<T>::get(int i, int j, int k, int l)
{
    assert(num_dims == 4);
    return data_[l + k * dims[3] + j * dims[2] * dims[3] + i * dims[1] * dims[2] * dims[3]];
}

template <typename T>
void Tensor<T>::set(int i, int j, int k, T value)
{
    assert(num_dims == 3);
    data_[k + j * dims[2] + i * dims[1] * dims[2]] = value;
}

template <typename T>
void Tensor<T>::set(int i, int j, int k, int l, T value)
{
    assert(num_dims == 4);
    data_[l + k * dims[3] + j * dims[2] * dims[3] + i * dims[1] * dims[2] * dims[3]] = value;
}

template <typename T>
void Tensor<T>::add(int i, int j, int k, int l, T value)
{
    assert(num_dims == 4);
    data_[l + k * dims[3] + j * dims[2] * dims[3] + i * dims[1] * dims[2] * dims[3]] += value;
}

template <typename T>
Tensor<T>::Tensor(const Tensor<T> &other) : size_(other.size_), num_dims(other.num_dims),
                                            data_(new T[other.size_])
{
    std::copy(other.data_, other.data_ + size_, data_);
    std::copy(other.dims, other.dims + 4, dims);
}

template <typename T>
Tensor<T>::~Tensor()
{
    delete[] data_;
}

// input.matmul(weights)
template <typename T>
Tensor<T> Tensor<T>::matmul(Tensor<T> other)
{
    assert(num_dims == 2 && other.num_dims == 2);
    assert(dims[1] == other.dims[0]);
    auto start = std::chrono::high_resolution_clock::now();

    int new_dims[] = {dims[0], other.dims[1]};
    Tensor<T> product(2, new_dims);
    // print size of product
    // printf("Product size: %d. new_dims: (%d, %d)\n", product.size_, new_dims[0], new_dims[1]);
    product.zero();
    // printf("Product values: %d\n", product.get(0, 0));
    // Assign empty values to product
    // product = (T*)malloc(sizeof(float) * product.size_);

    // Parallel collapse
    // #if defined(_OPENMP)
    // #endif
    // #pragma omp parallel for collapse(3)
    for (int i = 0; i < this->dims[0]; ++i)
    {
        for (int j = 0; j < other.dims[1]; ++j)
        {
            // T value = 0;
            for (int k = 0; k < other.dims[0]; ++k)
            {
                // Working without atomic
                T value = this->get(i, k) * other.get(k, j);
                product.add(i, j, value);
                // product.add(i, j, this->get(i, k) * other.get(k, j));
                // ERROR:
                // #pragm omp atomic
                // product.data_[j + i * dims[1]] += this->get(i, k) * other.get(k, j);
            }
        }
    }

    // #pragm omp parallel for collapse(2)
    // for (int i = 0; i < this->dims[0]; ++i)
    // {
    //     for (int j = 0; j < other.dims[1]; ++j)
    //     {
    //         T value = 0;
    //         for (int k = 0; k < other.dims[0]; ++k)
    //         {
    //             value += this->get(i, k) * other.get(k, j);
    //         }
    //         product.set(i, j, value);
    //     }
    // }
    auto stop = std::chrono::high_resolution_clock::now();
    // printf("matmul    Time taken: %ld\n", std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count());
    return product;
}

// Parallel performs worse
template <typename T>
Tensor<T> Tensor<T>::matrixTranspose()
{
    assert(num_dims == 2);
    int new_dims[] = {dims[1], dims[0]};
    Tensor<T> transpose(num_dims, new_dims);
    // printf("before transpose");
    // for (int i = 0; i < dims[0]; ++i)
    // {
    //     for (int j = 0; j < dims[1]; ++j)
    //     {
    //         std::cout << get(i, j) << " ";
    //     }
    //     printf("\n");
    // }
    // #pragma omp parallel for collapse(2)
    for (int i = 0; i < dims[0]; ++i)
    {
        for (int j = 0; j < dims[1]; ++j)
        {
            transpose.set(j, i, get(i, j));
        }
    }
    // printf("after transpose");
    // for (int i = 0; i < dims[1]; ++i)
    // {
    //     for (int j = 0; j < dims[0]; ++j)
    //     {
    //         std::cout << transpose.get(i, j) << " ";
    //     }
    //     printf("\n");
    // }

    return transpose;
}

// TODO: parallelize
// Parallel performs worse
template <typename T>
Tensor<T> Tensor<T>::relu()
{
    auto start = std::chrono::high_resolution_clock::now();
    Tensor<T> result(num_dims, dims);
    // #pragma omp parallel for
    for (int i = 0; i < size_; ++i)
    {
        T x = data_[i];
        result.data_[i] = x > 0 ? x : 0;
    }

    auto stop = std::chrono::high_resolution_clock::now();
    // printf("relu      Time taken: %ld\n", std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count());
    return result;
}

// TODO: parallelize

template <typename T>
T sigmoid(T x)
{
    return 1.0 / (1.0 + exp(-x));
}

template <typename T>
Tensor<T> Tensor<T>::sigmoid()
{
    Tensor<T> result(num_dims, dims);
    for (int i = 0; i < size_; ++i)
    {
        T x = data_[i];
        result.data_[i] = ::sigmoid(x);
    }

    return result;
}

// TODO: parallelize
template <typename T>
T sigmoidPrime(T x)
{
    return sigmoid(x) * (1.0 - sigmoid(x));
}

template <typename T>
Tensor<T> Tensor<T>::sigmoidPrime()
{
    Tensor<T> result(num_dims, dims);
    for (int i = 0; i < size_; ++i)
    {
        T x = data_[i];
        result.data_[i] = ::sigmoidPrime(x);
    }

    return result;
}

template <typename T>
T Tensor<T>::sum()
{
    T total = 0;
    for (int i = 0; i < size_; ++i)
    {
        total += data_[i];
    }
    return 0;
}

// TODO: parallelize
template <typename T>
Tensor<T> Tensor<T>::softmax()
{
    assert(num_dims == 2);
    // Softmax with max trick to avoid overflows
    int rows = dims[0], cols = dims[1];
    Tensor<T> probabilities(2, dims);
    for (int i = 0; i < rows; ++i)
    {
        T row_max = -1; // useless value so my IDE stops screaming at me, will always be replaced
        for (int j = 0; j < cols; ++j)
        {
            if (j == 0 || get(i, j) > row_max)
            {
                row_max = get(i, j);
            }
        }

        T denominator = 0;
        for (int j = 0; j < cols; ++j)
        {
            T x = get(i, j);
            denominator += exp(get(i, j) - row_max);
        }

        for (int j = 0; j < cols; ++j)
        {
            probabilities.set(i, j, exp(get(i, j) - row_max) / denominator);
        }
    }
    return probabilities;
}

// TODO: parallelize
template <typename T>
Tensor<T> Tensor<T>::reluPrime()
{
    Tensor<T> prime(num_dims, dims);
    // #pragma omp parallel for
    for (int i = 0; i < size_; ++i)
    {
        prime.data_[i] = data_[i] > 0 ? 1 : 0;
    }
    return prime;
}

template <typename T>
Tensor<T> Tensor<T>::operator+(Tensor<T> &other)
{
    if (other.num_dims == 1 && other.size_ == this->dims[1] && num_dims == 2)
    {
        // if other is a 1d tensor and this is a 2d tensor
        Tensor<T> sum(num_dims, dims);
        for (int k = 0; k < this->dims[0]; ++k)
        {
            for (int j = 0; j < this->dims[1]; ++j)
            {
                sum.set(k, j, get(k, j) + other.get(j));
            }
        }

        return sum;
    }
    else if (other.num_dims == num_dims && other.size_ == size_)
    {
        Tensor<T> sum(num_dims, dims);
        for (int i = 0; i < size_; ++i)
        {
            sum.data_[i] = data_[i] + other.data_[i];
        }
        return sum;
    }
    throw std::logic_error("Undefined sum");
}

template <typename T>
Tensor<T> Tensor<T>::operator*(Tensor<T> other)
{
    assert(size_ == other.size_);
    Tensor<T> product(num_dims, dims);
    for (int i = 0; i < size_; ++i)
    {
        product.data_[i] = data_[i] * other.data_[i];
    }
    return product;
}

template <typename T>
Tensor<T> Tensor<T>::operator*(T multiplier)
{
    Tensor<T> product(num_dims, dims);
    for (int i = 0; i < size_; ++i)
    {
        product.data_[i] = data_[i] * multiplier;
    }
    return product;
}

template <typename T>
Tensor<T> Tensor<T>::operator/(T divisor)
{
    Tensor<T> quotient(num_dims, dims);
    for (int i = 0; i < size_; ++i)
    {
        quotient.data_[i] = data_[i] / divisor;
    }
    return quotient;
}

template <typename T>
Tensor<T> Tensor<T>::operator-=(Tensor<T> difference)
{
    assert(size_ == difference.size_);
    for (int i = 0; i < size_; ++i)
    {
        data_[i] = data_[i] - difference.data_[i];
    }
    return *this;
}

// TODO: check
template <typename T>
Tensor<T> Tensor<T>::columnWiseSum()
{
    assert(num_dims == 2);
    int rows = dims[0], cols = dims[1];
    int sum_dims[] = {cols};
    Tensor<T> sum(1, sum_dims);
    // sum.zero();
    // #pragma omp parallel for
    // for (int i = 0; i < cols; ++i)
    // {
    //     for (int j = 0; j < rows; ++j)
    //     {
    //         sum.add(i, get(j, i));
    //     }
    // }
    for (int i = 0; i < cols; ++i)
    {
        T total = 0;
        for (int j = 0; j < rows; ++j)
        {
            total += get(j, i);
        }
        sum.set(i, total);
    }
    return sum;
}

template <>
void Tensor<double>::randn(std::default_random_engine generator, std::normal_distribution<double> distribution,
                           double multiplier)
{
    for (int i = 0; i < size_; ++i)
    {
        data_[i] = distribution(generator) * multiplier;
    }
}

template <>
void Tensor<double>::print()
{
    if (num_dims == 2)
    {
        int rows = dims[0], cols = dims[1];
        std::cout << "Tensor2D (" << rows << ", " << cols << ")\n[";
        for (int i = 0; i < rows; ++i)
        {
            if (i != 0)
                std::cout << " ";
            std::cout << "[";
            for (int j = 0; j < cols; ++j)
            {
                if (j == (cols - 1))
                {
                    printf("%.18lf", get(i, j));
                }
                else
                {
                    printf("%.18lf ", get(i, j));
                }
            }
            if (i == (rows - 1))
            {
                std::cout << "]]\n";
            }
            else
            {
                std::cout << "]\n";
            }
        }
    }
    else
    {
        printf("Tensor%dd (", num_dims);
        for (int i = 0; i < num_dims; ++i)
        {
            printf("%d", dims[i]);
            if (i != (num_dims - 1))
            {
                printf(",");
            }
        }
        printf(")\n[");
        for (int j = 0; j < size_; ++j)
        {
            printf("%lf ", data_[j]);
        }
        printf("]\n");
    }
}

template <typename T>
Tensor<T> &Tensor<T>::operator=(const Tensor<T> &other)
{
    if (this != &other)
    {
        T *new_data = new T[other.size_];
        std::copy(other.data_, other.data_ + other.size_, new_data);
        if (size_ != -1)
        {
            delete[] data_;
        }
        size_ = other.size_;
        std::copy(other.dims, other.dims + 4, dims);
        num_dims = other.num_dims;
        data_ = new_data;
    }

    return *this;
}

// TODO: parallelize
template <typename T>
void Tensor<T>::dropout(std::default_random_engine generator, std::uniform_real_distribution<> distribution, double p)
{
    for (int i = 0; i < size_; ++i)
    {
        data_[i] = (distribution(generator) < p) / p;
    }
}

// TODO: parallelize
//dim[0] = batch_size, dim[1] = channels
//dim[2] = height, dim[3] = width

//kernels =  {out_channels, in_channels, kernel_size, kernel_size};

__global__ void conv2d_kernel(float *input, float *kernels, float *output, float *bias,
                               int B, int C, int H, int W, // Input dimensions
                               int M, int KH, int KW, // Kernel dimensions
                               int outH, int outW, // Output dimensions
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
                        // Calculate the index in the input tensor considering the batch
                        float pixel = input[((b * C + c) * H + ih) * W + iw];
                        float weight = kernels[((m * C + c) * KH + kh) * KW + kw];
                        total += pixel * weight;
                    }
                }
            }
        }
        total += bias[m];
        // Calculate the index in the output tensor considering the batch
        output[((b * M + m) * outH + h) * outW + w] = total;
    }
}

template <typename T>
Tensor<T> Tensor<T>::convolve2d(Tensor<T> kernels, int stride, int padding, Tensor<T> bias)
{   
    assert(kernels.dims[1] == dims[1]);
    int w = ((dims[3] + 2 * padding - (kernels.dims[3] - 1) - 1) / stride) + 1;
    int h = ((dims[2] + 2 * padding - (kernels.dims[2] - 1) - 1) / stride) + 1;
    int result_dims[] = {dims[0], kernels.dims[0], h, w};
    Tensor<T> output(4, result_dims);   
    float *inp, *ker, *bi, *out;

    cudaError_t err = cudaMalloc(&inp, input.dims[0] * input.dims[1] * input.dims[2] * input.dims[3] * sizeof(float));
    if (err != cudaSuccess) 
    {
        cout << "CUDA malloc failed: " << cudaGetErrorString(err)
            << " [Requested dims: " << input.dims[0] << " x " << input.dims[1]
            << ", Size: " << (input.dims[0] * input.dims[1] * input.dims[2] * input.dims[3] * sizeof(float)) << " bytes]" << endl;
        exit(-1);
    }

    err = cudaMalloc(&ker, kernels.dims[2] * kernels.dims[3] * sizeof(float));
    if (err != cudaSuccess)
    {
        cout << "Dev Memory not allocated2" << endl;
        exit(-1);
    }

    err = cudaMalloc(&bi, bias.dims[0] * sizeof(float));
    if (err != cudaSuccess)
    {
        cout << "Dev Memory not allocated3" << endl;
        exit(-1);
    }

    err = cudaMalloc(&out, w * h * input.dims[0] * kernels.dim[0] * sizeof(float));
    if (err != cudaSuccess)
    {
        cout << "Dev Memory not allocated3" << endl;
        exit(-1);
    }

    cudaMemcpy(inp,
                input.data_,
                input.dims[0] * input.dims[1] * input.dims[2] * input.dims[3] * sizeof(float),
                cudaMemcpyHostToDevice);

    cudaMemcpy(ker,
                kernels.data_,
                kernels.dims[2] * kernels.dims[3] * sizeof(float),
                cudaMemcpyHostToDevice);
    
    cudaMemcpy(bi,
                bi.data_,
                bias.dims[0] * sizeof(float),
                cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((outW + 15) / 16, (outH + 15) / 16, M * B); // Updated for batch processing

    conv2d_kernel<<<numBlocks, threadsPerBlock>>>(inp, ker, out, bi,
                                                  input.dim[0], input.dim[1], input.dim[2], input.dim[3], kernels.dim[0], kernels.dim[2], kernels.dim[3], h, w, padding, stride);
    cudaDeviceSynchronize(); // Wait for the kernel to complete
    
    cudaMemcpy(output,
                out.data_,
                w * h * input.dims[0] * kernels.dim[0] * sizeof(float),
                cudaMemcpyDeviceToHost);

    // return output;

    // for (int i = 0; i < dims[0]; ++i)
    // { // pra cada img do batch
    //     for (int j = 0; j < kernels.dims[0]; ++j)
    //     { // pra cada output volume
    //         for (int k = 0; k < h; ++k)
    //         { // pra cada k vertical no output volume
    //             for (int l = 0; l < w; ++l)


    //             { // pra cada l horizontal no output volume
    //                 int im_si = stride * k - padding;
    //                 int im_sj = stride * l - padding;
    //                 T total = 0;
    //                 for (int m = 0; m < kernels.dims[1]; ++m)
    //                 { // pra cada canal do filtro
    //                     for (int n = 0; n < kernels.dims[2]; ++n)
    //                     {
    //                         for (int o = 0; o < kernels.dims[3]; ++o)
    //                         {
    //                             int x = im_si + n, y = im_sj + o;
    //                             if (x < 0 || x >= dims[2] || y < 0 || y >= dims[3])
    //                                 continue; // se for regiao do padding, pula (soma 0)
    //                             T a = data_[y + x * dims[3] + m * dims[2] * dims[3] + i * dims[1] * dims[2] * dims[3]];
    //                             T b = kernels.data_[o + n * dims[3] + m * dims[2] * dims[3] + j * dims[1] * dims[2] * dims[3]]; 
    //                             total += a * b;
    //                         }
    //                     }
    //                 }
            
    //                 output.data_[l + k * dims[3] + j * dims[2] * dims[3] + i * dims[1] * dims[2] * dims[3]] =  total + bias.data_[j];
            
    //             }


    //         }
    //     }
    // }
  
}