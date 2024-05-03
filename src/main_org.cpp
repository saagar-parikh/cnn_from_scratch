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

using namespace std::chrono;
using namespace std;

/*
 * Train a neural network on the MNIST data set and evaluate its performance
 */

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        throw runtime_error("Please provide the data directory path as an argument");
    }
    printf("Data directory: %s\n", argv[1]);
    string data_path = argv[1];

    printf("Loading training set... ");
    fflush(stdout);
    // MNISTDataLoader train_loader(data_path + "/mnist_train_images.ubyte", data_path + "/mnist_train_labels.ubyte", 32);
    MNISTDataLoader test_loader(data_path + "/mnist_test_images.ubyte", data_path + "/mnist_test_labels.ubyte", 64);
    printf("Loaded.\n");

    // in_channels, out_channels, kernel_size, stride, padding, seed

    int seed = 0;
    //vector<Module *> modules = {new FullyConnected(28 * 28, 2, seed), new ReLU()};

    vector<Module *> modules = {
    //     new Conv2d(1, 8, 3, 1, 0, seed), // 222x222
    //     new MaxPool(2, 2), // 111x111
    //     new ReLU(), 
    //     new FullyConnected(8*111*111, 30, seed), 
    //     new ReLU(),
    //     new FullyConnected(30, 2, seed) 
    // };
        new Conv2d(1, 32, 5, 4, 1, seed), //224x224x32
        new MaxPool(2,2), //112x112
        new Conv2d(32, 64, 3, 1, 1, seed), // 9 ms
        new Conv2d(64, 64, 3, 1, 1, seed),
        new Conv2d(64, 64, 3, 1, 1, seed),
        new MaxPool(2,2), //56x56
        new Conv2d(64, 128, 3, 1, 1, seed), // 11ms
        new Conv2d(128, 32, 3, 1, 1, seed),
        new MaxPool(2,2), 
        new FullyConnected(7*7*32, 1024, seed), 
        new ReLU(),
        new FullyConnected(1024, 256, seed), 
        new ReLU(),
        new FullyConnected(1024, 2, seed)
    };
    auto lr_sched = new LinearLRScheduler(0.2, -0.000005);
    NetworkModel model = NetworkModel(modules, new SoftmaxClassifier(), lr_sched);
    //    model.load("network.txt");
// #if defined(_OPENMP)
//     printf("Using OpenMP\n");
// #endif
    int epochs = 1;
    printf("Training for %d epoch(s).\n", epochs);
    // Train network
    int num_train_batches = test_loader.getNumBatches();
    auto start = high_resolution_clock::now();
    long int total_avg = 0;
    for (int k = 0; k < epochs; ++k)
    {
        auto avg_start = high_resolution_clock::now();

        printf("Epoch %d\n", k + 1);
        for (int i = 0; i < num_train_batches; ++i)
        {
            pair<Tensor<double>, vector<int>> xy = test_loader.nextBatch();
            Tensor<double> output = model.forward(xy.first);
            cout << "Done" << endl;
        }
        auto avg_end = high_resolution_clock::now();

        total_avg = total_avg + duration_cast<microseconds>(avg_end - avg_start).count();
        printf("\n");
    }
    auto stop = high_resolution_clock::now();
    printf("Time taken: %f\n", duration_cast<microseconds>(stop - start).count()/1e6);
    printf("Avg time taken: %f\n",total_avg/num_train_batches/1e6);
    
    // Save weights
    // model.save("network.txt");

    // printf("Loading testing set... ");
    // fflush(stdout);
    // MNISTDataLoader test_loader(data_path + "/mnist_test_images.ubyte", data_path + "/mnist_test_labels.ubyte", 32);

    // // MNISTDataLoader test_loader(data_path + "/t10k-images-idx3-ubyte", data_path + "/t10k-labels-idx1-ubyte", 32);
    // printf("Loaded.\n");

    // model.eval();

    // // Test and measure accuracy
    // int hits = 0;
    // int total = 0;
    // printf("Testing...\n");
    // int num_test_batches = test_loader.getNumBatches();
    // for (int i = 0; i < num_test_batches; ++i)
    // {
    //     if ((i + 1) % 10 == 0 || i == (num_test_batches - 1))
    //     {
    //         printf("\rIteration %d/%d", i + 1, num_test_batches);
    //         fflush(stdout);
    //     }
    //     pair<Tensor<double>, vector<int>> xy = test_loader.nextBatch();
    //     vector<int> predictions = model.predict(xy.first);
    //     for (int j = 0; j < predictions.size(); ++j)
    //     {
    //         if (predictions[j] == xy.second[j])
    //         {
    //             hits++;
    //         }
    //     }
    //     total += xy.second.size();
    // }
    // printf("\n");

    // printf("Accuracy: %.2f%% (%d/%d)\n", ((double)hits * 100) / total, hits, total);

    return 0;
}