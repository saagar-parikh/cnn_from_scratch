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

using namespace std;

/*
 * Train a neural network on the MNIST data set and evaluate its performance
 */

int main(int argc, char **argv) {
    if (argc < 2) {
        throw runtime_error("Please provide the data directory path as an argument");
    }
    printf("Data directory: %s\n", argv[1]);
    string data_path = argv[1];

    printf("Loading training set... ");
    fflush(stdout);
    MNISTDataLoader train_loader(data_path + "/train-images-idx3-ubyte", data_path + "/train-labels-idx1-ubyte", 32);
    printf("Loaded.\n");

    // in_channels, out_channels, kernel_size, stride, padding, seed

    int seed = 0;
    // vector<Module *> modules = { new Conv2d(1, 1, 3, 1, 1, seed), new ReLU(), new Conv2d(1, 8, 3, 1, 0, seed), new MaxPool(2, 2), new ReLU(), new FullyConnected(1352, 30, seed), new ReLU(),
    //                             new FullyConnected(30, 10, seed)};
    // vector<Module *> modules = {new FullyConnected(784, 10, seed)};
    vector<Module *> modules = {
        new Conv2d(1, 32, 3, 1, 1, seed),
        new ReLU(),
        new MaxPool(2, 2),
        new Conv2d(32, 64, 3, 1, 1, seed),
        new ReLU(),
        new MaxPool(2, 2),
        new FullyConnected(7 * 7 * 64, 128, seed),
        new ReLU(),
        new Dropout(0.5),
        new FullyConnected(128, 10, seed)
    };
    auto lr_sched = new LinearLRScheduler(0.2, -0.000005);
    NetworkModel model = NetworkModel(modules, new SoftmaxClassifier(), lr_sched);
//    model.load("network.txt");

    int epochs = 1;
    printf("Training for %d epoch(s).\n", epochs);
    // Train network
    int num_train_batches = train_loader.getNumBatches();
    for (int k = 0; k < epochs; ++k) {
        printf("Epoch %d\n", k + 1);
        for (int i = 0; i < num_train_batches; ++i) {
            pair<Tensor<double>, vector<int> > xy = train_loader.nextBatch();
            double loss = model.trainStep(xy.first, xy.second);
            if ((i + 1) % 10 == 0) {
                printf("\rIteration %d/%d - Batch Loss: %.4lf", i + 1, num_train_batches, loss);
                fflush(stdout);
            }
        }
        printf("\n");
    }
    // Save weights
    model.save("network.txt");

    printf("Loading testing set... ");
    fflush(stdout);
    MNISTDataLoader test_loader(data_path + "/t10k-images-idx3-ubyte", data_path + "/t10k-labels-idx1-ubyte", 32);
    printf("Loaded.\n");

    model.eval();

    // Test and measure accuracy
    int hits = 0;
    int total = 0;
    printf("Testing...\n");
    int num_test_batches = test_loader.getNumBatches();
    for (int i = 0; i < num_test_batches; ++i) {
        if ((i + 1) % 10 == 0 || i == (num_test_batches - 1)) {
            printf("\rIteration %d/%d", i + 1, num_test_batches);
            fflush(stdout);
        }
        pair<Tensor<double>, vector<int> > xy = test_loader.nextBatch();
        vector<int> predictions = model.predict(xy.first);
        for (int j = 0; j < predictions.size(); ++j) {
            if (predictions[j] == xy.second[j]) {
                hits++;
            }
        }
        total += xy.second.size();
    }
    printf("\n");

    printf("Accuracy: %.2f%% (%d/%d)\n", ((double) hits * 100) / total, hits, total);

    return 0;
}