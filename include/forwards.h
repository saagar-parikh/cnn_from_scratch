#include <iostream>
#include "../include/Module.h"
#include "../include/FullyConnected.h"
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