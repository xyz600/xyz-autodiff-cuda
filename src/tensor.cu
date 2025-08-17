#include "xyz_autodiff/tensor.h"
#include <cuda_runtime.h>

namespace xyz_autodiff {

Tensor::Tensor(int size) : size_(size) {
    cudaMalloc(&data_, size * sizeof(float));
}

Tensor::~Tensor() {
    if (data_) {
        cudaFree(data_);
    }
}

}