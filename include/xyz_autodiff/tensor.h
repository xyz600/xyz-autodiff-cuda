#pragma once

#include <vector>
#include <cuda_runtime.h>

namespace xyz_autodiff {

class Tensor {
public:
    Tensor(int size);
    ~Tensor();
    
    float* data() { return data_; }
    const float* data() const { return data_; }
    int size() const { return size_; }

private:
    float* data_;
    int size_;
};

}