#pragma once

#include "tensor.h"

namespace xyz_autodiff {

class Operations {
public:
    static Tensor add(const Tensor& a, const Tensor& b);
    static Tensor subtract(const Tensor& a, const Tensor& b);
    static Tensor multiply(const Tensor& a, const Tensor& b);
    static Tensor divide(const Tensor& a, const Tensor& b);
    
    static Tensor matmul(const Tensor& a, const Tensor& b);
    static Tensor sum(const Tensor& input);
    static Tensor mean(const Tensor& input);
    
    static Tensor relu(const Tensor& input);
    static Tensor sigmoid(const Tensor& input);
    static Tensor tanh(const Tensor& input);
    
    static Tensor reshape(const Tensor& input, const std::vector<int>& new_shape);
    static Tensor transpose(const Tensor& input);

private:
    static cublasHandle_t cublas_handle_;
    static bool initialized_;
    
    friend void initialize();
    friend void cleanup();
};

}