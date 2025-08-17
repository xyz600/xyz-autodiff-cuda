#pragma once

#include <cstddef>
#include <cmath>
#include <cuda_runtime.h>

namespace xyz_autodiff {

// Sigmoid関数のロジック実装
template <std::size_t Dim>
struct SigmoidLogic {
    static constexpr std::size_t outputDim = Dim;
    
    template <typename Output, typename Input>
    __host__ __device__ void forward(Output& output, const Input& input) const {
        for (std::size_t i = 0; i < outputDim; ++i) {
            // sigmoid(x) = 1 / (1 + exp(-x))
            float exp_neg_x = expf(-input[i]);
            output[i] = 1.0f / (1.0f + exp_neg_x);
        }
    }
    
    template <typename Output, typename Input>
    __host__ __device__ void backward(const Output& output, Input& input) const {
        for (std::size_t i = 0; i < outputDim; ++i) {
            // d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
            float exp_neg_x = expf(-input[i]);
            float sigmoid_val = 1.0f / (1.0f + exp_neg_x);
            float sigmoid_derivative = sigmoid_val * (1.0f - sigmoid_val);
            input.grad(i) += output.grad(i) * sigmoid_derivative;
        }
    }
};

} // namespace xyz_autodiff