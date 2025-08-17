#pragma once

#include <cstddef>
#include <cmath>
#include <cuda_runtime.h>

namespace xyz_autodiff {

// Exponential関数のロジック実装
template <std::size_t Dim>
struct ExpLogic {
    static constexpr std::size_t outputDim = Dim;
    
    template <typename Output, typename Input>
    __host__ __device__ void forward(Output& output, const Input& input) const {
        for (std::size_t i = 0; i < outputDim; ++i) {
            // exp(x)
            output[i] = expf(input[i]);
        }
    }
    
    template <typename Output, typename Input>
    __host__ __device__ void backward(const Output& output, Input& input) const {
        for (std::size_t i = 0; i < outputDim; ++i) {
            // d/dx exp(x) = exp(x)
            float exp_val = expf(input[i]);
            input.grad(i) += output.grad(i) * exp_val;
        }
    }
};

} // namespace xyz_autodiff