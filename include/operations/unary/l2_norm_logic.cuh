#pragma once

#include <cstddef>
#include <cuda_runtime.h>
#include "../operation.cuh"
#include "../math.cuh"
#include "../../concept/variable.cuh"

namespace xyz_autodiff {
namespace op {

// L2 norm operation: output = sqrt(sum(input[i]^2))
template <std::size_t InputDim>
struct L2NormLogic {
    static constexpr std::size_t outputDim = 1;  // scalar output
    
    template <typename Output, typename Input>
    __host__ __device__ void forward(Output& output, const Input& input) const {
        using T = typename Input::value_type;
        T sum_squares = T(0);
        for (std::size_t i = 0; i < InputDim; ++i) {
            sum_squares += input[i] * input[i];
        }
        output[0] = math::sqrt(sum_squares);
    }
    
    template <typename Output, typename Input>
    __host__ __device__ void backward(const Output& output, Input& input) const {
        using T = typename Input::value_type;
        const T& output_grad = output.grad(0);
        const T& norm_value = output[0];
        
        // d(sqrt(sum(x_i^2)))/dx_i = x_i / sqrt(sum(x_i^2))
        if (norm_value > T(1e-8)) {  // avoid division by zero
            for (std::size_t i = 0; i < InputDim; ++i) {
                input.add_grad(i, output_grad * input[i] / norm_value);
            }
        }
    }
};

// L2 norm関数のファクトリ
template <std::size_t Dim, DifferentiableVariableConcept Input>
requires (Input::size == Dim)
__host__ __device__ auto l2_norm(Input& input) {
    L2NormLogic<Dim> logic;
    
    auto op = UnaryOperation<1, L2NormLogic<Dim>, Input>(logic, input);
    return op;
}

// 型推論をサポートする版（入力のサイズから自動的にDimを決定）
template <DifferentiableVariableConcept Input>
__host__ __device__ auto l2_norm(Input& input) {
    constexpr std::size_t Dim = Input::size;
    return l2_norm<Dim>(input);
}

} // namespace op
} // namespace xyz_autodiff