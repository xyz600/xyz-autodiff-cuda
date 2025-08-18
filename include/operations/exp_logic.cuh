#pragma once

#include <cstddef>
#include <cuda_runtime.h>
#include "operation.cuh"
#include "math.cuh"
#include "../concept/variable.cuh"

namespace xyz_autodiff {

// Exponential関数のロジック実装
template <std::size_t Dim>
struct ExpLogic {
    static constexpr std::size_t outputDim = Dim;
    
    template <typename Output, typename Input>
    __host__ __device__ void forward(Output& output, const Input& input) const {
        for (std::size_t i = 0; i < outputDim; ++i) {
            // exp(x)
            using T = typename Input::value_type;
            output[i] = math::exp(input[i]);
        }
    }
    
    template <typename Output, typename Input>
    __host__ __device__ void backward(const Output& output, Input& input) const {
        for (std::size_t i = 0; i < outputDim; ++i) {
            // d/dx exp(x) = exp(x)
            using T = typename Input::value_type;
            T exp_val = math::exp(input[i]);
            input.add_grad(i, output.grad(i) * exp_val);
        }
    }
};

// Exponential関数のファクトリ
template <std::size_t Dim, DifferentiableVariableConcept Input>
__host__ __device__ auto exp(Input& input) {
    ExpLogic<Dim> logic;
    
    auto op = UnaryOperation<Dim, ExpLogic<Dim>, Input>(logic, input);
    op.forward();
    return op;
}

// 型推論をサポートする版（入力のサイズから自動的にDimを決定）
template <DifferentiableVariableConcept Input>
__host__ __device__ auto exp(Input& input) {
    constexpr std::size_t Dim = Input::size;
    return exp<Dim>(input);
}

} // namespace xyz_autodiff