#pragma once

#include <cstddef>
#include <cuda_runtime.h>
#include "operation.cuh"
#include "math.cuh"
#include "../concept/variable.cuh"

namespace xyz_autodiff {

// Sigmoid関数のロジック実装
template <std::size_t Dim>
struct SigmoidLogic {
    static constexpr std::size_t outputDim = Dim;
    
    template <typename Output, typename Input>
    __host__ __device__ void forward(Output& output, const Input& input) const {
        for (std::size_t i = 0; i < outputDim; ++i) {
            // sigmoid(x) = 1 / (1 + exp(-x))
            using T = typename Input::value_type;
            output[i] = math::sigmoid(input[i]);
        }
    }
    
    template <typename Output, typename Input>
    __host__ __device__ void backward(const Output& output, Input& input) const {
        for (std::size_t i = 0; i < outputDim; ++i) {
            // d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
            using T = typename Input::value_type;
            T sigmoid_val = math::sigmoid(input[i]);
            T sigmoid_derivative = sigmoid_val * (T{1} - sigmoid_val);
            input.add_grad(i, output.grad(i) * sigmoid_derivative);
        }
    }
};

// Sigmoid関数のファクトリ
template <std::size_t Dim, DifferentiableVariableConcept Input>
__host__ __device__ auto sigmoid(Input& input) {
    SigmoidLogic<Dim> logic;
    
    auto op = UnaryOperation<Dim, SigmoidLogic<Dim>, Input>(logic, input);
    return op;
}

// 型推論をサポートする版（入力のサイズから自動的にDimを決定）
template <DifferentiableVariableConcept Input>
__host__ __device__ auto sigmoid(Input& input) {
    constexpr std::size_t Dim = Input::size;
    return sigmoid<Dim>(input);
}

} // namespace xyz_autodiff