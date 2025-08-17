#pragma once

#include "operation.cuh"
#include "sigmoid_logic.cuh"
#include "exp_logic.cuh"

namespace xyz_autodiff {

// Sigmoid関数のファクトリ
template <int Dim, DifferentiableVariableConcept Input>
__host__ __device__ auto sigmoid(const Input& input) {
    constexpr std::size_t OutputSize = static_cast<std::size_t>(Dim);
    SigmoidLogic<Dim> logic;
    
    auto op = UnaryOperation<OutputSize, SigmoidLogic<Dim>, Input>(logic, input);
    op.forward();
    return op;
}

// Exponential関数のファクトリ
template <int Dim, DifferentiableVariableConcept Input>
__host__ __device__ auto exp(const Input& input) {
    constexpr std::size_t OutputSize = static_cast<std::size_t>(Dim);
    ExpLogic<Dim> logic;
    
    auto op = UnaryOperation<OutputSize, ExpLogic<Dim>, Input>(logic, input);
    op.forward();
    return op;
}

// 型推論をサポートする版（入力のサイズから自動的にDimを決定）
template <DifferentiableVariableConcept Input>
__host__ __device__ auto sigmoid(const Input& input) {
    constexpr int Dim = static_cast<int>(Input::size);
    return sigmoid<Dim>(input);
}

template <DifferentiableVariableConcept Input>
__host__ __device__ auto exp(const Input& input) {
    constexpr int Dim = static_cast<int>(Input::size);
    return exp<Dim>(input);
}

} // namespace xyz_autodiff