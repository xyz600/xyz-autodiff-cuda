#pragma once

#include "operation.cuh"
#include "sigmoid_logic.cuh"
#include "exp_logic.cuh"

namespace xyz_autodiff {

// Sigmoid関数のファクトリ
template <std::size_t Dim, DifferentiableVariableConcept Input>
__host__ __device__ auto sigmoid(const Input& input) {
    SigmoidLogic<Dim> logic;
    
    auto op = UnaryOperation<Dim, SigmoidLogic<Dim>, Input>(logic, input);
    op.forward();
    return op;
}

// Exponential関数のファクトリ
template <std::size_t Dim, DifferentiableVariableConcept Input>
__host__ __device__ auto exp(const Input& input) {
    ExpLogic<Dim> logic;
    
    auto op = UnaryOperation<Dim, ExpLogic<Dim>, Input>(logic, input);
    op.forward();
    return op;
}

// 型推論をサポートする版（入力のサイズから自動的にDimを決定）
template <DifferentiableVariableConcept Input>
__host__ __device__ auto sigmoid(const Input& input) {
    constexpr std::size_t Dim = Input::size;
    return sigmoid<Dim>(input);
}

template <DifferentiableVariableConcept Input>
__host__ __device__ auto exp(const Input& input) {
    constexpr std::size_t Dim = Input::size;
    return exp<Dim>(input);
}

} // namespace xyz_autodiff