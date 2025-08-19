#pragma once

#include <cstddef>
#include <cuda_runtime.h>
#include "../operation.cuh"
#include "../math.cuh"
#include "../../concept/variable.cuh"

namespace xyz_autodiff {
namespace op {

// Negation関数のロジック実装
template <std::size_t Dim>
struct NegLogic {
    static constexpr std::size_t outputDim = Dim;
    
    template <typename Output, typename Input>
    __host__ __device__ void forward(Output& output, const Input& input) const {
        for (std::size_t i = 0; i < outputDim; ++i) {
            // neg(x) = -x
            output[i] = -input[i];
        }
    }
    
    template <typename Output, typename Input>
    __host__ __device__ void backward(const Output& output, Input& input) const {
        for (std::size_t i = 0; i < outputDim; ++i) {
            // d/dx (-x) = -1
            using T = typename Input::value_type;
            input.add_grad(i, -output.grad(i));
        }
    }
};

// Negation関数のファクトリ
template <std::size_t Dim, DifferentiableVariableConcept Input>
__host__ __device__ auto neg(Input& input) {
    NegLogic<Dim> logic;
    
    auto op = UnaryOperation<Dim, NegLogic<Dim>, Input>(logic, input);
    return op;
}

// 型推論をサポートする版（入力のサイズから自動的にDimを決定）
template <DifferentiableVariableConcept Input>
__host__ __device__ auto neg(Input& input) {
    constexpr std::size_t Dim = Input::size;
    return neg<Dim>(input);
}

} // namespace op
} // namespace xyz_autodiff