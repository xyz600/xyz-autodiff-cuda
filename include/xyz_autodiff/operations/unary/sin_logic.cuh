#pragma once

#include <cstddef>
#include <cuda_runtime.h>
#include <xyz_autodiff/operations/operation.cuh>
#include <xyz_autodiff/operations/math.cuh>
#include <xyz_autodiff/concept/variable.cuh>

namespace xyz_autodiff {
namespace op {

// Sin関数のロジック実装
template <std::size_t Dim>
struct SinLogic {
    static constexpr std::size_t outputDim = Dim;
    
    template <typename Output, typename Input>
    __host__ __device__ void forward(Output& output, const Input& input) const {
        for (std::size_t i = 0; i < outputDim; ++i) {
            // sin(x)
            using T = typename Input::value_type;
            output[i] = math::sin(input[i]);
        }
    }
    
    template <typename Output, typename Input>
    __host__ __device__ void backward(const Output& output, Input& input) const {
        for (std::size_t i = 0; i < outputDim; ++i) {
            // d/dx sin(x) = cos(x)
            using T = typename Input::value_type;
            T cos_val = math::cos(input[i]);
            input.add_grad(i, output.grad(i) * cos_val);
        }
    }
};

// Sin関数のファクトリ
template <std::size_t Dim, DifferentiableVariableConcept Input>
__host__ __device__ auto sin(Input& input) {
    SinLogic<Dim> logic;
    
    auto op = UnaryOperation<Dim, SinLogic<Dim>, Input>(logic, input);
    return op;
}

// 型推論をサポートする版（入力のサイズから自動的にDimを決定）
template <DifferentiableVariableConcept Input>
__host__ __device__ auto sin(Input& input) {
    constexpr std::size_t Dim = Input::size;
    return sin<Dim>(input);
}

} // namespace op
} // namespace xyz_autodiff