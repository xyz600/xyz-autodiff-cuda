#pragma once

#include <cstddef>
#include <cuda_runtime.h>
#include <xyz_autodiff/operations/operation.cuh>
#include <xyz_autodiff/operations/math.cuh>
#include <xyz_autodiff/concept/variable.cuh>

namespace xyz_autodiff {
namespace op {

// L1 norm operation: output = sum(|input[i]|)
template <std::size_t InputDim>
struct L1NormLogic {
    static constexpr std::size_t outputDim = 1;  // scalar output
    
    template <typename Output, typename Input>
    __host__ __device__ void forward(Output& output, const Input& input) const {
        using T = typename Input::value_type;
        T sum = T(0);
        for (std::size_t i = 0; i < InputDim; ++i) {
            sum += math::abs(input[i]);
        }
        output[0] = sum;
    }
    
    template <typename Output, typename Input>
    __host__ __device__ void backward(const Output& output, Input& input) const {
        using T = typename Input::value_type;
        const T& output_grad = output.grad(0);
        for (std::size_t i = 0; i < InputDim; ++i) {
            // d(|x|)/dx = sign(x)
            T sign = (input[i] > T(0)) ? T(1) : ((input[i] < T(0)) ? T(-1) : T(0));
            input.add_grad(i, output_grad * sign);
        }
    }
};

// L1 norm関数のファクトリ
template <std::size_t Dim, DifferentiableVariableConcept Input>
requires (Input::size == Dim)
__host__ __device__ auto l1_norm(Input& input) {
    L1NormLogic<Dim> logic;
    
    auto op = UnaryOperation<1, L1NormLogic<Dim>, Input>(logic, input);
    return op;
}

// 型推論をサポートする版（入力のサイズから自動的にDimを決定）
template <DifferentiableVariableConcept Input>
__host__ __device__ auto l1_norm(Input& input) {
    constexpr std::size_t Dim = Input::size;
    return l1_norm<Dim>(input);
}

} // namespace op
} // namespace xyz_autodiff