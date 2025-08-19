#pragma once

#include <cuda_runtime.h>
#include "../../include/concept/core_logic.cuh"
#include "../../include/operations/operation.cuh"

namespace xyz_autodiff {
namespace op {

// NOTE: Standard 2-input element-wise multiplication is now handled by include/operations/binary/mul_logic.cuh
// This file contains only specialized multiplication operations for mini-gaussian-splatting

// Element-wise multiplication for three inputs: output[i] = input1[i] * input2[i] * input3[i]
template <typename Input1, typename Input2, typename Input3>
requires TernaryLogicParameterConcept<Input1, Input2, Input3> && 
         (Input1::size == Input2::size) && (Input2::size == Input3::size)
struct ElementWiseMultiply3Logic {
    using T = typename Input1::value_type;
    static_assert(std::is_same_v<T, typename Input2::value_type>, "Input types must match");
    static_assert(std::is_same_v<T, typename Input3::value_type>, "Input types must match");
    static constexpr std::size_t Dim = Input1::size;
    using Output = Variable<T, Dim>;
    
    static constexpr std::size_t outputDim = Dim;
    
    __host__ __device__ ElementWiseMultiply3Logic() = default;
    
    // forward: output[i] = input1[i] * input2[i] * input3[i]
    __device__ void forward(Output& output, const Input1& input1, const Input2& input2, const Input3& input3) const {
        for (std::size_t i = 0; i < Dim; ++i) {
            output[i] = input1[i] * input2[i] * input3[i];
        }
    }
    
    // backward: d(abc)/da = bc, d(abc)/db = ac, d(abc)/dc = ab
    __device__ void backward(const Output& output, Input1& input1, Input2& input2, Input3& input3) const {
        for (std::size_t i = 0; i < Dim; ++i) {
            const T& output_grad = output.grad(i);
            input1.add_grad(i, output_grad * input2[i] * input3[i]);
            input2.add_grad(i, output_grad * input1[i] * input3[i]);
            input3.add_grad(i, output_grad * input1[i] * input2[i]);
        }
    }
};

// Scalar multiplication: output[i] = scalar * input[i]
template <typename Input>
requires UnaryLogicParameterConcept<Input>
struct ScalarMultiplyLogic {
    using T = typename Input::value_type;
    static constexpr std::size_t Dim = Input::size;
    using Output = Variable<T, Dim>;
    
    static constexpr std::size_t outputDim = Dim;
    
    T scalar_value;
    
    __host__ __device__ explicit ScalarMultiplyLogic(T scalar) : scalar_value(scalar) {}
    
    // forward: output[i] = scalar * input[i]
    __device__ void forward(Output& output, const Input& input) const {
        for (std::size_t i = 0; i < Dim; ++i) {
            output[i] = scalar_value * input[i];
        }
    }
    
    // backward: d(s*x)/dx = s
    __device__ void backward(const Output& output, Input& input) const {
        for (std::size_t i = 0; i < Dim; ++i) {
            const T& output_grad = output.grad(i);
            input.add_grad(i, output_grad * scalar_value);
        }
    }
};

// NOTE: 2-input element_wise_multiply is now available as op::mul() from include/operations/binary/mul_logic.cuh

// Factory function for element-wise multiplication (3 inputs) - c * d * o operation
template <typename Input1, typename Input2, typename Input3>
requires TernaryLogicParameterConcept<Input1, Input2, Input3> && 
         (Input1::size == Input2::size) && (Input2::size == Input3::size)
__device__ auto element_wise_multiply_3(Input1& input1, Input2& input2, Input3& input3) {
    using LogicType = ElementWiseMultiply3Logic<Input1, Input2, Input3>;
    
    LogicType logic;
    return TernaryOperation<LogicType::outputDim, LogicType, Input1, Input2, Input3>(logic, input1, input2, input3);
}

// Factory function for scalar multiplication
template <typename Input>
requires UnaryLogicParameterConcept<Input>
__device__ auto scalar_multiply(Input& input, typename Input::value_type scalar) {
    using LogicType = ScalarMultiplyLogic<Input>;
    
    LogicType logic(scalar);
    return UnaryOperation<LogicType::outputDim, LogicType, Input>(logic, input);
}

} // namespace op
} // namespace xyz_autodiff