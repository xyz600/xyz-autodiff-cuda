#pragma once

#include <cuda_runtime.h>
#include "../../concept/core_logic.cuh"
#include "../operation.cuh"

namespace xyz_autodiff {
namespace op {

// Logic for subtraction operation: output = input1 - input2
template <typename Input1, typename Input2>
requires BinaryLogicParameterConcept<Input1, Input2>
struct SubLogic {
    using T = typename Input1::value_type;
    static constexpr std::size_t Dim = Input1::size;
    using Output = Variable<T, Dim>;
    
    // Define output dimension as constexpr
    static constexpr std::size_t outputDim = Dim;
    
    // Static assertion for concept validation
    static_assert(BinaryLogicParameterConcept<Input1, Input2>, 
                  "Input1 and Input2 must satisfy BinaryLogicParameterConcept");
    
    // Forward: output = input1 - input2
    __device__ void forward(Output& output, const Input1& input1, const Input2& input2) const {
        for (std::size_t i = 0; i < Dim; ++i) {
            output[i] = input1[i] - input2[i];
        }
    }
    
    // Backward: 
    // ∂L/∂input1 = ∂L/∂output * 1
    // ∂L/∂input2 = ∂L/∂output * (-1)
    __device__ void backward(const Output& output, Input1& input1, Input2& input2) const {
        for (std::size_t i = 0; i < Dim; ++i) {
            const T& output_grad = output.grad(i);
            input1.add_grad(i, output_grad);      // gradient w.r.t. input1
            input2.add_grad(i, -output_grad);     // gradient w.r.t. input2
        }
    }
};

// Factory function for subtraction operation
template <typename Input1, typename Input2>
requires BinaryLogicParameterConcept<Input1, Input2>
__device__ auto sub(Input1& input1, Input2& input2) {
    using LogicType = SubLogic<Input1, Input2>;
    
    LogicType logic;
    return BinaryOperation<LogicType::outputDim, LogicType, Input1, Input2>(logic, input1, input2);
}

// Overload for subtracting a scalar constant from a variable
template <typename Input>
requires UnaryLogicParameterConcept<Input>
struct SubConstantLogic {
    using T = typename Input::value_type;
    static constexpr std::size_t Dim = Input::size;
    using Output = Variable<T, Dim>;
    
    static constexpr std::size_t outputDim = Dim;
    
    T constant_c;  // The constant to subtract
    
    __host__ __device__ explicit SubConstantLogic(T c) : constant_c(c) {}
    
    // Forward: output = input - c
    __device__ void forward(Output& output, const Input& input) const {
        for (std::size_t i = 0; i < Dim; ++i) {
            output[i] = input[i] - constant_c;
        }
    }
    
    // Backward: ∂L/∂input = ∂L/∂output
    __device__ void backward(const Output& output, Input& input) const {
        for (std::size_t i = 0; i < Dim; ++i) {
            const T& output_grad = output.grad(i);
            input.add_grad(i, output_grad);
        }
    }
};

// Factory function for subtracting a constant
template <typename Input>
requires UnaryLogicParameterConcept<Input>
__device__ auto sub_constant(Input& input, typename Input::value_type constant) {
    using LogicType = SubConstantLogic<Input>;
    
    LogicType logic(constant);
    return UnaryOperation<LogicType::outputDim, LogicType, Input>(logic, input);
}

} // namespace op
} // namespace xyz_autodiff