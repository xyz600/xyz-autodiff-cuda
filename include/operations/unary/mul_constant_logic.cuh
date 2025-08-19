#pragma once

#include <cuda_runtime.h>
#include "../../concept/core_logic.cuh"
#include "../operation.cuh"

namespace xyz_autodiff {
namespace op {

// Overload for multiplying a variable by a scalar constant
template <typename Input>
requires UnaryLogicParameterConcept<Input>
struct MulConstantLogic {
    using T = typename Input::value_type;
    static constexpr std::size_t Dim = Input::size;
    using Output = Variable<T, Dim>;
    
    static constexpr std::size_t outputDim = Dim;
    
    T constant_c;  // The constant to multiply
    
    __host__ __device__ explicit MulConstantLogic(T c) : constant_c(c) {}
    
    // Forward: output = input * c
    __device__ void forward(Output& output, const Input& input) const {
        for (std::size_t i = 0; i < Dim; ++i) {
            output[i] = input[i] * constant_c;
        }
    }
    
    // Backward: ∂L/∂input = ∂L/∂output * c
    __device__ void backward(const Output& output, Input& input) const {
        for (std::size_t i = 0; i < Dim; ++i) {
            const T& output_grad = output.grad(i);
            input.add_grad(i, output_grad * constant_c);
        }
    }
};

// Factory function for multiplying by a constant
template <typename Input>
requires UnaryLogicParameterConcept<Input>
__device__ auto mul_constant(Input& input, typename Input::value_type constant) {
    using LogicType = MulConstantLogic<Input>;
    
    LogicType logic(constant);
    return UnaryOperation<LogicType::outputDim, LogicType, Input>(logic, input);
}

} // namespace op
} // namespace xyz_autodiff