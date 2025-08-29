#pragma once

#include <cuda_runtime.h>
#include <xyz_autodiff/concept/core_logic.cuh>
#include <xyz_autodiff/operations/operation.cuh>

namespace xyz_autodiff {
namespace op {

// Overload for adding a scalar constant to a variable
template <typename Input>
requires UnaryLogicParameterConcept<Input>
struct AddConstantLogic {
    using T = typename Input::value_type;
    static constexpr std::size_t Dim = Input::size;
    using Output = Variable<Dim, T>;
    
    static constexpr std::size_t outputDim = Dim;
    
    T constant_c;  // The constant to add
    
    __host__ __device__ explicit AddConstantLogic(T c) : constant_c(c) {}
    
    // Forward: output = input + c
    __device__ void forward(Output& output, const Input& input) const {
        for (std::size_t i = 0; i < Dim; ++i) {
            output[i] = input[i] + constant_c;
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

// Factory function for adding a constant
template <typename Input>
requires UnaryLogicParameterConcept<Input>
__device__ auto add_constant(Input& input, typename Input::value_type constant) {
    using LogicType = AddConstantLogic<Input>;
    
    LogicType logic(constant);
    return UnaryOperation<LogicType::outputDim, LogicType, Input>(logic, input);
}

} // namespace op
} // namespace xyz_autodiff