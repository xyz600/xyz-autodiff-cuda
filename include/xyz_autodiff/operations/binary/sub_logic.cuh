#pragma once

#include <cuda_runtime.h>
#include <xyz_autodiff/concept/core_logic.cuh>
#include <xyz_autodiff/operations/operation.cuh>

namespace xyz_autodiff {
namespace op {

// Logic for subtraction operation: output = input1 - input2
template <typename Input1, typename Input2>
requires BinaryLogicParameterConcept<Input1, Input2>
struct SubLogic {
    using T = typename Input1::value_type;
    static constexpr std::size_t Dim = Input1::size;
    using Output = Variable<Dim, T>;
    
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

} // namespace op
} // namespace xyz_autodiff