#pragma once

#include <cuda_runtime.h>
#include "../../concept/core_logic.cuh"
#include "../operation.cuh"

namespace xyz_autodiff {

// Logic for squared operation: output = input^2
template <std::size_t Dim>
struct SquaredLogic {
    // Define output dimension as constexpr
    static constexpr std::size_t outputDim = Dim;
    
    template <typename Input>
    __device__ void forward(Variable<typename Input::value_type, Dim>& output, const Input& input) const {
        using T = typename Input::value_type;
        for (std::size_t i = 0; i < Dim; ++i) {
            T val = input[i];
            output[i] = val * val;
        }
    }
    
    template <typename Input>
    __device__ void backward(const Variable<typename Input::value_type, Dim>& output, Input& input) const {
        using T = typename Input::value_type;
        for (std::size_t i = 0; i < Dim; ++i) {
            const T& output_grad = output.grad(i);
            // d(x^2)/dx = 2x
            input.add_grad(i, output_grad * 2.0 * input[i]);
        }
    }
};

// Factory function for squared operation
template <typename Input>
requires UnaryLogicParameterConcept<Input>
__device__ auto squared(Input& input) {
    constexpr std::size_t Dim = Input::size;
    using LogicType = SquaredLogic<Dim>;
    
    LogicType logic;
    return UnaryOperation<Dim, LogicType, Input>(logic, input);
}

} // namespace xyz_autodiff