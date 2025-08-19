#pragma once

#include <cstddef>
#include <cuda_runtime.h>
#include "../operation.cuh"
#include "../../concept/variable.cuh"

namespace xyz_autodiff {

// Broadcast operation: takes size-1 input and broadcasts to OutputDim
template <std::size_t OutputDim>
struct BroadcastLogic {
    static constexpr std::size_t outputDim = OutputDim;
    
    template <typename Output, typename Input>
    __host__ __device__ void forward(Output& output, const Input& input) const {
        static_assert(Input::size == 1, "Broadcast input must be size 1");
        
        // Broadcast the single input value to all output elements
        for (std::size_t i = 0; i < OutputDim; ++i) {
            output[i] = input[0];
        }
    }
    
    template <typename Output, typename Input>
    __host__ __device__ void backward(const Output& output, Input& input) const {
        static_assert(Input::size == 1, "Broadcast input must be size 1");
        
        // Sum all gradients from output back to single input
        // This is the correct gradient for broadcasting: d(broadcast)/dx = (1, 1, ..., 1)
        using T = typename Input::value_type;
        T grad_sum = T(0);
        for (std::size_t i = 0; i < OutputDim; ++i) {
            grad_sum += output.grad(i);
        }
        input.add_grad(0, grad_sum);
    }
};

// Factory function for broadcast operation
template <std::size_t OutputDim, DifferentiableVariableConcept Input>
requires (Input::size == 1)
__host__ __device__ auto broadcast(Input& input) {
    BroadcastLogic<OutputDim> logic;
    return UnaryOperation<OutputDim, BroadcastLogic<OutputDim>, Input>(logic, input);
}

} // namespace xyz_autodiff