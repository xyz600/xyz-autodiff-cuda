#pragma once

#include <cstddef>
#include <cuda_runtime.h>
#include "../operation.cuh"
#include "../math.cuh"
#include "../../concept/variable.cuh"
#include "const_array_concepts.cuh"

namespace xyz_autodiff {
namespace op {

// Element-wise sub with constant array: output[i] = input[i] - constant_array[i]
template <std::size_t Dim, typename ConstantArray>
requires ArrayLikeConcept<ConstantArray>
struct ConstArraySubLogic {
    static constexpr std::size_t outputDim = Dim;
    
    const ConstantArray& constant_array;
    
    __host__ __device__ explicit ConstArraySubLogic(const ConstantArray& arr) : constant_array(arr) {}
    
    template <typename Output, typename Input>
    __host__ __device__ void forward(Output& output, const Input& input) const {
        for (std::size_t i = 0; i < Dim; ++i) {
            output[i] = input[i] - constant_array[i];
        }
    }
    
    template <typename Output, typename Input>
    __host__ __device__ void backward(const Output& output, Input& input) const {
        // d(input[i] - const[i])/d(input[i]) = 1
        for (std::size_t i = 0; i < Dim; ++i) {
            const auto& output_grad = output.grad(i);
            input.add_grad(i, output_grad);
        }
    }
};

// Factory function for element-wise sub with constant array
template <DifferentiableVariableConcept Input, typename ConstantArray>
requires ArrayLikeConcept<ConstantArray>
__host__ __device__ auto const_sub(Input& input, const ConstantArray& constant_array) {
    constexpr std::size_t Dim = Input::size;
    ConstArraySubLogic<Dim, ConstantArray> logic(constant_array);
    
    return UnaryOperation<Dim, ConstArraySubLogic<Dim, ConstantArray>, Input>(logic, input);
}

} // namespace op
} // namespace xyz_autodiff