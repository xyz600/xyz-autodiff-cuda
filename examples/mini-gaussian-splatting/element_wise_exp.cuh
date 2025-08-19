#pragma once

#include <cuda_runtime.h>
#include <cmath>
#include "../../include/concept/core_logic.cuh"
#include "../../include/operations/operation.cuh"

namespace xyz_autodiff {
namespace op {

// Element-wise exponential operation: output[i] = exp(input[i])
template <typename Input>
requires UnaryLogicParameterConcept<Input>
struct ElementWiseExpLogic {
    using T = typename Input::value_type;
    static constexpr std::size_t Dim = Input::size;
    using Output = Variable<T, Dim>;
    
    static constexpr std::size_t outputDim = Dim;
    
    static_assert(UnaryLogicParameterConcept<Input>, "Input must satisfy UnaryLogicParameterConcept");
    
    __host__ __device__ ElementWiseExpLogic() = default;
    
    // forward: output[i] = exp(input[i])
    __device__ void forward(Output& output, const Input& input) const {
        for (std::size_t i = 0; i < Dim; ++i) {
            output[i] = exp(input[i]);
        }
    }
    
    // backward: d(exp(x))/dx = exp(x)
    __device__ void backward(const Output& output, Input& input) const {
        for (std::size_t i = 0; i < Dim; ++i) {
            const T& output_grad = output.grad(i);
            // derivative of exp(x) is exp(x), which is already computed in output[i]
            input.add_grad(i, output_grad * output[i]);
        }
    }
};

// Factory function for element-wise exp
template <typename Input>
requires UnaryLogicParameterConcept<Input>
__device__ auto element_wise_exp(Input& input) {
    using LogicType = ElementWiseExpLogic<Input>;
    
    LogicType logic;
    return UnaryOperation<LogicType::outputDim, LogicType, Input>(logic, input);
}

// Specialized exp(-x) operation
template <typename Input>
requires UnaryLogicParameterConcept<Input>
struct ElementWiseExpNegLogic {
    using T = typename Input::value_type;
    static constexpr std::size_t Dim = Input::size;
    using Output = Variable<T, Dim>;
    
    static constexpr std::size_t outputDim = Dim;
    
    static_assert(UnaryLogicParameterConcept<Input>, "Input must satisfy UnaryLogicParameterConcept");
    
    __host__ __device__ ElementWiseExpNegLogic() = default;
    
    // forward: output[i] = exp(-input[i])
    __device__ void forward(Output& output, const Input& input) const {
        for (std::size_t i = 0; i < Dim; ++i) {
            output[i] = exp(-input[i]);
        }
    }
    
    // backward: d(exp(-x))/dx = -exp(-x)
    __device__ void backward(const Output& output, Input& input) const {
        for (std::size_t i = 0; i < Dim; ++i) {
            const T& output_grad = output.grad(i);
            // derivative of exp(-x) is -exp(-x), which is -output[i]
            input.add_grad(i, output_grad * (-output[i]));
        }
    }
};

// Factory function for element-wise exp(-x)
template <typename Input>
requires UnaryLogicParameterConcept<Input>
__device__ auto element_wise_exp_neg(Input& input) {
    using LogicType = ElementWiseExpNegLogic<Input>;
    
    LogicType logic;
    return UnaryOperation<LogicType::outputDim, LogicType, Input>(logic, input);
}

} // namespace op
} // namespace xyz_autodiff