#pragma once

#include <cuda_runtime.h>
#include <cmath>
#include "../../include/concept/core_logic.cuh"
#include "../../include/operations/operation.cuh"

namespace xyz_autodiff {
namespace op {

// L1 norm operation: output = sum(|input[i]|)
template <typename Input>
requires UnaryLogicParameterConcept<Input>
struct L1NormLogic {
    using T = typename Input::value_type;
    static constexpr std::size_t Dim = Input::size;
    using Output = Variable<T, 1>;  // scalar output
    
    static constexpr std::size_t outputDim = 1;
    
    static_assert(UnaryLogicParameterConcept<Input>, "Input must satisfy UnaryLogicParameterConcept");
    
    __host__ __device__ L1NormLogic() = default;
    
    // forward: output = sum(|input[i]|)
    __device__ void forward(Output& output, const Input& input) const {
        T sum = T(0);
        for (std::size_t i = 0; i < Dim; ++i) {
            sum += abs(input[i]);
        }
        output[0] = sum;
    }
    
    // backward: d(|x|)/dx = sign(x)
    __device__ void backward(const Output& output, Input& input) const {
        const T& output_grad = output.grad(0);
        for (std::size_t i = 0; i < Dim; ++i) {
            T sign = (input[i] > T(0)) ? T(1) : ((input[i] < T(0)) ? T(-1) : T(0));
            input.add_grad(i, output_grad * sign);
        }
    }
};

// L2 norm operation: output = sqrt(sum(input[i]^2))
template <typename Input>
requires UnaryLogicParameterConcept<Input>
struct L2NormLogic {
    using T = typename Input::value_type;
    static constexpr std::size_t Dim = Input::size;
    using Output = Variable<T, 1>;  // scalar output
    
    static constexpr std::size_t outputDim = 1;
    
    static_assert(UnaryLogicParameterConcept<Input>, "Input must satisfy UnaryLogicParameterConcept");
    
    __host__ __device__ L2NormLogic() = default;
    
    // forward: output = sqrt(sum(input[i]^2))
    __device__ void forward(Output& output, const Input& input) const {
        T sum_squares = T(0);
        for (std::size_t i = 0; i < Dim; ++i) {
            sum_squares += input[i] * input[i];
        }
        output[0] = sqrt(sum_squares);
    }
    
    // backward: d(sqrt(sum(x_i^2)))/dx_i = x_i / sqrt(sum(x_i^2))
    __device__ void backward(const Output& output, Input& input) const {
        const T& output_grad = output.grad(0);
        const T& norm_value = output[0];
        
        if (norm_value > T(1e-8)) {  // avoid division by zero
            for (std::size_t i = 0; i < Dim; ++i) {
                input.add_grad(i, output_grad * input[i] / norm_value);
            }
        }
    }
};

// L2 squared norm operation: output = sum(input[i]^2)
template <typename Input>
requires UnaryLogicParameterConcept<Input>
struct L2SquaredNormLogic {
    using T = typename Input::value_type;
    static constexpr std::size_t Dim = Input::size;
    using Output = Variable<T, 1>;  // scalar output
    
    static constexpr std::size_t outputDim = 1;
    
    static_assert(UnaryLogicParameterConcept<Input>, "Input must satisfy UnaryLogicParameterConcept");
    
    __host__ __device__ L2SquaredNormLogic() = default;
    
    // forward: output = sum(input[i]^2)
    __device__ void forward(Output& output, const Input& input) const {
        T sum_squares = T(0);
        for (std::size_t i = 0; i < Dim; ++i) {
            sum_squares += input[i] * input[i];
        }
        output[0] = sum_squares;
    }
    
    // backward: d(sum(x_i^2))/dx_i = 2*x_i
    __device__ void backward(const Output& output, Input& input) const {
        const T& output_grad = output.grad(0);
        for (std::size_t i = 0; i < Dim; ++i) {
            input.add_grad(i, output_grad * T(2) * input[i]);
        }
    }
};

// Factory function for L1 norm
template <typename Input>
requires UnaryLogicParameterConcept<Input>
__device__ auto l1_norm(Input& input) {
    using LogicType = L1NormLogic<Input>;
    
    LogicType logic;
    return UnaryOperation<LogicType::outputDim, LogicType, Input>(logic, input);
}

// Factory function for L2 norm
template <typename Input>
requires UnaryLogicParameterConcept<Input>
__device__ auto l2_norm(Input& input) {
    using LogicType = L2NormLogic<Input>;
    
    LogicType logic;
    return UnaryOperation<LogicType::outputDim, LogicType, Input>(logic, input);
}

// Factory function for L2 squared norm
template <typename Input>
requires UnaryLogicParameterConcept<Input>
__device__ auto l2_squared_norm(Input& input) {
    using LogicType = L2SquaredNormLogic<Input>;
    
    LogicType logic;
    return UnaryOperation<LogicType::outputDim, LogicType, Input>(logic, input);
}

} // namespace op
} // namespace xyz_autodiff