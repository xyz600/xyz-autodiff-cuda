#pragma once

#include <cuda_runtime.h>
#include <cmath>
#include "../../include/concept/core_logic.cuh"
#include "../../include/operations/operation.cuh"

namespace xyz_autodiff {
namespace op {

// L1 + L2 norm regularization: output = ||input||_1 + ||input||_2
template <typename Input>
requires UnaryLogicParameterConcept<Input>
struct L1PlusL2NormLogic {
    using T = typename Input::value_type;
    static constexpr std::size_t Dim = Input::size;
    using Output = Variable<T, 1>;  // scalar output
    
    static constexpr std::size_t outputDim = 1;
    
    static_assert(UnaryLogicParameterConcept<Input>, "Input must satisfy UnaryLogicParameterConcept");
    
    __host__ __device__ L1PlusL2NormLogic() = default;
    
    // forward: output = sum(|input[i]|) + sqrt(sum(input[i]^2))
    __device__ void forward(Output& output, const Input& input) const {
        T l1_sum = T(0);
        T l2_sum_squares = T(0);
        
        for (std::size_t i = 0; i < Dim; ++i) {
            l1_sum += abs(input[i]);
            l2_sum_squares += input[i] * input[i];
        }
        
        T l2_norm = sqrt(l2_sum_squares);
        output[0] = l1_sum + l2_norm;
    }
    
    // backward: d(||x||_1 + ||x||_2)/dx_i = sign(x_i) + x_i/||x||_2
    __device__ void backward(const Output& output, Input& input) const {
        const T& output_grad = output.grad(0);
        
        // Compute L2 norm for gradient calculation
        T l2_sum_squares = T(0);
        for (std::size_t i = 0; i < Dim; ++i) {
            l2_sum_squares += input[i] * input[i];
        }
        T l2_norm = sqrt(l2_sum_squares);
        
        // Avoid division by zero for L2 gradient
        if (l2_norm < T(1e-8)) {
            l2_norm = T(1e-8);
        }
        
        for (std::size_t i = 0; i < Dim; ++i) {
            // L1 gradient: sign(x_i)
            T l1_grad = (input[i] > T(0)) ? T(1) : ((input[i] < T(0)) ? T(-1) : T(0));
            
            // L2 gradient: x_i / ||x||_2
            T l2_grad = input[i] / l2_norm;
            
            // Combined gradient
            input.add_grad(i, output_grad * (l1_grad + l2_grad));
        }
    }
};

// Weighted L1 + L2 norm: output = λ1 * ||input||_1 + λ2 * ||input||_2
template <typename Input>
requires UnaryLogicParameterConcept<Input>
struct WeightedL1PlusL2NormLogic {
    using T = typename Input::value_type;
    static constexpr std::size_t Dim = Input::size;
    using Output = Variable<T, 1>;  // scalar output
    
    static constexpr std::size_t outputDim = 1;
    
    static_assert(UnaryLogicParameterConcept<Input>, "Input must satisfy UnaryLogicParameterConcept");
    
    T lambda1, lambda2;  // weights for L1 and L2 norms
    
    __host__ __device__ explicit WeightedL1PlusL2NormLogic(T l1_weight, T l2_weight) 
        : lambda1(l1_weight), lambda2(l2_weight) {}
    
    // forward: output = λ1 * sum(|input[i]|) + λ2 * sqrt(sum(input[i]^2))
    __device__ void forward(Output& output, const Input& input) const {
        T l1_sum = T(0);
        T l2_sum_squares = T(0);
        
        for (std::size_t i = 0; i < Dim; ++i) {
            l1_sum += abs(input[i]);
            l2_sum_squares += input[i] * input[i];
        }
        
        T l2_norm = sqrt(l2_sum_squares);
        output[0] = lambda1 * l1_sum + lambda2 * l2_norm;
    }
    
    // backward: d(λ1*||x||_1 + λ2*||x||_2)/dx_i = λ1*sign(x_i) + λ2*x_i/||x||_2
    __device__ void backward(const Output& output, Input& input) const {
        const T& output_grad = output.grad(0);
        
        // Compute L2 norm for gradient calculation
        T l2_sum_squares = T(0);
        for (std::size_t i = 0; i < Dim; ++i) {
            l2_sum_squares += input[i] * input[i];
        }
        T l2_norm = sqrt(l2_sum_squares);
        
        // Avoid division by zero for L2 gradient
        if (l2_norm < T(1e-8)) {
            l2_norm = T(1e-8);
        }
        
        for (std::size_t i = 0; i < Dim; ++i) {
            // L1 gradient: λ1 * sign(x_i)
            T l1_grad = lambda1 * ((input[i] > T(0)) ? T(1) : ((input[i] < T(0)) ? T(-1) : T(0)));
            
            // L2 gradient: λ2 * x_i / ||x||_2
            T l2_grad = lambda2 * input[i] / l2_norm;
            
            // Combined gradient
            input.add_grad(i, output_grad * (l1_grad + l2_grad));
        }
    }
};

// Scalar addition: output = input1 + input2 (for scalar inputs)
template <typename Input1, typename Input2>
requires BinaryLogicParameterConcept<Input1, Input2> && 
         (Input1::size == 1) && (Input2::size == 1)
struct ScalarAdditionLogic {
    using T = typename Input1::value_type;
    static_assert(std::is_same_v<T, typename Input2::value_type>, "Input types must match");
    static constexpr std::size_t Dim = 1;
    using Output = Variable<T, Dim>;
    
    static constexpr std::size_t outputDim = Dim;
    
    __host__ __device__ ScalarAdditionLogic() = default;
    
    // forward: output = input1 + input2
    __device__ void forward(Output& output, const Input1& input1, const Input2& input2) const {
        output[0] = input1[0] + input2[0];
    }
    
    // backward: d(a+b)/da = 1, d(a+b)/db = 1
    __device__ void backward(const Output& output, Input1& input1, Input2& input2) const {
        const T& output_grad = output.grad(0);
        input1.add_grad(0, output_grad);
        input2.add_grad(0, output_grad);
    }
};

// Factory function for L1 + L2 norm
template <typename Input>
requires UnaryLogicParameterConcept<Input>
__device__ auto l1_plus_l2_norm(Input& input) {
    using LogicType = L1PlusL2NormLogic<Input>;
    
    LogicType logic;
    return UnaryOperation<LogicType::outputDim, LogicType, Input>(logic, input);
}

// Factory function for weighted L1 + L2 norm
template <typename Input>
requires UnaryLogicParameterConcept<Input>
__device__ auto weighted_l1_plus_l2_norm(Input& input, typename Input::value_type l1_weight, typename Input::value_type l2_weight) {
    using LogicType = WeightedL1PlusL2NormLogic<Input>;
    
    LogicType logic(l1_weight, l2_weight);
    return UnaryOperation<LogicType::outputDim, LogicType, Input>(logic, input);
}

// Factory function for scalar addition
template <typename Input1, typename Input2>
requires BinaryLogicParameterConcept<Input1, Input2> && 
         (Input1::size == 1) && (Input2::size == 1)
__device__ auto scalar_add(Input1& input1, Input2& input2) {
    using LogicType = ScalarAdditionLogic<Input1, Input2>;
    
    LogicType logic;
    return BinaryOperation<LogicType::outputDim, LogicType, Input1, Input2>(logic, input1, input2);
}

} // namespace op
} // namespace xyz_autodiff