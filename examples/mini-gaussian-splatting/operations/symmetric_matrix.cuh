#pragma once

// NOTE: Standard 2x2 symmetric matrix inverse is now handled by include/operations/unary/sym_matrix2_inv_logic.cuh
// This file contains only specialized matrix operations for mini-gaussian-splatting

#include <cuda_runtime.h>
#include <cmath>
#include "../../include/concept/core_logic.cuh"
#include "../../include/operations/operation.cuh"
#include "../../include/operations/unary/sym_matrix2_inv_logic.cuh"

namespace xyz_autodiff {
namespace op {

// Convert 3x3 matrix to 3-parameter symmetric representation for 2x2 covariance
// Input: 3x3 matrix, Output: 3 parameters [a, b, c] representing [[a, b], [b, c]]
template <typename Input>
requires UnaryLogicParameterConcept<Input> && (Input::size == 9)
struct MatrixToSymmetric3ParamLogic {
    using T = typename Input::value_type;
    static constexpr std::size_t Dim = 3;  // 3 parameters for 2x2 symmetric matrix
    using Output = Variable<Dim, T>;
    
    static constexpr std::size_t outputDim = Dim;
    
    static_assert(UnaryLogicParameterConcept<Input>, "Input must satisfy UnaryLogicParameterConcept");
    
    __host__ __device__ MatrixToSymmetric3ParamLogic() = default;
    
    // forward: Extract upper-left 2x2 symmetric part
    // Matrix layout: [0,1,2,3,4,5,6,7,8] = [[0,1,2],[3,4,5],[6,7,8]]
    // Extract: [[0,1],[3,4]] and symmetrize to [[0, (1+3)/2], [(1+3)/2, 4]]
    __device__ void forward(Output& output, const Input& input) const {
        output[0] = input[0];  // a = M[0,0]
        output[1] = (input[1] + input[3]) / T(2);  // b = (M[0,1] + M[1,0]) / 2
        output[2] = input[4];  // c = M[1,1]
    }
    
    // backward: Gradient propagation
    __device__ void backward(const Output& output, Input& input) const {
        const T& grad_a = output.grad(0);
        const T& grad_b = output.grad(1);
        const T& grad_c = output.grad(2);
        
        input.add_grad(0, grad_a);  // d/dM[0,0]
        input.add_grad(1, grad_b / T(2));  // d/dM[0,1]
        input.add_grad(3, grad_b / T(2));  // d/dM[1,0]
        input.add_grad(4, grad_c);  // d/dM[1,1]
        // Other matrix elements have zero gradient
    }
};

// Convert 3-parameter symmetric representation to 2x2 matrix
// Input: 3 parameters [a, b, c], Output: 4 elements [a, b, b, c]
template <typename Input>
requires UnaryLogicParameterConcept<Input> && (Input::size == 3)
struct Symmetric3ParamToMatrixLogic {
    using T = typename Input::value_type;
    static constexpr std::size_t Dim = 4;  // 2x2 matrix
    using Output = Variable<Dim, T>;
    
    static constexpr std::size_t outputDim = Dim;
    
    static_assert(UnaryLogicParameterConcept<Input>, "Input must satisfy UnaryLogicParameterConcept");
    
    __host__ __device__ Symmetric3ParamToMatrixLogic() = default;
    
    // forward: [a, b, c] -> [a, b, b, c]
    __device__ void forward(Output& output, const Input& input) const {
        output[0] = input[0];  // a
        output[1] = input[1];  // b
        output[2] = input[1];  // b
        output[3] = input[2];  // c
    }
    
    // backward: Gradient propagation
    __device__ void backward(const Output& output, Input& input) const {
        input.add_grad(0, output.grad(0));  // grad_a
        input.add_grad(1, output.grad(1) + output.grad(2));  // grad_b (from both positions)
        input.add_grad(2, output.grad(3));  // grad_c
    }
};

// Factory functions
template <typename Input>
requires UnaryLogicParameterConcept<Input> && (Input::size == 9)
__device__ auto matrix_to_symmetric_3param(Input& input) {
    using LogicType = MatrixToSymmetric3ParamLogic<Input>;
    LogicType logic;
    return UnaryOperation<LogicType::outputDim, LogicType, Input>(logic, input);
}

template <typename Input>
requires UnaryLogicParameterConcept<Input> && (Input::size == 3)
__device__ auto symmetric_3param_to_matrix(Input& input) {
    using LogicType = Symmetric3ParamToMatrixLogic<Input>;
    LogicType logic;
    return UnaryOperation<LogicType::outputDim, LogicType, Input>(logic, input);
}

// Use standard implementation from include/operations/unary/sym_matrix2_inv_logic.cuh
template <typename Input>
requires UnaryLogicParameterConcept<Input> && (Input::size == 3)
__device__ auto symmetric_matrix_2x2_inverse(Input& input) {
    return sym_matrix2_inv(input);
}

} // namespace op
} // namespace xyz_autodiff