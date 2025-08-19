#pragma once

#include <cuda_runtime.h>
#include "../../include/concept/core_logic.cuh"
#include "../../include/operations/operation.cuh"

namespace xyz_autodiff {
namespace op {

// 3x3 matrix multiplication: C = A * B
template <typename Input1, typename Input2>
requires BinaryLogicParameterConcept<Input1, Input2> && 
         (Input1::size == 9) && (Input2::size == 9)
struct MatrixMultiplication3x3Logic {
    using T = typename Input1::value_type;
    static_assert(std::is_same_v<T, typename Input2::value_type>, "Input types must match");
    static constexpr std::size_t Dim = 9;
    using Output = Variable<T, Dim>;
    
    static constexpr std::size_t outputDim = Dim;
    
    __host__ __device__ MatrixMultiplication3x3Logic() = default;
    
    // forward: C = A * B (3x3 matrix multiplication)
    __device__ void forward(Output& output, const Input1& A, const Input2& B) const {
        // Matrix multiplication: C[i,j] = Σ A[i,k] * B[k,j]
        // A, B, C are stored in row-major order: [0,1,2,3,4,5,6,7,8] = [[0,1,2],[3,4,5],[6,7,8]]
        
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                T sum = T(0);
                for (int k = 0; k < 3; k++) {
                    sum += A[i * 3 + k] * B[k * 3 + j];
                }
                output[i * 3 + j] = sum;
            }
        }
    }
    
    // backward: gradient propagation for matrix multiplication
    __device__ void backward(const Output& output, Input1& A, Input2& B) const {
        // dC/dA[i,k] = B[k,j] for C[i,j]
        // dC/dB[k,j] = A[i,k] for C[i,j]
        
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                const T& output_grad = output.grad(i * 3 + j);
                
                // Gradient w.r.t A
                for (int k = 0; k < 3; k++) {
                    A.add_grad(i * 3 + k, output_grad * B[k * 3 + j]);
                }
                
                // Gradient w.r.t B
                for (int k = 0; k < 3; k++) {
                    B.add_grad(k * 3 + j, output_grad * A[i * 3 + k]);
                }
            }
        }
    }
};

// Factory function for 3x3 matrix multiplication
template <typename Input1, typename Input2>
requires BinaryLogicParameterConcept<Input1, Input2> && 
         (Input1::size == 9) && (Input2::size == 9)
__device__ auto matrix_multiply_3x3(Input1& A, Input2& B) {
    using LogicType = MatrixMultiplication3x3Logic<Input1, Input2>;
    
    LogicType logic;
    return BinaryOperation<LogicType::outputDim, LogicType, Input1, Input2>(logic, A, B);
}

} // namespace op
} // namespace xyz_autodiff