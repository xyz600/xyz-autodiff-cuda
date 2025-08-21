#pragma once

#include <cuda_runtime.h>
#include "../../concept/core_logic.cuh"
#include "../operation.cuh"

namespace xyz_autodiff {
namespace op {

// Generalized matrix multiplication: C = A * B
// A is a×b matrix, B is b×c matrix, C is a×c matrix
// Template parameters: a (rows of A), b (cols of A / rows of B), c (cols of B)
template <std::size_t a, std::size_t b, std::size_t c, typename Input1, typename Input2>
requires BinaryLogicParameterConcept<Input1, Input2> && 
         (Input1::size == a * b) && (Input2::size == b * c)
struct MatMulLogic {
    using T = typename Input1::value_type;
    static_assert(std::is_same_v<T, typename Input2::value_type>, "Input types must match");
    
    static constexpr std::size_t rows_A = a;
    static constexpr std::size_t cols_A_rows_B = b;
    static constexpr std::size_t cols_B = c;
    static constexpr std::size_t output_size = a * c;
    using Output = Variable<output_size, T>;
    
    static constexpr std::size_t outputDim = output_size;
    
    __host__ __device__ MatMulLogic() = default;
    
    // forward: C = A * B (a×b * b×c = a×c matrix multiplication)
    __device__ void forward(Output& output, const Input1& A, const Input2& B) const {
        // Matrix multiplication: C[i,j] = Σ A[i,k] * B[k,j]
        // All matrices stored in row-major order
        
        for (std::size_t i = 0; i < a; i++) {
            for (std::size_t j = 0; j < c; j++) {
                T sum = T(0);
                for (std::size_t k = 0; k < b; k++) {
                    sum += A[i * b + k] * B[k * c + j];
                }
                output[i * c + j] = sum;
            }
        }
    }
    
    // backward: gradient propagation for matrix multiplication
    __device__ void backward(const Output& output, Input1& A, Input2& B) const {
        // For C[i,j] = Σ A[i,k] * B[k,j]:
        // dC/dA[i,k] = B[k,j] for each j
        // dC/dB[k,j] = A[i,k] for each i
        
        for (std::size_t i = 0; i < a; i++) {
            for (std::size_t j = 0; j < c; j++) {
                const T& output_grad = output.grad(i * c + j);
                
                // Gradient w.r.t A: dL/dA[i,k] += dL/dC[i,j] * dC/dA[i,k]
                //                                 += dL/dC[i,j] * B[k,j]
                for (std::size_t k = 0; k < b; k++) {
                    A.add_grad(i * b + k, output_grad * B[k * c + j]);
                }
                
                // Gradient w.r.t B: dL/dB[k,j] += dL/dC[i,j] * dC/dB[k,j]
                //                                 += dL/dC[i,j] * A[i,k]
                for (std::size_t k = 0; k < b; k++) {
                    B.add_grad(k * c + j, output_grad * A[i * b + k]);
                }
            }
        }
    }
};

// Factory function for generalized matrix multiplication
template <std::size_t a, std::size_t b, std::size_t c, typename Input1, typename Input2>
requires BinaryLogicParameterConcept<Input1, Input2> && 
         (Input1::size == a * b) && (Input2::size == b * c)
__device__ auto matmul(Input1& A, Input2& B) {
    using LogicType = MatMulLogic<a, b, c, Input1, Input2>;
    
    LogicType logic;
    return BinaryOperation<LogicType::outputDim, LogicType, Input1, Input2>(logic, A, B);
}

// Convenience specializations for common matrix sizes

// 2x2 matrix multiplication
template <typename Input1, typename Input2>
requires BinaryLogicParameterConcept<Input1, Input2> && 
         (Input1::size == 4) && (Input2::size == 4)
__device__ auto matmul_2x2(Input1& A, Input2& B) {
    return matmul<2, 2, 2>(A, B);
}

// 3x3 matrix multiplication (backward compatibility)
template <typename Input1, typename Input2>
requires BinaryLogicParameterConcept<Input1, Input2> && 
         (Input1::size == 9) && (Input2::size == 9)
__device__ auto matmul_3x3(Input1& A, Input2& B) {
    return matmul<3, 3, 3>(A, B);
}

// 4x4 matrix multiplication
template <typename Input1, typename Input2>
requires BinaryLogicParameterConcept<Input1, Input2> && 
         (Input1::size == 16) && (Input2::size == 16)
__device__ auto matmul_4x4(Input1& A, Input2& B) {
    return matmul<4, 4, 4>(A, B);
}

// Matrix-vector multiplication: A(m×n) * x(n×1) = b(m×1)
template <std::size_t m, std::size_t n, typename Input1, typename Input2>
requires BinaryLogicParameterConcept<Input1, Input2> && 
         (Input1::size == m * n) && (Input2::size == n)
__device__ auto matvec(Input1& A, Input2& x) {
    return matmul<m, n, 1>(A, x);
}

// Vector-matrix multiplication: x(1×m) * A(m×n) = b(1×n)
template <std::size_t m, std::size_t n, typename Input1, typename Input2>
requires BinaryLogicParameterConcept<Input1, Input2> && 
         (Input1::size == m) && (Input2::size == m * n)
__device__ auto vecmat(Input1& x, Input2& A) {
    return matmul<1, m, n>(x, A);
}

} // namespace op
} // namespace xyz_autodiff