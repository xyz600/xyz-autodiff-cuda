#pragma once

// This file now provides compatibility aliases for the generalized matrix multiplication
#include "../../include/operations/binary/matmul_logic.cuh"

namespace xyz_autodiff {
namespace op {

// Compatibility alias for 3x3 matrix multiplication
template <typename Input1, typename Input2>
requires BinaryLogicParameterConcept<Input1, Input2> && 
         (Input1::size == 9) && (Input2::size == 9)
using MatrixMultiplication3x3Logic = MatMulLogic<3, 3, 3, Input1, Input2>;

// Compatibility factory function for 3x3 matrix multiplication
template <typename Input1, typename Input2>
requires BinaryLogicParameterConcept<Input1, Input2> && 
         (Input1::size == 9) && (Input2::size == 9)
__device__ auto matrix_multiply_3x3(Input1& A, Input2& B) {
    return matmul<3, 3, 3>(A, B);
}

} // namespace op
} // namespace xyz_autodiff