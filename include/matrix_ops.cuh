#pragma once

#include "dense_matrix.cuh"
#include "concept/matrix.cuh"

namespace xyz_autodiff {

// MatrixView同士の行列乗算でDenseMatrixを返す
template <typename A, typename B>
requires MatrixView<A> && MatrixView<B> && (A::cols == B::rows)
__device__ auto operator*(const A& a, const B& b) {
    using ValueType = typename A::value_type;
    constexpr std::size_t ResultRows = A::rows;
    constexpr std::size_t ResultCols = B::cols;
    
    DenseMatrix<ValueType, ResultRows, ResultCols> result;
    
    // シングルスレッドでの行列乗算
    for (std::size_t i = 0; i < ResultRows; ++i) {
        for (std::size_t j = 0; j < ResultCols; ++j) {
            ValueType sum = 0;
            
            for (std::size_t k = 0; k < A::cols; ++k) {
                // 疎行列の最適化: 両方の要素がアクティブな場合のみ計算
                if (a.is_active_in_row(i, k) && b.is_active_in_col(k, j)) {
                    sum += a(i, k) * b(k, j);
                }
            }
            result(i, j) = sum;
        }
    }
   
    return result;
}

} // namespace xyz_autodiff