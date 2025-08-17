#pragma once

#include "dense_matrix.cuh"
#include "concept/matrix.cuh"

namespace xyz_autodiff {

// MatrixView同士の行列乗算でDenseMatrixを返す
template <typename A, typename B>
requires concept::MatrixView<A> && concept::MatrixView<B> && (A::cols == B::rows)
__device__ auto operator*(const A& a, const B& b) {
    using ValueType = typename A::value_type;
    constexpr std::size_t ResultRows = A::rows;
    constexpr std::size_t ResultCols = B::cols;
    
    DenseMatrix<ValueType, ResultRows, ResultCols> result;
    
    // シングルスレッドでの行列乗算
    for (std::size_t i = 0; i < ResultRows; ++i) {
        auto active_cols_a = a.active_cols_in_row(i);
        
        for (std::size_t j = 0; j < ResultCols; ++j) {
            auto active_rows_b = b.active_rows_in_col(j);
            ValueType sum = 0;
            
            for (std::size_t k = 0; k < A::cols; ++k) {
                // 疎行列の最適化: 両方の要素がアクティブな場合のみ計算
                if (active_cols_a[k] && active_rows_b[k]) {
                    sum += a(i, k) * b(k, j);
                }
            }
            result(i, j) = sum;
        }
    }
    
    return result;
}

} // namespace xyz_autodiff