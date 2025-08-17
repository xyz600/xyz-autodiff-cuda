#pragma once

#include <cstddef>
#include <span>
#include <cuda_runtime.h>

#include "concept/matrix.cuh"
#include "matrix_view.cuh"

namespace xyz_autodiff {

template <typename T, std::size_t Rows, std::size_t Cols>
class DenseMatrix {
public:
    using value_type = T;
    static constexpr std::size_t rows = Rows;
    static constexpr std::size_t cols = Cols;
    static constexpr std::size_t size = Rows * Cols;
    
private:
    T data_[Rows * Cols];
    T grad_[Rows * Cols];
    
public:
    // デフォルトコンストラクタ
    __host__ __device__ constexpr DenseMatrix() {
        for (std::size_t i = 0; i < size; ++i) {
            data_[i] = T{};
            grad_[i] = T{};
        }
    }
    
    // === Variable concept の要件 ===
    
    // データアクセサ
    __device__ T* data() { return data_; }
    __device__ const T* data() const { return data_; }
    
    // 勾配アクセサ
    __device__ T* grad() { return grad_; }
    __device__ const T* grad() const { return grad_; }
    
    // インデックスアクセス (値) - 1次元アクセス
    __device__ T& operator[](std::size_t i) { return data_[i]; }
    __device__ const T& operator[](std::size_t i) const { return data_[i]; }
    
    // インデックスアクセス (勾配) - 1次元アクセス
    __device__ T& grad(std::size_t i) { return grad_[i]; }
    __device__ const T& grad(std::size_t i) const { return grad_[i]; }
    
    // 勾配をゼロクリア
    __device__ void zero_grad() {
        for (std::size_t i = 0; i < size; ++i) {
            grad_[i] = T{};
        }
    }
    
    // 勾配を累積
    __device__ void accumulate_grad(const T* grad_values) {
        for (std::size_t i = 0; i < size; ++i) {
            grad_[i] += grad_values[i];
        }
    }
    
    // === MatrixView concept の要件 ===
    
    // 2次元アクセス (値)
    __device__ T& operator()(std::size_t row, std::size_t col) {
        return data_[row * cols + col];
    }
    
    __device__ const T& operator()(std::size_t row, std::size_t col) const {
        return data_[row * cols + col];
    }
    
    // transpose機能 - 新しいDenseMatrixを返す（データをコピー）
    __host__ __device__ DenseMatrix<T, Cols, Rows> transpose() const {
        DenseMatrix<T, Cols, Rows> result;
        for (std::size_t i = 0; i < Rows; ++i) {
            for (std::size_t j = 0; j < Cols; ++j) {
                result(j, i) = (*this)(i, j);
            }
        }
        return result;
    }
    
};

// 再帰テンプレートによるconstexprループ実装（疎行列サポート削除版）
template <typename A, typename B, std::size_t K, std::size_t MaxK>
__device__ constexpr typename A::value_type matrix_multiply_inner_impl(const A& a, const B& b, std::size_t i, std::size_t j) {
    using ValueType = typename A::value_type;
    
    if constexpr (K >= MaxK) {
        return ValueType{0};
    } else {
        ValueType current = a(i, K) * b(K, j);
        return current + matrix_multiply_inner_impl<A, B, K + 1, MaxK>(a, b, i, j);
    }
}

template <typename A, typename B, typename Result, std::size_t I, std::size_t J, std::size_t MaxI, std::size_t MaxJ>
__device__ constexpr void matrix_multiply_outer_impl(const A& a, const B& b, Result& result) {
    if constexpr (I < MaxI) {
        if constexpr (J < MaxJ) {
            constexpr std::size_t K_Max = A::cols;
            result[I * MaxJ + J] = matrix_multiply_inner_impl<A, B, 0, K_Max>(a, b, I, J);
            matrix_multiply_outer_impl<A, B, Result, I, J + 1, MaxI, MaxJ>(a, b, result);
        } else {
            matrix_multiply_outer_impl<A, B, Result, I + 1, 0, MaxI, MaxJ>(a, b, result);
        }
    }
}

// MatrixView同士の行列乗算でDenseMatrixを返す
template <typename A, typename B>
requires MatrixViewConcept<A> && MatrixViewConcept<B> && (A::cols == B::rows)
__device__ constexpr DenseMatrix<typename A::value_type, A::rows, B::cols> operator*(const A& a, const B& b) {
    using ValueType = typename A::value_type;
    constexpr std::size_t ResultRows = A::rows;
    constexpr std::size_t ResultCols = B::cols;
    
    DenseMatrix<ValueType, ResultRows, ResultCols> result;
    
    // constexpr再帰テンプレートによる行列乗算
    matrix_multiply_outer_impl<A, B, DenseMatrix<ValueType, ResultRows, ResultCols>, 0, 0, ResultRows, ResultCols>(a, b, result);
   
    return result;
}

} // namespace xyz_autodiff
