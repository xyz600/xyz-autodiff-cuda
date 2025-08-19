#pragma once

#include <cstddef>
#include <span>
#include <cuda_runtime.h>

#include "concept/matrix.cuh"

namespace xyz_autodiff {

template <typename T, std::size_t Rows, std::size_t Cols>
class DenseMatrix;

template <typename T, std::size_t Rows, std::size_t Cols>
class DenseMatrixTransposeView {
public:
    using value_type = T;
    static constexpr std::size_t rows = Cols;
    static constexpr std::size_t cols = Rows;
    static constexpr std::size_t size = Rows * Cols;
    
private:
    DenseMatrix<T, Rows, Cols>& original_matrix_;
    
public:
    __host__ __device__ constexpr DenseMatrixTransposeView(DenseMatrix<T, Rows, Cols>& matrix)
        : original_matrix_(matrix) {}
    
    // === Variable concept requirements ===
    
    // Data accessors
    __device__ T* data() { return original_matrix_.data(); }
    __device__ const T* data() const { return original_matrix_.data(); }
    
    // Gradient accessors
    __device__ T* grad() { return original_matrix_.grad(); }
    __device__ const T* grad() const { return original_matrix_.grad(); }
    
    // Index access (value) - 1D access with transposed linear indexing
    __device__ T& operator[](std::size_t i) {
        std::size_t row = i / cols;  // cols = original rows
        std::size_t col = i % cols;
        std::size_t original_idx = col * Cols + row;  // transpose indexing
        return original_matrix_[original_idx];
    }
    
    __device__ const T& operator[](std::size_t i) const {
        std::size_t row = i / cols;
        std::size_t col = i % cols;
        std::size_t original_idx = col * Cols + row;
        return original_matrix_[original_idx];
    }
    
    // Index access (gradient read-only)
    __device__ const T& grad(std::size_t i) const {
        std::size_t row = i / cols;
        std::size_t col = i % cols;
        std::size_t original_idx = col * Cols + row;
        return original_matrix_.grad(original_idx);
    }
    
    // Add to gradient
    __device__ void add_grad(std::size_t i, T value) {
        std::size_t row = i / cols;
        std::size_t col = i % cols;
        std::size_t original_idx = col * Cols + row;
        original_matrix_.add_grad(original_idx, value);
    }
    
    // Zero gradient (delegates to original matrix)
    __device__ void zero_grad() {
        original_matrix_.zero_grad();
    }
    
    // === MatrixView concept requirements ===
    
    // 2D access (value) with transposed coordinates
    __device__ T& operator()(std::size_t row, std::size_t col) {
        return original_matrix_(col, row);  // swap coordinates
    }
    
    __device__ const T& operator()(std::size_t row, std::size_t col) const {
        return original_matrix_(col, row);
    }
    
    // transpose function - returns view of original matrix (double transpose = identity)
    __host__ __device__ DenseMatrix<T, Rows, Cols>& transpose() const {
        return original_matrix_;
    }
};

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
    
    // インデックスアクセス (勾配の読み取り専用)
    __device__ const T& grad(std::size_t i) const { return grad_[i]; }
    
    // 勾配への加算
    __device__ void add_grad(std::size_t i, T value) { grad_[i] += value; }
    
    // 勾配をゼロクリア
    __device__ void zero_grad() {
        for (std::size_t i = 0; i < size; ++i) {
            grad_[i] = T{};
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
    
    // transpose機能 - Viewを返す（データを共有）
    __host__ __device__ DenseMatrixTransposeView<T, Rows, Cols> transpose() {
        return DenseMatrixTransposeView<T, Rows, Cols>(*this);
    }
    
    __host__ __device__ DenseMatrixTransposeView<T, Rows, Cols> transpose() const {
        return DenseMatrixTransposeView<T, Rows, Cols>(const_cast<DenseMatrix<T, Rows, Cols>&>(*this));
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
