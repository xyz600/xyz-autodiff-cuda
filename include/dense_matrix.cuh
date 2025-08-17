#pragma once

#include <cstddef>
#include <span>
#include <cuda_runtime.h>

#include "concept/matrix.cuh"

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
    __host__ __device__ DenseMatrix() {
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
    __device__ T operator()(std::size_t row, std::size_t col) {
        return data_[row * cols + col];
    }
    
    __device__ T operator()(std::size_t row, std::size_t col) const {
        return data_[row * cols + col];
    }
    
    // 疎行列サポート (密行列なので全て有効)
    __device__ bool is_active_cell(std::size_t row, std::size_t col) const {
        return true;
    }
};

// MatrixView同士の行列乗算でDenseMatrixを返す
template <typename A, typename B>
requires MatrixViewConcept<A> && MatrixViewConcept<B> && (A::cols == B::rows)
__device__ DenseMatrix<typename A::value_type, A::rows, B::cols> operator*(const A& a, const B& b) {
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
                if (a.is_active_cell(i, k) && b.is_active_cell(k, j)) {
                    sum += a(i, k) * b(k, j);
                }
            }
            result[i * ResultCols + j] = sum;
        }
    }
   
    return result;
}

} // namespace xyz_autodiff
