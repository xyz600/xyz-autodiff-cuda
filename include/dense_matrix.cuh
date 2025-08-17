#pragma once

#include <cstddef>
#include <span>
#include <cuda_runtime.h>
// #include "concept/variable.cuh"  // CUDA compiler concept limitations
// #include "concept/matrix.cuh"   // CUDA compiler concept limitations

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
    __device__ T& operator()(std::size_t row, std::size_t col) {
        return data_[row * cols + col];
    }
    
    __device__ const T& operator()(std::size_t row, std::size_t col) const {
        return data_[row * cols + col];
    }
    
    // 2次元アクセス (勾配)
    __device__ T& grad(std::size_t row, std::size_t col) {
        return grad_[row * cols + col];
    }
    
    __device__ const T& grad(std::size_t row, std::size_t col) const {
        return grad_[row * cols + col];
    }
    
    // 疎行列サポート (密行列なので全て有効)
    __device__ bool is_active_in_col(std::size_t row, std::size_t col) const {
        return true;  // 密行列なので全て有効
    }
    
    __device__ bool is_active_in_row(std::size_t row, std::size_t col) const {
        return true;  // 密行列なので全て有効
    }
};

// コンセプトチェック (CUDA compiler limitations)
// static_assert(concept::Variable<DenseMatrix<float, 3, 4>>);
// static_assert(concept::DifferentiableVariable<DenseMatrix<float, 3, 4>>);
// static_assert(concept::MatrixView<DenseMatrix<float, 3, 4>>);
// static_assert(concept::DifferentiableMatrixView<DenseMatrix<float, 3, 4>>);

} // namespace xyz_autodiff
