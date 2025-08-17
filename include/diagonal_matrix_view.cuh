#pragma once

#include <cstddef>
#include <cuda_runtime.h>
#include "variable.cuh"

namespace xyz_autodiff {

template <typename T, std::size_t N>
class DiagonalMatrixView {
public:
    using value_type = T;
    static constexpr std::size_t rows = N;
    static constexpr std::size_t cols = N;
    static constexpr std::size_t size = N;  // 対角要素のみ
    
private:
    Variable<T, N> variable_;
    
public:
    // Variableを受け取るコンストラクタ
    __host__ __device__ DiagonalMatrixView(const Variable<T, N>& var)
        : variable_(var) {}
    
    // === Variable concept の要件 ===
    
    // データアクセサ
    __device__ T* data() const { return variable_.data(); }
    
    // 勾配アクセサ
    __device__ T* grad() const { return variable_.grad(); }
    
    // インデックスアクセス (値) - 1次元アクセス（対角要素）
    __device__ T& operator[](std::size_t i) const { 
        return variable_[i]; 
    }
    
    // インデックスアクセス (勾配) - 1次元アクセス（対角要素）
    __device__ T& grad(std::size_t i) const { 
        return variable_.grad(i); 
    }
    
    // 勾配をゼロクリア
    __device__ void zero_grad() const {
        variable_.zero_grad();
    }
    
    // 勾配を累積
    __device__ void accumulate_grad(const T* grad_values) const {
        variable_.accumulate_grad(grad_values);
    }
    
    // === MatrixView concept の要件 ===
    
    // 2次元アクセス (値)    
    __device__ constexpr T operator()(std::size_t row, std::size_t col) {
        if (row == col) {
            return variable_[row];
        } else {
            return T{0};
        }
    }
    
    __device__ constexpr T operator()(std::size_t row, std::size_t col) const {
        if (row == col) {
            return variable_[row];
        } else {
            return T{0};
        }
    }

    // 疎行列サポート
    __device__ constexpr bool is_active_cell(std::size_t row, std::size_t col) const {
        return (row == col);  // 対角要素のみアクティブ
    }
    
    // Variable参照を取得
    __device__ const Variable<T, N>& variable() const { return variable_; }
};

} // namespace xyz_autodiff