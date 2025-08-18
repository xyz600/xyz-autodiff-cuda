#pragma once

#include <cstddef>
#include <cuda_runtime.h>
#include "variable.cuh"

namespace xyz_autodiff {

template <typename T, std::size_t N, typename VariableType = VariableRef<T, N>>
class DiagonalMatrixView {
public:
    using value_type = T;
    static constexpr std::size_t rows = N;
    static constexpr std::size_t cols = N;
    static constexpr std::size_t size = N;  // 対角要素のみ
    
private:
    VariableType& variable_;
    
public:
    // Variable または VariableRef の参照を受け取るコンストラクタ
    __host__ __device__ DiagonalMatrixView(VariableType& var)
        : variable_(var) {}
    
    // コピーコンストラクタ
    __host__ __device__ DiagonalMatrixView(const DiagonalMatrixView& other)
        : variable_(other.variable_) {}
    
    // コピー代入演算子とムーブ代入演算子は禁止（参照は再束縛できないため）
    DiagonalMatrixView& operator=(const DiagonalMatrixView&) = delete;
    DiagonalMatrixView& operator=(DiagonalMatrixView&&) = delete;
    
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
    
    
    // === MatrixView concept の要件 ===
    
    // 2次元アクセス (値)
    __device__ T operator()(std::size_t row, std::size_t col) {
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
    
    // transpose機能 - 対角行列は転置しても同じなので自分自身を返す
    __host__ __device__ DiagonalMatrixView transpose() const {
        return *this;
    }
    
    // Variable参照を取得
    __device__ VariableType& variable() { return variable_; }
    __device__ const VariableType& variable() const { return variable_; }
};

// ヘルパー関数
template <std::size_t N, typename T>
__host__ __device__ auto make_diagonal_matrix_view(VariableRef<T, N>& var) {
    return DiagonalMatrixView<T, N, VariableRef<T, N>>(var);
}

template <std::size_t N, typename T>
__host__ __device__ auto make_diagonal_matrix_view(Variable<T, N>& var) {
    return DiagonalMatrixView<T, N, Variable<T, N>>(var);
}

} // namespace xyz_autodiff