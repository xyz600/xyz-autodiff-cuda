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
    const Variable<T, N>& variable_;
    
public:
    // Variableの参照を受け取るコンストラクタ
    __host__ __device__ DiagonalMatrixView(const Variable<T, N>& var)
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
    
    // 勾配を累積
    __device__ void accumulate_grad(const T* grad_values) const {
        variable_.accumulate_grad(grad_values);
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
    __device__ const Variable<T, N>& variable() const { return variable_; }
};

} // namespace xyz_autodiff