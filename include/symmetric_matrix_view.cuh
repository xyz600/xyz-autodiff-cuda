#pragma once

#include <cstddef>
#include <cuda_runtime.h>
#include "concept/matrix.cuh"
#include "concept/variable.cuh"

namespace xyz_autodiff {

// 対称行列ビュー（N×N行列をN*(N+1)/2要素で表現）
template <typename T, std::size_t N, DifferentiableVariableConcept Variable>
class SymmetricMatrixView {
public:
    using value_type = T;
    static constexpr std::size_t rows = N;
    static constexpr std::size_t cols = N;
    static constexpr std::size_t size = N * N;
    static constexpr std::size_t storage_size = N * (N + 1) / 2;
    
private:
    Variable& variable_;
    
    // 上三角行列のインデックス計算: (i,j) where i <= j
    __host__ __device__ static constexpr std::size_t upper_triangular_index(std::size_t i, std::size_t j) {
        // i <= j の場合のインデックス計算
        // 行 i の開始位置: i * N - i * (i - 1) / 2
        // 列 j への追加オフセット: j - i
        return i * N - i * (i - 1) / 2 + (j - i);
    }
    
public:
    // コンストラクタ
    __host__ __device__ explicit SymmetricMatrixView(Variable& var) : variable_(var) {
        static_assert(Variable::size == storage_size, 
                     "Variable size must match N*(N+1)/2 for SymmetricMatrix");
        static_assert(std::is_same_v<typename Variable::value_type, T>,
                     "Variable value_type must match T");
    }
    
    // コピーコンストラクタ
    __host__ __device__ SymmetricMatrixView(const SymmetricMatrixView& other) : variable_(other.variable_) {}
    
    // 代入演算子
    __host__ __device__ SymmetricMatrixView& operator=(const SymmetricMatrixView& other) {
        variable_ = other.variable_;
        return *this;
    }
    
    // データアクセサ（MatrixViewConcept要件）
    __host__ __device__ T* data() { return variable_.data(); }
    __host__ __device__ const T* data() const { return variable_.data(); }
    
    // 2次元アクセス（対称性を考慮）
    __host__ __device__ T& operator()(std::size_t row, std::size_t col) {
        if (row <= col) {
            // 上三角または対角要素: 直接アクセス
            return variable_[upper_triangular_index(row, col)];
        } else {
            // 下三角要素: 対称性により上三角から参照
            return variable_[upper_triangular_index(col, row)];
        }
    }
    
    __host__ __device__ const T& operator()(std::size_t row, std::size_t col) const {
        if (row <= col) {
            return variable_[upper_triangular_index(row, col)];
        } else {
            return variable_[upper_triangular_index(col, row)];
        }
    }
    
    // 1次元アクセス（storage順序での直接アクセス）
    __host__ __device__ T& operator[](std::size_t i) { return variable_[i]; }
    __host__ __device__ const T& operator[](std::size_t i) const { return variable_[i]; }
    
    // transpose関数（対称行列なので自分自身を返す）
    __host__ __device__ const SymmetricMatrixView& transpose() const {
        return *this;
    }
    
    __host__ __device__ SymmetricMatrixView& transpose() {
        return *this;
    }
    
    // Variable conceptの要件（参照先に転送）
    __host__ __device__ T* grad() { return variable_.grad(); }
    __host__ __device__ const T* grad() const { return variable_.grad(); }
    
    __host__ __device__ T& grad(std::size_t i) { return variable_.grad(i); }
    __host__ __device__ const T& grad(std::size_t i) const { return variable_.grad(i); }
    
    __host__ __device__ void zero_grad() { variable_.zero_grad(); }
    
    
    // 基底Variableへのアクセス
    __host__ __device__ auto& variable() { return variable_; }
    __host__ __device__ const auto& variable() const { return variable_; }
};

// 型推論ヘルパー
template <std::size_t N, DifferentiableVariableConcept Variable>
__host__ __device__ auto make_symmetric_matrix_view(Variable& var) {
    using T = typename Variable::value_type;
    return SymmetricMatrixView<T, N, Variable>(var);
}

} // namespace xyz_autodiff