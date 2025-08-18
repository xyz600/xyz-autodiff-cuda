#pragma once

#include <cstddef>
#include <cuda_runtime.h>

namespace xyz_autodiff {

// VariableRef - 外部バッファへの参照版 (現在のVariableの設計)
template <typename T, std::size_t N>
class VariableRef {
public:
    using value_type = T;
    static constexpr std::size_t size = N;
    
private:
    T* const data_ptr_;
    T* const grad_ptr_;
    
public:
    // 外部メモリへの参照を受け取るコンストラクタ
    __host__ __device__ constexpr VariableRef(T* data, T* grad) 
        : data_ptr_(data), grad_ptr_(grad) {}
    
    // データアクセサ
    __device__ __forceinline__ T* data() const noexcept { return data_ptr_; }
    
    // 勾配アクセサ
    __device__ __forceinline__ T* grad() const noexcept { return grad_ptr_; }
    
    // インデックスアクセス (値)
    __device__ __forceinline__ constexpr T& operator[](std::size_t i) const noexcept { 
        return data_ptr_[i]; 
    }
    
    // インデックスアクセス (勾配の読み取り専用)
    __device__ __forceinline__ const T& grad(std::size_t i) const noexcept { 
        return grad_ptr_[i]; 
    }
    
    // 勾配への加算（shared/globalはatomicAdd、純粋ローカルのみ通常加算）
    __device__ __forceinline__ void add_grad(std::size_t i, T value) const noexcept {
        // shared memory または global memory の場合はatomicAdd
        if (__isShared(grad_ptr_ + i) || __isGlobal(grad_ptr_ + i)) {
            atomicAdd(&grad_ptr_[i], value);
        } else {
            // レジスタ・純粋ローカルの場合は通常加算
            grad_ptr_[i] += value;
        }
    }
    
    // 勾配をゼロクリア
    __device__ void zero_grad() const noexcept {
        #pragma unroll
        for (std::size_t i = 0; i < N; ++i) {
            grad_ptr_[i] = T{};
        }
    }
    
    
};

// Variable - 自身でバッファを持つ版
template <typename T, std::size_t N>
class Variable {
public:
    using value_type = T;
    static constexpr std::size_t size = N;
    
private:
    T data_[N];
    T grad_[N];
    
public:
    // デフォルトコンストラクタ
    __host__ __device__ Variable() {
        #pragma unroll
        for (std::size_t i = 0; i < N; ++i) {
            data_[i] = T{};
            grad_[i] = T{};
        }
    }
    
    // 初期値を指定するコンストラクタ
    __host__ __device__ Variable(const T& initial_value) {
        #pragma unroll
        for (std::size_t i = 0; i < N; ++i) {
            data_[i] = initial_value;
            grad_[i] = T{};
        }
    }
    
    // 配列からのコンストラクタ
    __host__ __device__ Variable(const T* values) {
        #pragma unroll
        for (std::size_t i = 0; i < N; ++i) {
            data_[i] = values[i];
            grad_[i] = T{};
        }
    }
    
    // コピーコンストラクタ
    __host__ __device__ Variable(const Variable& other) {
        #pragma unroll
        for (std::size_t i = 0; i < N; ++i) {
            data_[i] = other.data_[i];
            grad_[i] = other.grad_[i];
        }
    }
    
    // ムーブコンストラクタ
    __host__ __device__ Variable(Variable&& other) noexcept {
        #pragma unroll
        for (std::size_t i = 0; i < N; ++i) {
            data_[i] = other.data_[i];
            grad_[i] = other.grad_[i];
        }
    }
    
    // コピー代入演算子
    __host__ __device__ Variable& operator=(const Variable& other) {
        if (this != &other) {
            #pragma unroll
            for (std::size_t i = 0; i < N; ++i) {
                data_[i] = other.data_[i];
                grad_[i] = other.grad_[i];
            }
        }
        return *this;
    }
    
    // ムーブ代入演算子
    __host__ __device__ Variable& operator=(Variable&& other) noexcept {
        if (this != &other) {
            #pragma unroll
            for (std::size_t i = 0; i < N; ++i) {
                data_[i] = other.data_[i];
                grad_[i] = other.grad_[i];
            }
        }
        return *this;
    }
    
    // データアクセサ
    __device__ __forceinline__ T* data() noexcept { return data_; }
    __device__ __forceinline__ const T* data() const noexcept { return data_; }
    
    // 勾配アクセサ
    __device__ __forceinline__ T* grad() noexcept { return grad_; }
    __device__ __forceinline__ const T* grad() const noexcept { return grad_; }
    
    // インデックスアクセス (値)
    __device__ __forceinline__ T& operator[](std::size_t i) noexcept { 
        return data_[i]; 
    }
    __device__ __forceinline__ const T& operator[](std::size_t i) const noexcept { 
        return data_[i]; 
    }
    
    // インデックスアクセス (勾配の読み取り専用)
    __device__ __forceinline__ const T& grad(std::size_t i) const noexcept { 
        return grad_[i]; 
    }
    
    // 勾配への加算（通常の加算、atomic演算不要）
    __device__ __forceinline__ void add_grad(std::size_t i, T value) noexcept {
        grad_[i] += value;
    }
    
    // 勾配をゼロクリア
    __device__ void zero_grad() noexcept {
        #pragma unroll
        for (std::size_t i = 0; i < N; ++i) {
            grad_[i] = T{};
        }
    }
    
    
    // VariableRefに変換
    __device__ VariableRef<T, N> ref() noexcept {
        return VariableRef<T, N>(data_, grad_);
    }
    
    __device__ VariableRef<T, N> ref() const noexcept {
        return VariableRef<T, N>(const_cast<T*>(data_), const_cast<T*>(grad_));
    }
    
};

} // namespace xyz_autodiff