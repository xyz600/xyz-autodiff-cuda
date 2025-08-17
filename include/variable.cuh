#pragma once

#include <cstddef>
#include <cuda_runtime.h>

namespace xyz_autodiff {

template <typename T, std::size_t N>
class Variable {
public:
    using value_type = T;
    static constexpr std::size_t size = N;
    
private:
    T* const data_ptr_;
    T* const grad_ptr_;
    
public:
    // 外部メモリへの参照を受け取るコンストラクタ
    __host__ __device__ constexpr Variable(T* data, T* grad) 
        : data_ptr_(data), grad_ptr_(grad) {}
    
    // データアクセサ
    __device__ __forceinline__ T* data() const noexcept { return data_ptr_; }
    
    // 勾配アクセサ
    __device__ __forceinline__ T* grad() const noexcept { return grad_ptr_; }
    
    // インデックスアクセス (値)
    __device__ __forceinline__ T& operator[](std::size_t i) const noexcept { 
        return data_ptr_[i]; 
    }
    
    // インデックスアクセス (勾配)
    __device__ __forceinline__ T& grad(std::size_t i) const noexcept { 
        return grad_ptr_[i]; 
    }
    
    // 勾配をゼロクリア
    __device__ void zero_grad() const noexcept {
        #pragma unroll
        for (std::size_t i = 0; i < N; ++i) {
            grad_ptr_[i] = T{};
        }
    }
    
    // 勾配を累積
    __device__ void accumulate_grad(const T* const grad_values) const noexcept {
        for (std::size_t i = 0; i < N; ++i) {
            atomicAdd(&grad_ptr_[i], grad_values[i]);
        }
    }
};

} // namespace xyz_autodiff