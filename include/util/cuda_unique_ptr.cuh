#pragma once

#include <cuda_runtime.h>
#include <memory>
#include <utility>
#include <iostream>

template <typename T>
struct CudaDeleter
{
    void operator()(T *ptr) const
    {
        if (ptr)
        {
            cudaFree(ptr);
        }
    }
};

// Specialization for array types
template <typename T>
struct CudaDeleter<T[]>
{
    void operator()(T *ptr) const
    {
        if (ptr)
        {
            cudaFree(ptr);
        }
    }
};

template <typename T>
using cuda_unique_ptr = std::unique_ptr<T, CudaDeleter<T>>;

template <typename T>
cuda_unique_ptr<T> makeCudaUnique(size_t count = 1)
{
    T *ptr = nullptr;
    size_t bytes = sizeof(T) * count;
    cudaError_t err = cudaMalloc(&ptr, bytes);
    if (err != cudaSuccess)
    {
        std::cerr << "[CUDA ERROR] makeCudaUnique failed to allocate " << bytes << " bytes: " << cudaGetErrorString(err) << "\n";
        return nullptr;
    }
    std::cout << "[DEBUG] makeCudaUnique allocated " << bytes << " bytes at address: " << ptr << "\n";
    return cuda_unique_ptr<T>(ptr);
}

template <typename T>
cuda_unique_ptr<T[]> makeCudaUniqueArray(size_t count)
{
    T *ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, sizeof(T) * count);
    if (err != cudaSuccess)
    {
        return nullptr;
    }
    return cuda_unique_ptr<T[]>(ptr);
}
