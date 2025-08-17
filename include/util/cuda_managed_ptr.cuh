#pragma once

#include <cuda_runtime.h>
#include <memory>
#include <utility>
#include <iostream>

template <typename T>
struct CudaManagedDeleter
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
struct CudaManagedDeleter<T[]>
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
using cuda_managed_ptr = std::unique_ptr<T, CudaManagedDeleter<T>>;

template <typename T>
cuda_managed_ptr<T> makeCudaManagedUnique(size_t count = 1)
{
    T *ptr = nullptr;
    size_t bytes = sizeof(T) * count;
    cudaError_t err = cudaMallocManaged(&ptr, bytes);
    if (err != cudaSuccess)
    {
        std::cerr << "[CUDA ERROR] makeCudaManagedUnique failed to allocate " << bytes << " bytes: " << cudaGetErrorString(err) << "\n";
        return nullptr;
    }
    return cuda_managed_ptr<T>(ptr);
}

template <typename T>
cuda_managed_ptr<T[]> makeCudaManagedArray(size_t count)
{
    T *ptr = nullptr;
    const size_t BYTES = sizeof(T) * count;
    cudaError_t err = cudaMallocManaged(&ptr, BYTES);
    if (err != cudaSuccess)
    {
        std::cerr << "[CUDA ERROR] makeCudaManagedArray failed to allocate " << BYTES << " bytes: " << cudaGetErrorString(err) << "\n";
        return nullptr;
    }
    return cuda_managed_ptr<T[]>(ptr);
}
