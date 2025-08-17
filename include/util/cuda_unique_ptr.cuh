#pragma once

#include <cuda_runtime.h>
#include <memory>
#include <utility>
#include <iostream>

#include "error_checker.cuh"

template <typename T>
struct CudaDeleter
{
    void operator()(T *ptr) const
    {
        if (ptr)
        {
            CHECK_CUDA_ERROR(cudaFree(ptr));
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
        CHECK_CUDA_ERROR(cudaFree(ptr));
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
    CHECK_CUDA_ERROR(cudaMalloc(&ptr, bytes));
    return cuda_unique_ptr<T>(ptr);
}

template <typename T>
cuda_unique_ptr<T[]> makeCudaUniqueArray(size_t count)
{
    T *ptr = nullptr;
    CHECK_CUDA_ERROR(cudaMalloc(&ptr, sizeof(T) * count));
    return cuda_unique_ptr<T[]>(ptr);
}
