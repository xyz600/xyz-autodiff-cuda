#include "xyz_autodiff/operations.h"
#include "xyz_autodiff/backward.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <numeric>
#include <functional>

namespace xyz_autodiff {

cublasHandle_t Operations::cublas_handle_ = nullptr;
bool Operations::initialized_ = false;

__global__ void add_kernel(const float* a, const float* b, float* result, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = a[idx] + b[idx];
    }
}

__global__ void subtract_kernel(const float* a, const float* b, float* result, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = a[idx] - b[idx];
    }
}

__global__ void multiply_kernel(const float* a, const float* b, float* result, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = a[idx] * b[idx];
    }
}

__global__ void divide_kernel(const float* a, const float* b, float* result, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = a[idx] / b[idx];
    }
}

__global__ void relu_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

__global__ void sigmoid_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = 1.0f / (1.0f + expf(-input[idx]));
    }
}

__global__ void tanh_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = tanhf(input[idx]);
    }
}

__global__ void sum_kernel(const float* input, float* output, int size) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (idx < size) ? input[idx] : 0.0f;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}

Tensor Operations::add(const Tensor& a, const Tensor& b) {
    Tensor result(a.shape(), a.requires_grad() || b.requires_grad());
    
    int size = a.size();
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    
    add_kernel<<<grid_size, block_size>>>(a.data(), b.data(), result.data(), size);
    cudaDeviceSynchronize();
    
    if (result.requires_grad()) {
        auto a_ptr = std::make_shared<Tensor>(const_cast<Tensor&>(a));
        auto b_ptr = std::make_shared<Tensor>(const_cast<Tensor&>(b));
        result.parents_ = {a_ptr, b_ptr};
    }
    
    return result;
}

Tensor Operations::subtract(const Tensor& a, const Tensor& b) {
    Tensor result(a.shape(), a.requires_grad() || b.requires_grad());
    
    int size = a.size();
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    
    subtract_kernel<<<grid_size, block_size>>>(a.data(), b.data(), result.data(), size);
    cudaDeviceSynchronize();
    
    return result;
}

Tensor Operations::multiply(const Tensor& a, const Tensor& b) {
    Tensor result(a.shape(), a.requires_grad() || b.requires_grad());
    
    int size = a.size();
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    
    multiply_kernel<<<grid_size, block_size>>>(a.data(), b.data(), result.data(), size);
    cudaDeviceSynchronize();
    
    return result;
}

Tensor Operations::divide(const Tensor& a, const Tensor& b) {
    Tensor result(a.shape(), a.requires_grad() || b.requires_grad());
    
    int size = a.size();
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    
    divide_kernel<<<grid_size, block_size>>>(a.data(), b.data(), result.data(), size);
    cudaDeviceSynchronize();
    
    return result;
}

Tensor Operations::matmul(const Tensor& a, const Tensor& b) {
    if (a.shape().size() != 2 || b.shape().size() != 2) {
        throw std::runtime_error("Matrix multiplication requires 2D tensors");
    }
    
    int m = a.shape()[0];
    int k = a.shape()[1];
    int n = b.shape()[1];
    
    if (k != b.shape()[0]) {
        throw std::runtime_error("Matrix dimensions don't match for multiplication");
    }
    
    Tensor result({m, n}, a.requires_grad() || b.requires_grad());
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    cublasSgemm(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                n, m, k,
                &alpha,
                b.data(), n,
                a.data(), k,
                &beta,
                result.data(), n);
    
    return result;
}

Tensor Operations::sum(const Tensor& input) {
    Tensor result({1}, input.requires_grad());
    
    cudaMemset(result.data(), 0, sizeof(float));
    
    int size = input.size();
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    
    sum_kernel<<<grid_size, block_size, block_size * sizeof(float)>>>(
        input.data(), result.data(), size);
    cudaDeviceSynchronize();
    
    return result;
}

Tensor Operations::mean(const Tensor& input) {
    Tensor sum_result = sum(input);
    
    float size = static_cast<float>(input.size());
    float* host_data = new float[1];
    sum_result.to_cpu(host_data);
    host_data[0] /= size;
    sum_result.from_cpu(host_data);
    delete[] host_data;
    
    return sum_result;
}

Tensor Operations::relu(const Tensor& input) {
    Tensor result(input.shape(), input.requires_grad());
    
    int size = input.size();
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    
    relu_kernel<<<grid_size, block_size>>>(input.data(), result.data(), size);
    cudaDeviceSynchronize();
    
    return result;
}

Tensor Operations::sigmoid(const Tensor& input) {
    Tensor result(input.shape(), input.requires_grad());
    
    int size = input.size();
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    
    sigmoid_kernel<<<grid_size, block_size>>>(input.data(), result.data(), size);
    cudaDeviceSynchronize();
    
    return result;
}

Tensor Operations::tanh(const Tensor& input) {
    Tensor result(input.shape(), input.requires_grad());
    
    int size = input.size();
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    
    tanh_kernel<<<grid_size, block_size>>>(input.data(), result.data(), size);
    cudaDeviceSynchronize();
    
    return result;
}

Tensor Operations::reshape(const Tensor& input, const std::vector<int>& new_shape) {
    int new_size = std::accumulate(new_shape.begin(), new_shape.end(), 1, std::multiplies<int>());
    if (new_size != input.size()) {
        throw std::runtime_error("Reshape size mismatch");
    }
    
    Tensor result(new_shape, input.requires_grad());
    cudaMemcpy(result.data(), input.data(), input.size() * sizeof(float), cudaMemcpyDeviceToDevice);
    
    return result;
}

Tensor Operations::transpose(const Tensor& input) {
    if (input.shape().size() != 2) {
        throw std::runtime_error("Transpose only supports 2D tensors");
    }
    
    std::vector<int> new_shape = {input.shape()[1], input.shape()[0]};
    Tensor result(new_shape, input.requires_grad());
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    cublasSgeam(cublas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
                new_shape[1], new_shape[0],
                &alpha, input.data(), input.shape()[1],
                &beta, nullptr, new_shape[1],
                result.data(), new_shape[1]);
    
    return result;
}

}