#include "xyz_autodiff/tensor.h"
#include "xyz_autodiff/operations.h"
#include "xyz_autodiff/backward.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <numeric>
#include <functional>

namespace xyz_autodiff {

Tensor::Tensor(const std::vector<int>& shape, bool requires_grad)
    : shape_(shape), data_(nullptr), grad_(nullptr), 
      requires_grad_(requires_grad), owns_data_(true) {
    allocate_memory();
}

Tensor::Tensor(float* data, const std::vector<int>& shape, bool requires_grad)
    : shape_(shape), data_(data), grad_(nullptr), 
      requires_grad_(requires_grad), owns_data_(false) {
    if (requires_grad_) {
        cudaMalloc(&grad_, size() * sizeof(float));
        cudaMemset(grad_, 0, size() * sizeof(float));
    }
}

Tensor::~Tensor() {
    if (owns_data_) {
        free_memory();
    } else if (grad_) {
        cudaFree(grad_);
    }
}

Tensor::Tensor(const Tensor& other)
    : shape_(other.shape_), data_(nullptr), grad_(nullptr),
      requires_grad_(other.requires_grad_), owns_data_(true) {
    allocate_memory();
    cudaMemcpy(data_, other.data_, size() * sizeof(float), cudaMemcpyDeviceToDevice);
    if (grad_ && other.grad_) {
        cudaMemcpy(grad_, other.grad_, size() * sizeof(float), cudaMemcpyDeviceToDevice);
    }
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        if (owns_data_) {
            free_memory();
        }
        shape_ = other.shape_;
        requires_grad_ = other.requires_grad_;
        owns_data_ = true;
        allocate_memory();
        cudaMemcpy(data_, other.data_, size() * sizeof(float), cudaMemcpyDeviceToDevice);
        if (grad_ && other.grad_) {
            cudaMemcpy(grad_, other.grad_, size() * sizeof(float), cudaMemcpyDeviceToDevice);
        }
    }
    return *this;
}

Tensor::Tensor(Tensor&& other) noexcept
    : shape_(std::move(other.shape_)), data_(other.data_), grad_(other.grad_),
      requires_grad_(other.requires_grad_), owns_data_(other.owns_data_),
      parents_(std::move(other.parents_)) {
    other.data_ = nullptr;
    other.grad_ = nullptr;
    other.owns_data_ = false;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        if (owns_data_) {
            free_memory();
        }
        shape_ = std::move(other.shape_);
        data_ = other.data_;
        grad_ = other.grad_;
        requires_grad_ = other.requires_grad_;
        owns_data_ = other.owns_data_;
        parents_ = std::move(other.parents_);
        
        other.data_ = nullptr;
        other.grad_ = nullptr;
        other.owns_data_ = false;
    }
    return *this;
}

int Tensor::size() const {
    return std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<int>());
}

void Tensor::allocate_memory() {
    int total_size = size();
    cudaMalloc(&data_, total_size * sizeof(float));
    if (requires_grad_) {
        cudaMalloc(&grad_, total_size * sizeof(float));
        cudaMemset(grad_, 0, total_size * sizeof(float));
    }
}

void Tensor::free_memory() {
    if (data_) {
        cudaFree(data_);
        data_ = nullptr;
    }
    if (grad_) {
        cudaFree(grad_);
        grad_ = nullptr;
    }
}

void Tensor::zero_grad() {
    if (grad_) {
        cudaMemset(grad_, 0, size() * sizeof(float));
    }
}

void Tensor::backward() {
    BackwardEngine::backward(*this);
}

Tensor Tensor::operator+(const Tensor& other) const {
    return Operations::add(*this, other);
}

Tensor Tensor::operator-(const Tensor& other) const {
    return Operations::subtract(*this, other);
}

Tensor Tensor::operator*(const Tensor& other) const {
    return Operations::multiply(*this, other);
}

Tensor Tensor::operator/(const Tensor& other) const {
    return Operations::divide(*this, other);
}

Tensor Tensor::matmul(const Tensor& other) const {
    return Operations::matmul(*this, other);
}

Tensor Tensor::sum() const {
    return Operations::sum(*this);
}

Tensor Tensor::mean() const {
    return Operations::mean(*this);
}

Tensor Tensor::relu() const {
    return Operations::relu(*this);
}

Tensor Tensor::sigmoid() const {
    return Operations::sigmoid(*this);
}

Tensor Tensor::tanh() const {
    return Operations::tanh(*this);
}

void Tensor::to_cpu(float* host_data) const {
    cudaMemcpy(host_data, data_, size() * sizeof(float), cudaMemcpyDeviceToHost);
}

void Tensor::from_cpu(const float* host_data) {
    cudaMemcpy(data_, host_data, size() * sizeof(float), cudaMemcpyHostToDevice);
}

}