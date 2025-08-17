#pragma once

#include <vector>
#include <memory>
#include <cuda_runtime.h>
#include <cublas_v2.h>

namespace xyz_autodiff {

class Tensor {
public:
    Tensor(const std::vector<int>& shape, bool requires_grad = false);
    Tensor(float* data, const std::vector<int>& shape, bool requires_grad = false);
    ~Tensor();

    Tensor(const Tensor& other);
    Tensor& operator=(const Tensor& other);

    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;

    const std::vector<int>& shape() const { return shape_; }
    int size() const;
    bool requires_grad() const { return requires_grad_; }
    void set_requires_grad(bool value) { requires_grad_ = value; }

    float* data() { return data_; }
    const float* data() const { return data_; }
    
    float* grad() { return grad_; }
    const float* grad() const { return grad_; }

    void zero_grad();
    void backward();

    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator/(const Tensor& other) const;

    Tensor matmul(const Tensor& other) const;
    Tensor sum() const;
    Tensor mean() const;
    Tensor relu() const;
    Tensor sigmoid() const;
    Tensor tanh() const;

    void to_cpu(float* host_data) const;
    void from_cpu(const float* host_data);

private:
    std::vector<int> shape_;
    float* data_;
    float* grad_;
    bool requires_grad_;
    bool owns_data_;

    struct BackwardFunction;
    std::shared_ptr<BackwardFunction> backward_fn_;
    std::vector<std::shared_ptr<Tensor>> parents_;

    void allocate_memory();
    void free_memory();

    friend class Operations;
    friend class BackwardEngine;
};

}