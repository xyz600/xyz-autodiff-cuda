#pragma once

#include "tensor.h"
#include <functional>
#include <memory>

namespace xyz_autodiff {

class BackwardEngine {
public:
    static void backward(const Tensor& output);
    
private:
    static void topological_sort(const Tensor& tensor, 
                                std::vector<std::shared_ptr<Tensor>>& sorted_tensors,
                                std::set<const Tensor*>& visited);
};

struct Tensor::BackwardFunction {
    std::function<void(const Tensor& grad_output)> fn;
    std::vector<std::shared_ptr<Tensor>> inputs;
    
    BackwardFunction(std::function<void(const Tensor& grad_output)> f,
                    std::vector<std::shared_ptr<Tensor>> in)
        : fn(std::move(f)), inputs(std::move(in)) {}
};

}