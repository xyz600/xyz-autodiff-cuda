#include "xyz_autodiff/backward.h"
#include "xyz_autodiff/operations.h"
#include <set>
#include <queue>
#include <algorithm>

namespace xyz_autodiff {

void BackwardEngine::backward(const Tensor& output) {
    if (!output.requires_grad()) {
        return;
    }
    
    std::vector<std::shared_ptr<Tensor>> sorted_tensors;
    std::set<const Tensor*> visited;
    
    auto output_ptr = std::make_shared<Tensor>(const_cast<Tensor&>(output));
    topological_sort(*output_ptr, sorted_tensors, visited);
    
    float ones = 1.0f;
    cudaMemcpy(output_ptr->grad(), &ones, sizeof(float), cudaMemcpyHostToDevice);
    
    for (auto it = sorted_tensors.rbegin(); it != sorted_tensors.rend(); ++it) {
        auto& tensor = **it;
        if (tensor.backward_fn_) {
            tensor.backward_fn_->fn(tensor);
        }
    }
}

void BackwardEngine::topological_sort(const Tensor& tensor,
                                     std::vector<std::shared_ptr<Tensor>>& sorted_tensors,
                                     std::set<const Tensor*>& visited) {
    if (visited.find(&tensor) != visited.end()) {
        return;
    }
    
    visited.insert(&tensor);
    
    for (const auto& parent : tensor.parents_) {
        topological_sort(*parent, sorted_tensors, visited);
    }
    
    sorted_tensors.push_back(std::make_shared<Tensor>(const_cast<Tensor&>(tensor)));
}

void initialize() {
    if (!Operations::initialized_) {
        cublasCreate(&Operations::cublas_handle_);
        Operations::initialized_ = true;
    }
}

void cleanup() {
    if (Operations::initialized_) {
        cublasDestroy(Operations::cublas_handle_);
        Operations::cublas_handle_ = nullptr;
        Operations::initialized_ = false;
    }
}

}