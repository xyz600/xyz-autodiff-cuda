#pragma once

#include "tensor.h"
#include <functional>
#include <memory>
#include <set>
#include <vector>

namespace xyz_autodiff {

class BackwardEngine {
public:
    static void backward(const Tensor& output);
    
private:
    static void topological_sort(const Tensor& tensor, 
                                std::vector<std::shared_ptr<Tensor>>& sorted_tensors,
                                std::set<const Tensor*>& visited);
};


}