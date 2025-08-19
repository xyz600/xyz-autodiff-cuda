#pragma once

// NOTE: Standard L1 and L2 norm operations are now handled by include/operations/unary/l1_norm_logic.cuh and include/operations/unary/l2_norm_logic.cuh
// This file contains only specialized norm operations for mini-gaussian-splatting

#include <cuda_runtime.h>
#include <cmath>
#include "../../include/concept/core_logic.cuh"
#include "../../include/operations/operation.cuh"
#include "../../include/operations/unary/l1_norm_logic.cuh"
#include "../../include/operations/unary/l2_norm_logic.cuh"

namespace xyz_autodiff {
namespace op {

// L2 squared norm operation: output = sum(input[i]^2)
template <typename Input>
requires UnaryLogicParameterConcept<Input>
struct L2SquaredNormLogic {
    using T = typename Input::value_type;
    static constexpr std::size_t Dim = Input::size;
    using Output = Variable<T, 1>;  // scalar output
    
    static constexpr std::size_t outputDim = 1;
    
    static_assert(UnaryLogicParameterConcept<Input>, "Input must satisfy UnaryLogicParameterConcept");
    
    __host__ __device__ L2SquaredNormLogic() = default;
    
    // forward: output = sum(input[i]^2)
    __device__ void forward(Output& output, const Input& input) const {
        T sum_squares = T(0);
        for (std::size_t i = 0; i < Dim; ++i) {
            sum_squares += input[i] * input[i];
        }
        output[0] = sum_squares;
    }
    
    // backward: d(sum(x_i^2))/dx_i = 2*x_i
    __device__ void backward(const Output& output, Input& input) const {
        const T& output_grad = output.grad(0);
        for (std::size_t i = 0; i < Dim; ++i) {
            input.add_grad(i, output_grad * T(2) * input[i]);
        }
    }
};

// Factory function for L2 squared norm
template <typename Input>
requires UnaryLogicParameterConcept<Input>
__device__ auto l2_squared_norm(Input& input) {
    using LogicType = L2SquaredNormLogic<Input>;
    
    LogicType logic;
    return UnaryOperation<LogicType::outputDim, LogicType, Input>(logic, input);
}

} // namespace op
} // namespace xyz_autodiff