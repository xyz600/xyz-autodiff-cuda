#pragma once

#include <cstddef>
#include <cuda_runtime.h>
#include "../operation.cuh"
#include "../math.cuh"
#include "../../concept/variable.cuh"

namespace xyz_autodiff {
namespace op {

// Inverse of 3x3 symmetric matrix represented by 6 parameters
// Input: [a, b, c, d, e, f] representing [[a, b, c], [b, d, e], [c, e, f]]
// Output: [a', b', c', d', e', f'] representing inverse matrix
template <std::size_t InputDim>
requires (InputDim == 6)
struct SymMatrix3InvLogic {
    static constexpr std::size_t outputDim = 6;
    
    template <typename Output, typename Input>
    __host__ __device__ void forward(Output& output, const Input& input) const {
        using T = typename Input::value_type;
        
        // Extract matrix elements: [[a, b, c], [b, d, e], [c, e, f]]
        const T& a = input[0];  // [0,0]
        const T& b = input[1];  // [0,1] and [1,0]
        const T& c = input[2];  // [0,2] and [2,0]
        const T& d = input[3];  // [1,1]
        const T& e = input[4];  // [1,2] and [2,1]
        const T& f = input[5];  // [2,2]
        
        // Calculate determinant
        T det = a * (d * f - e * e) - b * (b * f - c * e) + c * (b * e - c * d);
        
        // Avoid division by zero
        if (math::abs(det) < T(1e-8)) {
            det = T(1e-8);  // regularization
        }
        
        T inv_det = T(1) / det;
        
        // Calculate inverse matrix elements
        // inv[0,0] = (d*f - e*e) / det
        output[0] = (d * f - e * e) * inv_det;
        
        // inv[0,1] = inv[1,0] = (c*e - b*f) / det
        output[1] = (c * e - b * f) * inv_det;
        
        // inv[0,2] = inv[2,0] = (b*e - c*d) / det
        output[2] = (b * e - c * d) * inv_det;
        
        // inv[1,1] = (a*f - c*c) / det
        output[3] = (a * f - c * c) * inv_det;
        
        // inv[1,2] = inv[2,1] = (b*c - a*e) / det
        output[4] = (b * c - a * e) * inv_det;
        
        // inv[2,2] = (a*d - b*b) / det
        output[5] = (a * d - b * b) * inv_det;
    }
    
    template <typename Output, typename Input>
    __host__ __device__ void backward(const Output& output, Input& input) const {
        using T = typename Input::value_type;
        
        // This is a complex operation - for now implement simplified version
        // Full analytical gradient computation for 3x3 matrix inverse is quite involved
        // This is a placeholder implementation
        
        // For production use, consider using numerical differentiation or 
        // implementing the full analytical derivatives
        
        // Zero gradients for now (placeholder)
        for (int i = 0; i < 6; ++i) {
            input.add_grad(i, T(0));
        }
    }
};

// 3x3対称行列逆変換のファクトリ
template <DifferentiableVariableConcept Input>
requires (Input::size == 6)
__host__ __device__ auto sym_matrix3_inv(Input& input) {
    SymMatrix3InvLogic<6> logic;
    
    auto op = UnaryOperation<6, SymMatrix3InvLogic<6>, Input>(logic, input);
    return op;
}

} // namespace op
} // namespace xyz_autodiff