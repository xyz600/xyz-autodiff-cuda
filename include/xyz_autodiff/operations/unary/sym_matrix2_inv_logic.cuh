#pragma once

#include <cstddef>
#include <cuda_runtime.h>
#include <xyz_autodiff/operations/operation.cuh>
#include <xyz_autodiff/operations/math.cuh>
#include <xyz_autodiff/concept/variable.cuh>

namespace xyz_autodiff {
namespace op {

// Inverse of 2x2 symmetric matrix represented by 3 parameters
// Input: [a, b, c] representing [[a, b], [b, c]]
// Output: [a', b', c'] representing inverse matrix
template <std::size_t InputDim>
requires (InputDim == 3)
struct SymMatrix2InvLogic {
    static constexpr std::size_t outputDim = 3;
    
    template <typename Output, typename Input>
    __host__ __device__ void forward(Output& output, const Input& input) const {
        using T = typename Input::value_type;
        const T& a = input[0];
        const T& b = input[1];
        const T& c = input[2];
        
        // Determinant: det = ac - b^2
        T det = a * c - b * b;
        
        // Avoid division by zero
        if (math::abs(det) < T(1e-8)) {
            det = T(1e-8);  // regularization
        }
        
        T inv_det = T(1) / det;
        
        // Inverse matrix: [[c, -b], [-b, a]] / det
        output[0] = c * inv_det;   // a'
        output[1] = -b * inv_det;  // b'
        output[2] = a * inv_det;   // c'
    }
    
    template <typename Output, typename Input>
    __host__ __device__ void backward(const Output& output, Input& input) const {
        using T = typename Input::value_type;
        const T& a = input[0];
        const T& b = input[1];
        const T& c = input[2];
        
        T det = a * c - b * b;
        if (math::abs(det) < T(1e-8)) {
            det = T(1e-8);  // same regularization as forward
        }
        
        T inv_det = T(1) / det;
        T inv_det2 = inv_det * inv_det;
        
        const T& grad_a_inv = output.grad(0);  // gradient w.r.t a'
        const T& grad_b_inv = output.grad(1);  // gradient w.r.t b'
        const T& grad_c_inv = output.grad(2);  // gradient w.r.t c'
        
        // Gradient computation for matrix inverse
        T da_inv_da = -c * c * inv_det2;
        T da_inv_db = T(2) * c * b * inv_det2;
        T da_inv_dc = inv_det - a * c * inv_det2;
        
        T db_inv_da = b * c * inv_det2;
        T db_inv_db = -inv_det - T(2) * b * b * inv_det2;
        T db_inv_dc = a * b * inv_det2;
        
        T dc_inv_da = inv_det - a * c * inv_det2;
        T dc_inv_db = T(2) * a * b * inv_det2;
        T dc_inv_dc = -a * a * inv_det2;
        
        input.add_grad(0, grad_a_inv * da_inv_da + grad_b_inv * db_inv_da + grad_c_inv * dc_inv_da);
        input.add_grad(1, grad_a_inv * da_inv_db + grad_b_inv * db_inv_db + grad_c_inv * dc_inv_db);
        input.add_grad(2, grad_a_inv * da_inv_dc + grad_b_inv * db_inv_dc + grad_c_inv * dc_inv_dc);
    }
};

// 2x2対称行列逆変換のファクトリ
template <DifferentiableVariableConcept Input>
requires (Input::size == 3)
__host__ __device__ auto sym_matrix2_inv(Input& input) {
    SymMatrix2InvLogic<3> logic;
    
    auto op = UnaryOperation<3, SymMatrix2InvLogic<3>, Input>(logic, input);
    return op;
}

} // namespace op
} // namespace xyz_autodiff