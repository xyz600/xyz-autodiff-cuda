#pragma once

#include <cuda_runtime.h>
#include <cmath>
#include "../../include/concept/core_logic.cuh"
#include "../../include/operations/operation.cuh"

namespace xyz_autodiff {
namespace op {

// Mahalanobis distance calculation for 2D points
// Input1: 2D point difference vector [dx, dy]
// Input2: 3-parameter inverse covariance matrix [a, b, c] representing [[a, b], [b, c]]
// Output: scalar Mahalanobis distance squared
template <typename Input1, typename Input2>
requires BinaryLogicParameterConcept<Input1, Input2> && 
         (Input1::size == 2) && (Input2::size == 3)
struct MahalanobisDistanceLogic {
    using T = typename Input1::value_type;
    static_assert(std::is_same_v<T, typename Input2::value_type>, "Input types must match");
    static constexpr std::size_t Dim = 1;  // scalar output
    using Output = Variable<Dim, T>;
    
    static constexpr std::size_t outputDim = Dim;
    
    __host__ __device__ MahalanobisDistanceLogic() = default;
    
    // forward: d^2 = [dx, dy] * [[a, b], [b, c]] * [dx, dy]^T
    //         = dx*(a*dx + b*dy) + dy*(b*dx + c*dy)
    //         = a*dx^2 + 2*b*dx*dy + c*dy^2
    __device__ void forward(Output& output, const Input1& diff, const Input2& inv_cov) const {
        const T& dx = diff[0];
        const T& dy = diff[1];
        const T& a = inv_cov[0];
        const T& b = inv_cov[1];
        const T& c = inv_cov[2];
        
        output[0] = a * dx * dx + T(2) * b * dx * dy + c * dy * dy;
    }
    
    // backward: Gradient propagation
    __device__ void backward(const Output& output, Input1& diff, Input2& inv_cov) const {
        const T& output_grad = output.grad(0);
        const T& dx = diff[0];
        const T& dy = diff[1];
        const T& a = inv_cov[0];
        const T& b = inv_cov[1];
        const T& c = inv_cov[2];
        
        // Gradient w.r.t diff vector
        // d/d(dx) = 2*a*dx + 2*b*dy
        // d/d(dy) = 2*b*dx + 2*c*dy
        diff.add_grad(0, output_grad * (T(2) * a * dx + T(2) * b * dy));
        diff.add_grad(1, output_grad * (T(2) * b * dx + T(2) * c * dy));
        
        // Gradient w.r.t inverse covariance matrix parameters
        // d/d(a) = dx^2
        // d/d(b) = 2*dx*dy
        // d/d(c) = dy^2
        inv_cov.add_grad(0, output_grad * dx * dx);
        inv_cov.add_grad(1, output_grad * T(2) * dx * dy);
        inv_cov.add_grad(2, output_grad * dy * dy);
    }
};

// Mahalanobis distance calculation with 2D center point
// Input1: 2D query point [x, y]
// Input2: 2D center point [cx, cy]
// Input3: 3-parameter inverse covariance matrix [a, b, c]
// Output: scalar Mahalanobis distance squared
template <typename Input1, typename Input2, typename Input3>
requires TernaryLogicParameterConcept<Input1, Input2, Input3> && 
         (Input1::size == 2) && (Input2::size == 2) && (Input3::size == 3)
struct MahalanobisDistanceWithCenterLogic {
    using T = typename Input1::value_type;
    static_assert(std::is_same_v<T, typename Input2::value_type>, "Input types must match");
    static_assert(std::is_same_v<T, typename Input3::value_type>, "Input types must match");
    static constexpr std::size_t Dim = 1;  // scalar output
    using Output = Variable<Dim, T>;
    
    static constexpr std::size_t outputDim = Dim;
    
    __host__ __device__ MahalanobisDistanceWithCenterLogic() = default;
    
    // forward: Compute difference then Mahalanobis distance
    __device__ void forward(Output& output, const Input1& point, const Input2& center, const Input3& inv_cov) const {
        const T dx = point[0] - center[0];
        const T dy = point[1] - center[1];
        const T& a = inv_cov[0];
        const T& b = inv_cov[1];
        const T& c = inv_cov[2];
        
        output[0] = a * dx * dx + T(2) * b * dx * dy + c * dy * dy;
    }
    
    // backward: Gradient propagation
    __device__ void backward(const Output& output, Input1& point, Input2& center, Input3& inv_cov) const {
        const T& output_grad = output.grad(0);
        const T dx = point[0] - center[0];
        const T dy = point[1] - center[1];
        const T& a = inv_cov[0];
        const T& b = inv_cov[1];
        const T& c = inv_cov[2];
        
        // Gradient w.r.t point
        T grad_dx = output_grad * (T(2) * a * dx + T(2) * b * dy);
        T grad_dy = output_grad * (T(2) * b * dx + T(2) * c * dy);
        
        point.add_grad(0, grad_dx);
        point.add_grad(1, grad_dy);
        
        // Gradient w.r.t center (opposite of point gradient)
        center.add_grad(0, -grad_dx);
        center.add_grad(1, -grad_dy);
        
        // Gradient w.r.t inverse covariance matrix parameters
        inv_cov.add_grad(0, output_grad * dx * dx);
        inv_cov.add_grad(1, output_grad * T(2) * dx * dy);
        inv_cov.add_grad(2, output_grad * dy * dy);
    }
};

// Factory function for Mahalanobis distance with difference vector
template <typename Input1, typename Input2>
requires BinaryLogicParameterConcept<Input1, Input2> && 
         (Input1::size == 2) && (Input2::size == 3)
__device__ auto mahalanobis_distance(Input1& diff, Input2& inv_cov) {
    using LogicType = MahalanobisDistanceLogic<Input1, Input2>;
    
    LogicType logic;
    return BinaryOperation<LogicType::outputDim, LogicType, Input1, Input2>(logic, diff, inv_cov);
}

// Factory function for Mahalanobis distance with center point
template <typename Input1, typename Input2, typename Input3>
requires TernaryLogicParameterConcept<Input1, Input2, Input3> && 
         (Input1::size == 2) && (Input2::size == 2) && (Input3::size == 3)
__device__ auto mahalanobis_distance_with_center(Input1& point, Input2& center, Input3& inv_cov) {
    using LogicType = MahalanobisDistanceWithCenterLogic<Input1, Input2, Input3>;
    
    LogicType logic;
    return TernaryOperation<LogicType::outputDim, LogicType, Input1, Input2, Input3>(logic, point, center, inv_cov);
}

} // namespace op
} // namespace xyz_autodiff