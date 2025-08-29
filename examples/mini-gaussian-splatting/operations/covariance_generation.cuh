#pragma once

#include <cuda_runtime.h>
#include <cmath>
#include <xyz_autodiff/concept/core_logic.cuh>
#include <xyz_autodiff/operations/operation.cuh>

namespace xyz_autodiff {
namespace op {

// Generate M matrix from scale and rotation for 2D Gaussian splatting
// Input1: 2D scale vector [sx, sy]
// Input2: 1D rotation angle [theta]
// Output: 2x2 matrix M = R * S where R is rotation matrix and S is diagonal scale matrix
template <typename Input1, typename Input2>
requires BinaryLogicParameterConcept<Input1, Input2> && 
         (Input1::size == 2) && (Input2::size == 1)
struct CovarianceMatrixGenerationLogic {
    using T = typename Input1::value_type;
    static_assert(std::is_same_v<T, typename Input2::value_type>, "Input types must match");
    static constexpr std::size_t Dim = 4;  // 2x2 matrix
    using Output = Variable<Dim, T>;
    
    static constexpr std::size_t outputDim = Dim;
    
    __host__ __device__ CovarianceMatrixGenerationLogic() = default;
    
    // forward: M = R * S = [[cos(θ), -sin(θ)], [sin(θ), cos(θ)]] * [[sx, 0], [0, sy]]
    //         = [[sx*cos(θ), -sy*sin(θ)], [sx*sin(θ), sy*cos(θ)]]
    __device__ void forward(Output& output, const Input1& scale, const Input2& rotation) const {
        const T& sx = scale[0];
        const T& sy = scale[1];
        const T& theta = rotation[0];
        
        T cos_theta = cos(theta);
        T sin_theta = sin(theta);
        
        // M = R * S (row-major storage: [0,1,2,3] = [[0,1],[2,3]])
        output[0] = sx * cos_theta;   // M[0,0]
        output[1] = -sy * sin_theta;  // M[0,1]
        output[2] = sx * sin_theta;   // M[1,0]
        output[3] = sy * cos_theta;   // M[1,1]
    }
    
    // backward: Gradient propagation
    __device__ void backward(const Output& output, Input1& scale, Input2& rotation) const {
        const T& sx = scale[0];
        const T& sy = scale[1];
        const T& theta = rotation[0];
        
        T cos_theta = cos(theta);
        T sin_theta = sin(theta);
        
        const T& grad_m00 = output.grad(0);
        const T& grad_m01 = output.grad(1);
        const T& grad_m10 = output.grad(2);
        const T& grad_m11 = output.grad(3);
        
        // Gradient w.r.t scale
        // dM[0,0]/dsx = cos(θ), dM[1,0]/dsx = sin(θ)
        scale.add_grad(0, grad_m00 * cos_theta + grad_m10 * sin_theta);
        
        // dM[0,1]/dsy = -sin(θ), dM[1,1]/dsy = cos(θ)
        scale.add_grad(1, grad_m01 * (-sin_theta) + grad_m11 * cos_theta);
        
        // Gradient w.r.t rotation
        // dM[0,0]/dθ = -sx*sin(θ), dM[0,1]/dθ = -sy*cos(θ)
        // dM[1,0]/dθ = sx*cos(θ), dM[1,1]/dθ = -sy*sin(θ)
        T grad_theta = grad_m00 * (-sx * sin_theta) + 
                       grad_m01 * (-sy * cos_theta) + 
                       grad_m10 * (sx * cos_theta) + 
                       grad_m11 * (-sy * sin_theta);
        
        rotation.add_grad(0, grad_theta);
    }
};

// Generate covariance matrix Σ = M * M^T from M matrix
// Input: 2x2 matrix M stored as 4 elements
// Output: 3-parameter symmetric matrix representing Σ
template <typename Input>
requires UnaryLogicParameterConcept<Input> && (Input::size == 4)
struct MatrixToCovariance3ParamLogic {
    using T = typename Input::value_type;
    static constexpr std::size_t Dim = 3;  // 3 parameters for symmetric 2x2 matrix
    using Output = Variable<Dim, T>;
    
    static constexpr std::size_t outputDim = Dim;
    
    static_assert(UnaryLogicParameterConcept<Input>, "Input must satisfy UnaryLogicParameterConcept");
    
    __host__ __device__ MatrixToCovariance3ParamLogic() = default;
    
    // forward: Σ = M * M^T for M = [[m00, m01], [m10, m11]]
    // Σ = [[m00*m00 + m01*m01, m00*m10 + m01*m11], 
    //      [m10*m00 + m11*m01, m10*m10 + m11*m11]]
    __device__ void forward(Output& output, const Input& M) const {
        const T& m00 = M[0];
        const T& m01 = M[1];
        const T& m10 = M[2];
        const T& m11 = M[3];
        
        // Σ[0,0] = m00^2 + m01^2
        output[0] = m00 * m00 + m01 * m01;
        
        // Σ[0,1] = Σ[1,0] = m00*m10 + m01*m11
        output[1] = m00 * m10 + m01 * m11;
        
        // Σ[1,1] = m10^2 + m11^2
        output[2] = m10 * m10 + m11 * m11;
    }
    
    // backward: Gradient propagation
    __device__ void backward(const Output& output, Input& M) const {
        const T& m00 = M[0];
        const T& m01 = M[1];
        const T& m10 = M[2];
        const T& m11 = M[3];
        
        const T& grad_s00 = output.grad(0);  // gradient w.r.t Σ[0,0]
        const T& grad_s01 = output.grad(1);  // gradient w.r.t Σ[0,1]
        const T& grad_s11 = output.grad(2);  // gradient w.r.t Σ[1,1]
        
        // dΣ[0,0]/dm00 = 2*m00, dΣ[0,1]/dm00 = m10
        M.add_grad(0, grad_s00 * T(2) * m00 + grad_s01 * m10);
        
        // dΣ[0,0]/dm01 = 2*m01, dΣ[0,1]/dm01 = m11
        M.add_grad(1, grad_s00 * T(2) * m01 + grad_s01 * m11);
        
        // dΣ[0,1]/dm10 = m00, dΣ[1,1]/dm10 = 2*m10
        M.add_grad(2, grad_s01 * m00 + grad_s11 * T(2) * m10);
        
        // dΣ[0,1]/dm11 = m01, dΣ[1,1]/dm11 = 2*m11
        M.add_grad(3, grad_s01 * m01 + grad_s11 * T(2) * m11);
    }
};

// Complete covariance generation from scale and rotation to 3-parameter representation
// Input1: 2D scale vector [sx, sy]
// Input2: 1D rotation angle [theta]
// Output: 3-parameter symmetric covariance matrix
template <typename Input1, typename Input2>
requires BinaryLogicParameterConcept<Input1, Input2> && 
         (Input1::size == 2) && (Input2::size == 1)
struct ScaleRotationToCovariance3ParamLogic {
    using T = typename Input1::value_type;
    static_assert(std::is_same_v<T, typename Input2::value_type>, "Input types must match");
    static constexpr std::size_t Dim = 3;  // 3 parameters for symmetric matrix
    using Output = Variable<Dim, T>;
    
    static constexpr std::size_t outputDim = Dim;
    
    __host__ __device__ ScaleRotationToCovariance3ParamLogic() = default;
    
    // forward: Complete computation from scale/rotation to covariance
    __device__ void forward(Output& output, const Input1& scale, const Input2& rotation) const {
        const T& sx = scale[0];
        const T& sy = scale[1];
        const T& theta = rotation[0];
        
        T cos_theta = cos(theta);
        T sin_theta = sin(theta);
        
        // M = R * S
        T m00 = sx * cos_theta;
        T m01 = -sy * sin_theta;
        T m10 = sx * sin_theta;
        T m11 = sy * cos_theta;
        
        // Σ = M * M^T
        output[0] = m00 * m00 + m01 * m01;  // Σ[0,0]
        output[1] = m00 * m10 + m01 * m11;  // Σ[0,1]
        output[2] = m10 * m10 + m11 * m11;  // Σ[1,1]
    }
    
    // backward: Combined gradient computation
    __device__ void backward(const Output& output, Input1& scale, Input2& rotation) const {
        const T& sx = scale[0];
        const T& sy = scale[1];
        const T& theta = rotation[0];
        
        T cos_theta = cos(theta);
        T sin_theta = sin(theta);
        
        T m00 = sx * cos_theta;
        T m01 = -sy * sin_theta;
        T m10 = sx * sin_theta;
        T m11 = sy * cos_theta;
        
        const T& grad_s00 = output.grad(0);
        const T& grad_s01 = output.grad(1);
        const T& grad_s11 = output.grad(2);
        
        // Chain rule: d/d(scale,rotation) = d/dM * dM/d(scale,rotation)
        T grad_m00 = grad_s00 * T(2) * m00 + grad_s01 * m10;
        T grad_m01 = grad_s00 * T(2) * m01 + grad_s01 * m11;
        T grad_m10 = grad_s01 * m00 + grad_s11 * T(2) * m10;
        T grad_m11 = grad_s01 * m01 + grad_s11 * T(2) * m11;
        
        // Gradient w.r.t scale
        scale.add_grad(0, grad_m00 * cos_theta + grad_m10 * sin_theta);
        scale.add_grad(1, grad_m01 * (-sin_theta) + grad_m11 * cos_theta);
        
        // Gradient w.r.t rotation
        T grad_theta = grad_m00 * (-sx * sin_theta) + 
                       grad_m01 * (-sy * cos_theta) + 
                       grad_m10 * (sx * cos_theta) + 
                       grad_m11 * (-sy * sin_theta);
        
        rotation.add_grad(0, grad_theta);
    }
};

// Factory functions
template <typename Input1, typename Input2>
requires BinaryLogicParameterConcept<Input1, Input2> && 
         (Input1::size == 2) && (Input2::size == 1)
__device__ auto generate_covariance_matrix(Input1& scale, Input2& rotation) {
    using LogicType = CovarianceMatrixGenerationLogic<Input1, Input2>;
    LogicType logic;
    return BinaryOperation<LogicType::outputDim, LogicType, Input1, Input2>(logic, scale, rotation);
}

template <typename Input>
requires UnaryLogicParameterConcept<Input> && (Input::size == 4)
__device__ auto matrix_to_covariance_3param(Input& M) {
    using LogicType = MatrixToCovariance3ParamLogic<Input>;
    LogicType logic;
    return UnaryOperation<LogicType::outputDim, LogicType, Input>(logic, M);
}

template <typename Input1, typename Input2>
requires BinaryLogicParameterConcept<Input1, Input2> && 
         (Input1::size == 2) && (Input2::size == 1)
__device__ auto scale_rotation_to_covariance_3param(Input1& scale, Input2& rotation) {
    using LogicType = ScaleRotationToCovariance3ParamLogic<Input1, Input2>;
    LogicType logic;
    return BinaryOperation<LogicType::outputDim, LogicType, Input1, Input2>(logic, scale, rotation);
}

} // namespace op
} // namespace xyz_autodiff