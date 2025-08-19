#pragma once

#include <cstddef>
#include <cuda_runtime.h>
#include "../operation.cuh"
#include "../../concept/variable.cuh"

namespace xyz_autodiff {

// Quaternion (x,y,z,w) から Rotation Matrix (3x3 = 9要素) への変換ロジック
template <std::size_t InputDim>
requires (InputDim == 4)  // Quaternionは4要素 (x,y,z,w)
struct QuaternionToRotationMatrixLogic {
    static constexpr std::size_t outputDim = 9;  // 3x3行列をflattenして9要素
    
    template <typename Output, typename Input>
    __host__ __device__ void forward(Output& output, const Input& input) const {
        using T = typename Input::value_type;
        
        // Quaternion要素を取得: (x, y, z, w)
        const T x = input[0];
        const T y = input[1]; 
        const T z = input[2];
        const T w = input[3];
        
        // 正規化されたquaternionを仮定
        // 回転行列の要素を計算
        // R = [r00 r01 r02]
        //     [r10 r11 r12]  
        //     [r20 r21 r22]
        
        // 第1行
        output[0] = T(1) - T(2) * (y*y + z*z);  // r00
        output[1] = T(2) * (x*y - z*w);         // r01
        output[2] = T(2) * (x*z + y*w);         // r02
        
        // 第2行
        output[3] = T(2) * (x*y + z*w);         // r10
        output[4] = T(1) - T(2) * (x*x + z*z);  // r11
        output[5] = T(2) * (y*z - x*w);         // r12
        
        // 第3行
        output[6] = T(2) * (x*z - y*w);         // r20
        output[7] = T(2) * (y*z + x*w);         // r21
        output[8] = T(1) - T(2) * (x*x + y*y);  // r22
    }
    
    template <typename Output, typename Input>
    __host__ __device__ void backward(const Output& output, Input& input) const {
        using T = typename Input::value_type;
        
        // Quaternion要素を取得
        const T x = input[0];
        const T y = input[1];
        const T z = input[2];
        const T w = input[3];
        
        // 回転行列の各要素に対する勾配から、quaternionの各要素への勾配を計算
        
        // ∂Loss/∂x の計算
        T grad_x = T(0);
        grad_x += output.grad(0) * T(0);                         // ∂r00/∂x = 0
        grad_x += output.grad(1) * T(2) * y;                     // ∂r01/∂x = 2y
        grad_x += output.grad(2) * T(2) * z;                     // ∂r02/∂x = 2z
        grad_x += output.grad(3) * T(2) * y;                     // ∂r10/∂x = 2y
        grad_x += output.grad(4) * T(-4) * x;                    // ∂r11/∂x = -4x
        grad_x += output.grad(5) * T(-2) * w;                    // ∂r12/∂x = -2w
        grad_x += output.grad(6) * T(2) * z;                     // ∂r20/∂x = 2z
        grad_x += output.grad(7) * T(2) * w;                     // ∂r21/∂x = 2w
        grad_x += output.grad(8) * T(-4) * x;                    // ∂r22/∂x = -4x
        
        // ∂Loss/∂y の計算
        T grad_y = T(0);
        grad_y += output.grad(0) * T(-4) * y;                    // ∂r00/∂y = -4y
        grad_y += output.grad(1) * T(2) * x;                     // ∂r01/∂y = 2x
        grad_y += output.grad(2) * T(2) * w;                     // ∂r02/∂y = 2w
        grad_y += output.grad(3) * T(2) * x;                     // ∂r10/∂y = 2x
        grad_y += output.grad(4) * T(0);                         // ∂r11/∂y = 0
        grad_y += output.grad(5) * T(2) * z;                     // ∂r12/∂y = 2z
        grad_y += output.grad(6) * T(-2) * w;                    // ∂r20/∂y = -2w
        grad_y += output.grad(7) * T(2) * z;                     // ∂r21/∂y = 2z
        grad_y += output.grad(8) * T(-4) * y;                    // ∂r22/∂y = -4y
        
        // ∂Loss/∂z の計算
        T grad_z = T(0);
        grad_z += output.grad(0) * T(-4) * z;                    // ∂r00/∂z = -4z
        grad_z += output.grad(1) * T(-2) * w;                    // ∂r01/∂z = -2w
        grad_z += output.grad(2) * T(2) * x;                     // ∂r02/∂z = 2x
        grad_z += output.grad(3) * T(2) * w;                     // ∂r10/∂z = 2w
        grad_z += output.grad(4) * T(-4) * z;                    // ∂r11/∂z = -4z
        grad_z += output.grad(5) * T(2) * y;                     // ∂r12/∂z = 2y
        grad_z += output.grad(6) * T(2) * x;                     // ∂r20/∂z = 2x
        grad_z += output.grad(7) * T(2) * y;                     // ∂r21/∂z = 2y
        grad_z += output.grad(8) * T(0);                         // ∂r22/∂z = 0
        
        // ∂Loss/∂w の計算
        T grad_w = T(0);
        grad_w += output.grad(0) * T(0);                         // ∂r00/∂w = 0
        grad_w += output.grad(1) * T(-2) * z;                    // ∂r01/∂w = -2z
        grad_w += output.grad(2) * T(2) * y;                     // ∂r02/∂w = 2y
        grad_w += output.grad(3) * T(2) * z;                     // ∂r10/∂w = 2z
        grad_w += output.grad(4) * T(0);                         // ∂r11/∂w = 0
        grad_w += output.grad(5) * T(-2) * x;                    // ∂r12/∂w = -2x
        grad_w += output.grad(6) * T(-2) * y;                    // ∂r20/∂w = -2y
        grad_w += output.grad(7) * T(2) * x;                     // ∂r21/∂w = 2x
        grad_w += output.grad(8) * T(0);                         // ∂r22/∂w = 0
        
        // 入力の勾配に加算
        input.add_grad(0, grad_x);
        input.add_grad(1, grad_y);
        input.add_grad(2, grad_z);
        input.add_grad(3, grad_w);
    }
};

// ヘルパー関数
namespace op {
    template <DifferentiableVariableConcept Input>
    requires (Input::size == 4)
    __device__ auto quaternion_to_rotation_matrix(Input& quaternion) {
        QuaternionToRotationMatrixLogic<4> logic;
        auto op = UnaryOperation<9, QuaternionToRotationMatrixLogic<4>, Input>(logic, quaternion);
        return op;
    }
}

} // namespace xyz_autodiff