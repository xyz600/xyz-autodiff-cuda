#pragma once

#include <cuda_runtime.h>
#include "../concept/operation.cuh"
#include "../concept/variable.cuh"

namespace xyz_autodiff {

// 加算Operation（2入力専用、型パラメータ付き）
template <typename T, typename Input1, typename Input2>
requires DifferentiableVariableConcept<Input1> && DifferentiableVariableConcept<Input2> && 
         (Input1::size == Input2::size) &&
         std::is_same_v<typename Input1::value_type, T> &&
         std::is_same_v<typename Input2::value_type, T>
class AddOperation {
public:
    using value_type = T;
    using input1_type = Input1;
    using input2_type = Input2;
    static constexpr std::size_t output_size = 1;
    
    // デフォルトコンストラクタ
    AddOperation() = default;
    
    // forward計算: result = input1 + input2
    __device__ void forward(const Input1& input1, const Input2& input2, T& result) const {
        result = input1[0] + input2[0];
    }
    
    // backward計算: 勾配を直接入力のgradに書き込み
    __device__ void backward(const T& output_grad, Input1& input1, Input2& input2) const {
        // d(a+b)/da = 1, d(a+b)/db = 1
        // 各入力の勾配に累積
        input1.accumulate_grad(&output_grad);
        input2.accumulate_grad(&output_grad);
    }
};

} // namespace xyz_autodiff