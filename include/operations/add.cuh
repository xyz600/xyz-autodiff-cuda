#pragma once

#include <cuda_runtime.h>
#include "../concept/operation.cuh"
#include "../concept/variable.cuh"

namespace xyz_autodiff {

// 加算Operation（2入力1出力、Variable based）
template <typename T, typename Input1, typename Input2, typename Output>
requires DifferentiableVariableConcept<Input1> && DifferentiableVariableConcept<Input2> && 
         VariableConcept<Output> &&
         (Input1::size == Input2::size) &&
         std::is_same_v<typename Input1::value_type, T> &&
         std::is_same_v<typename Input2::value_type, T> &&
         std::is_same_v<typename Output::value_type, T>
class AddOperation {
public:
    using value_type = T;
    using input1_type = Input1;
    using input2_type = Input2;
    using output_type = Output;
    static constexpr std::size_t output_size = 1;
    
    // デフォルトコンストラクタ
    AddOperation() = default;
    
    // forward計算: 2つの入力を受け取って出力Variableに結果を書き込む
    __device__ void forward(const Input1& input1, const Input2& input2, Output& output) const {
        output[0] = input1[0] + input2[0];
    }
    
    // backward計算: 出力gradと全入力を受け取って、各inputのgradにvjpを書き込む
    __device__ void backward(const Output& output, Input1& input1, Input2& input2) const {
        // d(a+b)/da = 1, d(a+b)/db = 1
        // 出力の勾配を各入力に伝播
        const T& output_grad = output.grad(0);
        input1.accumulate_grad(&output_grad);
        input2.accumulate_grad(&output_grad);
    }
};

// CTAD（Class Template Argument Deduction）用の推論ガイド
template <typename Input1, typename Input2, typename Output>
requires DifferentiableVariableConcept<Input1> && DifferentiableVariableConcept<Input2> && 
         VariableConcept<Output> &&
         (Input1::size == Input2::size) &&
         std::is_same_v<typename Input1::value_type, typename Input2::value_type> &&
         std::is_same_v<typename Input1::value_type, typename Output::value_type>
AddOperation(const Input1&, const Input2&, const Output&) 
    -> AddOperation<typename Input1::value_type, Input1, Input2, Output>;

} // namespace xyz_autodiff