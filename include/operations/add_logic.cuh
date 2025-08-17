#pragma once

#include <cuda_runtime.h>
#include "../concept/core_logic.cuh"
#include "../operation.cuh"

namespace xyz_autodiff {

// 加算ロジック（2入力1出力用）
template <typename T, typename Input1, typename Input2, typename Output>
requires BinaryLogicParameterConcept<Input1, Input2, Output, T>
struct AddLogic {
    // デフォルトコンストラクタ
    __host__ __device__ AddLogic() = default;
    
    // forward: 出力に結果を書き込む
    __device__ void forward(Output& output, const Input1& input1, const Input2& input2) const {
        output[0] = input1[0] + input2[0];
    }
    
    // backward: 入力の勾配に結果を書き込む
    __device__ void backward(const Output& output, Input1& input1, Input2& input2) const {
        // d(a+b)/da = 1, d(a+b)/db = 1
        // 出力の勾配を各入力に伝播
        const T& output_grad = output.grad(0);
        input1.accumulate_grad(&output_grad);
        input2.accumulate_grad(&output_grad);
    }
};

// 型推論のためのヘルパー関数
template <typename T, typename Input1, typename Input2, typename Output>
requires BinaryLogicParameterConcept<Input1, Input2, Output, T>
__host__ __device__ auto make_add_logic() {
    return AddLogic<T, Input1, Input2, Output>{};
}

// BinaryOperationを返すファクトリ関数
template <typename Input1, typename Input2, std::size_t OutputSize>
requires BinaryLogicParameterConcept<Input1, Input2, Variable<typename Input1::value_type, OutputSize>, typename Input1::value_type>
__host__ __device__ auto make_add(const Input1& input1, const Input2& input2) {
    using T = typename Input1::value_type;
    using Output = Variable<T, OutputSize>;
    AddLogic<T, Input1, Input2, Output> logic;
    return make_binary_op<AddLogic<T, Input1, Input2, Output>, Input1, Input2, OutputSize>(
        logic, input1, input2);
}

} // namespace xyz_autodiff
