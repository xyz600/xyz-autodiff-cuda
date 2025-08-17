#pragma once

#include <cuda_runtime.h>
#include "../concept/core_logic.cuh"
#include "operation.cuh"

namespace xyz_autodiff {
namespace op {
    
// 加算ロジック（2入力1出力用）
template <typename Input1, typename Input2>
requires BinaryLogicParameterConcept<Input1, Input2>
struct AddLogic {
    using T = typename Input1::value_type;
    using Output = Variable<T, 1>;
    
    // 出力次元をconstexprで定義
    static constexpr std::size_t outputDim = 1;
    
    // デフォルトコンストラクタ
    __host__ __device__ AddLogic() = default;
    
    // forward: 出力に結果を書き込む
    __device__ void forward(Output& output, const Input1& input1, const Input2& input2) const {
        output[0] = input1[0] + input2[0];
    }
    
    // backward: 入力の勾配に結果を書き込む
    __device__ void backward(const Output& output, Input1& input1, Input2& input2) const {
        const T& output_grad = output.grad(0);
        input1.accumulate_grad(&output_grad);
        input2.accumulate_grad(&output_grad);
    }
};

// BinaryOperationを返すファクトリ関数
template <typename Input1, typename Input2>
requires BinaryLogicParameterConcept<Input1, Input2>
__host__ __device__ auto add(const Input1& input1, const Input2& input2) {
    using LogicType = AddLogic<Input1, Input2>;
    LogicType logic;
    return BinaryOperation<LogicType::outputDim, LogicType, Input1, Input2>(logic, input1, input2);
}

// BinaryOperationRefを返すファクトリ関数（外部バッファ版）
template <typename Input1, typename Input2>
requires BinaryLogicParameterConcept<Input1, Input2>
__host__ __device__ auto add_ref(const Input1& input1, const Input2& input2, 
                                      Variable<typename Input1::value_type, AddLogic<Input1, Input2>::outputDim>& output) {
    using LogicType = AddLogic<Input1, Input2>;
    LogicType logic;
    return BinaryOperationRef<LogicType::outputDim, LogicType, Input1, Input2>(logic, input1, input2, output);
}

} // namespace op
} // namespace xyz_autodiff
