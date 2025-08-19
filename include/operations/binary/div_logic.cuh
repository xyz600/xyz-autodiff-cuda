#pragma once

#include <cuda_runtime.h>
#include "../../concept/core_logic.cuh"
#include "../operation.cuh"

namespace xyz_autodiff {
namespace op {
    
// 要素毎の除算ロジック（element-wise division）
template <typename Input1, typename Input2>
requires BinaryLogicParameterConcept<Input1, Input2> && (Input1::size == Input2::size)
struct DivLogic {
    using T = typename Input1::value_type;
    static constexpr std::size_t Dim = Input1::size;
    using Output = Variable<Dim, T>;
    
    // 出力次元をconstexprで定義
    static constexpr std::size_t outputDim = Dim;
    
    // デフォルトコンストラクタ
    __host__ __device__ DivLogic() = default;
    
    // forward: 出力に結果を書き込む（element-wise）
    __device__ void forward(Output& output, const Input1& input1, const Input2& input2) const {
        for (std::size_t i = 0; i < Dim; ++i) {
            output[i] = input1[i] / input2[i];
        }
    }
    
    // backward: 入力の勾配に結果を書き込む（element-wise）
    __device__ void backward(const Output& output, Input1& input1, Input2& input2) const {
        // 除算の微分: d/dx1(x1[i]/x2[i]) = 1/x2[i], d/dx2(x1[i]/x2[i]) = -x1[i]/(x2[i])^2
        for (std::size_t i = 0; i < Dim; ++i) {
            const T& output_grad = output.grad(i);
            input1.add_grad(i, output_grad / input2[i]);
            input2.add_grad(i, -output_grad * input1[i] / (input2[i] * input2[i]));
        }
    }
};

// BinaryOperationを返すファクトリ関数
template <typename Input1, typename Input2>
requires BinaryLogicParameterConcept<Input1, Input2>
__device__ auto div(Input1& input1, Input2& input2) {
    using LogicType = DivLogic<Input1, Input2>;
    LogicType logic;
    auto op = BinaryOperation<LogicType::outputDim, LogicType, Input1, Input2>(logic, input1, input2);
    return op;
}

} // namespace op
} // namespace xyz_autodiff