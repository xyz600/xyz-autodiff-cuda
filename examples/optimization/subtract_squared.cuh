#pragma once

#include <cuda_runtime.h>
#include "../../include/concept/core_logic.cuh"
#include "../../include/operations/operation.cuh"

namespace xyz_autodiff {
namespace op {

// f(x) = (x - c)^2 を表現するカスタムオペレーション
template <typename Input>
requires UnaryLogicParameterConcept<Input>
struct SubtractAndSquareLogic {
    using T = typename Input::value_type;
    static constexpr std::size_t Dim = Input::size;
    using Output = Variable<T, Dim>;
    
    // 出力次元をconstexprで定義
    static constexpr std::size_t outputDim = Dim;
    
    // static_assert for concept validation
    static_assert(UnaryLogicParameterConcept<Input>, "Input must satisfy UnaryLogicParameterConcept");
    
    T constant_c;  // 定数 c を格納
    
    // コンストラクタで定数cを受け取る
    __host__ __device__ explicit SubtractAndSquareLogic(T c) : constant_c(c) {}
    
    // forward: f(x) = (x - c)^2 を計算
    __device__ void forward(Output& output, const Input& input) const {
        for (std::size_t i = 0; i < Dim; ++i) {
            T diff = input[i] - constant_c;
            output[i] = diff * diff;
        }
    }
    
    // backward: df/dx = 2 * (x - c) を計算
    __device__ void backward(const Output& output, Input& input) const {
        for (std::size_t i = 0; i < Dim; ++i) {
            const T& output_grad = output.grad(i);
            T diff = input[i] - constant_c;
            input.add_grad(i, output_grad * 2.0 * diff);
        }
    }
};

// UnaryOperationを返すファクトリ関数
template <typename Input>
requires UnaryLogicParameterConcept<Input>
__device__ auto subtract_and_square(Input& input, typename Input::value_type constant_c) {
    using LogicType = SubtractAndSquareLogic<Input>;
    
    LogicType logic(constant_c);
    return UnaryOperation<LogicType::outputDim, LogicType, Input>(logic, input);
}

} // namespace op
} // namespace xyz_autodiff