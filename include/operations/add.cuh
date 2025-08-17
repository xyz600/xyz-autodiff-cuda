#pragma once

#include <cuda_runtime.h>
#include "../graph.cuh"
#include "../concept/operation.cuh"

namespace xyz_autodiff {

// 加算Operation（可変長引数対応）
template <typename T>
class AddOperation {
public:
    using value_type = T;
    static constexpr std::size_t output_size = 1;
    
    // デフォルトコンストラクタ
    __host__ __device__ AddOperation() = default;
    
    // 可変長引数版
    template <typename... Inputs>
    __host__ __device__ auto operator()(const Inputs&... inputs) const {
        return Node<AddOperation<T>, Inputs...>(*this, inputs...);
    }
    
    // forward計算: 2引数版（最適化）
    template <typename Input1, typename Input2>
    __device__ void forward(const Input1& input1, const Input2& input2, T& result) const {
        result = get_value(input1) + get_value(input2);
    }
    
    // forward計算: 可変長引数版
    template <typename... Inputs>
    requires (sizeof...(Inputs) > 2)
    __device__ void forward(const Inputs&... inputs, T& result) const {
        result = T{0};
        sum_inputs(result, inputs...);
    }
    
    // vjp計算: Vector-Jacobian Product for each input
    template <std::size_t Idx, typename... Inputs>
    __device__ T vjp(const T& output_grad, const Inputs&... inputs) const {
        // d(sum)/d(input_i) = 1 for all inputs
        return output_grad;
    }

private:
    // 再帰的にすべての入力を加算
    template <typename Input>
    __device__ void sum_inputs(T& result, const Input& input) const {
        result += get_value(input);
    }
    
    template <typename Input, typename... Rest>
    __device__ void sum_inputs(T& result, const Input& input, const Rest&... rest) const {
        result += get_value(input);
        sum_inputs(result, rest...);
    }
    
    // 値の取得: Variable or Node
    template <typename Input>
    __device__ auto get_value(const Input& input) const {
        if constexpr (requires { input.value(); }) {
            return input.value();  // Node
        } else {
            return input[0];  // Variable (assuming single element)
        }
    }
};

// ヘルパー関数
template <typename T>
constexpr AddOperation<T> add{};

} // namespace xyz_autodiff