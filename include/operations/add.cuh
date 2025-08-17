#pragma once

#include <cuda_runtime.h>
#include "../graph.cuh"
#include "../concept/operation.cuh"

namespace xyz_autodiff {

// 加算Operation
template <typename T>
class AddOperation {
public:
    using value_type = T;
    static constexpr std::size_t output_size = 1;
    
    // デフォルトコンストラクタ
    __host__ __device__ AddOperation() = default;
    
    template <typename Input1, typename Input2>
    __host__ __device__ auto operator()(const Input1& a, const Input2& b) const {
        return Node<AddOperation<T>, Input1, Input2>(*this, a, b);
    }
    
    // forward計算: result = a + b
    template <typename Input1, typename Input2>
    __device__ void forward(const Input1& a, const Input2& b, T& result) const {
        // 入力が値型かGraphかに応じて処理
        auto a_val = get_value(a);
        auto b_val = get_value(b);
        result = a_val + b_val;
    }
    
    // vjp計算: Vector-Jacobian Product for each input
    // vjp<0>: 第0引数(a)に対するvjp
    template <std::size_t idx, typename Input1, typename Input2>
    __device__ std::enable_if_t<idx == 0, T> vjp(const T& output_grad, const Input1& a, const Input2& b) const {
        // d(a+b)/da = 1
        return output_grad;
    }
    
    // vjp<1>: 第1引数(b)に対するvjp  
    template <std::size_t idx, typename Input1, typename Input2>
    __device__ std::enable_if_t<idx == 1, T> vjp(const T& output_grad, const Input1& a, const Input2& b) const {
        // d(a+b)/db = 1
        return output_grad;
    }

private:
    // 値の取得: Variable or Graph
    template <typename Input>
    __device__ auto get_value(const Input& input) const {
        if constexpr (requires { input.value(); }) {
            return input.value();  // Graph
        } else {
            return input[0];  // Variable (assuming single element)
        }
    }
};

// ヘルパー関数
template <typename T>
constexpr AddOperation<T> add{};

} // namespace xyz_autodiff