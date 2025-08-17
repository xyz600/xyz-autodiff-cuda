#pragma once

#include <cstddef>
#include <tuple>
#include <cuda_runtime.h>
#include "concept/operation.cuh"
#include "concept/variable.cuh"

namespace xyz_autodiff {

// 計算グラフのノード (即時評価版) - 2引数特化
template <typename Op, typename Input1, typename Input2>
class Node {
public:
    using value_type = typename Op::value_type;
    static constexpr std::size_t num_inputs = 2;
    static constexpr std::size_t output_size = Op::output_size;
    
private:
    Op operation_;
    const Input1& input1_;
    const Input2& input2_;
    value_type computed_value_;
    
public:
    // コンストラクタ（即時評価）
    __host__ __device__ Node(const Op& op, const Input1& input1, const Input2& input2)
        : operation_(op), input1_(input1), input2_(input2), computed_value_{} {
        // 即座にforward計算を実行
        compute_forward();
    }
    
    // コピー・ムーブ禁止
    Node(const Node&) = delete;
    Node(Node&&) = delete;
    Node& operator=(const Node&) = delete;
    Node& operator=(Node&&) = delete;
    
    // === Variable concept の要件 ===
    
    // サイズ（Operationによって決定）
    static constexpr std::size_t size = Op::output_size;
    
    // 値の取得（即座に利用可能）
    __device__ const value_type& value() const {
        return computed_value_;
    }
    
    // インデックスアクセス（単一要素の場合）
    template <std::size_t N = size>
    __device__ std::enable_if_t<N == 1, value_type&> operator[](std::size_t) const {
        return const_cast<value_type&>(computed_value_);
    }
    
    // === Node の要件 ===
    
    // forward計算（即時実行）
    __device__ void compute_forward() {
        operation_.forward(input1_, input2_, computed_value_);
    }
    
    // backward計算（自動微分の核心）
    __device__ void backward() const {
        // 勾配を初期化（自分に対する勾配は1）
        value_type unit_grad = value_type{1};
        
        // 各入力に対してvjpを実行
        auto vjp_grad1 = operation_.template vjp<0>(unit_grad, input1_, input2_);
        auto vjp_grad2 = operation_.template vjp<1>(unit_grad, input1_, input2_);
        
        // 勾配を累積
        accumulate_grad_to_input(input1_, vjp_grad1);
        accumulate_grad_to_input(input2_, vjp_grad2);
    }

private:
    
    // 入力に勾配を累積するヘルパー
    template <typename Input>
    __device__ void accumulate_grad_to_input(const Input& input, const value_type& grad) const {
        if constexpr (requires { input.accumulate_grad(&grad); }) {
            input.accumulate_grad(&grad);  // Variable
        } else if constexpr (requires { input.backward(); }) {
            // Node の場合はさらにbackwardを実行（チェーンルール）
            input.backward();
        }
    }

public:
    
    // Operation と inputs へのアクセス
    __device__ const Op& operation() const { return operation_; }
    __device__ const Input1& input1() const { return input1_; }
    __device__ const Input2& input2() const { return input2_; }
};

// ヘルパー関数：型推論を簡単にする
template <typename Op, typename Input1, typename Input2>
__host__ __device__ auto make_node(const Op& op, const Input1& input1, const Input2& input2) {
    return Node<Op, Input1, Input2>(op, input1, input2);
}

} // namespace xyz_autodiff