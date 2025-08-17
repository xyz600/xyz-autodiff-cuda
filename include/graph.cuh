#pragma once

#include <cstddef>
#include <tuple>
#include <cuda_runtime.h>
#include <utility>
#include "concept/operation.cuh"
#include "concept/variable.cuh"

namespace xyz_autodiff {

// 計算グラフのノード (即時評価版) - 可変長引数対応
template <typename Op, typename... Inputs>
class Node {
public:
    using value_type = typename Op::value_type;
    static constexpr std::size_t num_inputs = sizeof...(Inputs);
    static constexpr std::size_t output_size = Op::output_size;
    
private:
    Op operation_;
    std::tuple<const Inputs&...> inputs_;
    value_type computed_value_;
    
public:
    // コンストラクタ（即時評価）
    __host__ __device__ Node(const Op& op, const Inputs&... inputs)
        : operation_(op), inputs_(inputs...), computed_value_{} {
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
        compute_forward_impl(std::make_index_sequence<num_inputs>{});
    }
    
    // backward計算（自動微分の核心）
    __device__ void backward() const {
        // 勾配を初期化（自分に対する勾配は1）
        value_type unit_grad = value_type{1};
        
        // 各入力に対してvjpを実行
        backward_impl(unit_grad, std::make_index_sequence<num_inputs>{});
    }

private:
    // forward計算の実装（index_sequence を使って展開）
    template <std::size_t... Is>
    __device__ void compute_forward_impl(std::index_sequence<Is...>) {
        operation_.forward(std::get<Is>(inputs_)..., computed_value_);
    }
    
    // backward計算の実装（各入力に対してvjpを実行）
    template <std::size_t... Is>
    __device__ void backward_impl(const value_type& grad, std::index_sequence<Is...>) const {
        // 各入力に対してvjpを実行して勾配を累積
        (accumulate_grad_at_index<Is>(grad), ...);
    }
    
    // 特定のインデックスに対してvjpを実行
    template <std::size_t Idx>
    __device__ void accumulate_grad_at_index(const value_type& grad) const {
        auto vjp_grad = compute_vjp_at_index<Idx>(grad, std::make_index_sequence<num_inputs>{});
        accumulate_grad_to_input(std::get<Idx>(inputs_), vjp_grad);
    }
    
    // vjp計算（特定のインデックス）
    template <std::size_t Idx, std::size_t... Is>
    __device__ auto compute_vjp_at_index(const value_type& grad, std::index_sequence<Is...>) const {
        return operation_.template vjp<Idx>(grad, std::get<Is>(inputs_)...);
    }
    
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
    
    template <std::size_t Idx>
    __device__ const auto& input() const { 
        return std::get<Idx>(inputs_); 
    }
};

// ヘルパー関数：型推論を簡単にする
template <typename Op, typename... Inputs>
__host__ __device__ auto make_node(const Op& op, const Inputs&... inputs) {
    return Node<Op, Inputs...>(op, inputs...);
}

// 2引数特化版（後方互換性とCUDA最適化のため）
template <typename Op, typename Input1, typename Input2>
class Node<Op, Input1, Input2> {
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

} // namespace xyz_autodiff