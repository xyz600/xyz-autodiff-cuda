#pragma once

#include <concepts>
#include <type_traits>

namespace xyz_autodiff {

// 前方宣言
template <typename Op, typename Input1, typename Input2>
class Graph;

// Expression Template パターンのための Operation concept (CUDA compiler limitations)
/*
template <typename T, typename... Inputs>
concept OperationConcept = requires(T op, Inputs... inputs) {
    // デフォルトコンストラクタ
    T{};
    
    // 型情報
    typename T::value_type;
    
    // operator() でGraphを返す (Expression Template)
    { op(inputs...) } -> std::convertible_to<Graph<T, Inputs...>>;
    
    // forward計算: forward(inputs..., output)
    { op.forward(inputs..., std::declval<typename T::value_type&>()) } -> std::same_as<void>;
    
    // vjp計算: 各入力に対するVector-Jacobian Product
    // vjp<idx>(output_grad, inputs...) で idx番目の入力に対するvjpを計算
} && std::is_default_constructible_v<T>;
*/

// Operation要件:
// 1. デフォルトコンストラクタ
// 2. value_type型定義
// 3. operator()(inputs...) -> Graph<Op, Inputs...>
// 4. forward(inputs..., output&) -> void
// 5. vjp<idx>(output_grad, inputs...) -> value_type

// Graphが持つべき機能 (CUDA compiler limitations)
/*
template <typename T>
concept GraphConcept = requires(T graph) {
    // backward計算
    { graph.backward() } -> std::same_as<void>;
    
    // 値アクセス
    typename T::value_type;
    
    // Variable concept の要件も満たす
} && std::is_copy_constructible_v<T>;
*/

// Graph要件:
// 1. backward() -> void
// 2. value() -> const value_type&
// 3. Variable concept の要件

} // namespace xyz_autodiff
