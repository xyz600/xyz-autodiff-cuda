#pragma once

#include <concepts>
#include <type_traits>

namespace xyz_autodiff {

// 前方宣言
template <typename Op, typename Input1, typename Input2>
class Node;

// Expression Template パターンのための Operation concept

template <typename T, typename Input1, typename Input2>
concept OperationConcept = requires(T op, Input1 input1, Input2 input2) {
    // デフォルトコンストラクタ
    T{};
    
    // 型情報
    typename T::value_type;
    
    // output_size静的定数
    { T::output_size } -> std::convertible_to<std::size_t>;
    
    // operator() でNodeを返す (Expression Template)
    { op(input1, input2) } -> std::convertible_to<Node<T, Input1, Input2>>;
    
    // forward計算: forward(input1, input2, output)
    { op.forward(input1, input2, std::declval<typename T::value_type&>()) } -> std::same_as<void>;
    
    // vjp計算: 各入力に対するVector-Jacobian Product
    { op.template vjp<0>(std::declval<typename T::value_type&>(), input1, input2) } -> std::convertible_to<typename T::value_type>;
    { op.template vjp<1>(std::declval<typename T::value_type&>(), input1, input2) } -> std::convertible_to<typename T::value_type>;

} && std::is_default_constructible_v<T>;

} // namespace xyz_autodiff
