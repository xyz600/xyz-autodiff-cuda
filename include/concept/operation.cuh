#pragma once

#include <concepts>
#include <type_traits>

namespace xyz_autodiff {

// 前方宣言
template <typename Op, typename... Inputs>
class Node;

// Helper: 特定のインデックスに対するvjpが存在するかチェック
template <typename T, std::size_t Idx, typename... Inputs>
concept HasVjpAt = requires(T op, typename T::value_type grad, Inputs... inputs) {
    { op.template vjp<Idx>(grad, inputs...) } -> std::convertible_to<typename T::value_type>;
};

// Helper: インデックスシーケンスに対してすべてのvjpをチェック
template <typename T, typename IndexSeq, typename... Inputs>
struct CheckAllVjps;

template <typename T, std::size_t... Is, typename... Inputs>
struct CheckAllVjps<T, std::index_sequence<Is...>, Inputs...> {
    static constexpr bool value = (HasVjpAt<T, Is, Inputs...> && ...);
};

// Expression Template パターンのための Operation concept

template <typename T, typename... Inputs>
concept OperationConcept = requires(T op, Inputs... inputs) {
    // デフォルトコンストラクタ
    T{};
    
    // 型情報
    typename T::value_type;
    
    // output_size静的定数
    { T::output_size } -> std::convertible_to<std::size_t>;
    
    // operator() でNodeを返す (Expression Template)
    { op(inputs...) } -> std::convertible_to<Node<T, Inputs...>>;
    
    // forward計算: forward(inputs..., output)
    { op.forward(inputs..., std::declval<typename T::value_type&>()) } -> std::same_as<void>;
} && std::is_default_constructible_v<T> 
  && CheckAllVjps<T, std::make_index_sequence<sizeof...(Inputs)>, Inputs...>::value;

} // namespace xyz_autodiff
