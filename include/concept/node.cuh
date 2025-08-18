#pragma once

#include <cuda_runtime.h>
#include <concepts>
#include <cstddef>

namespace xyz_autodiff {

// 計算グラフのノードに要求されるコンセプト
template <typename T>
concept NodeConcept = requires(T node) {
    // 基本的な型情報
    typename T::value_type;
    { T::size } -> std::convertible_to<std::size_t>;
    
    // variable の取得（Variable は自分自身、Operation は output）
    { node.variable() } -> std::same_as<typename T::variable_type&>;
    { std::as_const(node).variable() } -> std::same_as<const typename T::variable_type&>;
    
    // 勾配初期化（自動伝播）
    { node.zero_grad() } -> std::same_as<void>;
    
    // 逆伝播（自動伝播）
    { node.backward() } -> std::same_as<void>;
    
    // 数値微分による逆伝播（自動伝播）
    { node.backward_numerical(typename T::value_type{}) } -> std::same_as<void>;
};

} // namespace xyz_autodiff