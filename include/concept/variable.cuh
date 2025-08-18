#pragma once

#include <concepts>
#include <type_traits>

namespace xyz_autodiff {

// Forward propagation に必要な要件
template <typename T>
concept VariableConcept = requires(T var) {
    // 基本的な型情報
    typename T::value_type;
    { T::size } -> std::convertible_to<std::size_t>;
    
    // 値データへのアクセス
    { var.data() } -> std::convertible_to<typename T::value_type*>;
    { var.data() } -> std::convertible_to<const typename T::value_type*>;
    
    // インデックスアクセス (値)
    { var[std::size_t{}] } -> std::convertible_to<typename T::value_type&>;
    { var[std::size_t{}] } -> std::convertible_to<const typename T::value_type&>;
    
    // 勾配初期化
    { var.zero_grad() } -> std::same_as<void>;
};

// Backward propagation に必要な要件
template <typename T>
concept DifferentiableVariableConcept = VariableConcept<T> && requires(T var) {
    // 勾配データへのアクセス
    { var.grad() } -> std::convertible_to<typename T::value_type*>;
    { var.grad() } -> std::convertible_to<const typename T::value_type*>;
    
    // インデックスアクセス (勾配)
    { var.grad(std::size_t{}) } -> std::convertible_to<typename T::value_type&>;
    { var.grad(std::size_t{}) } -> std::convertible_to<const typename T::value_type&>;
    
    // 勾配初期化
    { var.zero_grad() } -> std::same_as<void>;
};

} // namespace xyz_autodiff