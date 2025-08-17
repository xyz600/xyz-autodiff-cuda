#pragma once

#include <concepts>
#include <type_traits>

namespace xyz_autodiff {

// Forward propagation に必要な要件
template <typename T>
concept VariableConcept = requires(T var) {
    // コンパイル時に決定されるサイズ
    { T::size } -> std::convertible_to<std::size_t>;
    
    // 値データへのアクセス
    { var.data() } -> std::convertible_to<typename T::value_type*>;
    { var.data() } -> std::convertible_to<const typename T::value_type*>;
    
    // インデックスアクセス (値)
    { var[std::size_t{}] } -> std::convertible_to<typename T::value_type&>;
    { var[std::size_t{}] } -> std::convertible_to<const typename T::value_type&>;
    
    // 型情報
    typename T::value_type;
} && std::is_copy_constructible_v<T>;

// Backward propagation に必要な要件
template <typename T>
concept DifferentiableVariableConcept = VariableConcept<T> && requires(T var) {
    // 勾配データへのアクセス
    { var.grad() } -> std::convertible_to<typename T::value_type*>;
    { var.grad() } -> std::convertible_to<const typename T::value_type*>;
    
    // インデックスアクセス (勾配)
    { var.grad(std::size_t{}) } -> std::convertible_to<typename T::value_type&>;
    { var.grad(std::size_t{}) } -> std::convertible_to<const typename T::value_type&>;
    
    // 勾配操作
    { var.zero_grad() } -> std::same_as<void>;
    { var.accumulate_grad(std::declval<const typename T::value_type*>()) } -> std::same_as<void>;
};

} // namespace xyz_autodiff