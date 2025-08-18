#pragma once

#include <concepts>
#include <type_traits>
#include "node.cuh"

namespace xyz_autodiff {

// Forward propagation に必要な要件
template <typename T>
concept VariableConcept = NodeConcept<T> && requires(T var) {
    // 値データへのアクセス
    { var.data() } -> std::convertible_to<typename T::value_type*>;
    { var.data() } -> std::convertible_to<const typename T::value_type*>;
    
    // インデックスアクセス (値)
    { var[std::size_t{}] } -> std::convertible_to<typename T::value_type&>;
    { var[std::size_t{}] } -> std::convertible_to<const typename T::value_type&>;
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
};

} // namespace xyz_autodiff