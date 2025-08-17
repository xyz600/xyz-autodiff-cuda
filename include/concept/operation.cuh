#pragma once

#include <concepts>
#include <type_traits>

namespace xyz_autodiff {

// Operation concept - シンプル版

template <typename T, typename... Inputs>
concept OperationConcept = requires(T op, Inputs... inputs) {
    // 型情報
    typename T::value_type;
    
    // output_size静的定数
    { T::output_size } -> std::convertible_to<std::size_t>;
    
    // forward計算: forward(inputs..., output)
    { op.forward(inputs..., std::declval<typename T::value_type&>()) } -> std::same_as<void>;
    
    // backward計算: backward(output_grad, inputs...)
    { op.backward(std::declval<typename T::value_type&>(), inputs...) } -> std::same_as<void>;
};

} // namespace xyz_autodiff
