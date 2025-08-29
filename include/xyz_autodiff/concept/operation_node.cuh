#pragma once

#include <cuda_runtime.h>
#include <concepts>
#include <cstddef>

namespace xyz_autodiff {

// 計算グラフの中間ノード（Operation）に要求されるコンセプト
template <typename T>
concept OperationNode = requires(T node) {
    // 基本的な型情報
    typename T::value_type;
    { T::size } -> std::convertible_to<std::size_t>;
    
    { node.forward() } -> std::same_as<void>;

    // 勾配初期化
    { node.zero_grad() } -> std::same_as<void>;
    
    // 逆伝播
    { node.backward() } -> std::same_as<void>;
    
    // 数値微分による逆伝播
    { node.backward_numerical(typename T::value_type{}) } -> std::same_as<void>;
};

} // namespace xyz_autodiff