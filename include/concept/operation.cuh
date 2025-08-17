#pragma once

#include <concepts>
#include <type_traits>

namespace xyz_autodiff {

// Forward propagation のための Operation
template <typename T, typename Output, typename... Inputs>
concept OperationConcept = requires(T op, Output output, Inputs... inputs) {
    // オペレーションは複数の入力を受け取り、出力に結果を書き込む
    { op.forward(inputs..., output) } -> std::same_as<void>;
} && std::is_default_constructible_v<T>;

// ForwardとBackward両方をサポートする微分可能Operation
template <typename T, typename Output, typename... Inputs>
concept DifferentiableOperationConcept = OperationConcept<T, Output, Inputs...> && 
    requires(T op, Output output, Inputs... inputs) {
    // VJP (Vector-Jacobian Product) による逆伝播
    // outputのgradientを受け取って、inputsのgradientに累積
    { op.backward(output, inputs...) } -> std::same_as<void>;
};

// 数値微分をサポートするOperation (テスト用)
template <typename T, typename Output, typename... Inputs>
concept TestableOperationConcept = DifferentiableOperationConcept<T, Output, Inputs...> && 
    requires(T op, Output output, Inputs... inputs) {
    { op.numerical_backward(output, inputs...) } -> std::same_as<void>;
};

} // namespace xyz_autodiff