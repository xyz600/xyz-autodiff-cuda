#pragma once

#include <concepts>
#include <type_traits>
#include "variable.cuh"

namespace xyz_autodiff {

// 1入力1出力のOperation concept（UnaryOperationなど用）
template <typename T, typename Input, typename Output>
concept UnaryOperationConcept = 
    VariableConcept<Input> && VariableConcept<Output> &&
    requires(T op, Input input, Output output) {
    // 型情報
    typename T::value_type;
    typename T::input_type;
    typename T::output_type;
    
    // サイズ情報
    { T::output_size } -> std::convertible_to<std::size_t>;
    
    // forward計算: 1つの入力を受け取って出力Variableに結果を書き込む
    { op.forward(input, output) } -> std::same_as<void>;
    
    // backward計算: 出力gradと入力を受け取って、inputのgradにvjpを書き込む
    { op.backward(output, input) } -> std::same_as<void>;
};

// 2入力1出力のOperation concept（AddOperation、MulOperationなど用）
template <typename T, typename Input1, typename Input2, typename Output>
concept BinaryOperationConcept = 
    VariableConcept<Input1> && VariableConcept<Input2> && VariableConcept<Output> &&
    requires(T op, Input1 input1, Input2 input2, Output output) {
    // 型情報
    typename T::value_type;
    typename T::input1_type;
    typename T::input2_type;
    typename T::output_type;
    
    // サイズ情報
    { T::output_size } -> std::convertible_to<std::size_t>;
    
    // forward計算: 2つの入力を受け取って出力Variableに結果を書き込む
    { op.forward(input1, input2, output) } -> std::same_as<void>;
    
    // backward計算: 出力gradと全入力を受け取って、各inputのgradにvjpを書き込む
    { op.backward(output, input1, input2) } -> std::same_as<void>;
};

// 3入力1出力のOperation concept（FusedMultiplyAddなど用）
template <typename T, typename Input1, typename Input2, typename Input3, typename Output>
concept TernaryOperationConcept = 
    VariableConcept<Input1> && VariableConcept<Input2> && VariableConcept<Input3> && VariableConcept<Output> &&
    requires(T op, Input1 input1, Input2 input2, Input3 input3, Output output) {
    // 型情報
    typename T::value_type;
    typename T::input1_type;
    typename T::input2_type;
    typename T::input3_type;
    typename T::output_type;
    
    // サイズ情報
    { T::output_size } -> std::convertible_to<std::size_t>;
    
    // forward計算: 3つの入力を受け取って出力Variableに結果を書き込む
    { op.forward(input1, input2, input3, output) } -> std::same_as<void>;
    
    // backward計算: 出力gradと全入力を受け取って、各inputのgradにvjpを書き込む
    { op.backward(output, input1, input2, input3) } -> std::same_as<void>;
};

} // namespace xyz_autodiff