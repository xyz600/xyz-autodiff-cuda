#pragma once

#include <concepts>
#include <type_traits>
#include "variable.cuh"

namespace xyz_autodiff {

// 1入力1出力のコアロジック用concept
template <typename T, typename Input, typename Output>
concept UnaryLogicConcept = 
    VariableConcept<Input> && VariableConcept<Output> &&
    requires(T logic, Output& output, const Input& input,
             typename Input::value_type* input_grad, const typename Output::value_type* upstream_grad) {
    // constexpr outputDim を要求
    { T::outputDim } -> std::convertible_to<std::size_t>;
    
    // forward: 出力に結果を書き込む
    { logic.forward(output, input) } -> std::same_as<void>;
    
    // backward: 入力の勾配に結果を書き込む
    { logic.backward(input_grad, input, upstream_grad) } -> std::same_as<void>;
};

// 2入力1出力のコアロジック用concept
template <typename T, typename Input1, typename Input2, typename Output>
concept BinaryLogicConcept = 
    VariableConcept<Input1> && VariableConcept<Input2> && VariableConcept<Output> &&
    requires(T logic, Output& output, const Input1& input1, const Input2& input2,
             typename Input1::value_type* input1_grad, typename Input2::value_type* input2_grad,
             const typename Output::value_type* upstream_grad) {
    // constexpr outputDim を要求
    { T::outputDim } -> std::convertible_to<std::size_t>;
    
    // forward: 出力に結果を書き込む
    { logic.forward(output, input1, input2) } -> std::same_as<void>;
    
    // backward: 入力の勾配に結果を書き込む
    { logic.backward(input1_grad, input2_grad, input1, input2, upstream_grad) } -> std::same_as<void>;
};

// 3入力1出力のコアロジック用concept
template <typename T, typename Input1, typename Input2, typename Input3, typename Output>
concept TernaryLogicConcept = 
    VariableConcept<Input1> && VariableConcept<Input2> && VariableConcept<Input3> && VariableConcept<Output> &&
    requires(T logic, Output& output, const Input1& input1, const Input2& input2, const Input3& input3,
             typename Input1::value_type* input1_grad, typename Input2::value_type* input2_grad, 
             typename Input3::value_type* input3_grad, const typename Output::value_type* upstream_grad) {
    // constexpr outputDim を要求
    { T::outputDim } -> std::convertible_to<std::size_t>;
    
    // forward: 出力に結果を書き込む
    { logic.forward(output, input1, input2, input3) } -> std::same_as<void>;
    
    // backward: 入力の勾配に結果を書き込む
    { logic.backward(input1_grad, input2_grad, input3_grad, input1, input2, input3, upstream_grad) } -> std::same_as<void>;
};

// パラメータ制約用のconcept群

// 1入力のパラメータ制約
template <typename Input>
concept UnaryLogicParameterConcept = 
    DifferentiableVariableConcept<Input>;

// 2入力のパラメータ制約
template <typename Input1, typename Input2>
concept BinaryLogicParameterConcept = 
    DifferentiableVariableConcept<Input1> && DifferentiableVariableConcept<Input2> && 
    std::is_same_v<typename Input1::value_type, typename Input2::value_type>;

// 3入力のパラメータ制約
template <typename Input1, typename Input2, typename Input3>
concept TernaryLogicParameterConcept = 
    DifferentiableVariableConcept<Input1> && DifferentiableVariableConcept<Input2> && 
    DifferentiableVariableConcept<Input3> &&
    std::is_same_v<typename Input1::value_type, typename Input2::value_type> &&
    std::is_same_v<typename Input1::value_type, typename Input3::value_type>;

} // namespace xyz_autodiff