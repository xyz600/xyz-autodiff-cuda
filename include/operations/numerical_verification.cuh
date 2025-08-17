#pragma once

#include <cstddef>
#include <cuda_runtime.h>
#include <cmath>
#include "../variable.cuh"

namespace xyz_autodiff {
namespace numerical {

// 数値微分の設定
constexpr double NUMERICAL_DELTA = 1e-8;
constexpr double PRECISION_TOLERANCE = 1e-6;

// UnaryOperation用の数値微分検証
template <typename Logic, typename Input>
__device__ bool verify_unary_numerical_gradient(const Logic& logic, Input& input, std::size_t input_idx) {
    using T = typename Input::value_type;
    
    // 元の値を保存
    const T original_value = input[input_idx];
    
    // 出力用のVariable（Logic::outputDimサイズ）
    Variable<T, Logic::outputDim> output_plus(nullptr, nullptr);
    Variable<T, Logic::outputDim> output_minus(nullptr, nullptr);
    Variable<T, Logic::outputDim> output_center(nullptr, nullptr);
    
    // データのみのバッファ（勾配は使わない）
    T output_plus_data[Logic::outputDim];
    T output_minus_data[Logic::outputDim];
    T output_center_data[Logic::outputDim];
    T dummy_grad[Logic::outputDim];
    
    // 勾配バッファをゼロクリア
    for (std::size_t i = 0; i < Logic::outputDim; ++i) {
        dummy_grad[i] = T{0};
    }
    
    // 出力Variableの初期化
    output_plus = Variable<T, Logic::outputDim>(output_plus_data, dummy_grad);
    output_minus = Variable<T, Logic::outputDim>(output_minus_data, dummy_grad);
    output_center = Variable<T, Logic::outputDim>(output_center_data, dummy_grad);
    
    // f(x + delta)の計算
    input[input_idx] = original_value + static_cast<T>(NUMERICAL_DELTA);
    logic.forward(output_plus, input);
    
    // f(x - delta)の計算
    input[input_idx] = original_value - static_cast<T>(NUMERICAL_DELTA);
    logic.forward(output_minus, input);
    
    // 元の値を復元してf(x)の計算
    input[input_idx] = original_value;
    logic.forward(output_center, input);
    
    // 解析的勾配の計算
    // 出力の各要素に対して単位勾配を設定して逆伝播
    for (std::size_t output_idx = 0; output_idx < Logic::outputDim; ++output_idx) {
        // 勾配をリセット
        input.zero_grad();
        output_center.zero_grad();
        
        // 出力の該当要素に単位勾配を設定
        output_center.grad(output_idx) = T{1};
        
        // 解析的勾配計算
        logic.backward(output_center, input);
        T analytical_grad = input.grad(input_idx);
        
        // 数値勾配計算
        T numerical_grad = (output_plus[output_idx] - output_minus[output_idx]) / (2 * static_cast<T>(NUMERICAL_DELTA));
        
        // 相対誤差の計算
        T abs_diff = abs(analytical_grad - numerical_grad);
        T rel_error = abs_diff / (abs(analytical_grad) + static_cast<T>(1e-12));
        
        // 許容誤差内かチェック
        if (rel_error > static_cast<T>(PRECISION_TOLERANCE)) {
            return false;
        }
    }
    
    return true;
}

// BinaryOperation用の数値微分検証
template <typename Logic, typename Input1, typename Input2>
__device__ bool verify_binary_numerical_gradient(const Logic& logic, Input1& input1, Input2& input2, 
                                                std::size_t input_num, std::size_t input_idx) {
    using T = typename Input1::value_type;
    
    // 出力用のVariable
    Variable<T, Logic::outputDim> output_plus(nullptr, nullptr);
    Variable<T, Logic::outputDim> output_minus(nullptr, nullptr);
    Variable<T, Logic::outputDim> output_center(nullptr, nullptr);
    
    T output_plus_data[Logic::outputDim];
    T output_minus_data[Logic::outputDim];
    T output_center_data[Logic::outputDim];
    T dummy_grad[Logic::outputDim];
    
    for (std::size_t i = 0; i < Logic::outputDim; ++i) {
        dummy_grad[i] = T{0};
    }
    
    output_plus = Variable<T, Logic::outputDim>(output_plus_data, dummy_grad);
    output_minus = Variable<T, Logic::outputDim>(output_minus_data, dummy_grad);
    output_center = Variable<T, Logic::outputDim>(output_center_data, dummy_grad);
    
    T original_value;
    
    // どちらの入力を摂動するかを決定
    if (input_num == 1) {
        original_value = input1[input_idx];
        
        // f(x + delta, y)
        input1[input_idx] = original_value + static_cast<T>(NUMERICAL_DELTA);
        logic.forward(output_plus, input1, input2);
        
        // f(x - delta, y)
        input1[input_idx] = original_value - static_cast<T>(NUMERICAL_DELTA);
        logic.forward(output_minus, input1, input2);
        
        // f(x, y)
        input1[input_idx] = original_value;
        logic.forward(output_center, input1, input2);
    } else {
        original_value = input2[input_idx];
        
        // f(x, y + delta)
        input2[input_idx] = original_value + static_cast<T>(NUMERICAL_DELTA);
        logic.forward(output_plus, input1, input2);
        
        // f(x, y - delta)
        input2[input_idx] = original_value - static_cast<T>(NUMERICAL_DELTA);
        logic.forward(output_minus, input1, input2);
        
        // f(x, y)
        input2[input_idx] = original_value;
        logic.forward(output_center, input1, input2);
    }
    
    // 各出力要素に対して勾配をチェック
    for (std::size_t output_idx = 0; output_idx < Logic::outputDim; ++output_idx) {
        // 勾配をリセット
        input1.zero_grad();
        input2.zero_grad();
        output_center.zero_grad();
        
        // 出力の該当要素に単位勾配を設定
        output_center.grad(output_idx) = T{1};
        
        // 解析的勾配計算
        logic.backward(output_center, input1, input2);
        
        T analytical_grad;
        if (input_num == 1) {
            analytical_grad = input1.grad(input_idx);
        } else {
            analytical_grad = input2.grad(input_idx);
        }
        
        // 数値勾配計算
        T numerical_grad = (output_plus[output_idx] - output_minus[output_idx]) / (2 * static_cast<T>(NUMERICAL_DELTA));
        
        // 相対誤差の計算
        T abs_diff = abs(analytical_grad - numerical_grad);
        T rel_error = abs_diff / (abs(analytical_grad) + static_cast<T>(1e-12));
        
        if (rel_error > static_cast<T>(PRECISION_TOLERANCE)) {
            return false;
        }
    }
    
    return true;
}

// ランダムな値でのテスト
template <typename Logic>
__device__ bool test_unary_logic_random(const Logic& logic, unsigned int seed) {
    using T = double;  // 高精度でテスト
    
    // 簡単な線形合同法によるランダム数生成
    unsigned int state = seed;
    auto next_random = [&state]() -> T {
        state = state * 1103515245 + 12345;
        return static_cast<T>(state % 10000) / 5000.0 - 1.0;  // -1.0 to 1.0
    };
    
    // テスト用のデータとグラデーション配列
    T input_data[3];
    T input_grad[3];
    
    // ランダムな入力値を生成
    for (std::size_t i = 0; i < 3; ++i) {
        input_data[i] = next_random();
        input_grad[i] = T{0};
    }
    
    Variable<T, 3> input(input_data, input_grad);
    
    // 各入力要素について数値微分をテスト
    for (std::size_t i = 0; i < 3; ++i) {
        if (!verify_unary_numerical_gradient(logic, input, i)) {
            return false;
        }
    }
    
    return true;
}

template <typename Logic>
__device__ bool test_binary_logic_random(const Logic& logic, unsigned int seed) {
    using T = double;
    
    unsigned int state = seed;
    auto next_random = [&state]() -> T {
        state = state * 1103515245 + 12345;
        return static_cast<T>(state % 10000) / 5000.0 - 1.0;
    };
    
    T input1_data[3], input1_grad[3];
    T input2_data[3], input2_grad[3];
    
    for (std::size_t i = 0; i < 3; ++i) {
        input1_data[i] = next_random();
        input1_grad[i] = T{0};
        input2_data[i] = next_random();
        input2_grad[i] = T{0};
    }
    
    Variable<T, 3> input1(input1_data, input1_grad);
    Variable<T, 3> input2(input2_data, input2_grad);
    
    // 両方の入力の各要素についてテスト
    for (std::size_t i = 0; i < 1; ++i) {  // サイズ1の場合
        if (!verify_binary_numerical_gradient(logic, input1, input2, 1, i)) {
            return false;
        }
        if (!verify_binary_numerical_gradient(logic, input1, input2, 2, i)) {
            return false;
        }
    }
    
    return true;
}

} // namespace numerical
} // namespace xyz_autodiff