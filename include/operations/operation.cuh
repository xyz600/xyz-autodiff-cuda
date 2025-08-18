#pragma once

#include <cuda_runtime.h>
#include <type_traits>
#include "../concept/core_logic.cuh"
#include "../concept/variable.cuh"
#include "../variable.cuh"

namespace xyz_autodiff {

// 1入力1出力のOperation
template <std::size_t OutputSize, typename Logic, typename Input>
requires UnaryLogicConcept<Logic, Input, Variable<typename Input::value_type, OutputSize>>
class UnaryOperation {
public:
    using input_type = Input;
    using output_type = Variable<typename Input::value_type, OutputSize>;
    using value_type = typename Input::value_type;
    static constexpr std::size_t output_size = OutputSize;
    static constexpr std::size_t size = OutputSize;

private:
    Logic logic_;
    const Input& input_;
    output_type output_;  // Variableが自身でバッファを持つ

public:
    // デフォルトコンストラクタを禁止
    UnaryOperation() = delete;
    
    // コピーコンストラクタを禁止
    UnaryOperation(const UnaryOperation&) = delete;
    
    // ムーブコンストラクタ
    __host__ __device__ UnaryOperation(UnaryOperation&& other) noexcept
        : logic_(other.logic_), input_(other.input_), output_(std::move(other.output_)) {
    }
    
    // 代入演算子を禁止
    UnaryOperation& operator=(const UnaryOperation&) = delete;
    UnaryOperation& operator=(UnaryOperation&&) = delete;
    
    // コアロジックと入力を受け取るコンストラクタ
    __host__ __device__ UnaryOperation(const Logic& logic, const Input& input)
        : logic_(logic), input_(input), output_() {
    }
    
    // Forward pass
    __device__ void forward() {
        logic_.forward(output_, input_);
    }
    
    // Backward pass
    __device__ void backward() {
        logic_.backward(output_, const_cast<Input&>(input_));
    }
    
    // 数値微分（デバッグ用）
    __device__ void numerical_backward(const value_type delta = value_type(1e-5)) {
        // forward の結果を退避
        output_type original_output = output_;

        const_cast<Input&>(input_).zero_grad();
        
        for (std::size_t i = 0; i < Input::size; i++) {
            const auto orig = input_[i];

            const_cast<Input&>(input_)[i] = orig + delta;
            forward();
            output_type plus_out = output_;

            const_cast<Input&>(input_)[i] = orig - delta;
            forward();
            output_type minus_out = output_;

            const_cast<Input&>(input_)[i] = orig;

            // 退避した forward の結果を戻す
            output_ = original_output;

            // 勾配の数値計算
            for (std::size_t j = 0; j < OutputSize; j++) {
                const auto dj_di = (plus_out[j] - minus_out[j]) / (value_type(2.0) * delta);
                const_cast<Input&>(input_).grad(i) += output_.grad(j) * dj_di;
            }
        }
    }
    
    // 出力への参照を取得
    __device__ output_type& output() { return output_; }
    __device__ const output_type& output() const { return output_; }
    
    // Variable concept 対応
    __device__ __forceinline__ value_type& operator[](std::size_t i) { 
        return output_[i]; 
    }
    __device__ __forceinline__ const value_type& operator[](std::size_t i) const { 
        return output_[i]; 
    }
    
    __device__ __forceinline__ value_type& grad(std::size_t i) { 
        return output_.grad(i); 
    }
    __device__ __forceinline__ const value_type& grad(std::size_t i) const { 
        return output_.grad(i); 
    }
    
    __device__ value_type* data() { return output_.data(); }
    __device__ const value_type* data() const { return output_.data(); }
    
    __device__ value_type* grad() { return output_.grad(); }
    __device__ const value_type* grad() const { return output_.grad(); }
    
    __device__ void zero_grad() { output_.zero_grad(); }
    
    __device__ void accumulate_grad(const value_type* const grad_values) { 
        output_.accumulate_grad(grad_values); 
    }
};

// 2入力1出力のOperation
template <std::size_t OutputSize, typename Logic, typename Input1, typename Input2>
requires BinaryLogicConcept<Logic, Input1, Input2, Variable<typename Input1::value_type, OutputSize>>
class BinaryOperation {
public:
    using input1_type = Input1;
    using input2_type = Input2;
    using output_type = Variable<typename Input1::value_type, OutputSize>;
    using value_type = typename Input1::value_type;
    static constexpr std::size_t output_size = OutputSize;
    static constexpr std::size_t size = OutputSize;

private:
    Logic logic_;
    const Input1& input1_;
    const Input2& input2_;
    output_type output_;

public:
    // デフォルトコンストラクタを禁止
    BinaryOperation() = delete;
    
    // コピーコンストラクタを禁止
    BinaryOperation(const BinaryOperation&) = delete;
    
    // ムーブコンストラクタ
    __host__ __device__ BinaryOperation(BinaryOperation&& other) noexcept
        : logic_(other.logic_), input1_(other.input1_), input2_(other.input2_), 
          output_(std::move(other.output_)) {
    }
    
    // 代入演算子を禁止
    BinaryOperation& operator=(const BinaryOperation&) = delete;
    BinaryOperation& operator=(BinaryOperation&&) = delete;
    
    // コアロジックと入力を受け取るコンストラクタ
    __host__ __device__ BinaryOperation(const Logic& logic, const Input1& input1, const Input2& input2)
        : logic_(logic), input1_(input1), input2_(input2), output_() {
    }
    
    // Forward pass
    __device__ void forward() {
        logic_.forward(output_, input1_, input2_);
    }
    
    // Backward pass
    __device__ void backward() {
        logic_.backward(output_, const_cast<Input1&>(input1_), const_cast<Input2&>(input2_));
    }
    
    // 出力への参照を取得
    __device__ output_type& output() { return output_; }
    __device__ const output_type& output() const { return output_; }
    
    // Variable concept 対応
    __device__ __forceinline__ value_type& operator[](std::size_t i) { 
        return output_[i]; 
    }
    __device__ __forceinline__ const value_type& operator[](std::size_t i) const { 
        return output_[i]; 
    }
    
    __device__ __forceinline__ value_type& grad(std::size_t i) { 
        return output_.grad(i); 
    }
    __device__ __forceinline__ const value_type& grad(std::size_t i) const { 
        return output_.grad(i); 
    }
    
    __device__ value_type* data() { return output_.data(); }
    __device__ const value_type* data() const { return output_.data(); }
    
    __device__ value_type* grad() { return output_.grad(); }
    __device__ const value_type* grad() const { return output_.grad(); }
    
    __device__ void zero_grad() { output_.zero_grad(); }
    
    __device__ void accumulate_grad(const value_type* const grad_values) { 
        output_.accumulate_grad(grad_values); 
    }
};

// 3入力1出力のOperation
template <std::size_t OutputSize, typename Logic, typename Input1, typename Input2, typename Input3>
requires TernaryLogicConcept<Logic, Input1, Input2, Input3, Variable<typename Input1::value_type, OutputSize>>
class TernaryOperation {
public:
    using input1_type = Input1;
    using input2_type = Input2;
    using input3_type = Input3;
    using output_type = Variable<typename Input1::value_type, OutputSize>;
    using value_type = typename Input1::value_type;
    static constexpr std::size_t output_size = OutputSize;
    static constexpr std::size_t size = OutputSize;

private:
    Logic logic_;
    const Input1& input1_;
    const Input2& input2_;
    const Input3& input3_;
    output_type output_;

public:
    // デフォルトコンストラクタを禁止
    TernaryOperation() = delete;
    
    // コピーコンストラクタを禁止
    TernaryOperation(const TernaryOperation&) = delete;
    
    // ムーブコンストラクタ
    __host__ __device__ TernaryOperation(TernaryOperation&& other) noexcept
        : logic_(other.logic_), input1_(other.input1_), input2_(other.input2_), input3_(other.input3_),
          output_(std::move(other.output_)) {
    }
    
    // 代入演算子を禁止
    TernaryOperation& operator=(const TernaryOperation&) = delete;
    TernaryOperation& operator=(TernaryOperation&&) = delete;
    
    // コアロジックと入力を受け取るコンストラクタ
    __host__ __device__ TernaryOperation(const Logic& logic, const Input1& input1, 
                                         const Input2& input2, const Input3& input3)
        : logic_(logic), input1_(input1), input2_(input2), input3_(input3), output_() {
    }
    
    // Forward pass
    __device__ void forward() {
        logic_.forward(output_, input1_, input2_, input3_);
    }
    
    // Backward pass
    __device__ void backward() {
        logic_.backward(output_, const_cast<Input1&>(input1_), 
                        const_cast<Input2&>(input2_), const_cast<Input3&>(input3_));
    }
    
    // 出力への参照を取得
    __device__ output_type& output() { return output_; }
    __device__ const output_type& output() const { return output_; }
    
    // Variable concept 対応
    __device__ __forceinline__ value_type& operator[](std::size_t i) { 
        return output_[i]; 
    }
    __device__ __forceinline__ const value_type& operator[](std::size_t i) const { 
        return output_[i]; 
    }
    
    __device__ __forceinline__ value_type& grad(std::size_t i) { 
        return output_.grad(i); 
    }
    __device__ __forceinline__ const value_type& grad(std::size_t i) const { 
        return output_.grad(i); 
    }
    
    __device__ value_type* data() { return output_.data(); }
    __device__ const value_type* data() const { return output_.data(); }
    
    __device__ value_type* grad() { return output_.grad(); }
    __device__ const value_type* grad() const { return output_.grad(); }
    
    __device__ void zero_grad() { output_.zero_grad(); }
    
    __device__ void accumulate_grad(const value_type* const grad_values) { 
        output_.accumulate_grad(grad_values); 
    }
};

} // namespace xyz_autodiff