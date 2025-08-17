#pragma once

#include <cuda_runtime.h>
#include <type_traits>
#include "concept/core_logic.cuh"
#include "concept/variable.cuh"

namespace xyz_autodiff {

// 1入力1出力のOperation
template <typename Logic, typename Input, std::size_t OutputSize>
requires UnaryLogicConcept<Logic, Input, Variable<typename Input::value_type, OutputSize>>
class UnaryOperation {
public:
    using input_type = Input;
    using output_type = Variable<typename Input::value_type, OutputSize>;
    using value_type = typename Input::value_type;
    static constexpr std::size_t output_size = OutputSize;

private:
    Logic logic_;
    const Input& input_;
    value_type output_data_[OutputSize];
    value_type output_grad_[OutputSize];
    output_type output_;

public:
    // デフォルトコンストラクタを禁止
    UnaryOperation() = delete;
    
    // コピーコンストラクタを禁止
    UnaryOperation(const UnaryOperation&) = delete;
    
    // ムーブコンストラクタを禁止
    UnaryOperation(UnaryOperation&&) = delete;
    
    // 代入演算子を禁止
    UnaryOperation& operator=(const UnaryOperation&) = delete;
    UnaryOperation& operator=(UnaryOperation&&) = delete;
    
    // コアロジックと入力を受け取るコンストラクタ
    __host__ __device__ UnaryOperation(const Logic& logic, const Input& input)
        : logic_(logic), input_(input), output_(output_data_, output_grad_) {
        // バッファをゼロ初期化
        for (std::size_t i = 0; i < OutputSize; ++i) {
            output_data_[i] = value_type{};
            output_grad_[i] = value_type{};
        }
    }
    
    // forward計算
    __device__ void forward() {
        logic_.forward(output_, input_);
    }
    
    // backward計算
    __device__ void backward() {
        // inputは非constの参照が必要なのでconst_castを使用
        Input& input_ref = const_cast<Input&>(input_);
        logic_.backward(output_, input_ref);
    }
    
    // 出力への参照を取得
    __host__ __device__ const output_type& output() const { return output_; }
    __host__ __device__ output_type& output() { return output_; }
};

// 2入力1出力のOperation
template <typename Logic, typename Input1, typename Input2, std::size_t OutputSize>
requires BinaryLogicConcept<Logic, Input1, Input2, Variable<typename Input1::value_type, OutputSize>>
class BinaryOperation {
public:
    using input1_type = Input1;
    using input2_type = Input2;
    using output_type = Variable<typename Input1::value_type, OutputSize>;
    using value_type = typename Input1::value_type;
    static constexpr std::size_t output_size = OutputSize;

private:
    Logic logic_;
    const Input1& input1_;
    const Input2& input2_;
    value_type output_data_[OutputSize];
    value_type output_grad_[OutputSize];
    output_type output_;

public:
    // デフォルトコンストラクタを禁止
    BinaryOperation() = delete;
    
    // コピーコンストラクタを禁止
    BinaryOperation(const BinaryOperation&) = delete;
    
    // ムーブコンストラクタを禁止
    BinaryOperation(BinaryOperation&&) = delete;
    
    // 代入演算子を禁止
    BinaryOperation& operator=(const BinaryOperation&) = delete;
    BinaryOperation& operator=(BinaryOperation&&) = delete;
    
    // コアロジックと入力を受け取るコンストラクタ
    __host__ __device__ BinaryOperation(const Logic& logic, const Input1& input1, const Input2& input2)
        : logic_(logic), input1_(input1), input2_(input2), output_(output_data_, output_grad_) {
        // バッファをゼロ初期化
        for (std::size_t i = 0; i < OutputSize; ++i) {
            output_data_[i] = value_type{};
            output_grad_[i] = value_type{};
        }
    }
    
    // forward計算
    __device__ void forward() {
        logic_.forward(output_, input1_, input2_);
    }
    
    // backward計算
    __device__ void backward() {
        // inputsは非constの参照が必要なのでconst_castを使用
        Input1& input1_ref = const_cast<Input1&>(input1_);
        Input2& input2_ref = const_cast<Input2&>(input2_);
        logic_.backward(output_, input1_ref, input2_ref);
    }
    
    // 出力への参照を取得
    __host__ __device__ const output_type& output() const { return output_; }
    __host__ __device__ output_type& output() { return output_; }
};

// 3入力1出力のOperation
template <typename Logic, typename Input1, typename Input2, typename Input3, std::size_t OutputSize>
requires TernaryLogicConcept<Logic, Input1, Input2, Input3, Variable<typename Input1::value_type, OutputSize>>
class TernaryOperation {
public:
    using input1_type = Input1;
    using input2_type = Input2;
    using input3_type = Input3;
    using output_type = Variable<typename Input1::value_type, OutputSize>;
    using value_type = typename Input1::value_type;
    static constexpr std::size_t output_size = OutputSize;

private:
    Logic logic_;
    const Input1& input1_;
    const Input2& input2_;
    const Input3& input3_;
    value_type output_data_[OutputSize];
    value_type output_grad_[OutputSize];
    output_type output_;

public:
    // デフォルトコンストラクタを禁止
    TernaryOperation() = delete;
    
    // コピーコンストラクタを禁止
    TernaryOperation(const TernaryOperation&) = delete;
    
    // ムーブコンストラクタを禁止
    TernaryOperation(TernaryOperation&&) = delete;
    
    // 代入演算子を禁止
    TernaryOperation& operator=(const TernaryOperation&) = delete;
    TernaryOperation& operator=(TernaryOperation&&) = delete;
    
    // コアロジックと入力を受け取るコンストラクタ
    __host__ __device__ TernaryOperation(const Logic& logic, const Input1& input1, const Input2& input2, const Input3& input3)
        : logic_(logic), input1_(input1), input2_(input2), input3_(input3), output_(output_data_, output_grad_) {
        // バッファをゼロ初期化
        for (std::size_t i = 0; i < OutputSize; ++i) {
            output_data_[i] = value_type{};
            output_grad_[i] = value_type{};
        }
    }
    
    // forward計算
    __device__ void forward() {
        logic_.forward(output_, input1_, input2_, input3_);
    }
    
    // backward計算
    __device__ void backward() {
        // inputsは非constの参照が必要なのでconst_castを使用
        Input1& input1_ref = const_cast<Input1&>(input1_);
        Input2& input2_ref = const_cast<Input2&>(input2_);
        Input3& input3_ref = const_cast<Input3&>(input3_);
        logic_.backward(output_, input1_ref, input2_ref, input3_ref);
    }
    
    // 出力への参照を取得
    __host__ __device__ const output_type& output() const { return output_; }
    __host__ __device__ output_type& output() { return output_; }
};

// ファクトリメソッド

// UnaryOperationのファクトリ
template <typename Logic, typename Input, std::size_t OutputSize>
requires UnaryLogicConcept<Logic, Input, Variable<typename Input::value_type, OutputSize>>
__host__ __device__ auto make_unary_op(const Logic& logic, const Input& input) {
    return UnaryOperation<Logic, Input, OutputSize>(logic, input);
}

// BinaryOperationのファクトリ
template <typename Logic, typename Input1, typename Input2, std::size_t OutputSize>
requires BinaryLogicConcept<Logic, Input1, Input2, Variable<typename Input1::value_type, OutputSize>>
__host__ __device__ auto make_binary_op(const Logic& logic, const Input1& input1, const Input2& input2) {
    return BinaryOperation<Logic, Input1, Input2, OutputSize>(logic, input1, input2);
}

// TernaryOperationのファクトリ
template <typename Logic, typename Input1, typename Input2, typename Input3, std::size_t OutputSize>
requires TernaryLogicConcept<Logic, Input1, Input2, Input3, Variable<typename Input1::value_type, OutputSize>>
__host__ __device__ auto make_ternary_op(const Logic& logic, const Input1& input1, const Input2& input2, const Input3& input3) {
    return TernaryOperation<Logic, Input1, Input2, Input3, OutputSize>(logic, input1, input2, input3);
}

} // namespace xyz_autodiff