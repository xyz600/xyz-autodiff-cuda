#pragma once

#include <cuda_runtime.h>
#include <type_traits>
#include "../concept/core_logic.cuh"
#include "../concept/variable.cuh"

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
    value_type output_data_[OutputSize];
    value_type output_grad_[OutputSize];
    output_type output_;

public:
    // デフォルトコンストラクタを禁止
    UnaryOperation() = delete;
    
    // コピーコンストラクタを禁止
    UnaryOperation(const UnaryOperation&) = delete;
    
    // ムーブコンストラクタ
    __host__ __device__ UnaryOperation(UnaryOperation&& other) noexcept
        : logic_(other.logic_), input_(other.input_), output_(output_data_, output_grad_) {
        // バッファをコピー
        for (std::size_t i = 0; i < OutputSize; ++i) {
            output_data_[i] = other.output_data_[i];
            output_grad_[i] = other.output_grad_[i];
        }
    }
    
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
template <std::size_t OutputSize, typename Logic, typename Input1, typename Input2>
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
    
    // ムーブコンストラクタ
    __host__ __device__ BinaryOperation(BinaryOperation&& other) noexcept
        : logic_(other.logic_), input1_(other.input1_), input2_(other.input2_), output_(output_data_, output_grad_) {
        // バッファをコピー
        for (std::size_t i = 0; i < OutputSize; ++i) {
            output_data_[i] = other.output_data_[i];
            output_grad_[i] = other.output_grad_[i];
        }
    }
    
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
        forward();
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
    
    // ムーブコンストラクタ
    __host__ __device__ TernaryOperation(TernaryOperation&& other) noexcept
        : logic_(other.logic_), input1_(other.input1_), input2_(other.input2_), input3_(other.input3_), output_(output_data_, output_grad_) {
        // バッファをコピー
        for (std::size_t i = 0; i < OutputSize; ++i) {
            output_data_[i] = other.output_data_[i];
            output_grad_[i] = other.output_grad_[i];
        }
    }
    
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
template <std::size_t OutputSize, typename Logic, typename Input>
requires UnaryLogicConcept<Logic, Input, Variable<typename Input::value_type, OutputSize>>
__host__ __device__ auto make_unary_op(const Logic& logic, const Input& input) {
    return UnaryOperation<OutputSize, Logic, Input>(logic, input);
}

// BinaryOperationのファクトリ
template <std::size_t OutputSize, typename Logic, typename Input1, typename Input2>
requires BinaryLogicConcept<Logic, Input1, Input2, Variable<typename Input1::value_type, OutputSize>>
__host__ __device__ auto make_binary_op(const Logic& logic, const Input1& input1, const Input2& input2) {
    return BinaryOperation<OutputSize, Logic, Input1, Input2>(logic, input1, input2);
}

// TernaryOperationのファクトリ
template <std::size_t OutputSize, typename Logic, typename Input1, typename Input2, typename Input3>
requires TernaryLogicConcept<Logic, Input1, Input2, Input3, Variable<typename Input1::value_type, OutputSize>>
__host__ __device__ auto make_ternary_op(const Logic& logic, const Input1& input1, const Input2& input2, const Input3& input3) {
    return TernaryOperation<OutputSize, Logic, Input1, Input2, Input3>(logic, input1, input2, input3);
}

// 参照型Operation（内部バッファを持たない版）

// 1入力1出力のOperationRef
template <std::size_t OutputSize, typename Logic, typename Input>
requires UnaryLogicConcept<Logic, Input, Variable<typename Input::value_type, OutputSize>>
class UnaryOperationRef {
public:
    using input_type = Input;
    using output_type = Variable<typename Input::value_type, OutputSize>;
    using value_type = typename Input::value_type;
    static constexpr std::size_t output_size = OutputSize;

private:
    Logic logic_;
    const Input& input_;
    output_type& output_;

public:
    // デフォルトコンストラクタを禁止
    UnaryOperationRef() = delete;
    
    // コピーコンストラクタを禁止
    UnaryOperationRef(const UnaryOperationRef&) = delete;
    
    // ムーブコンストラクタを禁止（参照なのでムーブ不要）
    UnaryOperationRef(UnaryOperationRef&&) = delete;
    
    // 代入演算子を禁止
    UnaryOperationRef& operator=(const UnaryOperationRef&) = delete;
    UnaryOperationRef& operator=(UnaryOperationRef&&) = delete;
    
    // コアロジック、入力、出力を受け取るコンストラクタ
    __host__ __device__ UnaryOperationRef(const Logic& logic, const Input& input, output_type& output)
        : logic_(logic), input_(input), output_(output) {}
    
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

// 2入力1出力のOperationRef
template <std::size_t OutputSize, typename Logic, typename Input1, typename Input2>
requires BinaryLogicConcept<Logic, Input1, Input2, Variable<typename Input1::value_type, OutputSize>>
class BinaryOperationRef {
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
    output_type& output_;

public:
    // デフォルトコンストラクタを禁止
    BinaryOperationRef() = delete;
    
    // コピーコンストラクタを禁止
    BinaryOperationRef(const BinaryOperationRef&) = delete;
    
    // ムーブコンストラクタを禁止（参照なのでムーブ不要）
    BinaryOperationRef(BinaryOperationRef&&) = delete;
    
    // 代入演算子を禁止
    BinaryOperationRef& operator=(const BinaryOperationRef&) = delete;
    BinaryOperationRef& operator=(BinaryOperationRef&&) = delete;
    
    // コアロジック、入力、出力を受け取るコンストラクタ
    __host__ __device__ BinaryOperationRef(const Logic& logic, const Input1& input1, const Input2& input2, output_type& output)
        : logic_(logic), input1_(input1), input2_(input2), output_(output) {}
    
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

// 3入力1出力のOperationRef
template <std::size_t OutputSize, typename Logic, typename Input1, typename Input2, typename Input3>
requires TernaryLogicConcept<Logic, Input1, Input2, Input3, Variable<typename Input1::value_type, OutputSize>>
class TernaryOperationRef {
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
    output_type& output_;

public:
    // デフォルトコンストラクタを禁止
    TernaryOperationRef() = delete;
    
    // コピーコンストラクタを禁止
    TernaryOperationRef(const TernaryOperationRef&) = delete;
    
    // ムーブコンストラクタを禁止（参照なのでムーブ不要）
    TernaryOperationRef(TernaryOperationRef&&) = delete;
    
    // 代入演算子を禁止
    TernaryOperationRef& operator=(const TernaryOperationRef&) = delete;
    TernaryOperationRef& operator=(TernaryOperationRef&&) = delete;
    
    // コアロジック、入力、出力を受け取るコンストラクタ
    __host__ __device__ TernaryOperationRef(const Logic& logic, const Input1& input1, const Input2& input2, const Input3& input3, output_type& output)
        : logic_(logic), input1_(input1), input2_(input2), input3_(input3), output_(output) {}
    
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

// OperationRefファクトリメソッド

// UnaryOperationRefのファクトリ
template <std::size_t OutputSize, typename Logic, typename Input>
requires UnaryLogicConcept<Logic, Input, Variable<typename Input::value_type, OutputSize>>
__host__ __device__ auto make_unary_op_ref(const Logic& logic, const Input& input, Variable<typename Input::value_type, OutputSize>& output) {
    return UnaryOperationRef<OutputSize, Logic, Input>(logic, input, output);
}

// BinaryOperationRefのファクトリ
template <std::size_t OutputSize, typename Logic, typename Input1, typename Input2>
requires BinaryLogicConcept<Logic, Input1, Input2, Variable<typename Input1::value_type, OutputSize>>
__host__ __device__ auto make_binary_op_ref(const Logic& logic, const Input1& input1, const Input2& input2, Variable<typename Input1::value_type, OutputSize>& output) {
    return BinaryOperationRef<OutputSize, Logic, Input1, Input2>(logic, input1, input2, output);
}

// TernaryOperationRefのファクトリ
template <std::size_t OutputSize, typename Logic, typename Input1, typename Input2, typename Input3>
requires TernaryLogicConcept<Logic, Input1, Input2, Input3, Variable<typename Input1::value_type, OutputSize>>
__host__ __device__ auto make_ternary_op_ref(const Logic& logic, const Input1& input1, const Input2& input2, const Input3& input3, Variable<typename Input1::value_type, OutputSize>& output) {
    return TernaryOperationRef<OutputSize, Logic, Input1, Input2, Input3>(logic, input1, input2, input3, output);
}

} // namespace xyz_autodiff