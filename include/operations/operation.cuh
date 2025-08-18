#pragma once

#include <cuda_runtime.h>
#include <type_traits>
#include "../concept/core_logic.cuh"
#include "../concept/variable.cuh"
#include "../concept/operation_node.cuh"
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
    Input& input_;
    output_type output_;  // Variableが自身でバッファを持つ
    mutable std::uint8_t ref_count_ = 0;  // DAG対応: 参照カウント

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
    __host__ __device__ UnaryOperation(const Logic& logic, Input& input)
        : logic_(logic), input_(input), output_() {
        // forwardは明示的に呼ばれるまで実行しない
    }
    
    // Forward pass
    __device__ void forward() {
        // 1. inputがOperationの場合、先にforwardと参照カウント処理
        if constexpr (OperationNode<Input>) {
            input_.forward();
            input_.increment_ref_count();
        }
        
        // 2. 自身のlogic.forwardを実行
        logic_.forward(output_, input_);
    }
    
    // Backward pass
    __device__ void backward() {
        logic_.backward(output_, input_);
        
        // 入力がOperationNodeの場合、参照カウントを減らしてから条件付きbackward
        if constexpr (OperationNode<Input>) {
            if (input_.decrement_ref_count_and_check()) {
                input_.backward();
            }
        }
    }
    
    // 参照カウント管理メソッド
    __device__ void increment_ref_count() const {
        ref_count_++;
    }
    
    __device__ bool decrement_ref_count_and_check() const {
        return --ref_count_ == 0;
    }
    
    // 数値微分による backward pass
    __device__ void backward_numerical(const value_type delta = value_type(1e-5)) {
        // forward の結果を退避
        output_type original_output = output_;

        // 入力の勾配をクリア（OperationNodeの場合とVariableRefの場合の両方に対応）
        if constexpr (OperationNode<Input>) {
            input_.zero_grad();
        } else {
            // VariableRefなどの場合は直接勾配をクリア
            for (std::size_t i = 0; i < Input::size; ++i) {
                input_.grad()[i] = value_type(0);
            }
        }
        
        for (std::size_t i = 0; i < Input::size; i++) {
            const auto orig = input_[i];

            input_[i] = orig + delta;
            logic_.forward(output_, input_);  // 直接logic.forwardを呼ぶ
            output_type plus_out = output_;

            input_[i] = orig - delta;
            logic_.forward(output_, input_);  // 直接logic.forwardを呼ぶ
            output_type minus_out = output_;

            input_[i] = orig;

            // 退避した forward の結果を戻す
            output_ = original_output;

            // 勾配の数値計算
            for (std::size_t j = 0; j < OutputSize; j++) {
                const auto dj_di = (plus_out[j] - minus_out[j]) / (value_type(2.0) * delta);
                input_.add_grad(i, output_.grad(j) * dj_di);
            }
        }
        
        // 入力がOperationNodeの場合、参照カウントを減らしてから条件付きbackward_numerical
        if constexpr (OperationNode<Input>) {
            if (input_.decrement_ref_count_and_check()) {
                input_.backward_numerical(delta);
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
    
    __device__ __forceinline__ const value_type& grad(std::size_t i) const { 
        return output_.grad(i); 
    }
    
    __device__ __forceinline__ void add_grad(std::size_t i, value_type value) { 
        output_.add_grad(i, value); 
    }
    
    __device__ value_type* data() { return output_.data(); }
    __device__ const value_type* data() const { return output_.data(); }
    
    __device__ value_type* grad() { return output_.grad(); }
    __device__ const value_type* grad() const { return output_.grad(); }
    
    __device__ void zero_grad() { 
        output_.zero_grad(); 
    }
    
    // forward -> zero_grad -> add_grad(all 1.0) -> backward の定型処理
    __device__ void run() {
        forward();
        zero_grad();
        // 全ての出力次元に対して1.0の勾配を設定
        for (std::size_t i = 0; i < OutputSize; ++i) {
            add_grad(i, value_type(1.0));
        }
        backward();
    }
    
    // forward -> zero_grad -> add_grad(all 1.0) -> backward_numerical の定型処理
    __device__ void run_numerical(const value_type delta = value_type(1e-5)) {
        forward();
        zero_grad();
        // 全ての出力次元に対して1.0の勾配を設定
        for (std::size_t i = 0; i < OutputSize; ++i) {
            add_grad(i, value_type(1.0));
        }
        backward_numerical(delta);
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
    Input1& input1_;
    Input2& input2_;
    output_type output_;
    mutable std::uint8_t ref_count_ = 0;  // DAG対応: 参照カウント

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
    __host__ __device__ BinaryOperation(const Logic& logic, Input1& input1, Input2& input2)
        : logic_(logic), input1_(input1), input2_(input2), output_() {
        // forwardは明示的に呼ばれるまで実行しない
    }
    
    // Forward pass
    __device__ void forward() {
        // 1. inputがOperationの場合、先にforwardと参照カウント処理
        if constexpr (OperationNode<Input1>) {
            input1_.forward();
            input1_.increment_ref_count();
        }
        if constexpr (OperationNode<Input2>) {
            input2_.forward();
            input2_.increment_ref_count();
        }
        
        // 2. 自身のlogic.forwardを実行
        logic_.forward(output_, input1_, input2_);
    }
    
    // Backward pass
    __device__ void backward() {
        logic_.backward(output_, input1_, input2_);
        
        // 入力がOperationNodeの場合、参照カウントを減らしてから条件付きbackward
        if constexpr (OperationNode<Input1>) {
            if (input1_.decrement_ref_count_and_check()) {
                input1_.backward();
            }
        }
        if constexpr (OperationNode<Input2>) {
            if (input2_.decrement_ref_count_and_check()) {
                input2_.backward();
            }
        }
    }
    
    // 参照カウント管理メソッド
    __device__ void increment_ref_count() const {
        ref_count_++;
    }
    
    __device__ bool decrement_ref_count_and_check() const {
        return --ref_count_ == 0;
    }
    
    // 数値微分による backward pass
    __device__ void backward_numerical(const value_type delta = value_type(1e-5)) {
        // forward の結果を退避
        output_type original_output = output_;

        // 入力の勾配をクリア（OperationNodeの場合とVariableRefの場合の両方に対応）
        if constexpr (OperationNode<Input1>) {
            input1_.zero_grad();
        } else {
            // VariableRefなどの場合は直接勾配をクリア
            for (std::size_t i = 0; i < Input1::size; ++i) {
                input1_.grad()[i] = value_type(0);
            }
        }
        if constexpr (OperationNode<Input2>) {
            input2_.zero_grad();
        } else {
            // VariableRefなどの場合は直接勾配をクリア
            for (std::size_t i = 0; i < Input2::size; ++i) {
                input2_.grad()[i] = value_type(0);
            }
        }
        
        // Input1 に対する数値微分
        for (std::size_t i = 0; i < Input1::size; i++) {
            const auto orig = input1_[i];

            input1_[i] = orig + delta;
            logic_.forward(output_, input1_, input2_);  // 直接logic.forwardを呼ぶ
            output_type plus_out = output_;

            input1_[i] = orig - delta;
            logic_.forward(output_, input1_, input2_);  // 直接logic.forwardを呼ぶ
            output_type minus_out = output_;

            input1_[i] = orig;

            // 退避した forward の結果を戻す
            output_ = original_output;

            // 勾配の数値計算
            for (std::size_t j = 0; j < OutputSize; j++) {
                const auto dj_di = (plus_out[j] - minus_out[j]) / (value_type(2.0) * delta);
                input1_.add_grad(i, output_.grad(j) * dj_di);
            }
        }
        
        // Input2 に対する数値微分
        for (std::size_t i = 0; i < Input2::size; i++) {
            const auto orig = input2_[i];

            input2_[i] = orig + delta;
            logic_.forward(output_, input1_, input2_);  // 直接logic.forwardを呼ぶ
            output_type plus_out = output_;

            input2_[i] = orig - delta;
            logic_.forward(output_, input1_, input2_);  // 直接logic.forwardを呼ぶ
            output_type minus_out = output_;

            input2_[i] = orig;

            // 退避した forward の結果を戻す
            output_ = original_output;

            // 勾配の数値計算
            for (std::size_t j = 0; j < OutputSize; j++) {
                const auto dj_di = (plus_out[j] - minus_out[j]) / (value_type(2.0) * delta);
                input2_.add_grad(i, output_.grad(j) * dj_di);
            }
        }
        
        // 入力がOperationNodeの場合、参照カウントを減らしてから条件付きbackward_numerical
        if constexpr (OperationNode<Input1>) {
            if (input1_.decrement_ref_count_and_check()) {
                input1_.backward_numerical(delta);
            }
        }
        if constexpr (OperationNode<Input2>) {
            if (input2_.decrement_ref_count_and_check()) {
                input2_.backward_numerical(delta);
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
    
    __device__ __forceinline__ const value_type& grad(std::size_t i) const { 
        return output_.grad(i); 
    }
    
    __device__ __forceinline__ void add_grad(std::size_t i, value_type value) { 
        output_.add_grad(i, value); 
    }
    
    __device__ value_type* data() { return output_.data(); }
    __device__ const value_type* data() const { return output_.data(); }
    
    __device__ value_type* grad() { return output_.grad(); }
    __device__ const value_type* grad() const { return output_.grad(); }
    
    __device__ void zero_grad() { 
        output_.zero_grad(); 
    }
    
    // forward -> zero_grad -> add_grad(all 1.0) -> backward の定型処理
    __device__ void run() {
        forward();
        zero_grad();
        // 全ての出力次元に対して1.0の勾配を設定
        for (std::size_t i = 0; i < OutputSize; ++i) {
            add_grad(i, value_type(1.0));
        }
        backward();
    }
    
    // forward -> zero_grad -> add_grad(all 1.0) -> backward_numerical の定型処理
    __device__ void run_numerical(const value_type delta = value_type(1e-5)) {
        forward();
        zero_grad();
        // 全ての出力次元に対して1.0の勾配を設定
        for (std::size_t i = 0; i < OutputSize; ++i) {
            add_grad(i, value_type(1.0));
        }
        backward_numerical(delta);
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
    Input1& input1_;
    Input2& input2_;
    Input3& input3_;
    output_type output_;
    mutable std::uint8_t ref_count_ = 0;  // DAG対応: 参照カウント

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
    __host__ __device__ TernaryOperation(const Logic& logic, Input1& input1, 
                                         Input2& input2, Input3& input3)
        : logic_(logic), input1_(input1), input2_(input2), input3_(input3), output_() {
        // forwardは明示的に呼ばれるまで実行しない
    }
    
    // Forward pass
    __device__ void forward() {
        // 1. inputがOperationの場合、先にforwardと参照カウント処理
        if constexpr (OperationNode<Input1>) {
            input1_.forward();
            input1_.increment_ref_count();
        }
        if constexpr (OperationNode<Input2>) {
            input2_.forward();
            input2_.increment_ref_count();
        }
        if constexpr (OperationNode<Input3>) {
            input3_.forward();
            input3_.increment_ref_count();
        }
        
        // 2. 自身のlogic.forwardを実行
        logic_.forward(output_, input1_, input2_, input3_);
    }
    
    // Backward pass
    __device__ void backward() {
        logic_.backward(output_, input1_, input2_, input3_);
        
        // 入力がOperationNodeの場合、参照カウントを減らしてから条件付きbackward
        if constexpr (OperationNode<Input1>) {
            if (input1_.decrement_ref_count_and_check()) {
                input1_.backward();
            }
        }
        if constexpr (OperationNode<Input2>) {
            if (input2_.decrement_ref_count_and_check()) {
                input2_.backward();
            }
        }
        if constexpr (OperationNode<Input3>) {
            if (input3_.decrement_ref_count_and_check()) {
                input3_.backward();
            }
        }
    }
    
    // 参照カウント管理メソッド
    __device__ void increment_ref_count() const {
        ref_count_++;
    }
    
    __device__ bool decrement_ref_count_and_check() const {
        return --ref_count_ == 0;
    }
    
    // 数値微分による backward pass
    __device__ void backward_numerical(const value_type delta = value_type(1e-5)) {
        // forward の結果を退避
        output_type original_output = output_;

        // 入力の勾配をクリア（OperationNodeの場合とVariableRefの場合の両方に対応）
        if constexpr (OperationNode<Input1>) {
            input1_.zero_grad();
        } else {
            // VariableRefなどの場合は直接勾配をクリア
            for (std::size_t i = 0; i < Input1::size; ++i) {
                input1_.grad()[i] = value_type(0);
            }
        }
        if constexpr (OperationNode<Input2>) {
            input2_.zero_grad();
        } else {
            // VariableRefなどの場合は直接勾配をクリア
            for (std::size_t i = 0; i < Input2::size; ++i) {
                input2_.grad()[i] = value_type(0);
            }
        }
        if constexpr (OperationNode<Input3>) {
            input3_.zero_grad();
        } else {
            // VariableRefなどの場合は直接勾配をクリア
            for (std::size_t i = 0; i < Input3::size; ++i) {
                input3_.grad()[i] = value_type(0);
            }
        }
        
        // Input1 に対する数値微分
        for (std::size_t i = 0; i < Input1::size; i++) {
            const auto orig = input1_[i];

            input1_[i] = orig + delta;
            logic_.forward(output_, input1_, input2_);  // 直接logic.forwardを呼ぶ
            output_type plus_out = output_;

            input1_[i] = orig - delta;
            logic_.forward(output_, input1_, input2_);  // 直接logic.forwardを呼ぶ
            output_type minus_out = output_;

            input1_[i] = orig;

            // 退避した forward の結果を戻す
            output_ = original_output;

            // 勾配の数値計算
            for (std::size_t j = 0; j < OutputSize; j++) {
                const auto dj_di = (plus_out[j] - minus_out[j]) / (value_type(2.0) * delta);
                input1_.add_grad(i, output_.grad(j) * dj_di);
            }
        }
        
        // Input2 に対する数値微分
        for (std::size_t i = 0; i < Input2::size; i++) {
            const auto orig = input2_[i];

            input2_[i] = orig + delta;
            logic_.forward(output_, input1_, input2_, input3_);  // 直接logic.forwardを呼ぶ
            output_type plus_out = output_;

            input2_[i] = orig - delta;
            logic_.forward(output_, input1_, input2_, input3_);  // 直接logic.forwardを呼ぶ
            output_type minus_out = output_;

            input2_[i] = orig;

            // 退避した forward の結果を戻す
            output_ = original_output;

            // 勾配の数値計算
            for (std::size_t j = 0; j < OutputSize; j++) {
                const auto dj_di = (plus_out[j] - minus_out[j]) / (value_type(2.0) * delta);
                input2_.add_grad(i, output_.grad(j) * dj_di);
            }
        }
        
        // Input3 に対する数値微分
        for (std::size_t i = 0; i < Input3::size; i++) {
            const auto orig = input3_[i];

            input3_[i] = orig + delta;
            logic_.forward(output_, input1_, input2_, input3_);  // 直接logic.forwardを呼ぶ
            output_type plus_out = output_;

            input3_[i] = orig - delta;
            logic_.forward(output_, input1_, input2_, input3_);  // 直接logic.forwardを呼ぶ
            output_type minus_out = output_;

            input3_[i] = orig;

            // 退避した forward の結果を戻す
            output_ = original_output;

            // 勾配の数値計算
            for (std::size_t j = 0; j < OutputSize; j++) {
                const auto dj_di = (plus_out[j] - minus_out[j]) / (value_type(2.0) * delta);
                input3_.add_grad(i, output_.grad(j) * dj_di);
            }
        }
        
        // 入力がOperationNodeの場合、参照カウントを減らしてから条件付きbackward_numerical
        if constexpr (OperationNode<Input1>) {
            if (input1_.decrement_ref_count_and_check()) {
                input1_.backward_numerical(delta);
            }
        }
        if constexpr (OperationNode<Input2>) {
            if (input2_.decrement_ref_count_and_check()) {
                input2_.backward_numerical(delta);
            }
        }
        if constexpr (OperationNode<Input3>) {
            if (input3_.decrement_ref_count_and_check()) {
                input3_.backward_numerical(delta);
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
    
    __device__ __forceinline__ const value_type& grad(std::size_t i) const { 
        return output_.grad(i); 
    }
    
    __device__ __forceinline__ void add_grad(std::size_t i, value_type value) { 
        output_.add_grad(i, value); 
    }
    
    __device__ value_type* data() { return output_.data(); }
    __device__ const value_type* data() const { return output_.data(); }
    
    __device__ value_type* grad() { return output_.grad(); }
    __device__ const value_type* grad() const { return output_.grad(); }
    
    __device__ void zero_grad() { 
        output_.zero_grad(); 
    }
    
    // forward -> zero_grad -> add_grad(all 1.0) -> backward の定型処理
    __device__ void run() {
        forward();
        zero_grad();
        // 全ての出力次元に対して1.0の勾配を設定
        for (std::size_t i = 0; i < OutputSize; ++i) {
            add_grad(i, value_type(1.0));
        }
        backward();
    }
    
    // forward -> zero_grad -> add_grad(all 1.0) -> backward_numerical の定型処理
    __device__ void run_numerical(const value_type delta = value_type(1e-5)) {
        forward();
        zero_grad();
        // 全ての出力次元に対して1.0の勾配を設定
        for (std::size_t i = 0; i < OutputSize; ++i) {
            add_grad(i, value_type(1.0));
        }
        backward_numerical(delta);
    }
    
    
};

} // namespace xyz_autodiff