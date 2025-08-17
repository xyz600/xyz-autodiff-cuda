#pragma once

#include <cuda_runtime.h>
#include "../concept/operation.cuh"
#include "../concept/variable.cuh"

namespace xyz_autodiff {

// 加算Operation（2入力専用、型パラメータ付き）
template <typename T, typename Input1, typename Input2>
requires DifferentiableVariableConcept<Input1> && DifferentiableVariableConcept<Input2> && 
         (Input1::size == Input2::size) &&
         std::is_same_v<typename Input1::value_type, T> &&
         std::is_same_v<typename Input2::value_type, T>
class AddOperation {
private:
    const Input1& input1_ref_;
    const Input2& input2_ref_;
    mutable T result_;  // 内部結果保存

public:
    using value_type = T;
    using input1_type = Input1;
    using input2_type = Input2;
    static constexpr std::size_t output_size = 1;
    
    // 参照コンストラクタのみ（型推論用）
    __host__ __device__ AddOperation(const Input1& input1, const Input2& input2) 
        : input1_ref_(input1), input2_ref_(input2), result_{} {
        // 構築時に即座にforward計算を実行
        forward();
    }
    
    // デフォルトコンストラクタ禁止
    AddOperation() = delete;
    
    // コピー・ムーブ禁止
    AddOperation(const AddOperation&) = delete;
    AddOperation(AddOperation&&) = delete;
    AddOperation& operator=(const AddOperation&) = delete;
    AddOperation& operator=(AddOperation&&) = delete;
    
    // forward計算: 内部結果に書き込み
    __device__ void forward() const {
        result_ = input1_ref_[0] + input2_ref_[0];
    }
    
    // 結果の取得
    __device__ const T& value() const {
        return result_;
    }
    
    // インデックスアクセス（単一要素の場合）
    __device__ const T& operator[](std::size_t) const {
        return result_;
    }
    
    // backward計算: 勾配を直接入力のgradに書き込み
    __device__ void backward(const T& output_grad = T{1}) const {
        // d(a+b)/da = 1, d(a+b)/db = 1
        // 各入力の勾配に累積
        const_cast<Input1&>(input1_ref_).accumulate_grad(&output_grad);
        const_cast<Input2&>(input2_ref_).accumulate_grad(&output_grad);
    }
};

// CTAD（Class Template Argument Deduction）用の推論ガイド
template <typename Input1, typename Input2>
requires DifferentiableVariableConcept<Input1> && DifferentiableVariableConcept<Input2> && 
         (Input1::size == Input2::size) &&
         std::is_same_v<typename Input1::value_type, typename Input2::value_type>
AddOperation(const Input1&, const Input2&) 
    -> AddOperation<typename Input1::value_type, Input1, Input2>;

} // namespace xyz_autodiff