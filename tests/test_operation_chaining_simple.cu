#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "../include/variable.cuh"
#include "../include/operations/operation.cuh"
#include "../include/operations/add_logic.cuh"

using namespace xyz_autodiff;

// 単純なホスト側チェーンテスト
class OperationChainingSimpleTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaError_t err = cudaSetDevice(0);
        if (err != cudaSuccess) {
            GTEST_SKIP() << "CUDA device not available";
        }
    }
};

// ホスト側でのVariable Conceptテスト
TEST_F(OperationChainingSimpleTest, HostVariableConcept) {
    using namespace op;
    using Var1 = Variable<float, 1>;
    using AddLogicType = AddLogic<Var1, Var1>;
    using AddOp = BinaryOperation<AddLogicType::outputDim, AddLogicType, Var1, Var1>;
    
    // Variable Concept の要件チェック
    static_assert(VariableConcept<AddOp>);
    static_assert(DifferentiableVariableConcept<AddOp>);
    
    // サイズ情報のチェック
    static_assert(AddOp::size == 1);
    static_assert(AddOp::output_size == 1);
    
    // 型情報のチェック
    static_assert(std::is_same_v<AddOp::value_type, float>);
    static_assert(std::is_same_v<AddOp::output_type, Variable<float, 1>>);
    
    SUCCEED() << "All static assertions passed";
}

// ホスト側でのチェーンテスト（コンパイル時のみ）
TEST_F(OperationChainingSimpleTest, HostChainingCompileTest) {
    // ホスト側でVariable作成
    float data1[1] = {3.0f};
    float grad1[1] = {0.0f};
    float data2[1] = {4.0f};
    float grad2[1] = {0.0f};
    float data3[1] = {5.0f};
    float grad3[1] = {0.0f};
    
    Variable<float, 1> var1(data1, grad1);
    Variable<float, 1> var2(data2, grad2);
    Variable<float, 1> var3(data3, grad3);
    
    // これはコンパイルできることの確認のみ（実際にdevice関数は呼ばない）
    // auto node1 = op::add(var1, var2);  // Device function call
    // auto node2 = op::add(var3, node1);
    
    // Variable Conceptのメソッドアクセスが型推論で正しく動作することをテスト
    using AddOpType = decltype(op::add(var1, var2));
    
    // データアクセス方法の型チェック
    static_assert(std::is_same_v<decltype(std::declval<AddOpType>().data()), float*>);
    static_assert(std::is_same_v<decltype(std::declval<AddOpType>().grad()), float*>);
    static_assert(std::is_same_v<decltype(std::declval<AddOpType>()[0]), float&>);
    static_assert(std::is_same_v<decltype(std::declval<AddOpType>().grad(0)), float&>);
    
    SUCCEED() << "Type deduction for chaining works correctly";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}