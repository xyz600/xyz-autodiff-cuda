#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "../include/variable.cuh"
#include "../include/operations/add_logic.cuh"
#include "../include/operations/mul_logic.cuh"
#include "../include/operations/exp_logic.cuh"
#include "../include/concept/variable.cuh"
#include "../include/concept/operation_node.cuh"

using namespace xyz_autodiff;

// テスト用のOperation型を定義  
using TestVariable = VariableRef<float, 2>;
using TestUnaryOp = UnaryOperation<2, ExpLogic<2>, TestVariable>;
using TestBinaryOp = BinaryOperation<2, op::MulLogic<TestVariable, TestVariable>, TestVariable, TestVariable>;

class OperationConceptTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaError_t err = cudaSetDevice(0);
        if (err != cudaSuccess) {
            GTEST_SKIP() << "CUDA device not available";
        }
    }
};

// Variable関連のstatic_assert
static_assert(VariableConcept<Variable<float, 3>>, 
    "Variable should satisfy VariableConcept");
static_assert(VariableConcept<VariableRef<double, 5>>, 
    "VariableRef should satisfy VariableConcept");

static_assert(DifferentiableVariableConcept<Variable<float, 3>>, 
    "Variable should satisfy DifferentiableVariableConcept");
static_assert(DifferentiableVariableConcept<VariableRef<double, 5>>, 
    "VariableRef should satisfy DifferentiableVariableConcept");

// Operation関連のstatic_assert
static_assert(VariableConcept<TestUnaryOp>, 
    "UnaryOperation should satisfy VariableConcept");
static_assert(VariableConcept<TestBinaryOp>, 
    "BinaryOperation should satisfy VariableConcept");

static_assert(DifferentiableVariableConcept<TestUnaryOp>, 
    "UnaryOperation should satisfy DifferentiableVariableConcept");
static_assert(DifferentiableVariableConcept<TestBinaryOp>, 
    "BinaryOperation should satisfy DifferentiableVariableConcept");

static_assert(OperationNode<TestUnaryOp>, 
    "UnaryOperation should satisfy OperationNode");
static_assert(OperationNode<TestBinaryOp>, 
    "BinaryOperation should satisfy OperationNode");

// Variable は OperationNode ではないことを確認
static_assert(!OperationNode<Variable<float, 3>>, 
    "Variable should NOT satisfy OperationNode");
static_assert(!OperationNode<VariableRef<double, 5>>, 
    "VariableRef should NOT satisfy OperationNode");

// 具体的な型でのテスト
using TestVariable4d = Variable<double, 4>;
using TestVariable2f = Variable<float, 2>;
static_assert(VariableConcept<UnaryOperation<4, ExpLogic<4>, TestVariable4d>>, 
    "Specific UnaryOperation should satisfy VariableConcept");
static_assert(VariableConcept<BinaryOperation<2, op::MulLogic<TestVariable2f, TestVariable2f>, TestVariable2f, TestVariable2f>>, 
    "Specific BinaryOperation should satisfy VariableConcept");

TEST_F(OperationConceptTest, ConceptComplianceBasicTest) {
    // このテストは主にコンパイル時の確認のためのもの
    // static_assertがすべて通ればテストは成功
    EXPECT_TRUE(true);
}

// Variableのzero_gradテスト用カーネル
template<typename T>
__global__ void test_variable_zero_grad_kernel(T* result) {
    Variable<float, 3> var;
    var.zero_grad(); // コンパイルが通ることを確認
    
    VariableRef<float, 3> var_ref(var.data(), var.grad());
    var_ref.zero_grad(); // コンパイルが通ることを確認
    
    *result = 1.0f; // 成功マーカー
}

TEST_F(OperationConceptTest, VariableZeroGradInterface) {
    float* device_result;
    ASSERT_EQ(cudaMalloc(&device_result, sizeof(float)), cudaSuccess);
    
    test_variable_zero_grad_kernel<<<1, 1>>>(device_result);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    float host_result;
    ASSERT_EQ(cudaMemcpy(&host_result, device_result, sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(host_result, 1.0f);
    
    cudaFree(device_result);
}

// Operationのインターフェーステスト用カーネル
__global__ void test_operation_interface_kernel(float* result) {
    float data1[2] = {1.0f, 2.0f};
    float grad1[2] = {0.0f, 0.0f};
    float data2[2] = {3.0f, 4.0f};
    float grad2[2] = {0.0f, 0.0f};
    
    VariableRef<float, 2> var1(data1, grad1);
    VariableRef<float, 2> var2(data2, grad2);
    
    // UnaryOperationを作成（ExpLogicを使用）
    ExpLogic<2> logic;
    UnaryOperation<2, ExpLogic<2>, VariableRef<float, 2>> op(logic, var1);
    
    // VariableConceptのインターフェースが使えることを確認
    op.zero_grad();  // zero_grad
    constexpr auto size = op.size;  // size
    auto* data = op.data();  // data()
    auto* grad = op.grad();  // grad()
    auto value = op[0];  // operator[]
    auto grad_value = op.grad(0);  // grad(size_t)
    
    // 結果を設定（サイズが正しいことを確認）
    *result = (size == 2 && data != nullptr && grad != nullptr) ? 1.0f : 0.0f;
}

TEST_F(OperationConceptTest, OperationVariableInterface) {
    float* device_result;
    ASSERT_EQ(cudaMalloc(&device_result, sizeof(float)), cudaSuccess);
    
    test_operation_interface_kernel<<<1, 1>>>(device_result);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    float host_result;
    ASSERT_EQ(cudaMemcpy(&host_result, device_result, sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(host_result, 1.0f);
    
    cudaFree(device_result);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}