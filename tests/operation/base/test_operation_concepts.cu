#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "../../../include/variable.cuh"
#include "../../../include/operations/binary/add_logic.cuh"
#include "../../../include/operations/binary/mul_logic.cuh"
#include "../../../include/operations/unary/exp_logic.cuh"
#include "../../../include/operations/unary/to_rotation_matrix_logic.cuh"
#include "../../../include/concept/variable.cuh"
#include "../../../include/concept/operation_node.cuh"
#include "../../../include/util/cuda_unique_ptr.cuh"

using namespace xyz_autodiff;

// テスト用のOperation型を定義  
using TestVariable = VariableRef<2, float>;
using TestUnaryOp = UnaryOperation<2, op::ExpLogic<2>, TestVariable>;
using TestBinaryOp = BinaryOperation<2, op::MulLogic<TestVariable, TestVariable>, TestVariable, TestVariable>;

// QuaternionToRotationMatrix用のテスト型
using TestQuaternion = Variable<4, float>;
using TestQuatToMatOp = UnaryOperation<9, op::QuaternionToRotationMatrixLogic<4>, TestQuaternion>;

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
static_assert(VariableConcept<Variable<3, float>>, 
    "Variable should satisfy VariableConcept");
static_assert(VariableConcept<VariableRef<5, double>>, 
    "VariableRef should satisfy VariableConcept");

static_assert(DifferentiableVariableConcept<Variable<3, float>>, 
    "Variable should satisfy DifferentiableVariableConcept");
static_assert(DifferentiableVariableConcept<VariableRef<5, double>>, 
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
static_assert(!OperationNode<Variable<3, float>>, 
    "Variable should NOT satisfy OperationNode");
static_assert(!OperationNode<VariableRef<5, double>>, 
    "VariableRef should NOT satisfy OperationNode");

// 具体的な型でのテスト
using TestVariable4d = Variable<4, double>;
using TestVariable2f = Variable<2, float>;
static_assert(VariableConcept<UnaryOperation<4, op::ExpLogic<4>, TestVariable4d>>, 
    "Specific UnaryOperation should satisfy VariableConcept");
static_assert(VariableConcept<BinaryOperation<2, op::MulLogic<TestVariable2f, TestVariable2f>, TestVariable2f, TestVariable2f>>, 
    "Specific BinaryOperation should satisfy VariableConcept");

// QuaternionToRotationMatrix Operation関連のstatic_assert
static_assert(VariableConcept<TestQuatToMatOp>, 
    "QuaternionToRotationMatrix Operation should satisfy VariableConcept");
static_assert(DifferentiableVariableConcept<TestQuatToMatOp>, 
    "QuaternionToRotationMatrix Operation should satisfy DifferentiableVariableConcept");
static_assert(OperationNode<TestQuatToMatOp>, 
    "QuaternionToRotationMatrix Operation should satisfy OperationNode");

// 具体的なQuaternionToRotationMatrix型でのテスト
using TestQuaternionFloat = Variable<4, float>;
using TestQuaternionDouble = Variable<4, double>;
static_assert(VariableConcept<UnaryOperation<9, op::QuaternionToRotationMatrixLogic<4>, TestQuaternionFloat>>, 
    "Float QuaternionToRotationMatrix Operation should satisfy VariableConcept");
static_assert(VariableConcept<UnaryOperation<9, op::QuaternionToRotationMatrixLogic<4>, TestQuaternionDouble>>, 
    "Double QuaternionToRotationMatrix Operation should satisfy VariableConcept");
static_assert(OperationNode<UnaryOperation<9, op::QuaternionToRotationMatrixLogic<4>, TestQuaternionFloat>>, 
    "Float QuaternionToRotationMatrix Operation should satisfy OperationNode");
static_assert(OperationNode<UnaryOperation<9, op::QuaternionToRotationMatrixLogic<4>, TestQuaternionDouble>>, 
    "Double QuaternionToRotationMatrix Operation should satisfy OperationNode");

// QuaternionToRotationMatrixLogic自体のテスト（型チェック）
static_assert(op::QuaternionToRotationMatrixLogic<4>::outputDim == 9, 
    "QuaternionToRotationMatrixLogic should have outputDim = 9");

// Quaternion変数は OperationNode ではないことを再確認
static_assert(!OperationNode<TestQuaternionFloat>, 
    "Quaternion Variable should NOT satisfy OperationNode");
static_assert(!OperationNode<TestQuaternionDouble>, 
    "Quaternion Variable should NOT satisfy OperationNode");

TEST_F(OperationConceptTest, ConceptComplianceBasicTest) {
    // このテストは主にコンパイル時の確認のためのもの
    // static_assertがすべて通ればテストは成功
    EXPECT_TRUE(true);
}

// Variableのzero_gradテスト用カーネル
template<typename T>
__global__ void test_variable_zero_grad_kernel(T* result) {
    Variable<3, float> var;
    var.zero_grad(); // コンパイルが通ることを確認
    
    VariableRef<3, float> var_ref(var.data(), var.grad());
    var_ref.zero_grad(); // コンパイルが通ることを確認
    
    *result = 1.0f; // 成功マーカー
}

TEST_F(OperationConceptTest, VariableZeroGradInterface) {
    auto device_result = makeCudaUnique<float>();
    
    test_variable_zero_grad_kernel<<<1, 1>>>(device_result.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    float host_result;
    ASSERT_EQ(cudaMemcpy(&host_result, device_result.get(), sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(host_result, 1.0f);
}

// Operationのインターフェーステスト用カーネル
__global__ void test_operation_interface_kernel(float* result) {
    float data1[2] = {1.0f, 2.0f};
    float grad1[2] = {0.0f, 0.0f};
    float data2[2] = {3.0f, 4.0f};
    float grad2[2] = {0.0f, 0.0f};
    
    VariableRef<2, float> var1(data1, grad1);
    VariableRef<2, float> var2(data2, grad2);
    
    // UnaryOperationを作成（ExpLogicを使用）
    op::ExpLogic<2> logic;
    UnaryOperation<2, op::ExpLogic<2>, VariableRef<2, float>> op(logic, var1);
    
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
    auto device_result = makeCudaUnique<float>();
    
    test_operation_interface_kernel<<<1, 1>>>(device_result.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    float host_result;
    ASSERT_EQ(cudaMemcpy(&host_result, device_result.get(), sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(host_result, 1.0f);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}