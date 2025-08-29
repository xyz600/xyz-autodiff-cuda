#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <xyz_autodiff/variable.cuh>
#include <xyz_autodiff/operations/binary/add_logic.cuh>
#include <xyz_autodiff/operations/binary/mul_logic.cuh>
#include <xyz_autodiff/operations/unary/exp_logic.cuh>
#include <xyz_autodiff/operations/unary/to_rotation_matrix_logic.cuh>
#include <xyz_autodiff/concept/variable.cuh>
#include <xyz_autodiff/concept/operation_node.cuh>
#include <xyz_autodiff/util/cuda_unique_ptr.cuh>

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

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}