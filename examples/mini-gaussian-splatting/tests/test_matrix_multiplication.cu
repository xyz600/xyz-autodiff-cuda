#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include "../../../include/variable.cuh"
#include "../../../include/concept/variable.cuh"
#include "../../../include/concept/operation_node.cuh"
#include "../../../include/operations/binary/matmul_logic.cuh"
#include "../../../include/util/cuda_unique_ptr.cuh"
#include "../../../tests/utility/binary_gradient_tester.cuh"

using namespace xyz_autodiff;

// ===========================================
// Static Assert Tests for Concept Compliance
// ===========================================

// Test types
using TestMatrix9 = Variable<9, float>;
using TestMatrixRef9 = VariableRef<9, float>;
using MatMulOp = BinaryOperation<9, op::MatMulLogic<3, 3, 3, TestMatrixRef9, TestMatrixRef9>, TestMatrixRef9, TestMatrixRef9>;

// Static assertions for concept compliance
static_assert(VariableConcept<TestMatrix9>, 
    "Variable<9, float> should satisfy VariableConcept");
static_assert(DifferentiableVariableConcept<TestMatrix9>, 
    "Variable<9, float> should satisfy DifferentiableVariableConcept");

static_assert(VariableConcept<MatMulOp>, 
    "MatrixMultiplication Operation should satisfy VariableConcept");
static_assert(DifferentiableVariableConcept<MatMulOp>, 
    "MatrixMultiplication Operation should satisfy DifferentiableVariableConcept");
static_assert(OperationNode<MatMulOp>, 
    "MatrixMultiplication Operation should satisfy OperationNode");

// Ensure Variable is not an OperationNode
static_assert(!OperationNode<TestMatrix9>, 
    "Variable should NOT satisfy OperationNode");

// ===========================================
// Test Class
// ===========================================

class MatrixMultiplicationTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaError_t err = cudaSetDevice(0);
        if (err != cudaSuccess) {
            GTEST_SKIP() << "CUDA device not available";
        }
    }
};

// ===========================================
// Gradient Verification Tests
// ===========================================

// Use the binary gradient tester utility
TEST_F(MatrixMultiplicationTest, GradientVerification) {
    using Logic = op::MatMulLogic<3, 3, 3, VariableRef<9, double>, VariableRef<9, double>>;
    test::BinaryGradientTester<Logic, 9, 9, 9>::test_custom(
        "MatrixMultiplication3x3", 
        50,      // num_tests
        1e-5,    // tolerance
        1e-6,    // delta
        -2.0,    // input_min
        2.0      // input_max
    );
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}