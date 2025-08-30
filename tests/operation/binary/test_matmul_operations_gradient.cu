#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include <xyz_autodiff/variable.cuh>
#include <xyz_autodiff/concept/variable.cuh>
#include <xyz_autodiff/concept/operation_node.cuh>
#include <xyz_autodiff/operations/binary/matmul_logic.cuh>
#include <xyz_autodiff/util/cuda_unique_ptr.cuh>
#include <xyz_autodiff/variable_operators.cuh>
#include "../../utility/binary_gradient_tester.cuh"

using namespace xyz_autodiff;

// ===========================================
// Static Assert Tests for Concept Compliance
// ===========================================

// Test types for different matrix sizes
using TestMatrix4 = Variable<4, float>;   // 2x2 matrix
using TestMatrixRef4 = VariableRef<4, float>;
using TestMatrix9 = Variable<9, float>;   // 3x3 matrix
using TestMatrixRef9 = VariableRef<9, float>;
using TestMatrix16 = Variable<16, float>; // 4x4 matrix
using TestMatrixRef16 = VariableRef<16, float>;

// Different matrix multiplication operation types
using MatMul2x2Op = BinaryOperation<4, op::MatMulLogic<2, 2, 2, TestMatrixRef4, TestMatrixRef4>, TestMatrixRef4, TestMatrixRef4>;
using MatMul3x3Op = BinaryOperation<9, op::MatMulLogic<3, 3, 3, TestMatrixRef9, TestMatrixRef9>, TestMatrixRef9, TestMatrixRef9>;
using MatMul4x4Op = BinaryOperation<16, op::MatMulLogic<4, 4, 4, TestMatrixRef16, TestMatrixRef16>, TestMatrixRef16, TestMatrixRef16>;

// Matrix-vector and vector-matrix operations
using TestVector2 = Variable<2, float>;
using TestVectorRef2 = VariableRef<2, float>;
using TestVector3 = Variable<3, float>;
using TestVectorRef3 = VariableRef<3, float>;

using MatVec2x2Op = BinaryOperation<2, op::MatMulLogic<2, 2, 1, TestMatrixRef4, TestVectorRef2>, TestMatrixRef4, TestVectorRef2>;
using VecMat2x2Op = BinaryOperation<2, op::MatMulLogic<1, 2, 2, TestVectorRef2, TestMatrixRef4>, TestVectorRef2, TestMatrixRef4>;

// Non-square matrix operations
using TestMatrix6 = Variable<6, float>;   // 2x3 matrix
using TestMatrixRef6 = VariableRef<6, float>;
using TestMatrix12 = Variable<12, float>; // 3x4 matrix  
using TestMatrixRef12 = VariableRef<12, float>;
using TestMatrix8 = Variable<8, float>;   // 2x4 matrix
using TestMatrixRef8 = VariableRef<8, float>;

using MatMul2x3_3x4Op = BinaryOperation<8, op::MatMulLogic<2, 3, 4, TestMatrixRef6, TestMatrixRef12>, TestMatrixRef6, TestMatrixRef12>;

// Static assertions for concept compliance
static_assert(VariableConcept<MatMul2x2Op>, "MatMul2x2 Operation should satisfy VariableConcept");
static_assert(DifferentiableVariableConcept<MatMul2x2Op>, "MatMul2x2 Operation should satisfy DifferentiableVariableConcept");
static_assert(OperationNode<MatMul2x2Op>, "MatMul2x2 Operation should satisfy OperationNode");

static_assert(VariableConcept<MatMul3x3Op>, "MatMul3x3 Operation should satisfy VariableConcept");
static_assert(DifferentiableVariableConcept<MatMul3x3Op>, "MatMul3x3 Operation should satisfy DifferentiableVariableConcept");
static_assert(OperationNode<MatMul3x3Op>, "MatMul3x3 Operation should satisfy OperationNode");

static_assert(VariableConcept<MatVec2x2Op>, "MatVec2x2 Operation should satisfy VariableConcept");
static_assert(DifferentiableVariableConcept<MatVec2x2Op>, "MatVec2x2 Operation should satisfy DifferentiableVariableConcept");
static_assert(OperationNode<MatVec2x2Op>, "MatVec2x2 Operation should satisfy OperationNode");

// Ensure Variable is not an OperationNode
static_assert(!OperationNode<TestMatrix4>, "Variable should NOT satisfy OperationNode");

// ===========================================
// Test Class
// ===========================================

class MatMulOperationsGradientTest : public ::testing::Test {
protected:
    void SetUp() override {
        int device_count;
        cudaGetDeviceCount(&device_count);
        if (device_count == 0) {
            GTEST_SKIP() << "No CUDA devices available";
        }
    }
};

// ===========================================
// Gradient Verification Tests
// ===========================================

// Test 2x2 matrix multiplication
TEST_F(MatMulOperationsGradientTest, MatMul2x2GradientVerification) {
    using Logic = op::MatMulLogic<2, 2, 2, VariableRef<4, double>, VariableRef<4, double>>;
    test::BinaryGradientTester<Logic, 4, 4, 4>::test_custom(
        "MatMul2x2Logic", 
        50,      // num_tests
        1e-5,    // tolerance
        1e-7,    // delta
        -2.0,    // input_min
        2.0      // input_max
    );
}

// Test 3x3 matrix multiplication
TEST_F(MatMulOperationsGradientTest, MatMul3x3GradientVerification) {
    using Logic = op::MatMulLogic<3, 3, 3, VariableRef<9, double>, VariableRef<9, double>>;
    test::BinaryGradientTester<Logic, 9, 9, 9>::test_custom(
        "MatMul3x3Logic", 
        50,      // num_tests
        1e-5,    // tolerance
        1e-7,    // delta
        -2.0,    // input_min
        2.0      // input_max
    );
}

// Test 4x4 matrix multiplication
TEST_F(MatMulOperationsGradientTest, MatMul4x4GradientVerification) {
    using Logic = op::MatMulLogic<4, 4, 4, VariableRef<16, double>, VariableRef<16, double>>;
    test::BinaryGradientTester<Logic, 16, 16, 16>::test_custom(
        "MatMul4x4Logic", 
        30,      // num_tests (fewer for larger matrices)
        1e-5,    // tolerance
        1e-6,    // delta (slightly larger for numerical stability)
        -1.5,    // input_min (smaller range for stability)
        1.5      // input_max
    );
}

// Test 2x2 matrix-vector multiplication (2x2 * 2x1 = 2x1)
TEST_F(MatMulOperationsGradientTest, MatVec2x2GradientVerification) {
    using Logic = op::MatMulLogic<2, 2, 1, VariableRef<4, double>, VariableRef<2, double>>;
    test::BinaryGradientTester<Logic, 4, 2, 2>::test_custom(
        "MatVec2x2Logic", 
        50,      // num_tests
        1e-5,    // tolerance
        1e-7,    // delta
        -3.0,    // input_min
        3.0      // input_max
    );
}

// Test 2x2 vector-matrix multiplication (1x2 * 2x2 = 1x2)
TEST_F(MatMulOperationsGradientTest, VecMat2x2GradientVerification) {
    using Logic = op::MatMulLogic<1, 2, 2, VariableRef<2, double>, VariableRef<4, double>>;
    test::BinaryGradientTester<Logic, 2, 4, 2>::test_custom(
        "VecMat2x2Logic", 
        50,      // num_tests
        1e-5,    // tolerance
        1e-7,    // delta
        -3.0,    // input_min
        3.0      // input_max
    );
}

// Test non-square matrix multiplication (2x3 * 3x4 = 2x4)
TEST_F(MatMulOperationsGradientTest, MatMul2x3_3x4GradientVerification) {
    using Logic = op::MatMulLogic<2, 3, 4, VariableRef<6, double>, VariableRef<12, double>>;
    test::BinaryGradientTester<Logic, 6, 12, 8>::test_custom(
        "MatMul2x3_3x4Logic", 
        40,      // num_tests
        1e-5,    // tolerance
        1e-6,    // delta
        -2.0,    // input_min
        2.0      // input_max
    );
}

// Test 3x1 matrix-vector multiplication (3x1 * 1x1 = 3x1)
TEST_F(MatMulOperationsGradientTest, MatVec3x1GradientVerification) {
    using Logic = op::MatMulLogic<3, 1, 1, VariableRef<3, double>, VariableRef<1, double>>;
    test::BinaryGradientTester<Logic, 3, 1, 3>::test_custom(
        "MatVec3x1Logic", 
        50,      // num_tests
        1e-5,    // tolerance
        1e-7,    // delta
        -4.0,    // input_min
        4.0      // input_max
    );
}

// Test 1x3 vector-matrix multiplication (1x3 * 3x2 = 1x2)
TEST_F(MatMulOperationsGradientTest, VecMat1x3_3x2GradientVerification) {
    using Logic = op::MatMulLogic<1, 3, 2, VariableRef<3, double>, VariableRef<6, double>>;
    test::BinaryGradientTester<Logic, 3, 6, 2>::test_custom(
        "VecMat1x3_3x2Logic", 
        50,      // num_tests
        1e-5,    // tolerance
        1e-7,    // delta
        -2.5,    // input_min
        2.5      // input_max
    );
}

// Test edge case: 1x1 matrix multiplication (scalar multiplication)
TEST_F(MatMulOperationsGradientTest, MatMul1x1GradientVerification) {
    using Logic = op::MatMulLogic<1, 1, 1, VariableRef<1, double>, VariableRef<1, double>>;
    test::BinaryGradientTester<Logic, 1, 1, 1>::test_custom(
        "MatMul1x1Logic", 
        50,      // num_tests
        1e-5,    // tolerance
        1e-7,    // delta
        -10.0,   // input_min (can handle larger range for scalars)
        10.0     // input_max
    );
}

// Test rectangular matrix: 4x2 * 2x3 = 4x3
TEST_F(MatMulOperationsGradientTest, MatMul4x2_2x3GradientVerification) {
    using Logic = op::MatMulLogic<4, 2, 3, VariableRef<8, double>, VariableRef<6, double>>;
    test::BinaryGradientTester<Logic, 8, 6, 12>::test_custom(
        "MatMul4x2_2x3Logic", 
        30,      // num_tests (fewer for larger output)
        1e-5,    // tolerance
        1e-6,    // delta
        -1.5,    // input_min (smaller range for stability)
        1.5      // input_max
    );
}