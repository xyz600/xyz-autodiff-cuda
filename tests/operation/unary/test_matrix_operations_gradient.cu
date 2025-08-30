#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include <xyz_autodiff/variable.cuh>
#include <xyz_autodiff/concept/variable.cuh>
#include <xyz_autodiff/concept/operation_node.cuh>
#include <xyz_autodiff/operations/unary/sym_matrix2_inv_logic.cuh>
#include <xyz_autodiff/util/cuda_unique_ptr.cuh>
#include <xyz_autodiff/variable_operators.cuh>
#include "../../utility/unary_gradient_tester.cuh"

using namespace xyz_autodiff;

// ===========================================
// Static Assert Tests for Concept Compliance
// ===========================================

// Test types
using TestMatrix2x2 = Variable<3, float>;  // 2x2 symmetric matrix as 3-element vector [a,b,c] for [[a,b],[b,c]]
using TestMatrixRef2x2 = VariableRef<3, float>;

// Matrix operation types
using SymMatrix2InvOp = UnaryOperation<3, op::SymMatrix2InvLogic<3>, TestMatrixRef2x2>;

// Static assertions for concept compliance
static_assert(VariableConcept<SymMatrix2InvOp>, "SymMatrix2InvOperation should satisfy VariableConcept");
static_assert(DifferentiableVariableConcept<SymMatrix2InvOp>, "SymMatrix2InvOperation should satisfy DifferentiableVariableConcept");
static_assert(OperationNode<SymMatrix2InvOp>, "SymMatrix2InvOperation should satisfy OperationNode");

// Ensure Variable is NOT an OperationNode
static_assert(!OperationNode<TestMatrix2x2>, "Variable should NOT be OperationNode");

// ===========================================
// Test Class
// ===========================================

class MatrixOperationsGradientTest : public ::testing::Test {
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

TEST_F(MatrixOperationsGradientTest, SymMatrix2InvGradientVerification) {
    using Logic = op::SymMatrix2InvLogic<3>;
    test::UnaryGradientTester<Logic, 3, 3>::test_custom(
        "SymMatrix2InvLogic", 
        30,      // num_tests (fewer due to complexity)
        1e-5,    // tolerance (minimum allowed for double precision)
        1e-6,    // delta
        0.1,     // input_min (avoid singular matrices)
        5.0      // input_max
    );
}

// Test with well-conditioned matrices
TEST_F(MatrixOperationsGradientTest, SymMatrix2InvWellConditioned) {
    using Logic = op::SymMatrix2InvLogic<3>;
    test::UnaryGradientTester<Logic, 3, 3>::test_custom(
        "SymMatrix2InvLogicWellConditioned", 
        40,      // num_tests
        1e-5,    // tolerance
        1e-7,    // delta
        1.0,     // input_min (well-conditioned range)
        10.0     // input_max
    );
}

// Test with identity-like matrices (near-diagonal dominant)
TEST_F(MatrixOperationsGradientTest, SymMatrix2InvNearIdentity) {
    using Logic = op::SymMatrix2InvLogic<3>;
    test::UnaryGradientTester<Logic, 3, 3>::test_custom(
        "SymMatrix2InvLogicNearIdentity", 
        30,      // num_tests
        1e-5,    // tolerance
        1e-7,    // delta
        0.5,     // input_min (diagonal elements dominant)
        2.0      // input_max
    );
}

// Test with moderately conditioned matrices
TEST_F(MatrixOperationsGradientTest, SymMatrix2InvModerateCondition) {
    using Logic = op::SymMatrix2InvLogic<3>;
    test::UnaryGradientTester<Logic, 3, 3>::test_custom(
        "SymMatrix2InvLogicModerateCondition", 
        25,      // num_tests
        1e-5,    // tolerance
        1e-6,    // delta
        0.2,     // input_min (moderate conditioning)
        3.0      // input_max
    );
}