#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include <xyz_autodiff/variable.cuh>
#include <xyz_autodiff/concept/variable.cuh>
#include <xyz_autodiff/concept/operation_node.cuh>
#include <xyz_autodiff/operations/unary/l1_norm_logic.cuh>
#include <xyz_autodiff/operations/unary/l2_norm_logic.cuh>
#include <xyz_autodiff/util/cuda_unique_ptr.cuh>
#include <xyz_autodiff/variable_operators.cuh>
#include <xyz_autodiff/testing/unary_gradient_tester.cuh>

using namespace xyz_autodiff;

// ===========================================
// Static Assert Tests for Concept Compliance
// ===========================================

// Test types
using TestVector3 = Variable<3, float>;
using TestVectorRef3 = VariableRef<3, float>;

// Norm operation types
using L1NormOp = UnaryOperation<1, op::L1NormLogic<3>, TestVectorRef3>;
using L2NormOp = UnaryOperation<1, op::L2NormLogic<3>, TestVectorRef3>;

// Static assertions for concept compliance
static_assert(VariableConcept<L1NormOp>, "L1NormOperation should satisfy VariableConcept");
static_assert(DifferentiableVariableConcept<L1NormOp>, "L1NormOperation should satisfy DifferentiableVariableConcept");
static_assert(OperationNode<L1NormOp>, "L1NormOperation should satisfy OperationNode");

static_assert(VariableConcept<L2NormOp>, "L2NormOperation should satisfy VariableConcept");
static_assert(DifferentiableVariableConcept<L2NormOp>, "L2NormOperation should satisfy DifferentiableVariableConcept");
static_assert(OperationNode<L2NormOp>, "L2NormOperation should satisfy OperationNode");

// Ensure Variable is NOT an OperationNode
static_assert(!OperationNode<TestVector3>, "Variable should NOT be OperationNode");

// ===========================================
// Test Class
// ===========================================

class NormOperationsGradientTest : public ::testing::Test {
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

TEST_F(NormOperationsGradientTest, L1NormGradientVerification) {
    using Logic = op::L1NormLogic<3>;
    xyz_autodiff::testing::UnaryGradientTester<Logic, 3, 1>::test_custom(
        "L1NormLogic", 
        50,      // num_tests
        1e-5,    // tolerance (minimum allowed for double precision)
        1e-6,    // delta
        -5.0,    // input_min
        5.0      // input_max
    );
}

TEST_F(NormOperationsGradientTest, L2NormGradientVerification) {
    using Logic = op::L2NormLogic<3>;
    xyz_autodiff::testing::UnaryGradientTester<Logic, 3, 1>::test_custom(
        "L2NormLogic", 
        50,      // num_tests
        1e-5,    // tolerance
        1e-7,    // delta
        0.1,     // input_min (avoid zero for L2 norm)
        5.0      // input_max
    );
}

// Test with different dimensions
TEST_F(NormOperationsGradientTest, L1NormGradientVerification5D) {
    using Logic = op::L1NormLogic<5>;
    xyz_autodiff::testing::UnaryGradientTester<Logic, 5, 1>::test_custom(
        "L1NormLogic5D", 
        30,      // num_tests
        1e-5,    // tolerance (minimum allowed for double precision)
        1e-6,    // delta
        -3.0,    // input_min
        3.0      // input_max
    );
}

TEST_F(NormOperationsGradientTest, L2NormGradientVerification2D) {
    using Logic = op::L2NormLogic<2>;
    xyz_autodiff::testing::UnaryGradientTester<Logic, 2, 1>::test_custom(
        "L2NormLogic2D", 
        30,      // num_tests
        1e-5,    // tolerance
        1e-7,    // delta
        0.5,     // input_min (avoid zero)
        10.0     // input_max
    );
}

// Edge case tests
TEST_F(NormOperationsGradientTest, L1NormNearZero) {
    using Logic = op::L1NormLogic<3>;
    xyz_autodiff::testing::UnaryGradientTester<Logic, 3, 1>::test_custom(
        "L1NormLogicNearZero", 
        30,      // num_tests
        1e-5,    // tolerance (minimum allowed for double precision)
        1e-7,    // delta (smaller for near-zero)
        -0.5,    // input_min (near zero but not exactly zero)
        0.5      // input_max
    );
}

TEST_F(NormOperationsGradientTest, L2NormMediumRange) {
    using Logic = op::L2NormLogic<4>;
    xyz_autodiff::testing::UnaryGradientTester<Logic, 4, 1>::test_custom(
        "L2NormLogicMediumRange", 
        40,      // num_tests
        1e-5,    // tolerance
        1e-7,    // delta
        1.0,     // input_min (avoid issues near zero)
        5.0      // input_max
    );
}

// Test large dimensions
TEST_F(NormOperationsGradientTest, L2NormLargeDimension) {
    using Logic = op::L2NormLogic<9>;
    xyz_autodiff::testing::UnaryGradientTester<Logic, 9, 1>::test_custom(
        "L2NormLogic9D", 
        20,      // num_tests (fewer for high dimension)
        1e-5,    // tolerance
        1e-6,    // delta
        0.2,     // input_min (avoid zero)
        2.0      // input_max
    );
}