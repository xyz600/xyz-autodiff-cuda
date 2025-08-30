#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include <xyz_autodiff/variable.cuh>
#include <xyz_autodiff/concept/variable.cuh>
#include <xyz_autodiff/concept/operation_node.cuh>
#include <xyz_autodiff/operations/unary/squared_logic.cuh>
#include <xyz_autodiff/operations/unary/neg_logic.cuh>
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

// Unary operation types
using SquaredOp = UnaryOperation<3, op::SquaredLogic<3>, TestVectorRef3>;
using NegOp = UnaryOperation<3, op::NegLogic<3>, TestVectorRef3>;

// Static assertions for concept compliance
static_assert(VariableConcept<SquaredOp>, "SquaredOperation should satisfy VariableConcept");
static_assert(DifferentiableVariableConcept<SquaredOp>, "SquaredOperation should satisfy DifferentiableVariableConcept");
static_assert(OperationNode<SquaredOp>, "SquaredOperation should satisfy OperationNode");

static_assert(VariableConcept<NegOp>, "NegOperation should satisfy VariableConcept");
static_assert(DifferentiableVariableConcept<NegOp>, "NegOperation should satisfy DifferentiableVariableConcept");
static_assert(OperationNode<NegOp>, "NegOperation should satisfy OperationNode");

// Ensure Variable is NOT an OperationNode
static_assert(!OperationNode<TestVector3>, "Variable should NOT be OperationNode");

// ===========================================
// Test Class
// ===========================================

class BasicUnaryOperationsGradientTest : public ::testing::Test {
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

TEST_F(BasicUnaryOperationsGradientTest, SquaredGradientVerification) {
    using Logic = op::SquaredLogic<3>;
    xyz_autodiff::testing::UnaryGradientTester<Logic, 3, 3>::test_custom(
        "SquaredLogic", 
        50,      // num_tests
        1e-5,    // tolerance
        1e-7,    // delta
        -5.0,    // input_min
        5.0      // input_max
    );
}

TEST_F(BasicUnaryOperationsGradientTest, NegGradientVerification) {
    using Logic = op::NegLogic<3>;
    xyz_autodiff::testing::UnaryGradientTester<Logic, 3, 3>::test_custom(
        "NegLogic", 
        50,      // num_tests
        1e-5,    // tolerance
        1e-7,    // delta
        -10.0,   // input_min
        10.0     // input_max
    );
}

// Test with different dimensions
TEST_F(BasicUnaryOperationsGradientTest, SquaredGradientVerification2D) {
    using Logic = op::SquaredLogic<2>;
    xyz_autodiff::testing::UnaryGradientTester<Logic, 2, 2>::test_custom(
        "SquaredLogic2D", 
        30,      // num_tests
        1e-5,    // tolerance
        1e-7,    // delta
        -3.0,    // input_min
        3.0      // input_max
    );
}

TEST_F(BasicUnaryOperationsGradientTest, NegGradientVerification1D) {
    using Logic = op::NegLogic<1>;
    xyz_autodiff::testing::UnaryGradientTester<Logic, 1, 1>::test_custom(
        "NegLogic1D", 
        30,      // num_tests
        1e-5,    // tolerance
        1e-7,    // delta
        -15.0,   // input_min
        15.0     // input_max
    );
}

// Edge case tests
TEST_F(BasicUnaryOperationsGradientTest, SquaredNearZero) {
    using Logic = op::SquaredLogic<3>;
    xyz_autodiff::testing::UnaryGradientTester<Logic, 3, 3>::test_custom(
        "SquaredLogicNearZero", 
        30,      // num_tests
        1e-5,    // tolerance
        1e-8,    // delta (smaller for near-zero)
        -0.1,    // input_min (near zero)
        0.1      // input_max
    );
}

TEST_F(BasicUnaryOperationsGradientTest, SquaredLargeValues) {
    using Logic = op::SquaredLogic<2>;
    xyz_autodiff::testing::UnaryGradientTester<Logic, 2, 2>::test_custom(
        "SquaredLogicLargeValues", 
        20,      // num_tests
        1e-5,    // tolerance (minimum allowed for double precision)
        1e-6,    // delta
        -10.0,   // input_min
        10.0     // input_max
    );
}