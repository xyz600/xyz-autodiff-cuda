#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include <xyz_autodiff/variable.cuh>
#include <xyz_autodiff/concept/variable.cuh>
#include <xyz_autodiff/concept/operation_node.cuh>
#include <xyz_autodiff/operations/unary/exp_logic.cuh>
#include <xyz_autodiff/operations/unary/sigmoid_logic.cuh>
#include <xyz_autodiff/util/cuda_unique_ptr.cuh>
#include "../../utility/unary_gradient_tester.cuh"

using namespace xyz_autodiff;

// ===========================================
// Static Assert Tests for Concept Compliance
// ===========================================

// Test types
using TestVector3 = Variable<3, float>;
using TestVectorRef3 = VariableRef<3, float>;

// Exponential operation types
using ExpOp = UnaryOperation<3, op::ExpLogic<3>, TestVectorRef3>;
using SigmoidOp = UnaryOperation<3, op::SigmoidLogic<3>, TestVectorRef3>;

// Static assertions for concept compliance
static_assert(VariableConcept<ExpOp>, "ExpOperation should satisfy VariableConcept");
static_assert(DifferentiableVariableConcept<ExpOp>, "ExpOperation should satisfy DifferentiableVariableConcept");
static_assert(OperationNode<ExpOp>, "ExpOperation should satisfy OperationNode");

static_assert(VariableConcept<SigmoidOp>, "SigmoidOperation should satisfy VariableConcept");
static_assert(DifferentiableVariableConcept<SigmoidOp>, "SigmoidOperation should satisfy DifferentiableVariableConcept");
static_assert(OperationNode<SigmoidOp>, "SigmoidOperation should satisfy OperationNode");

// Ensure Variable is NOT an OperationNode
static_assert(!OperationNode<TestVector3>, "Variable should NOT be OperationNode");

// ===========================================
// Test Class
// ===========================================

class ExponentialOperationsGradientTest : public ::testing::Test {
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

TEST_F(ExponentialOperationsGradientTest, ExpGradientVerification) {
    using Logic = op::ExpLogic<3>;
    test::UnaryGradientTester<Logic, 3, 3>::test_custom(
        "ExpLogic", 
        50,      // num_tests
        1e-5,    // tolerance
        1e-7,    // delta
        -5.0,    // input_min (avoid overflow)
        5.0      // input_max (avoid overflow)
    );
}

TEST_F(ExponentialOperationsGradientTest, SigmoidGradientVerification) {
    using Logic = op::SigmoidLogic<3>;
    test::UnaryGradientTester<Logic, 3, 3>::test_custom(
        "SigmoidLogic", 
        50,      // num_tests
        1e-5,    // tolerance
        1e-7,    // delta
        -10.0,   // input_min (sigmoid is well-behaved for large range)
        10.0     // input_max
    );
}

// Test with different dimensions
TEST_F(ExponentialOperationsGradientTest, ExpGradientVerification2D) {
    using Logic = op::ExpLogic<2>;
    test::UnaryGradientTester<Logic, 2, 2>::test_custom(
        "ExpLogic2D", 
        30,      // num_tests
        1e-5,    // tolerance
        1e-7,    // delta
        -3.0,    // input_min (smaller range to avoid overflow)
        3.0      // input_max
    );
}

TEST_F(ExponentialOperationsGradientTest, SigmoidGradientVerification1D) {
    using Logic = op::SigmoidLogic<1>;
    test::UnaryGradientTester<Logic, 1, 1>::test_custom(
        "SigmoidLogic1D", 
        30,      // num_tests
        1e-5,    // tolerance
        1e-7,    // delta
        -20.0,   // input_min (sigmoid handles large negative values well)
        20.0     // input_max (sigmoid handles large positive values well)
    );
}

// Edge case tests
TEST_F(ExponentialOperationsGradientTest, ExpNearZero) {
    using Logic = op::ExpLogic<3>;
    test::UnaryGradientTester<Logic, 3, 3>::test_custom(
        "ExpLogicNearZero", 
        30,      // num_tests
        1e-5,    // tolerance
        1e-8,    // delta (smaller for near-zero)
        -0.1,    // input_min (near zero)
        0.1      // input_max
    );
}

TEST_F(ExponentialOperationsGradientTest, SigmoidSteepTransition) {
    using Logic = op::SigmoidLogic<2>;
    test::UnaryGradientTester<Logic, 2, 2>::test_custom(
        "SigmoidLogicSteepTransition", 
        30,      // num_tests
        1e-5,    // tolerance
        1e-7,    // delta
        -2.0,    // input_min (around steep transition)
        2.0      // input_max
    );
}

// Test small values where exp might underflow
TEST_F(ExponentialOperationsGradientTest, ExpSmallValues) {
    using Logic = op::ExpLogic<2>;
    test::UnaryGradientTester<Logic, 2, 2>::test_custom(
        "ExpLogicSmallValues", 
        30,      // num_tests
        1e-5,    // tolerance
        1e-7,    // delta
        -10.0,   // input_min (exp approaches zero but doesn't underflow)
        -1.0     // input_max
    );
}

// Test sigmoid in saturation regions
TEST_F(ExponentialOperationsGradientTest, SigmoidSaturation) {
    using Logic = op::SigmoidLogic<3>;
    test::UnaryGradientTester<Logic, 3, 3>::test_custom(
        "SigmoidLogicSaturation", 
        40,      // num_tests
        1e-5,    // tolerance
        1e-6,    // delta
        -15.0,   // input_min (saturation region)
        15.0     // input_max (saturation region)
    );
}