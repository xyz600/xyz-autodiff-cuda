#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include <xyz_autodiff/variable.cuh>
#include <xyz_autodiff/concept/variable.cuh>
#include <xyz_autodiff/concept/operation_node.cuh>
#include <xyz_autodiff/operations/unary/sin_logic.cuh>
#include <xyz_autodiff/operations/unary/cos_logic.cuh>
#include <xyz_autodiff/util/cuda_unique_ptr.cuh>
#include "../../utility/unary_gradient_tester.cuh"

using namespace xyz_autodiff;

// ===========================================
// Static Assert Tests for Concept Compliance
// ===========================================

// Test types
using TestVector3 = Variable<3, float>;
using TestVectorRef3 = VariableRef<3, float>;

// Trigonometric operation types
using SinOp = UnaryOperation<3, op::SinLogic<3>, TestVectorRef3>;
using CosOp = UnaryOperation<3, op::CosLogic<3>, TestVectorRef3>;

// Static assertions for concept compliance
static_assert(VariableConcept<SinOp>, "SinOperation should satisfy VariableConcept");
static_assert(DifferentiableVariableConcept<SinOp>, "SinOperation should satisfy DifferentiableVariableConcept");
static_assert(OperationNode<SinOp>, "SinOperation should satisfy OperationNode");

static_assert(VariableConcept<CosOp>, "CosOperation should satisfy VariableConcept");
static_assert(DifferentiableVariableConcept<CosOp>, "CosOperation should satisfy DifferentiableVariableConcept");
static_assert(OperationNode<CosOp>, "CosOperation should satisfy OperationNode");

// Ensure Variable is NOT an OperationNode
static_assert(!OperationNode<TestVector3>, "Variable should NOT be OperationNode");

// ===========================================
// Test Class
// ===========================================

class TrigonometricOperationsGradientTest : public ::testing::Test {
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

TEST_F(TrigonometricOperationsGradientTest, SinGradientVerification) {
    using Logic = op::SinLogic<3>;
    test::UnaryGradientTester<Logic, 3, 3>::test_custom(
        "SinLogic", 
        50,      // num_tests
        1e-5,    // tolerance
        1e-7,    // delta
        -3.14159, // input_min (-π)
        3.14159   // input_max (π)
    );
}

TEST_F(TrigonometricOperationsGradientTest, CosGradientVerification) {
    using Logic = op::CosLogic<3>;
    test::UnaryGradientTester<Logic, 3, 3>::test_custom(
        "CosLogic", 
        50,      // num_tests
        1e-5,    // tolerance
        1e-7,    // delta
        -3.14159, // input_min (-π)
        3.14159   // input_max (π)
    );
}

// Test with different dimensions
TEST_F(TrigonometricOperationsGradientTest, SinGradientVerification2D) {
    using Logic = op::SinLogic<2>;
    test::UnaryGradientTester<Logic, 2, 2>::test_custom(
        "SinLogic2D", 
        30,      // num_tests
        1e-5,    // tolerance
        1e-7,    // delta
        -6.28318, // input_min (-2π)
        6.28318   // input_max (2π)
    );
}

TEST_F(TrigonometricOperationsGradientTest, CosGradientVerification1D) {
    using Logic = op::CosLogic<1>;
    test::UnaryGradientTester<Logic, 1, 1>::test_custom(
        "CosLogic1D", 
        30,      // num_tests
        1e-5,    // tolerance
        1e-7,    // delta
        -1.5708,  // input_min (-π/2)
        1.5708    // input_max (π/2)
    );
}

// Test edge cases - near critical points
TEST_F(TrigonometricOperationsGradientTest, SinNearZero) {
    using Logic = op::SinLogic<3>;
    test::UnaryGradientTester<Logic, 3, 3>::test_custom(
        "SinLogicNearZero", 
        30,      // num_tests
        1e-5,    // tolerance
        1e-8,    // delta (smaller for near-zero)
        -0.1,    // input_min (near zero)
        0.1      // input_max
    );
}

TEST_F(TrigonometricOperationsGradientTest, CosNearPiHalf) {
    using Logic = op::CosLogic<2>;
    test::UnaryGradientTester<Logic, 2, 2>::test_custom(
        "CosLogicNearPiHalf", 
        30,      // num_tests
        1e-5,    // tolerance
        1e-7,    // delta
        1.4708,   // input_min (near π/2)
        1.6708    // input_max
    );
}

// Test larger range
TEST_F(TrigonometricOperationsGradientTest, SinLargeRange) {
    using Logic = op::SinLogic<2>;
    test::UnaryGradientTester<Logic, 2, 2>::test_custom(
        "SinLogicLargeRange", 
        40,      // num_tests
        1e-5,    // tolerance
        1e-7,    // delta
        -10.0,   // input_min
        10.0     // input_max
    );
}

TEST_F(TrigonometricOperationsGradientTest, CosLargeRange) {
    using Logic = op::CosLogic<2>;
    test::UnaryGradientTester<Logic, 2, 2>::test_custom(
        "CosLogicLargeRange", 
        40,      // num_tests
        1e-5,    // tolerance
        1e-7,    // delta
        -10.0,   // input_min
        10.0     // input_max
    );
}