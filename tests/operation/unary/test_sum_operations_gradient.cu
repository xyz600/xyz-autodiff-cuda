#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include <xyz_autodiff/variable.cuh>
#include <xyz_autodiff/concept/variable.cuh>
#include <xyz_autodiff/concept/operation_node.cuh>
#include <xyz_autodiff/operations/unary/sum_logic.cuh>
#include <xyz_autodiff/util/cuda_unique_ptr.cuh>
#include "../../utility/unary_gradient_tester.cuh"

using namespace xyz_autodiff;

// ===========================================
// Static Assert Tests for Concept Compliance
// ===========================================

// Test types
using TestVector3 = Variable<3, float>;
using TestVectorRef3 = VariableRef<3, float>;

// Sum operation types
using SumOp = UnaryOperation<1, op::SumLogic<3>, TestVectorRef3>;

// Static assertions for concept compliance
static_assert(VariableConcept<SumOp>, "SumOperation should satisfy VariableConcept");
static_assert(DifferentiableVariableConcept<SumOp>, "SumOperation should satisfy DifferentiableVariableConcept");
static_assert(OperationNode<SumOp>, "SumOperation should satisfy OperationNode");

// Ensure Variable is NOT an OperationNode
static_assert(!OperationNode<TestVector3>, "Variable should NOT be OperationNode");

// ===========================================
// Test Class
// ===========================================

class SumOperationsGradientTest : public ::testing::Test {
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

TEST_F(SumOperationsGradientTest, SumGradientVerification) {
    using Logic = op::SumLogic<3>;
    test::UnaryGradientTester<Logic, 3, 1>::test_custom(
        "SumLogic", 
        50,      // num_tests
        1e-5,    // tolerance (minimum allowed for double precision)
        1e-7,    // delta
        -5.0,    // input_min
        5.0      // input_max
    );
}

// Test with different dimensions
TEST_F(SumOperationsGradientTest, SumGradientVerification5D) {
    using Logic = op::SumLogic<5>;
    test::UnaryGradientTester<Logic, 5, 1>::test_custom(
        "SumLogic5D", 
        30,      // num_tests
        1e-5,    // tolerance (minimum allowed for double precision)
        1e-7,    // delta
        -3.0,    // input_min
        3.0      // input_max
    );
}

TEST_F(SumOperationsGradientTest, SumGradientVerification2D) {
    using Logic = op::SumLogic<2>;
    test::UnaryGradientTester<Logic, 2, 1>::test_custom(
        "SumLogic2D", 
        30,      // num_tests
        1e-5,    // tolerance
        1e-7,    // delta
        -10.0,   // input_min
        10.0     // input_max
    );
}

// Edge case tests
TEST_F(SumOperationsGradientTest, SumNearZero) {
    using Logic = op::SumLogic<3>;
    test::UnaryGradientTester<Logic, 3, 1>::test_custom(
        "SumLogicNearZero", 
        30,      // num_tests
        1e-5,    // tolerance (minimum allowed for double precision)
        1e-7,    // delta
        -0.5,    // input_min (near zero)
        0.5      // input_max
    );
}

TEST_F(SumOperationsGradientTest, SumLargeRange) {
    using Logic = op::SumLogic<4>;
    test::UnaryGradientTester<Logic, 4, 1>::test_custom(
        "SumLogicLargeRange", 
        40,      // num_tests
        1e-5,    // tolerance
        1e-7,    // delta
        -100.0,  // input_min (large range)
        100.0    // input_max
    );
}

// Test large dimensions
TEST_F(SumOperationsGradientTest, SumLargeDimension) {
    using Logic = op::SumLogic<9>;
    test::UnaryGradientTester<Logic, 9, 1>::test_custom(
        "SumLogic9D", 
        20,      // num_tests (fewer for high dimension)
        1e-5,    // tolerance
        1e-7,    // delta
        -2.0,    // input_min
        2.0      // input_max
    );
}

// Test single dimension
TEST_F(SumOperationsGradientTest, SumGradientVerification1D) {
    using Logic = op::SumLogic<1>;
    test::UnaryGradientTester<Logic, 1, 1>::test_custom(
        "SumLogic1D", 
        20,      // num_tests
        1e-5,    // tolerance
        1e-7,    // delta
        -10.0,   // input_min
        10.0     // input_max
    );
}