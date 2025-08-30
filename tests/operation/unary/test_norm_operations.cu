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
using L1NormOp = UnaryOperation<1, op::L1NormLogic<3>, TestVectorRef3>;
using L2NormOp = UnaryOperation<1, op::L2NormLogic<3>, TestVectorRef3>;

// Static assertions for concept compliance
static_assert(VariableConcept<TestVector3>, 
    "Variable<3, float> should satisfy VariableConcept");
static_assert(DifferentiableVariableConcept<TestVector3>, 
    "Variable<3, float> should satisfy DifferentiableVariableConcept");

static_assert(VariableConcept<L1NormOp>, 
    "L1Norm Operation should satisfy VariableConcept");
static_assert(DifferentiableVariableConcept<L1NormOp>, 
    "L1Norm Operation should satisfy DifferentiableVariableConcept");
static_assert(OperationNode<L1NormOp>, 
    "L1Norm Operation should satisfy OperationNode");

static_assert(VariableConcept<L2NormOp>, 
    "L2Norm Operation should satisfy VariableConcept");
static_assert(DifferentiableVariableConcept<L2NormOp>, 
    "L2Norm Operation should satisfy DifferentiableVariableConcept");
static_assert(OperationNode<L2NormOp>, 
    "L2Norm Operation should satisfy OperationNode");

// Ensure Variable is not an OperationNode
static_assert(!OperationNode<TestVector3>, 
    "Variable should NOT satisfy OperationNode");

// ===========================================
// Test Class
// ===========================================

class NormOperationsTest : public ::testing::Test {
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
// Gradient Verification Tests - L1 Norm
// ===========================================

TEST_F(NormOperationsTest, L1NormGradientVerification) {
    using Logic = op::L1NormLogic<3>;
    xyz_autodiff::testing::UnaryGradientTester<Logic, 3, 1>::test_custom(
        "L1Norm", 
        50,      // num_tests
        1e-5,    // tolerance (L1 norm has non-smooth gradients at zero)
        1e-7,    // delta
        -2.0,    // input_min
        2.0      // input_max
    );
}

// ===========================================
// Gradient Verification Tests - L2 Norm
// ===========================================

TEST_F(NormOperationsTest, L2NormGradientVerification) {
    using Logic = op::L2NormLogic<3>;
    xyz_autodiff::testing::UnaryGradientTester<Logic, 3, 1>::test_custom(
        "L2Norm", 
        50,      // num_tests
        1e-5,    // tolerance
        1e-7,    // delta
        -2.0,    // input_min
        2.0      // input_max
    );
}
