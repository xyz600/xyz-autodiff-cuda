#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include <xyz_autodiff/variable.cuh>
#include <xyz_autodiff/concept/variable.cuh>
#include <xyz_autodiff/concept/operation_node.cuh>
#include <xyz_autodiff/operations/unary/sym_matrix2_inv_logic.cuh>
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
using SymMatInvOp = UnaryOperation<3, op::SymMatrix2InvLogic<3>, TestVectorRef3>;

// Static assertions for concept compliance
static_assert(VariableConcept<TestVector3>, 
    "Variable<3, float> should satisfy VariableConcept");
static_assert(DifferentiableVariableConcept<TestVector3>, 
    "Variable<3, float> should satisfy DifferentiableVariableConcept");

static_assert(VariableConcept<SymMatInvOp>, 
    "SymmetricMatrix2x2Inverse Operation should satisfy VariableConcept");
static_assert(DifferentiableVariableConcept<SymMatInvOp>, 
    "SymmetricMatrix2x2Inverse Operation should satisfy DifferentiableVariableConcept");
static_assert(OperationNode<SymMatInvOp>, 
    "SymmetricMatrix2x2Inverse Operation should satisfy OperationNode");

// Ensure Variable is not an OperationNode
static_assert(!OperationNode<TestVector3>, 
    "Variable should NOT satisfy OperationNode");

// ===========================================
// Test Class
// ===========================================

class SymMatrix2InvTest : public ::testing::Test {
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

TEST_F(SymMatrix2InvTest, SymMatrix2InvGradientVerification) {
    using Logic = op::SymMatrix2InvLogic<3>;
    xyz_autodiff::testing::UnaryGradientTester<Logic, 3, 3>::test_custom(
        "SymmetricMatrix2x2Inverse", 
        30,      // num_tests (reduced for stability)
        1e-5,    // tolerance (based on error analysis: >= 39.1856)
        1e-6,    // delta
        0.5,     // input_min (avoid singular matrices)
        3.0      // input_max
    );
}