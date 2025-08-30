#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include <xyz_autodiff/variable.cuh>
#include <xyz_autodiff/concept/variable.cuh>
#include <xyz_autodiff/concept/operation_node.cuh>
#include <xyz_autodiff/operations/unary/to_rotation_matrix_logic.cuh>
#include <xyz_autodiff/operations/unary/broadcast.cuh>
#include <xyz_autodiff/util/cuda_unique_ptr.cuh>
#include <xyz_autodiff/variable_operators.cuh>
#include <xyz_autodiff/testing/unary_gradient_tester.cuh>

using namespace xyz_autodiff;

// ===========================================
// Static Assert Tests for Concept Compliance
// ===========================================

// Test types
using TestVector4 = Variable<4, float>;
using TestVectorRef4 = VariableRef<4, float>;
using TestVector1 = Variable<1, float>;
using TestVectorRef1 = VariableRef<1, float>;

// Specialized operation types
using QuatToRotOp = UnaryOperation<9, op::QuaternionToRotationMatrixLogic<4>, TestVectorRef4>;
using BroadcastOp = op::BroadcastOperator<TestVectorRef1, 4>; // Broadcast from 1 to 4

// Static assertions for concept compliance
static_assert(VariableConcept<QuatToRotOp>, "QuaternionToRotationMatrixOperation should satisfy VariableConcept");
static_assert(DifferentiableVariableConcept<QuatToRotOp>, "QuaternionToRotationMatrixOperation should satisfy DifferentiableVariableConcept");
static_assert(OperationNode<QuatToRotOp>, "QuaternionToRotationMatrixOperation should satisfy OperationNode");

static_assert(VariableConcept<BroadcastOp>, "BroadcastOperation should satisfy VariableConcept");
static_assert(DifferentiableVariableConcept<BroadcastOp>, "BroadcastOperation should satisfy DifferentiableVariableConcept");
static_assert(OperationNode<BroadcastOp>, "BroadcastOperation should satisfy OperationNode");

// Ensure Variable is NOT an OperationNode
static_assert(!OperationNode<TestVector4>, "Variable should NOT be OperationNode");

// ===========================================
// Test Class
// ===========================================

class SpecializedOperationsGradientTest : public ::testing::Test {
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

TEST_F(SpecializedOperationsGradientTest, QuaternionToRotationMatrixGradientVerification) {
    using Logic = op::QuaternionToRotationMatrixLogic<4>;
    xyz_autodiff::testing::UnaryGradientTester<Logic, 4, 9>::test_custom(
        "QuaternionToRotationMatrixLogic", 
        30,      // num_tests (fewer due to complexity)
        1e-5,    // tolerance (minimum allowed for double precision)
        1e-6,    // delta
        -1.0,    // input_min
        1.0      // input_max
    );
}


// Test quaternion normalization edge case
TEST_F(SpecializedOperationsGradientTest, QuaternionToRotationMatrixNormalizedInput) {
    using Logic = op::QuaternionToRotationMatrixLogic<4>;
    
    // Test with pre-normalized quaternions (unit quaternions)
    xyz_autodiff::testing::UnaryGradientTester<Logic, 4, 9>::test_custom(
        "QuaternionToRotationMatrixLogicNormalized", 
        100,      // num_tests
        1e-5,    // tolerance (minimum allowed for double precision)
        1e-7,    // delta
        -0.707,  // input_min (components of unit quaternions)
        0.707    // input_max
    );
}
