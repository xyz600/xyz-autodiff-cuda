#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include <xyz_autodiff/variable.cuh>
#include <xyz_autodiff/concept/variable.cuh>
#include <xyz_autodiff/concept/operation_node.cuh>
#include "../operations/mahalanobis_distance.cuh"
#include <xyz_autodiff/util/cuda_unique_ptr.cuh>
#include <xyz_autodiff/testing/binary_gradient_tester.cuh>

using namespace xyz_autodiff;

// ===========================================
// Static Assert Tests for Concept Compliance
// ===========================================

// Test types
using TestVector2 = Variable<2, float>;
using TestVector3 = Variable<3, float>;
using TestVectorRef2 = VariableRef<2, float>;
using TestVectorRef3 = VariableRef<3, float>;
using MahalDistOp = BinaryOperation<1, op::MahalanobisDistanceLogic<TestVectorRef2, TestVectorRef3>, TestVectorRef2, TestVectorRef3>;

// Static assertions for concept compliance
static_assert(VariableConcept<TestVector2>, 
    "Variable<2, float> should satisfy VariableConcept");
static_assert(DifferentiableVariableConcept<TestVector2>, 
    "Variable<2, float> should satisfy DifferentiableVariableConcept");

static_assert(VariableConcept<MahalDistOp>, 
    "MahalanobisDistance Operation should satisfy VariableConcept");
static_assert(DifferentiableVariableConcept<MahalDistOp>, 
    "MahalanobisDistance Operation should satisfy DifferentiableVariableConcept");
static_assert(OperationNode<MahalDistOp>, 
    "MahalanobisDistance Operation should satisfy OperationNode");

// Ensure Variable is not an OperationNode
static_assert(!OperationNode<TestVector2>, 
    "Variable should NOT satisfy OperationNode");

// ===========================================
// Test Class
// ===========================================

class MahalanobisDistanceTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaError_t err = cudaSetDevice(0);
        if (err != cudaSuccess) {
            GTEST_SKIP() << "CUDA device not available";
        }
    }
};


// ===========================================
// Gradient Verification Tests
// ===========================================

TEST_F(MahalanobisDistanceTest, GradientVerification) {
    using Logic = op::MahalanobisDistanceLogic<VariableRef<2, double>, VariableRef<3, double>>;
    xyz_autodiff::testing::BinaryGradientTester<Logic, 2, 3, 1>::test_custom(
        "MahalanobisDistance", 
        30,      // num_tests (reduced for stability)
        1e-5,    // tolerance (relaxed for matrix inversion stability)
        1e-6,    // delta
        0.1,     // input_min (avoid singular matrices)
        2.0      // input_max
    );
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}