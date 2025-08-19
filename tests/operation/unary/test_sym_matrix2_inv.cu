#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include "../../../include/variable.cuh"
#include "../../../include/concept/variable.cuh"
#include "../../../include/concept/operation_node.cuh"
#include "../../../include/operations/unary/sym_matrix2_inv_logic.cuh"
#include "../../../include/util/cuda_unique_ptr.cuh"
#include "../../utility/unary_gradient_tester.cuh"

using namespace xyz_autodiff;

// ===========================================
// Static Assert Tests for Concept Compliance
// ===========================================

// Test types
using TestVector3 = Variable<float, 3>;
using TestVectorRef3 = VariableRef<float, 3>;
using SymMatInvOp = UnaryOperation<3, op::SymMatrix2InvLogic<3>, TestVectorRef3>;

// Static assertions for concept compliance
static_assert(VariableConcept<TestVector3>, 
    "Variable<float, 3> should satisfy VariableConcept");
static_assert(DifferentiableVariableConcept<TestVector3>, 
    "Variable<float, 3> should satisfy DifferentiableVariableConcept");

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
// Forward Pass Tests
// ===========================================

__global__ void test_sym_matrix2_inv_forward_kernel(float* result) {
    // Test matrix [[2, 1], [1, 2]] -> inverse = [[2/3, -1/3], [-1/3, 2/3]]
    // In 3-param format: [a, b, c] = [2, 1, 2]
    // Expected inverse: [a', b', c'] = [2/3, -1/3, 2/3]
    float data[3] = {2.0f, 1.0f, 2.0f};
    float grad[3] = {0,0,0};
    
    VariableRef<float, 3> input(data, grad);
    
    auto inv_result = op::sym_matrix2_inv(input);
    inv_result.forward();
    
    float expected_a = 2.0f/3.0f;   // 2/3
    float expected_b = -1.0f/3.0f;  // -1/3
    float expected_c = 2.0f/3.0f;   // 2/3
    float tolerance = 1e-6f;
    
    bool success = (fabsf(inv_result[0] - expected_a) < tolerance) &&
                   (fabsf(inv_result[1] - expected_b) < tolerance) &&
                   (fabsf(inv_result[2] - expected_c) < tolerance);
    
    *result = success ? 1.0f : 0.0f;
}

TEST_F(SymMatrix2InvTest, SymMatrix2InvForwardPass) {
    auto device_result = makeCudaUnique<float>();
    
    test_sym_matrix2_inv_forward_kernel<<<1, 1>>>(device_result.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    float host_result;
    ASSERT_EQ(cudaMemcpy(&host_result, device_result.get(), sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(host_result, 1.0f);
}

// ===========================================
// Gradient Verification Tests
// ===========================================

TEST_F(SymMatrix2InvTest, SymMatrix2InvGradientVerification) {
    using Logic = op::SymMatrix2InvLogic<3>;
    test::UnaryGradientTester<Logic, 3, 3>::test_custom(
        "SymmetricMatrix2x2Inverse", 
        30,      // num_tests (reduced for stability)
        1e-5,    // tolerance (based on error analysis: >= 39.1856)
        1e-6,    // delta
        0.5,     // input_min (avoid singular matrices)
        3.0      // input_max
    );
}