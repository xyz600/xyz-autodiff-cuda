#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include "../../../include/variable.cuh"
#include "../../../include/concept/variable.cuh"
#include "../../../include/concept/operation_node.cuh"
#include "../../../include/operations/unary/l1_norm_logic.cuh"
#include "../../../include/operations/unary/l2_norm_logic.cuh"
#include "../../../include/util/cuda_unique_ptr.cuh"
#include "../../utility/unary_gradient_tester.cuh"

using namespace xyz_autodiff;

// ===========================================
// Static Assert Tests for Concept Compliance
// ===========================================

// Test types
using TestVector3 = Variable<float, 3>;
using TestVectorRef3 = VariableRef<float, 3>;
using L1NormOp = UnaryOperation<1, op::L1NormLogic<3>, TestVectorRef3>;
using L2NormOp = UnaryOperation<1, op::L2NormLogic<3>, TestVectorRef3>;

// Static assertions for concept compliance
static_assert(VariableConcept<TestVector3>, 
    "Variable<float, 3> should satisfy VariableConcept");
static_assert(DifferentiableVariableConcept<TestVector3>, 
    "Variable<float, 3> should satisfy DifferentiableVariableConcept");

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
// Forward Pass Tests - L1 Norm
// ===========================================

__global__ void test_l1_norm_forward_kernel(float* result) {
    // Test vector [3, -4, 5] -> L1 norm = 3 + 4 + 5 = 12
    float data[3] = {3.0f, -4.0f, 5.0f};
    float grad[3] = {0,0,0};
    
    VariableRef<float, 3> vec(data, grad);
    
    auto l1_result = op::l1_norm(vec);
    l1_result.forward();
    
    float expected = 12.0f;
    float tolerance = 1e-6f;
    bool success = (fabsf(l1_result[0] - expected) < tolerance);
    
    *result = success ? 1.0f : 0.0f;
}

TEST_F(NormOperationsTest, L1NormForwardPass) {
    auto device_result = makeCudaUnique<float>();
    
    test_l1_norm_forward_kernel<<<1, 1>>>(device_result.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    float host_result;
    ASSERT_EQ(cudaMemcpy(&host_result, device_result.get(), sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(host_result, 1.0f);
}

// ===========================================
// Forward Pass Tests - L2 Norm
// ===========================================

__global__ void test_l2_norm_forward_kernel(float* result) {
    // Test vector [3, 4] -> L2 norm = sqrt(9 + 16) = 5
    float data[2] = {3.0f, 4.0f};
    float grad[2] = {0,0};
    
    VariableRef<float, 2> vec(data, grad);
    
    auto l2_result = op::l2_norm(vec);
    l2_result.forward();
    
    float expected = 5.0f;
    float tolerance = 1e-6f;
    bool success = (fabsf(l2_result[0] - expected) < tolerance);
    
    *result = success ? 1.0f : 0.0f;
}

TEST_F(NormOperationsTest, L2NormForwardPass) {
    auto device_result = makeCudaUnique<float>();
    
    test_l2_norm_forward_kernel<<<1, 1>>>(device_result.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    float host_result;
    ASSERT_EQ(cudaMemcpy(&host_result, device_result.get(), sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(host_result, 1.0f);
}

// ===========================================
// Gradient Verification Tests - L1 Norm
// ===========================================

TEST_F(NormOperationsTest, L1NormGradientVerification) {
    using Logic = op::L1NormLogic<3>;
    test::UnaryGradientTester<Logic, 3, 1>::test_custom(
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
    test::UnaryGradientTester<Logic, 3, 1>::test_custom(
        "L2Norm", 
        50,      // num_tests
        1e-5,    // tolerance
        1e-7,    // delta
        -2.0,    // input_min
        2.0      // input_max
    );
}

// ===========================================
// Interface Tests
// ===========================================

__global__ void test_norm_interface_kernel(float* result) {
    float data[3] = {1.0f, 2.0f, 3.0f};
    float grad[3] = {0,0,0};
    
    VariableRef<float, 3> input(data, grad);
    
    // Test all norm operations
    auto l1_op = op::l1_norm(input);
    auto l2_op = op::l2_norm(input);
    
    // Test VariableConcept interface on L1 norm
    l1_op.zero_grad();
    constexpr auto size = decltype(l1_op)::size;
    auto* l1_data = l1_op.data();
    auto* l1_grad = l1_op.grad();
    auto value = l1_op[0];
    auto grad_value = l1_op.grad(0);
    
    bool success = (size == 1) && 
                   (l1_data != nullptr) && 
                   (l1_grad != nullptr) &&
                   (grad_value == 0.0f);
    
    *result = success ? 1.0f : 0.0f;
}

TEST_F(NormOperationsTest, NormInterfaceTest) {
    auto device_result = makeCudaUnique<float>();
    
    test_norm_interface_kernel<<<1, 1>>>(device_result.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    float host_result;
    ASSERT_EQ(cudaMemcpy(&host_result, device_result.get(), sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(host_result, 1.0f);
}