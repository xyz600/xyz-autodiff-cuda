#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include "../../../include/variable.cuh"
#include "../../../include/concept/variable.cuh"
#include "../../../include/concept/operation_node.cuh"
#include "../../../include/operations/unary/broadcast_logic.cuh"
#include "../../../include/util/cuda_unique_ptr.cuh"
#include "../../utility/unary_gradient_tester.cuh"

using namespace xyz_autodiff;

// ===========================================
// Static Assert Tests for Concept Compliance
// ===========================================

// Test types
using TestScalar = Variable<float, 1>;
using TestScalarRef = VariableRef<float, 1>;
using BroadcastOp3 = UnaryOperation<3, BroadcastLogic<3>, TestScalarRef>;
using BroadcastOp5 = UnaryOperation<5, BroadcastLogic<5>, TestScalarRef>;

// Static assertions for concept compliance
static_assert(VariableConcept<TestScalar>, 
    "Variable<float, 1> should satisfy VariableConcept");
static_assert(DifferentiableVariableConcept<TestScalar>, 
    "Variable<float, 1> should satisfy DifferentiableVariableConcept");

static_assert(VariableConcept<BroadcastOp3>, 
    "Broadcast<3> Operation should satisfy VariableConcept");
static_assert(DifferentiableVariableConcept<BroadcastOp3>, 
    "Broadcast<3> Operation should satisfy DifferentiableVariableConcept");
static_assert(OperationNode<BroadcastOp3>, 
    "Broadcast<3> Operation should satisfy OperationNode");

static_assert(VariableConcept<BroadcastOp5>, 
    "Broadcast<5> Operation should satisfy VariableConcept");
static_assert(DifferentiableVariableConcept<BroadcastOp5>, 
    "Broadcast<5> Operation should satisfy DifferentiableVariableConcept");
static_assert(OperationNode<BroadcastOp5>, 
    "Broadcast<5> Operation should satisfy OperationNode");

// Ensure Variable is not an OperationNode
static_assert(!OperationNode<TestScalar>, 
    "Variable should NOT satisfy OperationNode");

// ===========================================
// Test Class
// ===========================================

class BroadcastTest : public ::testing::Test {
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

__global__ void test_broadcast_forward_kernel(float* result) {
    // Test broadcast from size 1 to size 3: input = 2.5 -> output = (2.5, 2.5, 2.5)
    float input_data[1] = {2.5f};
    float input_grad[1] = {0.0f};
    
    VariableRef<float, 1> input(input_data, input_grad);
    
    auto broadcast_result = broadcast<3>(input);
    broadcast_result.forward();
    
    // Check that all output elements equal input value
    bool success = true;
    float tolerance = 1e-6f;
    for (int i = 0; i < 3; ++i) {
        if (fabsf(broadcast_result[i] - 2.5f) > tolerance) {
            success = false;
            break;
        }
    }
    
    *result = success ? 1.0f : 0.0f;
}

TEST_F(BroadcastTest, ForwardPassSize3) {
    auto device_result = makeCudaUnique<float>();
    
    test_broadcast_forward_kernel<<<1, 1>>>(device_result.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    float host_result;
    ASSERT_EQ(cudaMemcpy(&host_result, device_result.get(), sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(host_result, 1.0f);
}

__global__ void test_broadcast_forward_size5_kernel(float* result) {
    // Test broadcast from size 1 to size 5: input = 1.5 -> output = (1.5, 1.5, 1.5, 1.5, 1.5)
    float input_data[1] = {1.5f};
    float input_grad[1] = {0.0f};
    
    VariableRef<float, 1> input(input_data, input_grad);
    
    auto broadcast_result = broadcast<5>(input);
    broadcast_result.forward();
    
    // Check that all output elements equal input value
    bool success = true;
    float tolerance = 1e-6f;
    for (int i = 0; i < 5; ++i) {
        if (fabsf(broadcast_result[i] - 1.5f) > tolerance) {
            success = false;
            break;
        }
    }
    
    *result = success ? 1.0f : 0.0f;
}

TEST_F(BroadcastTest, ForwardPassSize5) {
    auto device_result = makeCudaUnique<float>();
    
    test_broadcast_forward_size5_kernel<<<1, 1>>>(device_result.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    float host_result;
    ASSERT_EQ(cudaMemcpy(&host_result, device_result.get(), sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(host_result, 1.0f);
}

// ===========================================
// Backward Pass Tests
// ===========================================

__global__ void test_broadcast_backward_kernel(double* result) {
    // Test broadcast backward: gradient should sum from all output elements to input
    double input_data[1] = {3.0};
    double input_grad[1] = {0.0};
    
    VariableRef<double, 1> input(input_data, input_grad);
    
    auto broadcast_op = broadcast<3>(input);
    
    // Forward pass
    broadcast_op.forward();
    
    // Set different gradients for each output element
    broadcast_op.zero_grad();
    broadcast_op.add_grad(0, 1.0);  // grad = 1.0
    broadcast_op.add_grad(1, 2.0);  // grad = 2.0
    broadcast_op.add_grad(2, 3.0);  // grad = 3.0
    
    // Backward pass
    broadcast_op.backward();
    
    // Expected input gradient: sum of all output gradients = 1.0 + 2.0 + 3.0 = 6.0
    double expected_grad = 6.0;
    double tolerance = 1e-10;
    bool success = (fabs(input.grad(0) - expected_grad) < tolerance);
    
    *result = success ? 1.0 : 0.0;
}

TEST_F(BroadcastTest, BackwardPassGradientSum) {
    auto device_result = makeCudaUnique<double>();
    
    test_broadcast_backward_kernel<<<1, 1>>>(device_result.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    double host_result;
    ASSERT_EQ(cudaMemcpy(&host_result, device_result.get(), sizeof(double), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(host_result, 1.0);
}

// ===========================================
// Gradient Verification Tests
// ===========================================

TEST_F(BroadcastTest, BroadcastSize3GradientVerification) {
    using Logic = BroadcastLogic<3>;
    test::UnaryGradientTester<Logic, 1, 3>::test_custom(
        "Broadcast<3>", 
        50,      // num_tests
        1e-5,    // tolerance
        1e-7,    // delta
        -5.0,    // input_min
        5.0      // input_max
    );
}

TEST_F(BroadcastTest, BroadcastSize5GradientVerification) {
    using Logic = BroadcastLogic<5>;
    test::UnaryGradientTester<Logic, 1, 5>::test_custom(
        "Broadcast<5>", 
        50,      // num_tests
        1e-5,    // tolerance
        1e-7,    // delta
        -5.0,    // input_min
        5.0      // input_max
    );
}

// ===========================================
// Interface Tests
// ===========================================

__global__ void test_broadcast_interface_kernel(float* result) {
    float input_data[1] = {2.0f};
    float input_grad[1] = {0.0f};
    
    VariableRef<float, 1> input(input_data, input_grad);
    
    // Test broadcast operation interface
    auto broadcast_op = broadcast<4>(input);
    
    // Test VariableConcept interface
    broadcast_op.zero_grad();
    constexpr auto size = decltype(broadcast_op)::size;
    auto* data = broadcast_op.data();
    auto* grad = broadcast_op.grad();
    auto value = broadcast_op[0];
    auto grad_value = broadcast_op.grad(0);
    
    bool success = (size == 4) && 
                   (data != nullptr) && 
                   (grad != nullptr) &&
                   (grad_value == 0.0f);
    
    *result = success ? 1.0f : 0.0f;
}

TEST_F(BroadcastTest, InterfaceTest) {
    auto device_result = makeCudaUnique<float>();
    
    test_broadcast_interface_kernel<<<1, 1>>>(device_result.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    float host_result;
    ASSERT_EQ(cudaMemcpy(&host_result, device_result.get(), sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(host_result, 1.0f);
}

// ===========================================
// Use Case Tests (Mini Gaussian Splatting Pattern)
// ===========================================

__global__ void test_broadcast_opacity_pattern_kernel(float* result) {
    // Test the exact pattern used in mini Gaussian splatting: opacity broadcasting
    float opacity_value = 0.8f;
    float opacity_data[1] = {opacity_value};
    float opacity_grad[1] = {0.0f};
    
    VariableRef<float, 1> opacity(opacity_data, opacity_grad);
    
    // Use broadcast instead of manual array creation
    auto opacity_broadcast = broadcast<3>(opacity);
    opacity_broadcast.forward();
    
    // Verify all elements are the same as input
    bool success = true;
    float tolerance = 1e-6f;
    for (int i = 0; i < 3; ++i) {
        if (fabsf(opacity_broadcast[i] - opacity_value) > tolerance) {
            success = false;
            break;
        }
    }
    
    *result = success ? 1.0f : 0.0f;
}

TEST_F(BroadcastTest, OpacityBroadcastPattern) {
    auto device_result = makeCudaUnique<float>();
    
    test_broadcast_opacity_pattern_kernel<<<1, 1>>>(device_result.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    float host_result;
    ASSERT_EQ(cudaMemcpy(&host_result, device_result.get(), sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(host_result, 1.0f);
}