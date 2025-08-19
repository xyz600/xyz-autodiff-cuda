#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include "../../../include/variable.cuh"
#include "../../../include/concept/variable.cuh"
#include "../../../include/concept/operation_node.cuh"
#include "../../../include/operations/unary/to_rotation_matrix_logic.cuh"
#include "../../../include/operations/unary/broadcast.cuh"
#include "../../../include/util/cuda_unique_ptr.cuh"
#include "../../utility/unary_gradient_tester.cuh"

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
// Forward Pass Tests
// ===========================================

__global__ void test_quaternion_to_rotation_matrix_forward_kernel(float* result) {
    // Identity quaternion (0, 0, 0, 1) should give identity matrix
    float quat_data[4] = {0.0f, 0.0f, 0.0f, 1.0f}; // (x, y, z, w)
    float quat_grad[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    
    VariableRef<4, float> quat(quat_data, quat_grad);
    
    auto rot_matrix_op = op::quaternion_to_rotation_matrix(quat);
    rot_matrix_op.forward();
    
    // Expected: Identity matrix
    // [1, 0, 0]
    // [0, 1, 0] 
    // [0, 0, 1]
    // Flattened: [1, 0, 0, 0, 1, 0, 0, 0, 1]
    float tolerance = 1e-6f;
    bool success = (fabsf(rot_matrix_op[0] - 1.0f) < tolerance &&  // (0,0)
                   fabsf(rot_matrix_op[1] - 0.0f) < tolerance &&  // (0,1)
                   fabsf(rot_matrix_op[2] - 0.0f) < tolerance &&  // (0,2)
                   fabsf(rot_matrix_op[3] - 0.0f) < tolerance &&  // (1,0)
                   fabsf(rot_matrix_op[4] - 1.0f) < tolerance &&  // (1,1)
                   fabsf(rot_matrix_op[5] - 0.0f) < tolerance &&  // (1,2)
                   fabsf(rot_matrix_op[6] - 0.0f) < tolerance &&  // (2,0)
                   fabsf(rot_matrix_op[7] - 0.0f) < tolerance &&  // (2,1)
                   fabsf(rot_matrix_op[8] - 1.0f) < tolerance);   // (2,2)
    
    result[0] = success ? 1.0f : 0.0f;
}

__global__ void test_broadcast_forward_kernel(float* result) {
    float input_data[1] = {3.5f};
    float input_grad[1] = {0.0f};
    
    VariableRef<1, float> input(input_data, input_grad);
    
    auto broadcast_op = op::broadcast<4>(input);
    broadcast_op.forward();
    
    // Expected: [3.5, 3.5, 3.5, 3.5]
    float tolerance = 1e-6f;
    bool success = (fabsf(broadcast_op[0] - 3.5f) < tolerance &&
                   fabsf(broadcast_op[1] - 3.5f) < tolerance &&
                   fabsf(broadcast_op[2] - 3.5f) < tolerance &&
                   fabsf(broadcast_op[3] - 3.5f) < tolerance);
    
    result[0] = success ? 1.0f : 0.0f;
}

// ===========================================
// Test Cases
// ===========================================

TEST_F(SpecializedOperationsGradientTest, QuaternionToRotationMatrixForwardPass) {
    auto device_result = makeCudaUnique<float>();
    test_quaternion_to_rotation_matrix_forward_kernel<<<1, 1>>>(device_result.get());
    cudaDeviceSynchronize();
    
    float host_result;
    cudaMemcpy(&host_result, device_result.get(), sizeof(float), cudaMemcpyDeviceToHost);
    EXPECT_EQ(host_result, 1.0f);
}

TEST_F(SpecializedOperationsGradientTest, BroadcastForwardPass) {
    auto device_result = makeCudaUnique<float>();
    test_broadcast_forward_kernel<<<1, 1>>>(device_result.get());
    cudaDeviceSynchronize();
    
    float host_result;
    cudaMemcpy(&host_result, device_result.get(), sizeof(float), cudaMemcpyDeviceToHost);
    EXPECT_EQ(host_result, 1.0f);
}

// ===========================================
// Gradient Verification Tests
// ===========================================

TEST_F(SpecializedOperationsGradientTest, QuaternionToRotationMatrixGradientVerification) {
    using Logic = op::QuaternionToRotationMatrixLogic<4>;
    test::UnaryGradientTester<Logic, 4, 9>::test_custom(
        "QuaternionToRotationMatrixLogic", 
        30,      // num_tests (fewer due to complexity)
        1e-5,    // tolerance (minimum allowed for double precision)
        1e-6,    // delta
        -1.0,    // input_min
        1.0      // input_max
    );
}

__global__ void test_broadcast_1to3_gradient_kernel(double* result) {
    double input_data[1] = {2.5};
    double input_grad[1] = {0.0};
    
    VariableRef<1, double> input(input_data, input_grad);
    auto broadcast_op = op::broadcast<3>(input);
    
    // Forward pass
    broadcast_op.forward();
    
    // Set upstream gradients
    broadcast_op.zero_grad();
    broadcast_op.add_grad(0, 1.0);
    broadcast_op.add_grad(1, 2.0);
    broadcast_op.add_grad(2, 3.0);
    
    // Backward pass
    broadcast_op.backward();
    
    // For broadcast, gradients should sum: 1.0 + 2.0 + 3.0 = 6.0
    bool success = (abs(input_grad[0] - 6.0) < 1e-10);
    result[0] = success ? 1.0 : 0.0;
}

TEST_F(SpecializedOperationsGradientTest, BroadcastGradientVerification1To3) {
    // Custom gradient tester for broadcast operation since it's not a traditional logic-based operation
    // We'll test the broadcast gradient manually
    
    // Test that broadcast correctly sums gradients in backward pass
    auto device_result = makeCudaUnique<double>();
    
    test_broadcast_1to3_gradient_kernel<<<1, 1>>>(device_result.get());
    cudaDeviceSynchronize();
    
    double host_result;
    cudaMemcpy(&host_result, device_result.get(), sizeof(double), cudaMemcpyDeviceToHost);
    EXPECT_EQ(host_result, 1.0);
}

__global__ void test_broadcast_1to4_gradient_kernel(double* result) {
    double input_data[1] = {1.0};
    double input_grad[1] = {0.0};
    
    VariableRef<1, double> input(input_data, input_grad);
    auto broadcast_op = op::broadcast<4>(input);
    
    // Forward pass
    broadcast_op.forward();
    
    // Set uniform upstream gradient
    broadcast_op.zero_grad();
    for (int i = 0; i < 4; ++i) {
        broadcast_op.add_grad(i, 1.5);
    }
    
    // Backward pass
    broadcast_op.backward();
    
    // For broadcast, gradients should sum: 1.5 * 4 = 6.0
    bool success = (abs(input_grad[0] - 6.0) < 1e-10);
    result[0] = success ? 1.0 : 0.0;
}

TEST_F(SpecializedOperationsGradientTest, BroadcastGradientVerification1To4) {
    auto device_result = makeCudaUnique<double>();
    
    test_broadcast_1to4_gradient_kernel<<<1, 1>>>(device_result.get());
    cudaDeviceSynchronize();
    
    double host_result;
    cudaMemcpy(&host_result, device_result.get(), sizeof(double), cudaMemcpyDeviceToHost);
    EXPECT_EQ(host_result, 1.0);
}

// Test quaternion normalization edge case
TEST_F(SpecializedOperationsGradientTest, QuaternionToRotationMatrixNormalizedInput) {
    using Logic = op::QuaternionToRotationMatrixLogic<4>;
    
    // Test with pre-normalized quaternions (unit quaternions)
    test::UnaryGradientTester<Logic, 4, 9>::test_custom(
        "QuaternionToRotationMatrixLogicNormalized", 
        100,      // num_tests
        1e-5,    // tolerance (minimum allowed for double precision)
        1e-7,    // delta
        -0.707,  // input_min (components of unit quaternions)
        0.707    // input_max
    );
}

__global__ void test_broadcast_1to8_gradient_kernel(double* result) {
    double input_data[1] = {-2.5};
    double input_grad[1] = {0.0};
    
    VariableRef<1, double> input(input_data, input_grad);
    auto broadcast_op = op::broadcast<8>(input);
    
    // Forward pass
    broadcast_op.forward();
    
    // Check forward result
    bool forward_correct = true;
    for (int i = 0; i < 8; ++i) {
        if (abs(broadcast_op[i] - (-2.5)) > 1e-10) {
            forward_correct = false;
            break;
        }
    }
    
    // Set different upstream gradients
    broadcast_op.zero_grad();
    double total_grad = 0.0;
    for (int i = 0; i < 8; ++i) {
        double grad_val = 0.5 + i * 0.25; // [0.5, 0.75, 1.0, 1.25, ...]
        broadcast_op.add_grad(i, grad_val);
        total_grad += grad_val;
    }
    
    // Backward pass
    broadcast_op.backward();
    
    // Check gradient accumulation
    bool gradient_correct = (abs(input_grad[0] - total_grad) < 1e-10);
    
    result[0] = (forward_correct && gradient_correct) ? 1.0 : 0.0;
}

// Test broadcast with different output sizes
TEST_F(SpecializedOperationsGradientTest, BroadcastGradientVerification1To8) {
    auto device_result = makeCudaUnique<double>();
    
    test_broadcast_1to8_gradient_kernel<<<1, 1>>>(device_result.get());
    cudaDeviceSynchronize();
    
    double host_result;
    cudaMemcpy(&host_result, device_result.get(), sizeof(double), cudaMemcpyDeviceToHost);
    EXPECT_EQ(host_result, 1.0);
}

// Test quaternion rotation properties
__global__ void test_quaternion_rotation_properties_kernel(double* result) {
    // Test 90-degree rotation around Z-axis
    // Quaternion for 90° rotation around Z: (0, 0, sin(π/4), cos(π/4)) = (0, 0, 0.707, 0.707)
    double quat_data[4] = {0.0, 0.0, 0.7071067811865476, 0.7071067811865476};
    double quat_grad[4] = {0.0, 0.0, 0.0, 0.0};
    
    VariableRef<4, double> quat(quat_data, quat_grad);
    auto rot_matrix_op = op::quaternion_to_rotation_matrix(quat);
    
    // Forward pass
    rot_matrix_op.forward();
    
    // Expected rotation matrix for 90° around Z:
    // [0, -1, 0]
    // [1,  0, 0]
    // [0,  0, 1]
    double tolerance = 1e-6;
    bool success = (abs(rot_matrix_op[0] - 0.0) < tolerance &&   // (0,0)
                   abs(rot_matrix_op[1] - (-1.0)) < tolerance && // (0,1)
                   abs(rot_matrix_op[2] - 0.0) < tolerance &&   // (0,2)
                   abs(rot_matrix_op[3] - 1.0) < tolerance &&   // (1,0)
                   abs(rot_matrix_op[4] - 0.0) < tolerance &&   // (1,1)
                   abs(rot_matrix_op[5] - 0.0) < tolerance &&   // (1,2)
                   abs(rot_matrix_op[6] - 0.0) < tolerance &&   // (2,0)
                   abs(rot_matrix_op[7] - 0.0) < tolerance &&   // (2,1)
                   abs(rot_matrix_op[8] - 1.0) < tolerance);    // (2,2)
    
    result[0] = success ? 1.0 : 0.0;
}

TEST_F(SpecializedOperationsGradientTest, QuaternionRotationProperties) {
    auto device_result = makeCudaUnique<double>();
    test_quaternion_rotation_properties_kernel<<<1, 1>>>(device_result.get());
    cudaDeviceSynchronize();
    
    double host_result;
    cudaMemcpy(&host_result, device_result.get(), sizeof(double), cudaMemcpyDeviceToHost);
    EXPECT_EQ(host_result, 1.0);
}