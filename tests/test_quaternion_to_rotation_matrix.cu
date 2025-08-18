#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include "../include/variable.cuh"
#include "../include/operations/quaternion_to_rotation_matrix_logic.cuh"
#include "../include/concept/variable.cuh"
#include "../include/concept/operation_node.cuh"
#include "../include/util/cuda_unique_ptr.cuh"

using namespace xyz_autodiff;

class QuaternionToRotationMatrixTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaError_t err = cudaSetDevice(0);
        if (err != cudaSuccess) {
            GTEST_SKIP() << "CUDA device not available";
        }
    }
};

// Static assert tests for concept compliance
using TestQuaternion = Variable<float, 4>;
using QuatToMatOp = UnaryOperation<9, QuaternionToRotationMatrixLogic<4>, TestQuaternion>;

static_assert(VariableConcept<TestQuaternion>, 
    "Quaternion Variable should satisfy VariableConcept");
static_assert(DifferentiableVariableConcept<TestQuaternion>, 
    "Quaternion Variable should satisfy DifferentiableVariableConcept");

static_assert(VariableConcept<QuatToMatOp>, 
    "QuaternionToRotationMatrix Operation should satisfy VariableConcept");
static_assert(DifferentiableVariableConcept<QuatToMatOp>, 
    "QuaternionToRotationMatrix Operation should satisfy DifferentiableVariableConcept");
static_assert(OperationNode<QuatToMatOp>, 
    "QuaternionToRotationMatrix Operation should satisfy OperationNode");

// Ensure quaternion is not an OperationNode
static_assert(!OperationNode<TestQuaternion>, 
    "Quaternion Variable should NOT satisfy OperationNode");

// Forward pass test kernel
__global__ void test_quaternion_to_rotation_matrix_forward_kernel(float* result) {
    // Identity quaternion (0, 0, 0, 1) should produce identity matrix
    float quat_data[4] = {0.0f, 0.0f, 0.0f, 1.0f};
    float quat_grad[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    
    VariableRef<float, 4> quaternion(quat_data, quat_grad);
    
    // Create operation - forward() is called automatically in helper function
    auto rotation_matrix = op::quaternion_to_rotation_matrix(quaternion);
    
    // Expected identity matrix: [1,0,0,0,1,0,0,0,1]
    float expected[9] = {1.0f, 0.0f, 0.0f,
                        0.0f, 1.0f, 0.0f,
                        0.0f, 0.0f, 1.0f};
    
    float tolerance = 1e-6f;
    bool success = true;
    
    for (int i = 0; i < 9; i++) {
        float diff = fabsf(rotation_matrix[i] - expected[i]);
        if (diff > tolerance) {
            success = false;
            break;
        }
    }
    
    *result = success ? 1.0f : 0.0f;
}

TEST_F(QuaternionToRotationMatrixTest, ForwardPassIdentityQuaternion) {
    auto device_result = makeCudaUnique<float>();
    
    test_quaternion_to_rotation_matrix_forward_kernel<<<1, 1>>>(device_result.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    float host_result;
    ASSERT_EQ(cudaMemcpy(&host_result, device_result.get(), sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(host_result, 1.0f);
}

// 90-degree rotation around Z-axis test kernel
__global__ void test_quaternion_90_degree_z_rotation_kernel(float* result) {
    // 90-degree rotation around Z-axis: (0, 0, sin(π/4), cos(π/4)) = (0, 0, √2/2, √2/2)
    float sqrt2_2 = 0.7071067811865476f; // √2/2
    float quat_data[4] = {0.0f, 0.0f, sqrt2_2, sqrt2_2};
    float quat_grad[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    
    VariableRef<float, 4> quaternion(quat_data, quat_grad);
    
    // Create operation
    auto rotation_matrix = op::quaternion_to_rotation_matrix(quaternion);
    
    // Expected 90-degree Z rotation matrix: [0,-1,0,1,0,0,0,0,1]
    float expected[9] = {0.0f, -1.0f, 0.0f,
                        1.0f,  0.0f, 0.0f,
                        0.0f,  0.0f, 1.0f};
    
    float tolerance = 1e-6f;
    bool success = true;
    
    for (int i = 0; i < 9; i++) {
        float diff = fabsf(rotation_matrix[i] - expected[i]);
        if (diff > tolerance) {
            success = false;
            break;
        }
    }
    
    *result = success ? 1.0f : 0.0f;
}

TEST_F(QuaternionToRotationMatrixTest, ForwardPass90DegreeZRotation) {
    auto device_result = makeCudaUnique<float>();
    
    test_quaternion_90_degree_z_rotation_kernel<<<1, 1>>>(device_result.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    float host_result;
    ASSERT_EQ(cudaMemcpy(&host_result, device_result.get(), sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(host_result, 1.0f);
}

// Gradient verification test kernel using numerical differentiation with double precision
__global__ void test_quaternion_gradient_verification_kernel(float* result) {
    // Use double precision for accurate gradient verification
    double quat_data[4] = {0.1, 0.0, 0.0, 0.995}; // Approximately normalized
    double quat_grad[4] = {0.0, 0.0, 0.0, 0.0};
    
    VariableRef<double, 4> quaternion(quat_data, quat_grad);
    
    // Create operation
    auto rotation_matrix = op::quaternion_to_rotation_matrix(quaternion);
    
    // Set up output gradient (gradient flows from a scalar loss)
    // Single zero_grad call propagates through computation graph
    rotation_matrix.zero_grad();
    for (int i = 0; i < 9; i++) {
        rotation_matrix.add_grad(i, 1.0); // Simple uniform gradient
    }
    
    // Compute analytical gradients
    rotation_matrix.backward();
    
    // Save analytical gradients
    double analytical_grad[4];
    for (int i = 0; i < 4; i++) {
        analytical_grad[i] = quaternion.grad(i);
    }
    
    // Reset gradients for numerical computation
    // Single zero_grad call propagates through computation graph
    rotation_matrix.zero_grad();
    for (int i = 0; i < 9; i++) {
        rotation_matrix.add_grad(i, 1.0);
    }
    
    // Compute numerical gradients with smaller step for double precision
    rotation_matrix.backward_numerical(1e-8);
    
    // Compare analytical vs numerical gradients with strict tolerance
    double tolerance = 1e-5; // Strict tolerance for double precision
    bool success = true;
    
    for (int i = 0; i < 4; i++) {
        double diff = fabs(analytical_grad[i] - quaternion.grad(i));
        if (diff > tolerance) {
            success = false;
            break;
        }
    }
    
    *result = success ? 1.0f : 0.0f;
}

TEST_F(QuaternionToRotationMatrixTest, GradientVerification) {
    auto device_result = makeCudaUnique<float>();
    
    test_quaternion_gradient_verification_kernel<<<1, 1>>>(device_result.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    float host_result;
    ASSERT_EQ(cudaMemcpy(&host_result, device_result.get(), sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(host_result, 1.0f);
}

// Interface compliance test kernel
__global__ void test_quaternion_operation_interface_kernel(float* result) {
    float quat_data[4] = {0.0f, 0.0f, 0.0f, 1.0f};
    float quat_grad[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    
    VariableRef<float, 4> quaternion(quat_data, quat_grad);
    
    // Create operation using helper function
    auto rotation_matrix = op::quaternion_to_rotation_matrix(quaternion);
    
    // Test VariableConcept interface
    rotation_matrix.zero_grad();  // zero_grad
    constexpr auto size = decltype(rotation_matrix)::size;  // size
    auto* data = rotation_matrix.data();  // data()
    auto* grad = rotation_matrix.grad();  // grad()
    auto value = rotation_matrix[0];  // operator[]
    auto grad_value = rotation_matrix.grad(0);  // grad(size_t)
    
    // Test OperationNode interface
    rotation_matrix.backward();  // backward
    rotation_matrix.backward_numerical(1e-5f);  // backward_numerical
    
    // Verify expected behavior
    bool success = (size == 9 && data != nullptr && grad != nullptr);
    
    *result = success ? 1.0f : 0.0f;
}

TEST_F(QuaternionToRotationMatrixTest, OperationInterfaceCompliance) {
    auto device_result = makeCudaUnique<float>();
    
    test_quaternion_operation_interface_kernel<<<1, 1>>>(device_result.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    float host_result;
    ASSERT_EQ(cudaMemcpy(&host_result, device_result.get(), sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(host_result, 1.0f);
}

// Operation chaining test kernel
__global__ void test_quaternion_operation_chaining_kernel(float* result) {
    // Use a non-identity quaternion to test gradient propagation
    float quat_data[4] = {0.1f, 0.2f, 0.3f, 0.926f};  // Approximately normalized
    float quat_grad[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    
    VariableRef<float, 4> quaternion(quat_data, quat_grad);
    
    // Create quaternion to rotation matrix operation
    auto rotation_matrix = op::quaternion_to_rotation_matrix(quaternion);
    
    // Test automatic backward propagation
    rotation_matrix.zero_grad();  // Should propagate to quaternion
    
    // Set gradients on multiple outputs to ensure non-zero gradient flow
    for (int i = 0; i < 9; i++) {
        rotation_matrix.add_grad(i, 1.0f);
    }
    
    // Backward should automatically propagate
    rotation_matrix.backward();
    
    // Check that gradient propagated to quaternion
    bool gradient_propagated = false;
    for (int i = 0; i < 4; i++) {
        if (fabsf(quaternion.grad(i)) > 1e-7f) {
            gradient_propagated = true;
            break;
        }
    }
    
    *result = gradient_propagated ? 1.0f : 0.0f;
}

TEST_F(QuaternionToRotationMatrixTest, OperationChaining) {
    auto device_result = makeCudaUnique<float>();
    
    test_quaternion_operation_chaining_kernel<<<1, 1>>>(device_result.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    float host_result;
    ASSERT_EQ(cudaMemcpy(&host_result, device_result.get(), sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(host_result, 1.0f);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}