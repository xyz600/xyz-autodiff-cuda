#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include "../../../include/variable.cuh"
#include "../operations/covariance_generation.cuh"
#include "../operations/mahalanobis_distance.cuh"
#include "../../../include/operations/quaternion_to_rotation_matrix_logic.cuh"
#include "../../../include/util/cuda_unique_ptr.cuh"

using namespace xyz_autodiff;

class GaussianSplattingTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaError_t err = cudaSetDevice(0);
        if (err != cudaSuccess) {
            GTEST_SKIP() << "CUDA device not available";
        }
    }
};

// Test kernel for covariance matrix generation from scale and rotation
__global__ void test_covariance_generation_kernel(float* result) {
    // Test case: scale = [2, 1], rotation = 0 (no rotation)
    // Expected M = [[2, 0], [0, 1]]
    float scale_data[2] = {2.0f, 1.0f};
    float scale_grad[2] = {0,0};
    float rotation_data[1] = {0.0f};  // 0 radians
    float rotation_grad[1] = {0};
    
    VariableRef<float, 2> scale(scale_data, scale_grad);
    VariableRef<float, 1> rotation(rotation_data, rotation_grad);
    
    auto M = op::generate_covariance_matrix(scale, rotation);
    M.forward();
    
    // Expected: M = [[2*cos(0), -1*sin(0)], [2*sin(0), 1*cos(0)]] = [[2,0],[0,1]]
    float tolerance = 1e-6f;
    bool success = (fabsf(M[0] - 2.0f) < tolerance &&  // M[0,0]
                   fabsf(M[1] - 0.0f) < tolerance &&  // M[0,1]
                   fabsf(M[2] - 0.0f) < tolerance &&  // M[1,0]
                   fabsf(M[3] - 1.0f) < tolerance);   // M[1,1]
    
    *result = success ? 1.0f : 0.0f;
}

TEST_F(GaussianSplattingTest, CovarianceGenerationIdentityRotation) {
    auto device_result = makeCudaUnique<float>();
    
    test_covariance_generation_kernel<<<1, 1>>>(device_result.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    float host_result;
    ASSERT_EQ(cudaMemcpy(&host_result, device_result.get(), sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(host_result, 1.0f);
}

// Test kernel for covariance matrix generation with 90-degree rotation
__global__ void test_covariance_generation_90deg_kernel(float* result) {
    // Test case: scale = [2, 1], rotation = π/2 (90 degrees)
    // Expected M = [[2*cos(π/2), -1*sin(π/2)], [2*sin(π/2), 1*cos(π/2)]] = [[0,-1],[2,0]]
    float scale_data[2] = {2.0f, 1.0f};
    float scale_grad[2] = {0,0};
    float rotation_data[1] = {1.5707963f};  // π/2 radians
    float rotation_grad[1] = {0};
    
    VariableRef<float, 2> scale(scale_data, scale_grad);
    VariableRef<float, 1> rotation(rotation_data, rotation_grad);
    
    auto M = op::generate_covariance_matrix(scale, rotation);
    M.forward();
    
    float tolerance = 1e-5f;
    bool success = (fabsf(M[0] - 0.0f) < tolerance &&  // M[0,0]
                   fabsf(M[1] - (-1.0f)) < tolerance &&  // M[0,1]
                   fabsf(M[2] - 2.0f) < tolerance &&  // M[1,0]
                   fabsf(M[3] - 0.0f) < tolerance);   // M[1,1]
    
    *result = success ? 1.0f : 0.0f;
}

TEST_F(GaussianSplattingTest, CovarianceGeneration90DegreeRotation) {
    auto device_result = makeCudaUnique<float>();
    
    test_covariance_generation_90deg_kernel<<<1, 1>>>(device_result.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    float host_result;
    ASSERT_EQ(cudaMemcpy(&host_result, device_result.get(), sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(host_result, 1.0f);
}

// Test kernel for scale-rotation to covariance 3-parameter conversion
__global__ void test_scale_rotation_to_covariance_3param_kernel(float* result) {
    // Test case: scale = [1, 1], rotation = 0
    // M = [[1,0],[0,1]], Σ = M*M^T = [[1,0],[0,1]] -> [1, 0, 1]
    float scale_data[2] = {1.0f, 1.0f};
    float scale_grad[2] = {0,0};
    float rotation_data[1] = {0.0f};
    float rotation_grad[1] = {0};
    
    VariableRef<float, 2> scale(scale_data, scale_grad);
    VariableRef<float, 1> rotation(rotation_data, rotation_grad);
    
    auto cov_3param = op::scale_rotation_to_covariance_3param(scale, rotation);
    cov_3param.forward();
    
    float tolerance = 1e-6f;
    bool success = (fabsf(cov_3param[0] - 1.0f) < tolerance &&  // Σ[0,0]
                   fabsf(cov_3param[1] - 0.0f) < tolerance &&  // Σ[0,1]
                   fabsf(cov_3param[2] - 1.0f) < tolerance);   // Σ[1,1]
    
    *result = success ? 1.0f : 0.0f;
}

TEST_F(GaussianSplattingTest, ScaleRotationToCovariance3Param) {
    auto device_result = makeCudaUnique<float>();
    
    test_scale_rotation_to_covariance_3param_kernel<<<1, 1>>>(device_result.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    float host_result;
    ASSERT_EQ(cudaMemcpy(&host_result, device_result.get(), sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(host_result, 1.0f);
}

// Test kernel for Mahalanobis distance calculation
__global__ void test_mahalanobis_distance_kernel(float* result) {
    // Test case: diff = [1, 0], inv_cov = [1, 0, 1] (identity matrix)
    // Expected distance^2 = [1,0] * [[1,0],[0,1]] * [1,0]^T = 1
    float diff_data[2] = {1.0f, 0.0f};
    float diff_grad[2] = {0,0};
    float inv_cov_data[3] = {1.0f, 0.0f, 1.0f};  // identity matrix
    float inv_cov_grad[3] = {0,0,0};
    
    VariableRef<float, 2> diff(diff_data, diff_grad);
    VariableRef<float, 3> inv_cov(inv_cov_data, inv_cov_grad);
    
    auto distance_sq = op::mahalanobis_distance(diff, inv_cov);
    distance_sq.forward();
    
    float tolerance = 1e-6f;
    bool success = (fabsf(distance_sq[0] - 1.0f) < tolerance);
    
    *result = success ? 1.0f : 0.0f;
}

TEST_F(GaussianSplattingTest, MahalanobisDistanceIdentityMatrix) {
    auto device_result = makeCudaUnique<float>();
    
    test_mahalanobis_distance_kernel<<<1, 1>>>(device_result.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    float host_result;
    ASSERT_EQ(cudaMemcpy(&host_result, device_result.get(), sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(host_result, 1.0f);
}

// Test kernel for Mahalanobis distance with center point
__global__ void test_mahalanobis_distance_with_center_kernel(float* result) {
    // Test case: point = [2, 1], center = [1, 1], inv_cov = identity
    // diff = [1, 0], distance^2 = 1
    float point_data[2] = {2.0f, 1.0f};
    float point_grad[2] = {0,0};
    float center_data[2] = {1.0f, 1.0f};
    float center_grad[2] = {0,0};
    float inv_cov_data[3] = {1.0f, 0.0f, 1.0f};
    float inv_cov_grad[3] = {0,0,0};
    
    VariableRef<float, 2> point(point_data, point_grad);
    VariableRef<float, 2> center(center_data, center_grad);
    VariableRef<float, 3> inv_cov(inv_cov_data, inv_cov_grad);
    
    auto distance_sq = op::mahalanobis_distance_with_center(point, center, inv_cov);
    distance_sq.forward();
    
    float tolerance = 1e-6f;
    bool success = (fabsf(distance_sq[0] - 1.0f) < tolerance);
    
    *result = success ? 1.0f : 0.0f;
}

TEST_F(GaussianSplattingTest, MahalanobisDistanceWithCenter) {
    auto device_result = makeCudaUnique<float>();
    
    test_mahalanobis_distance_with_center_kernel<<<1, 1>>>(device_result.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    float host_result;
    ASSERT_EQ(cudaMemcpy(&host_result, device_result.get(), sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(host_result, 1.0f);
}

// Test kernel for quaternion to rotation matrix (reusing existing implementation)
__global__ void test_quaternion_to_rotation_matrix_integration_kernel(float* result) {
    // Test identity quaternion [0, 0, 0, 1] -> identity rotation matrix
    float quat_data[4] = {0.0f, 0.0f, 0.0f, 1.0f};
    float quat_grad[4] = {0,0,0,0};
    
    VariableRef<float, 4> quaternion(quat_data, quat_grad);
    
    auto rotation_matrix = op::quaternion_to_rotation_matrix(quaternion);
    rotation_matrix.forward();
    
    // Expected 3x3 identity matrix: [1,0,0,0,1,0,0,0,1]
    float tolerance = 1e-6f;
    bool success = (fabsf(rotation_matrix[0] - 1.0f) < tolerance &&  // [0,0]
                   fabsf(rotation_matrix[1] - 0.0f) < tolerance &&  // [0,1]
                   fabsf(rotation_matrix[2] - 0.0f) < tolerance &&  // [0,2]
                   fabsf(rotation_matrix[3] - 0.0f) < tolerance &&  // [1,0]
                   fabsf(rotation_matrix[4] - 1.0f) < tolerance &&  // [1,1]
                   fabsf(rotation_matrix[5] - 0.0f) < tolerance &&  // [1,2]
                   fabsf(rotation_matrix[6] - 0.0f) < tolerance &&  // [2,0]
                   fabsf(rotation_matrix[7] - 0.0f) < tolerance &&  // [2,1]
                   fabsf(rotation_matrix[8] - 1.0f) < tolerance);   // [2,2]
    
    *result = success ? 1.0f : 0.0f;
}

TEST_F(GaussianSplattingTest, QuaternionToRotationMatrixIntegration) {
    auto device_result = makeCudaUnique<float>();
    
    test_quaternion_to_rotation_matrix_integration_kernel<<<1, 1>>>(device_result.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    float host_result;
    ASSERT_EQ(cudaMemcpy(&host_result, device_result.get(), sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(host_result, 1.0f);
}

// Test kernel for gradient verification of complete Gaussian splatting pipeline
__global__ void test_gaussian_splatting_gradient_verification_kernel(float* result) {
    // Use double precision for accurate gradient verification
    double scale_data[2] = {1.2, 0.8};
    double scale_grad[2] = {0.0, 0.0};
    double rotation_data[1] = {0.3};  // small rotation
    double rotation_grad[1] = {0.0};
    
    VariableRef<double, 2> scale(scale_data, scale_grad);
    VariableRef<double, 1> rotation(rotation_data, rotation_grad);
    
    // Generate covariance -> invert -> use in Mahalanobis distance
    auto cov_3param = op::scale_rotation_to_covariance_3param(scale, rotation);
    
    // Run analytical gradient computation
    cov_3param.run();
    
    // Save analytical gradients
    double analytical_scale_grad[2];
    double analytical_rotation_grad[1];
    for (int i = 0; i < 2; i++) {
        analytical_scale_grad[i] = scale.grad(i);
    }
    analytical_rotation_grad[0] = rotation.grad(0);
    
    // Reset gradients for numerical computation
    for (int i = 0; i < 2; i++) {
        scale_grad[i] = 0.0;
    }
    rotation_grad[0] = 0.0;
    
    // Run numerical gradient computation
    cov_3param.run_numerical(1e-8);
    
    // Compare analytical vs numerical gradients
    double tolerance = 1e-4;
    bool success = true;
    
    for (int i = 0; i < 2; i++) {
        double diff = fabs(analytical_scale_grad[i] - scale.grad(i));
        if (diff > tolerance) {
            success = false;
            break;
        }
    }
    
    if (success) {
        double diff = fabs(analytical_rotation_grad[0] - rotation.grad(0));
        if (diff > tolerance) {
            success = false;
        }
    }
    
    *result = success ? 1.0f : 0.0f;
}

TEST_F(GaussianSplattingTest, GradientVerification) {
    auto device_result = makeCudaUnique<float>();
    
    test_gaussian_splatting_gradient_verification_kernel<<<1, 1>>>(device_result.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    float host_result;
    ASSERT_EQ(cudaMemcpy(&host_result, device_result.get(), sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(host_result, 1.0f);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}