#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include "../../../include/variable.cuh"
#include "../matrix_multiplication.cuh"
#include "../symmetric_matrix.cuh"
#include "../../../include/util/cuda_unique_ptr.cuh"

using namespace xyz_autodiff;

class MatrixOperationsTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaError_t err = cudaSetDevice(0);
        if (err != cudaSuccess) {
            GTEST_SKIP() << "CUDA device not available";
        }
    }
};

// Test kernel for 3x3 matrix multiplication
__global__ void test_matrix_multiply_3x3_kernel(float* result) {
    // Test case: A = [[1,2,3],[4,5,6],[7,8,9]], B = [[9,8,7],[6,5,4],[3,2,1]]
    float a_data[9] = {1,2,3,4,5,6,7,8,9};
    float a_grad[9] = {0,0,0,0,0,0,0,0,0};
    float b_data[9] = {9,8,7,6,5,4,3,2,1};
    float b_grad[9] = {0,0,0,0,0,0,0,0,0};
    
    VariableRef<float, 9> A(a_data, a_grad);
    VariableRef<float, 9> B(b_data, b_grad);
    
    auto C = op::matrix_multiply_3x3(A, B);
    C.forward();
    
    // Expected result: C = [[30,24,18],[84,69,54],[138,114,90]]
    float expected[9] = {30,24,18,84,69,54,138,114,90};
    
    bool success = true;
    float tolerance = 1e-6f;
    
    for (int i = 0; i < 9; i++) {
        if (fabsf(C[i] - expected[i]) > tolerance) {
            success = false;
            break;
        }
    }
    
    *result = success ? 1.0f : 0.0f;
}

TEST_F(MatrixOperationsTest, MatrixMultiplication3x3Forward) {
    auto device_result = makeCudaUnique<float>();
    
    test_matrix_multiply_3x3_kernel<<<1, 1>>>(device_result.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    float host_result;
    ASSERT_EQ(cudaMemcpy(&host_result, device_result.get(), sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(host_result, 1.0f);
}

// Test kernel for matrix multiplication gradient
__global__ void test_matrix_multiply_gradient_kernel(float* result) {
    float a_data[9] = {1,2,3,4,5,6,7,8,9};
    float a_grad[9] = {0,0,0,0,0,0,0,0,0};
    float b_data[9] = {9,8,7,6,5,4,3,2,1};
    float b_grad[9] = {0,0,0,0,0,0,0,0,0};
    
    VariableRef<float, 9> A(a_data, a_grad);
    VariableRef<float, 9> B(b_data, b_grad);
    
    auto C = op::matrix_multiply_3x3(A, B);
    C.run();  // forward -> zero_grad -> add_grad(all 1.0) -> backward
    
    // Check that gradients were computed
    bool gradients_computed = false;
    for (int i = 0; i < 9; i++) {
        if (fabsf(A.grad(i)) > 1e-7f || fabsf(B.grad(i)) > 1e-7f) {
            gradients_computed = true;
            break;
        }
    }
    
    *result = gradients_computed ? 1.0f : 0.0f;
}

TEST_F(MatrixOperationsTest, MatrixMultiplicationGradient) {
    auto device_result = makeCudaUnique<float>();
    
    test_matrix_multiply_gradient_kernel<<<1, 1>>>(device_result.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    float host_result;
    ASSERT_EQ(cudaMemcpy(&host_result, device_result.get(), sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(host_result, 1.0f);
}

// Test kernel for symmetric matrix operations
__global__ void test_symmetric_matrix_kernel(float* result) {
    // Test 3-parameter symmetric matrix operations
    float cov_data[3] = {4.0f, 1.0f, 2.0f};  // [[4,1],[1,2]]
    float cov_grad[3] = {0,0,0};
    
    VariableRef<float, 3> cov_params(cov_data, cov_grad);
    
    // Convert to 2x2 matrix
    auto matrix_2x2 = op::symmetric_3param_to_matrix(cov_params);
    matrix_2x2.forward();
    
    // Expected: [4, 1, 1, 2]
    float tolerance = 1e-6f;
    bool success = (fabsf(matrix_2x2[0] - 4.0f) < tolerance &&
                   fabsf(matrix_2x2[1] - 1.0f) < tolerance &&
                   fabsf(matrix_2x2[2] - 1.0f) < tolerance &&
                   fabsf(matrix_2x2[3] - 2.0f) < tolerance);
    
    *result = success ? 1.0f : 0.0f;
}

TEST_F(MatrixOperationsTest, SymmetricMatrixConversion) {
    auto device_result = makeCudaUnique<float>();
    
    test_symmetric_matrix_kernel<<<1, 1>>>(device_result.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    float host_result;
    ASSERT_EQ(cudaMemcpy(&host_result, device_result.get(), sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(host_result, 1.0f);
}

// Test kernel for symmetric matrix inverse
__global__ void test_symmetric_matrix_inverse_kernel(float* result) {
    // Test inverse of [[2,1],[1,1]] which should be [[1,-1],[-1,2]]
    float cov_data[3] = {2.0f, 1.0f, 1.0f};
    float cov_grad[3] = {0,0,0};
    
    VariableRef<float, 3> cov_params(cov_data, cov_grad);
    
    auto inv_cov = op::symmetric_matrix_2x2_inverse(cov_params);
    inv_cov.forward();
    
    // Expected inverse: [[1,-1],[-1,2]] -> [1, -1, 2]
    float tolerance = 1e-5f;
    bool success = (fabsf(inv_cov[0] - 1.0f) < tolerance &&
                   fabsf(inv_cov[1] - (-1.0f)) < tolerance &&
                   fabsf(inv_cov[2] - 2.0f) < tolerance);
    
    *result = success ? 1.0f : 0.0f;
}

TEST_F(MatrixOperationsTest, SymmetricMatrixInverse) {
    auto device_result = makeCudaUnique<float>();
    
    test_symmetric_matrix_inverse_kernel<<<1, 1>>>(device_result.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    float host_result;
    ASSERT_EQ(cudaMemcpy(&host_result, device_result.get(), sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(host_result, 1.0f);
}

// Test kernel for gradient verification using numerical differentiation
__global__ void test_matrix_operations_gradient_verification_kernel(float* result) {
    // Use double precision for accurate gradient verification
    double cov_data[3] = {2.1, 0.5, 1.8};
    double cov_grad[3] = {0.0, 0.0, 0.0};
    
    VariableRef<double, 3> cov_params(cov_data, cov_grad);
    
    auto inv_cov = op::symmetric_matrix_2x2_inverse(cov_params);
    
    // Run analytical gradient computation
    inv_cov.run();
    
    // Save analytical gradients
    double analytical_grad[3];
    for (int i = 0; i < 3; i++) {
        analytical_grad[i] = cov_params.grad(i);
    }
    
    // Reset gradients for numerical computation
    for (int i = 0; i < 3; i++) {
        cov_grad[i] = 0.0;
    }
    
    // Run numerical gradient computation
    inv_cov.run_numerical(1e-8);
    
    // Compare analytical vs numerical gradients
    double tolerance = 1e-4;
    bool success = true;
    
    for (int i = 0; i < 3; i++) {
        double diff = fabs(analytical_grad[i] - cov_params.grad(i));
        if (diff > tolerance) {
            success = false;
            break;
        }
    }
    
    *result = success ? 1.0f : 0.0f;
}

TEST_F(MatrixOperationsTest, GradientVerification) {
    auto device_result = makeCudaUnique<float>();
    
    test_matrix_operations_gradient_verification_kernel<<<1, 1>>>(device_result.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    float host_result;
    ASSERT_EQ(cudaMemcpy(&host_result, device_result.get(), sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(host_result, 1.0f);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}