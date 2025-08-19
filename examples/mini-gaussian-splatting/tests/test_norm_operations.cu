#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include "../../../include/variable.cuh"
#include "../norm_operations.cuh"
#include "../norm_addition.cuh"
#include "../element_wise_exp.cuh"
#include "../element_wise_multiply.cuh"
#include "../../../include/util/cuda_unique_ptr.cuh"

using namespace xyz_autodiff;

class NormOperationsTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaError_t err = cudaSetDevice(0);
        if (err != cudaSuccess) {
            GTEST_SKIP() << "CUDA device not available";
        }
    }
};

// Test kernel for L1 norm
__global__ void test_l1_norm_kernel(float* result) {
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

TEST_F(NormOperationsTest, L1NormForward) {
    auto device_result = makeCudaUnique<float>();
    
    test_l1_norm_kernel<<<1, 1>>>(device_result.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    float host_result;
    ASSERT_EQ(cudaMemcpy(&host_result, device_result.get(), sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(host_result, 1.0f);
}

// Test kernel for L2 norm
__global__ void test_l2_norm_kernel(float* result) {
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

TEST_F(NormOperationsTest, L2NormForward) {
    auto device_result = makeCudaUnique<float>();
    
    test_l2_norm_kernel<<<1, 1>>>(device_result.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    float host_result;
    ASSERT_EQ(cudaMemcpy(&host_result, device_result.get(), sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(host_result, 1.0f);
}

// Test kernel for L2 squared norm
__global__ void test_l2_squared_norm_kernel(float* result) {
    // Test vector [3, 4] -> L2 squared norm = 9 + 16 = 25
    float data[2] = {3.0f, 4.0f};
    float grad[2] = {0,0};
    
    VariableRef<float, 2> vec(data, grad);
    
    auto l2_sq_result = op::l2_squared_norm(vec);
    l2_sq_result.forward();
    
    float expected = 25.0f;
    float tolerance = 1e-6f;
    bool success = (fabsf(l2_sq_result[0] - expected) < tolerance);
    
    *result = success ? 1.0f : 0.0f;
}

TEST_F(NormOperationsTest, L2SquaredNormForward) {
    auto device_result = makeCudaUnique<float>();
    
    test_l2_squared_norm_kernel<<<1, 1>>>(device_result.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    float host_result;
    ASSERT_EQ(cudaMemcpy(&host_result, device_result.get(), sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(host_result, 1.0f);
}

// Test kernel for L1 + L2 norm combination
__global__ void test_l1_plus_l2_norm_kernel(float* result) {
    // Test vector [3, 4] -> L1 norm = 7, L2 norm = 5, L1 + L2 = 12
    float data[2] = {3.0f, 4.0f};
    float grad[2] = {0,0};
    
    VariableRef<float, 2> vec(data, grad);
    
    auto combined_result = op::l1_plus_l2_norm(vec);
    combined_result.forward();
    
    float expected = 7.0f + 5.0f;  // L1 + L2
    float tolerance = 1e-6f;
    bool success = (fabsf(combined_result[0] - expected) < tolerance);
    
    *result = success ? 1.0f : 0.0f;
}

TEST_F(NormOperationsTest, L1PlusL2NormForward) {
    auto device_result = makeCudaUnique<float>();
    
    test_l1_plus_l2_norm_kernel<<<1, 1>>>(device_result.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    float host_result;
    ASSERT_EQ(cudaMemcpy(&host_result, device_result.get(), sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(host_result, 1.0f);
}

// Test kernel for element-wise exp
__global__ void test_element_wise_exp_kernel(float* result) {
    // Test vector [0, 1, 2] -> exp result = [1, e, e^2]
    float data[3] = {0.0f, 1.0f, 2.0f};
    float grad[3] = {0,0,0};
    
    VariableRef<float, 3> vec(data, grad);
    
    auto exp_result = op::element_wise_exp(vec);
    exp_result.forward();
    
    float tolerance = 1e-5f;
    bool success = (fabsf(exp_result[0] - 1.0f) < tolerance &&
                   fabsf(exp_result[1] - expf(1.0f)) < tolerance &&
                   fabsf(exp_result[2] - expf(2.0f)) < tolerance);
    
    *result = success ? 1.0f : 0.0f;
}

TEST_F(NormOperationsTest, ElementWiseExpForward) {
    auto device_result = makeCudaUnique<float>();
    
    test_element_wise_exp_kernel<<<1, 1>>>(device_result.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    float host_result;
    ASSERT_EQ(cudaMemcpy(&host_result, device_result.get(), sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(host_result, 1.0f);
}

// Test kernel for element-wise exp(-x)
__global__ void test_element_wise_exp_neg_kernel(float* result) {
    // Test vector [0, 1, 2] -> exp(-x) result = [1, e^(-1), e^(-2)]
    float data[3] = {0.0f, 1.0f, 2.0f};
    float grad[3] = {0,0,0};
    
    VariableRef<float, 3> vec(data, grad);
    
    auto exp_neg_result = op::element_wise_exp_neg(vec);
    exp_neg_result.forward();
    
    float tolerance = 1e-5f;
    bool success = (fabsf(exp_neg_result[0] - 1.0f) < tolerance &&
                   fabsf(exp_neg_result[1] - expf(-1.0f)) < tolerance &&
                   fabsf(exp_neg_result[2] - expf(-2.0f)) < tolerance);
    
    *result = success ? 1.0f : 0.0f;
}

TEST_F(NormOperationsTest, ElementWiseExpNegForward) {
    auto device_result = makeCudaUnique<float>();
    
    test_element_wise_exp_neg_kernel<<<1, 1>>>(device_result.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    float host_result;
    ASSERT_EQ(cudaMemcpy(&host_result, device_result.get(), sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(host_result, 1.0f);
}

// Test kernel for element-wise multiplication (3 inputs)
__global__ void test_element_wise_multiply_3_kernel(float* result) {
    // Test c * d * o: [1,2] * [3,4] * [5,6] = [15, 48]
    float c_data[2] = {1.0f, 2.0f};
    float c_grad[2] = {0,0};
    float d_data[2] = {3.0f, 4.0f};
    float d_grad[2] = {0,0};
    float o_data[2] = {5.0f, 6.0f};
    float o_grad[2] = {0,0};
    
    VariableRef<float, 2> c(c_data, c_grad);
    VariableRef<float, 2> d(d_data, d_grad);
    VariableRef<float, 2> o(o_data, o_grad);
    
    auto product = op::element_wise_multiply_3(c, d, o);
    product.forward();
    
    float tolerance = 1e-6f;
    bool success = (fabsf(product[0] - 15.0f) < tolerance &&
                   fabsf(product[1] - 48.0f) < tolerance);
    
    *result = success ? 1.0f : 0.0f;
}

TEST_F(NormOperationsTest, ElementWiseMultiply3Forward) {
    auto device_result = makeCudaUnique<float>();
    
    test_element_wise_multiply_3_kernel<<<1, 1>>>(device_result.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    float host_result;
    ASSERT_EQ(cudaMemcpy(&host_result, device_result.get(), sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(host_result, 1.0f);
}

// Test kernel for scalar addition
__global__ void test_scalar_addition_kernel(float* result) {
    // Test 3.5 + 2.5 = 6.0
    float a_data[1] = {3.5f};
    float a_grad[1] = {0};
    float b_data[1] = {2.5f};
    float b_grad[1] = {0};
    
    VariableRef<float, 1> a(a_data, a_grad);
    VariableRef<float, 1> b(b_data, b_grad);
    
    auto sum = op::scalar_add(a, b);
    sum.forward();
    
    float tolerance = 1e-6f;
    bool success = (fabsf(sum[0] - 6.0f) < tolerance);
    
    *result = success ? 1.0f : 0.0f;
}

TEST_F(NormOperationsTest, ScalarAdditionForward) {
    auto device_result = makeCudaUnique<float>();
    
    test_scalar_addition_kernel<<<1, 1>>>(device_result.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    float host_result;
    ASSERT_EQ(cudaMemcpy(&host_result, device_result.get(), sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(host_result, 1.0f);
}

// Test kernel for gradient verification using numerical differentiation
__global__ void test_norm_gradient_verification_kernel(float* result) {
    // Use double precision for accurate gradient verification
    double data[3] = {1.5, -2.3, 0.8};
    double grad[3] = {0.0, 0.0, 0.0};
    
    VariableRef<double, 3> vec(data, grad);
    
    auto l1_plus_l2 = op::l1_plus_l2_norm(vec);
    
    // Run analytical gradient computation
    l1_plus_l2.run();
    
    // Save analytical gradients
    double analytical_grad[3];
    for (int i = 0; i < 3; i++) {
        analytical_grad[i] = vec.grad(i);
    }
    
    // Reset gradients for numerical computation
    for (int i = 0; i < 3; i++) {
        grad[i] = 0.0;
    }
    
    // Run numerical gradient computation
    l1_plus_l2.run_numerical(1e-8);
    
    // Compare analytical vs numerical gradients
    double tolerance = 1e-4;
    bool success = true;
    
    for (int i = 0; i < 3; i++) {
        double diff = fabs(analytical_grad[i] - vec.grad(i));
        if (diff > tolerance) {
            success = false;
            break;
        }
    }
    
    *result = success ? 1.0f : 0.0f;
}

TEST_F(NormOperationsTest, GradientVerification) {
    auto device_result = makeCudaUnique<float>();
    
    test_norm_gradient_verification_kernel<<<1, 1>>>(device_result.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    float host_result;
    ASSERT_EQ(cudaMemcpy(&host_result, device_result.get(), sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(host_result, 1.0f);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}