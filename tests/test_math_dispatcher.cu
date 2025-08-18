#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "../include/operations/math.cuh"
#include "../include/util/cuda_unique_ptr.cuh"

using namespace xyz_autodiff;

// Test kernel for math functions with float
template <typename T>
__global__ void test_math_functions_kernel(T* input, T* output) {
    T x = input[0];
    
    // Test various math functions
    output[0] = math::exp(x);
    output[1] = math::log(x + T{1});  // log(x+1) to avoid log(0)
    output[2] = math::sin(x);
    output[3] = math::cos(x);
    output[4] = math::sqrt(math::abs(x));
    output[5] = math::sigmoid(x);
    output[6] = math::relu(x);
    output[7] = math::tanh(x);
}

class MathDispatcherTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaError_t err = cudaSetDevice(0);
        if (err != cudaSuccess) {
            GTEST_SKIP() << "CUDA device not available";
        }
    }
};

TEST_F(MathDispatcherTest, FloatFunctions) {
    using T = float;
    
    T host_input = 1.0f;
    T host_output[8];
    
    auto device_input = makeCudaUniqueArray<T>(1);
    auto device_output = makeCudaUniqueArray<T>(8);
    
    cudaMemcpy(device_input.get(), &host_input, sizeof(T), cudaMemcpyHostToDevice);
    
    test_math_functions_kernel<T><<<1, 1>>>(device_input.get(), device_output.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    cudaMemcpy(host_output, device_output.get(), 8 * sizeof(T), cudaMemcpyDeviceToHost);
    
    // Verify results (approximately)
    EXPECT_NEAR(host_output[0], std::exp(1.0f), 1e-5f);      // exp(1)
    EXPECT_NEAR(host_output[1], std::log(2.0f), 1e-5f);      // log(2)
    EXPECT_NEAR(host_output[2], std::sin(1.0f), 1e-5f);      // sin(1)
    EXPECT_NEAR(host_output[3], std::cos(1.0f), 1e-5f);      // cos(1)
    EXPECT_NEAR(host_output[4], 1.0f, 1e-5f);                // sqrt(1)
    EXPECT_NEAR(host_output[5], 1.0f/(1.0f + std::exp(-1.0f)), 1e-5f); // sigmoid(1)
    EXPECT_NEAR(host_output[6], 1.0f, 1e-5f);                // relu(1)
    EXPECT_NEAR(host_output[7], std::tanh(1.0f), 1e-5f);     // tanh(1)
}

TEST_F(MathDispatcherTest, DoubleFunctions) {
    using T = double;
    
    T host_input = 1.0;
    T host_output[8];
    
    auto device_input = makeCudaUniqueArray<T>(1);
    auto device_output = makeCudaUniqueArray<T>(8);
    
    cudaMemcpy(device_input.get(), &host_input, sizeof(T), cudaMemcpyHostToDevice);
    
    test_math_functions_kernel<T><<<1, 1>>>(device_input.get(), device_output.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    cudaMemcpy(host_output, device_output.get(), 8 * sizeof(T), cudaMemcpyDeviceToHost);
    
    // Verify results (approximately)
    EXPECT_NEAR(host_output[0], std::exp(1.0), 1e-10);       // exp(1)
    EXPECT_NEAR(host_output[1], std::log(2.0), 1e-10);       // log(2)
    EXPECT_NEAR(host_output[2], std::sin(1.0), 1e-10);       // sin(1)
    EXPECT_NEAR(host_output[3], std::cos(1.0), 1e-10);       // cos(1)
    EXPECT_NEAR(host_output[4], 1.0, 1e-10);                 // sqrt(1)
    EXPECT_NEAR(host_output[5], 1.0/(1.0 + std::exp(-1.0)), 1e-10); // sigmoid(1)
    EXPECT_NEAR(host_output[6], 1.0, 1e-10);                 // relu(1)
    EXPECT_NEAR(host_output[7], std::tanh(1.0), 1e-10);      // tanh(1)
}

// Kernel for testing activation functions
template <typename T>
__global__ void test_activation_functions_kernel(T* input, T* output) {
    for (int i = 0; i < 3; ++i) {
        T x = input[i];
        output[i * 4 + 0] = math::sigmoid(x);
        output[i * 4 + 1] = math::relu(x);
        output[i * 4 + 2] = math::tanh(x);
        output[i * 4 + 3] = math::softplus(x);
    }
}

TEST_F(MathDispatcherTest, ActivationFunctions) {
    using T = float;
    
    T host_input[3] = {-1.0f, 0.0f, 1.0f};
    T host_output[3 * 4]; // 4 activation functions for each input
    
    auto device_input = makeCudaUniqueArray<T>(3);
    auto device_output = makeCudaUniqueArray<T>(3 * 4);
    
    cudaMemcpy(device_input.get(), host_input, 3 * sizeof(T), cudaMemcpyHostToDevice);
    
    test_activation_functions_kernel<T><<<1, 1>>>(device_input.get(), device_output.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    cudaMemcpy(host_output, device_output.get(), 3 * 4 * sizeof(T), cudaMemcpyDeviceToHost);
    
    // Test sigmoid values
    EXPECT_NEAR(host_output[0], 1.0f/(1.0f + std::exp(1.0f)), 1e-5f); // sigmoid(-1)
    EXPECT_NEAR(host_output[4], 0.5f, 1e-5f);                          // sigmoid(0)
    EXPECT_NEAR(host_output[8], 1.0f/(1.0f + std::exp(-1.0f)), 1e-5f); // sigmoid(1)
    
    // Test ReLU values
    EXPECT_FLOAT_EQ(host_output[1], 0.0f);  // relu(-1)
    EXPECT_FLOAT_EQ(host_output[5], 0.0f);  // relu(0)
    EXPECT_FLOAT_EQ(host_output[9], 1.0f);  // relu(1)
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}