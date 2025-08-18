#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "../include/variable.cuh"
#include "../include/operations/sigmoid_logic.cuh"
#include "../include/operations/exp_logic.cuh"
#include "../include/util/cuda_unique_ptr.cuh"

using namespace xyz_autodiff;

// Test kernel for sigmoid factory function
template <typename T>
__global__ void test_sigmoid_factory_kernel(T* input_data, T* output_data) {
    VariableRef<T, 3> input(input_data, input_data + 3);
    
    // Test factory function from sigmoid_logic.cuh
    auto result = sigmoid(input);
    
    for (int i = 0; i < 3; ++i) {
        output_data[i] = result[i];
    }
}

// Test kernel for exp factory function
template <typename T>
__global__ void test_exp_factory_kernel(T* input_data, T* output_data) {
    VariableRef<T, 3> input(input_data, input_data + 3);
    
    // Test factory function from exp_logic.cuh
    auto result = exp(input);
    
    for (int i = 0; i < 3; ++i) {
        output_data[i] = result[i];
    }
}

class UnaryOperationsTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaError_t err = cudaSetDevice(0);
        if (err != cudaSuccess) {
            GTEST_SKIP() << "CUDA device not available";
        }
    }
};

TEST_F(UnaryOperationsTest, SigmoidFactory) {
    using T = float;
    
    T host_input[3] = {0.0f, 1.0f, -1.0f};
    auto device_input = makeCudaUniqueArray<T>(6);  // 3 for data, 3 for grad
    auto device_output = makeCudaUniqueArray<T>(3);
    
    cudaMemcpy(device_input.get(), host_input, 3 * sizeof(T), cudaMemcpyHostToDevice);
    
    test_sigmoid_factory_kernel<T><<<1, 1>>>(device_input.get(), device_output.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    T host_output[3];
    cudaMemcpy(host_output, device_output.get(), 3 * sizeof(T), cudaMemcpyDeviceToHost);
    
    // Verify sigmoid(0) = 0.5
    EXPECT_NEAR(host_output[0], 0.5f, 1e-5f);
    // Verify sigmoid(1) ≈ 0.731
    EXPECT_NEAR(host_output[1], 0.731f, 1e-3f);
    // Verify sigmoid(-1) ≈ 0.269
    EXPECT_NEAR(host_output[2], 0.269f, 1e-3f);
}

TEST_F(UnaryOperationsTest, ExpFactory) {
    using T = float;
    
    T host_input[3] = {0.0f, 1.0f, 2.0f};
    auto device_input = makeCudaUniqueArray<T>(6);  // 3 for data, 3 for grad
    auto device_output = makeCudaUniqueArray<T>(3);
    
    cudaMemcpy(device_input.get(), host_input, 3 * sizeof(T), cudaMemcpyHostToDevice);
    
    test_exp_factory_kernel<T><<<1, 1>>>(device_input.get(), device_output.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    T host_output[3];
    cudaMemcpy(host_output, device_output.get(), 3 * sizeof(T), cudaMemcpyDeviceToHost);
    
    // Verify exp(0) = 1
    EXPECT_NEAR(host_output[0], 1.0f, 1e-5f);
    // Verify exp(1) ≈ 2.718
    EXPECT_NEAR(host_output[1], 2.718f, 1e-3f);
    // Verify exp(2) ≈ 7.389
    EXPECT_NEAR(host_output[2], 7.389f, 1e-3f);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}