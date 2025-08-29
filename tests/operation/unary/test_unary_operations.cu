#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <xyz_autodiff/variable.cuh>
#include <xyz_autodiff/operations/unary/sigmoid_logic.cuh>
#include <xyz_autodiff/operations/unary/exp_logic.cuh>
#include <xyz_autodiff/util/cuda_unique_ptr.cuh>

using namespace xyz_autodiff;

// テスト用バッファ構造体
template <typename T>
struct UnaryTestBuffers {
    T input[6];   // 3 for data, 3 for grad
    T output[3];  // 3 output values
};

// Test kernel for sigmoid factory function
template <typename T>
__global__ void test_sigmoid_factory_kernel(T* input_data, T* output_data) {
    VariableRef<3, T> input(input_data, input_data + 3);
    
    // Test factory function from sigmoid_logic.cuh
    auto result = op::sigmoid(input);
    result.forward();
    
    for (int i = 0; i < 3; ++i) {
        output_data[i] = result[i];
    }
}

// Test kernel for exp factory function
template <typename T>
__global__ void test_exp_factory_kernel(T* input_data, T* output_data) {
    VariableRef<3, T> input(input_data, input_data + 3);
    
    // Test factory function from exp_logic.cuh
    auto result = op::exp(input);
    result.forward();
    
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
    
    auto device_buffers = makeCudaUnique<UnaryTestBuffers<T>>();
    
    // 入力データをセット
    UnaryTestBuffers<T> host_buffers;
    host_buffers.input[0] = 0.0f;  // data[0]
    host_buffers.input[1] = 1.0f;  // data[1]
    host_buffers.input[2] = -1.0f; // data[2]
    
    cudaMemcpy(device_buffers.get(), &host_buffers, sizeof(UnaryTestBuffers<T>), cudaMemcpyHostToDevice);
    
    test_sigmoid_factory_kernel<T><<<1, 1>>>(device_buffers.get()->input, device_buffers.get()->output);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    // 結果をホストにコピー
    cudaMemcpy(&host_buffers, device_buffers.get(), sizeof(UnaryTestBuffers<T>), cudaMemcpyDeviceToHost);
    
    // Verify sigmoid(0) = 0.5
    EXPECT_NEAR(host_buffers.output[0], 0.5f, 1e-5f);
    // Verify sigmoid(1) ≈ 0.731
    EXPECT_NEAR(host_buffers.output[1], 0.731f, 1e-3f);
    // Verify sigmoid(-1) ≈ 0.269
    EXPECT_NEAR(host_buffers.output[2], 0.269f, 1e-3f);
}

TEST_F(UnaryOperationsTest, ExpFactory) {
    using T = float;
    
    auto device_buffers = makeCudaUnique<UnaryTestBuffers<T>>();
    
    // 入力データをセット
    UnaryTestBuffers<T> host_buffers;
    host_buffers.input[0] = 0.0f;  // data[0]
    host_buffers.input[1] = 1.0f;  // data[1]
    host_buffers.input[2] = 2.0f;  // data[2]
    
    cudaMemcpy(device_buffers.get(), &host_buffers, sizeof(UnaryTestBuffers<T>), cudaMemcpyHostToDevice);
    
    test_exp_factory_kernel<T><<<1, 1>>>(device_buffers.get()->input, device_buffers.get()->output);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    // 結果をホストにコピー
    cudaMemcpy(&host_buffers, device_buffers.get(), sizeof(UnaryTestBuffers<T>), cudaMemcpyDeviceToHost);
    
    // Verify exp(0) = 1
    EXPECT_NEAR(host_buffers.output[0], 1.0f, 1e-5f);
    // Verify exp(1) ≈ 2.718
    EXPECT_NEAR(host_buffers.output[1], 2.718f, 1e-3f);
    // Verify exp(2) ≈ 7.389
    EXPECT_NEAR(host_buffers.output[2], 7.389f, 1e-3f);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}