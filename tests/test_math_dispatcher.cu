#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <xyz_autodiff/operations/math.cuh>
#include <xyz_autodiff/util/cuda_unique_ptr.cuh>
#include <xyz_autodiff/variable_operators.cuh>

using namespace xyz_autodiff;

// テスト用バッファ構造体
template <typename T>
struct MathTestBuffers {
    T input[3];      // 最大3個の入力値
    T output[12];    // 最大12個の出力値 (3*4)
};

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
    
    auto device_buffers = makeCudaUnique<MathTestBuffers<T>>();
    
    // 入力データをセット
    MathTestBuffers<T> host_buffers;
    host_buffers.input[0] = 1.0f;
    
    cudaMemcpy(device_buffers.get(), &host_buffers, sizeof(MathTestBuffers<T>), cudaMemcpyHostToDevice);
    
    test_math_functions_kernel<T><<<1, 1>>>(device_buffers.get()->input, device_buffers.get()->output);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    // 結果をホストにコピー
    cudaMemcpy(&host_buffers, device_buffers.get(), sizeof(MathTestBuffers<T>), cudaMemcpyDeviceToHost);
    
    // Verify results (approximately)
    EXPECT_NEAR(host_buffers.output[0], std::exp(1.0f), 1e-5f);      // exp(1)
    EXPECT_NEAR(host_buffers.output[1], std::log(2.0f), 1e-5f);      // log(2)
    EXPECT_NEAR(host_buffers.output[2], std::sin(1.0f), 1e-5f);      // sin(1)
    EXPECT_NEAR(host_buffers.output[3], std::cos(1.0f), 1e-5f);      // cos(1)
    EXPECT_NEAR(host_buffers.output[4], 1.0f, 1e-5f);                // sqrt(1)
    EXPECT_NEAR(host_buffers.output[5], 1.0f/(1.0f + std::exp(-1.0f)), 1e-5f); // sigmoid(1)
    EXPECT_NEAR(host_buffers.output[6], 1.0f, 1e-5f);                // relu(1)
    EXPECT_NEAR(host_buffers.output[7], std::tanh(1.0f), 1e-5f);     // tanh(1)
}

TEST_F(MathDispatcherTest, DoubleFunctions) {
    using T = double;
    
    auto device_buffers = makeCudaUnique<MathTestBuffers<T>>();
    
    // 入力データをセット
    MathTestBuffers<T> host_buffers;
    host_buffers.input[0] = 1.0;
    
    cudaMemcpy(device_buffers.get(), &host_buffers, sizeof(MathTestBuffers<T>), cudaMemcpyHostToDevice);
    
    test_math_functions_kernel<T><<<1, 1>>>(device_buffers.get()->input, device_buffers.get()->output);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    // 結果をホストにコピー
    cudaMemcpy(&host_buffers, device_buffers.get(), sizeof(MathTestBuffers<T>), cudaMemcpyDeviceToHost);
    
    // Verify results (approximately)
    EXPECT_NEAR(host_buffers.output[0], std::exp(1.0), 1e-10);       // exp(1)
    EXPECT_NEAR(host_buffers.output[1], std::log(2.0), 1e-10);       // log(2)
    EXPECT_NEAR(host_buffers.output[2], std::sin(1.0), 1e-10);       // sin(1)
    EXPECT_NEAR(host_buffers.output[3], std::cos(1.0), 1e-10);       // cos(1)
    EXPECT_NEAR(host_buffers.output[4], 1.0, 1e-10);                 // sqrt(1)
    EXPECT_NEAR(host_buffers.output[5], 1.0/(1.0 + std::exp(-1.0)), 1e-10); // sigmoid(1)
    EXPECT_NEAR(host_buffers.output[6], 1.0, 1e-10);                 // relu(1)
    EXPECT_NEAR(host_buffers.output[7], std::tanh(1.0), 1e-10);      // tanh(1)
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
    
    auto device_buffers = makeCudaUnique<MathTestBuffers<T>>();
    
    // 入力データをセット
    MathTestBuffers<T> host_buffers;
    host_buffers.input[0] = -1.0f;
    host_buffers.input[1] = 0.0f;
    host_buffers.input[2] = 1.0f;
    
    cudaMemcpy(device_buffers.get(), &host_buffers, sizeof(MathTestBuffers<T>), cudaMemcpyHostToDevice);
    
    test_activation_functions_kernel<T><<<1, 1>>>(device_buffers.get()->input, device_buffers.get()->output);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    // 結果をホストにコピー
    cudaMemcpy(&host_buffers, device_buffers.get(), sizeof(MathTestBuffers<T>), cudaMemcpyDeviceToHost);
    
    // Test sigmoid values
    EXPECT_NEAR(host_buffers.output[0], 1.0f/(1.0f + std::exp(1.0f)), 1e-5f); // sigmoid(-1)
    EXPECT_NEAR(host_buffers.output[4], 0.5f, 1e-5f);                          // sigmoid(0)
    EXPECT_NEAR(host_buffers.output[8], 1.0f/(1.0f + std::exp(-1.0f)), 1e-5f); // sigmoid(1)
    
    // Test ReLU values
    EXPECT_FLOAT_EQ(host_buffers.output[1], 0.0f);  // relu(-1)
    EXPECT_FLOAT_EQ(host_buffers.output[5], 0.0f);  // relu(0)
    EXPECT_FLOAT_EQ(host_buffers.output[9], 1.0f);  // relu(1)
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}