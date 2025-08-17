#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include "../include/variable.cuh"
// #include "../include/concept/variable.cuh"  // CUDA compiler concept limitations

using namespace xyz_autodiff;

// Variable テスト用のCUDAカーネル
template <typename T, std::size_t N>
__global__ void test_variable_kernel(T* data, T* grad, T* output) {
    // Variable作成
    Variable<T, N> var(data, grad);
    
    // データアクセステスト
    for (std::size_t i = 0; i < N; ++i) {
        var[i] = static_cast<T>(i + 1);  // 1, 2, 3, ...
    }
    
    // 勾配テスト
    for (std::size_t i = 0; i < N; ++i) {
        var.grad(i) = static_cast<T>(i * 2);  // 0, 2, 4, ...
    }
    
    // アクセサテスト
    T* data_ptr = var.data();
    T* grad_ptr = var.grad();
    
    // 結果をoutputに保存（検証用）
    for (std::size_t i = 0; i < N; ++i) {
        output[i] = data_ptr[i];           // データ値
        output[N + i] = grad_ptr[i];       // 勾配値
    }
}

template <typename T, std::size_t N>
__global__ void test_variable_operations_kernel(T* data, T* grad, T* grad_values, T* output) {
    Variable<T, N> var(data, grad);
    
    // zero_gradテスト
    var.zero_grad();
    
    // accumulate_gradテスト
    var.accumulate_grad(grad_values);
    
    // 結果を保存
    for (std::size_t i = 0; i < N; ++i) {
        output[i] = var.grad(i);
    }
}

class VariableTest : public ::testing::Test {
protected:
    void SetUp() override {
        // CUDA初期化チェック
        cudaError_t err = cudaSetDevice(0);
        if (err != cudaSuccess) {
            GTEST_SKIP() << "CUDA device not available";
        }
    }
};

TEST_F(VariableTest, BasicConstruction) {
    constexpr std::size_t N = 4;
    using T = float;
    
    // ホストメモリ
    std::vector<T> host_data(N, 0);
    std::vector<T> host_grad(N, 0);
    std::vector<T> host_output(2 * N, 0);
    
    // デバイスメモリ確保
    T* device_data;
    T* device_grad;
    T* device_output;
    
    ASSERT_EQ(cudaMalloc(&device_data, N * sizeof(T)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&device_grad, N * sizeof(T)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&device_output, 2 * N * sizeof(T)), cudaSuccess);
    
    // カーネル実行
    test_variable_kernel<T, N><<<1, 1>>>(device_data, device_grad, device_output);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    // 結果をホストにコピー
    ASSERT_EQ(cudaMemcpy(host_output.data(), device_output, 2 * N * sizeof(T), cudaMemcpyDeviceToHost), cudaSuccess);
    
    // 結果検証
    for (std::size_t i = 0; i < N; ++i) {
        EXPECT_FLOAT_EQ(host_output[i], static_cast<T>(i + 1));        // データ値
        EXPECT_FLOAT_EQ(host_output[N + i], static_cast<T>(i * 2));    // 勾配値
    }
    
    // メモリ解放
    cudaFree(device_data);
    cudaFree(device_grad);
    cudaFree(device_output);
}

TEST_F(VariableTest, GradientOperations) {
    constexpr std::size_t N = 3;
    using T = float;
    
    // ホストメモリ
    std::vector<T> host_data(N, 0);
    std::vector<T> host_grad(N, 1.0f);  // 初期値1.0
    std::vector<T> host_grad_values = {2.0f, 3.0f, 4.0f};
    std::vector<T> host_output(N, 0);
    
    // デバイスメモリ確保
    T* device_data;
    T* device_grad;
    T* device_grad_values;
    T* device_output;
    
    ASSERT_EQ(cudaMalloc(&device_data, N * sizeof(T)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&device_grad, N * sizeof(T)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&device_grad_values, N * sizeof(T)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&device_output, N * sizeof(T)), cudaSuccess);
    
    // データをデバイスにコピー
    ASSERT_EQ(cudaMemcpy(device_grad, host_grad.data(), N * sizeof(T), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(device_grad_values, host_grad_values.data(), N * sizeof(T), cudaMemcpyHostToDevice), cudaSuccess);
    
    // カーネル実行
    test_variable_operations_kernel<T, N><<<1, 1>>>(device_data, device_grad, device_grad_values, device_output);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    // 結果をホストにコピー
    ASSERT_EQ(cudaMemcpy(host_output.data(), device_output, N * sizeof(T), cudaMemcpyDeviceToHost), cudaSuccess);
    
    // zero_grad → accumulate_gradの結果を検証
    for (std::size_t i = 0; i < N; ++i) {
        EXPECT_FLOAT_EQ(host_output[i], host_grad_values[i]);
    }
    
    // メモリ解放
    cudaFree(device_data);
    cudaFree(device_grad);
    cudaFree(device_grad_values);
    cudaFree(device_output);
}

TEST_F(VariableTest, ConceptCheck) {
    // コンパイル時概念チェック (CUDA compiler limitations)
    // static_assert(concept::Variable<Variable<float, 4>>);
    // static_assert(concept::DifferentiableVariable<Variable<float, 4>>);
    
    // サイズチェック
    EXPECT_EQ((xyz_autodiff::Variable<float, 4>::size), 4);
    EXPECT_EQ((xyz_autodiff::Variable<double, 10>::size), 10);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}