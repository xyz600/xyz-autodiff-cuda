#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include "../include/variable.cuh"
#include "../include/util/cuda_unique_ptr.cuh"
#include "concept/variable.cuh"

using namespace xyz_autodiff;

// テスト用バッファ構造体（固定サイズ版）
template <typename T>
struct VariableTestBuffers {
    T data[10];   // 最大4要素までサポート
    T grad[10];   // 最大4要素までサポート
    T output[20]; // for storing results (2 * 10)
};

// Variable テスト用のCUDAカーネル
template <typename T, std::size_t N>
__global__ void test_variable_kernel(T* data, T* grad, T* output) {
    // VariableRef作成 (外部バッファへの参照)
    VariableRef<T, N> var(data, grad);
    
    // データアクセステスト
    for (std::size_t i = 0; i < N; ++i) {
        var[i] = static_cast<T>(i + 1);  // 1, 2, 3, ...
    }
    
    // 勾配テスト
    var.zero_grad();
    for (std::size_t i = 0; i < N; ++i) {
        var.add_grad(i, static_cast<T>(i * 2));  // 0, 2, 4, ...
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
__global__ void test_variable_operations_kernel(T* data, T* grad, T* output) {
    VariableRef<T, N> var(data, grad);
    
    // zero_gradテスト
    var.zero_grad();
    
    // 結果を保存
    for (std::size_t i = 0; i < N; ++i) {
        output[i] = var.grad(i);
    }
}

// Variable (自己バッファ) テスト用のCUDAカーネル
template <typename T, std::size_t N>
__global__ void test_variable_self_buffer_kernel(T* output) {
    // Variable作成 (自己バッファ)
    Variable<T, N> var;
    
    // データ設定
    var.zero_grad();
    for (std::size_t i = 0; i < N; ++i) {
        var[i] = static_cast<T>(i + 10);  // 10, 11, 12, ...
        var.add_grad(i, static_cast<T>(i * 3));  // 0, 3, 6, ...
    }
    
    // 結果をoutputに保存
    for (std::size_t i = 0; i < N; ++i) {
        output[i] = var[i];           // データ値
        output[N + i] = var.grad(i);  // 勾配値
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
    
    // デバイスメモリ確保 (cuda_unique_ptr使用、単一確保)
    auto device_buffers = makeCudaUnique<VariableTestBuffers<T>>();
    
    ASSERT_NE(device_buffers, nullptr);
    
    // カーネル実行
    test_variable_kernel<T, N><<<1, 1>>>(device_buffers.get()->data, device_buffers.get()->grad, device_buffers.get()->output);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    // 結果をホストにコピー
    VariableTestBuffers<T> host_buffers;
    ASSERT_EQ(cudaMemcpy(&host_buffers, device_buffers.get(), sizeof(VariableTestBuffers<T>), cudaMemcpyDeviceToHost), cudaSuccess);
    
    // 結果検証
    for (std::size_t i = 0; i < N; ++i) {
        EXPECT_FLOAT_EQ(host_buffers.output[i], static_cast<T>(i + 1));        // データ値
        EXPECT_FLOAT_EQ(host_buffers.output[N + i], static_cast<T>(i * 2));    // 勾配値
    }
    
    // メモリは自動解放される
}

TEST_F(VariableTest, ZeroGradOperation) {
    constexpr std::size_t N = 3;
    using T = float;
    
    // ホストメモリ
    std::vector<T> host_data(N, 0);
    std::vector<T> host_grad(N, 1.0f);  // 初期値1.0
    std::vector<T> host_output(N, 0);
    
    // デバイスメモリ確保 (cuda_unique_ptr使用、単一確保)
    auto device_buffers = makeCudaUnique<VariableTestBuffers<T>>();
    
    ASSERT_NE(device_buffers, nullptr);
    
    // ホストバッファを初期化
    VariableTestBuffers<T> host_buffers = {};
    for (std::size_t i = 0; i < N; ++i) {
        host_buffers.grad[i] = host_grad[i];  // 初期値1.0
    }
    
    // データをデバイスにコピー
    ASSERT_EQ(cudaMemcpy(device_buffers.get(), &host_buffers, sizeof(VariableTestBuffers<T>), cudaMemcpyHostToDevice), cudaSuccess);
    
    // カーネル実行
    test_variable_operations_kernel<T, N><<<1, 1>>>(device_buffers.get()->data, device_buffers.get()->grad, device_buffers.get()->output);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    // 結果をホストにコピー
    ASSERT_EQ(cudaMemcpy(&host_buffers, device_buffers.get(), sizeof(VariableTestBuffers<T>), cudaMemcpyDeviceToHost), cudaSuccess);
    
    // zero_gradの結果を検証（すべて0になっているはず）
    for (std::size_t i = 0; i < N; ++i) {
        EXPECT_FLOAT_EQ(host_buffers.output[i], 0.0f);
    }
    
    // メモリは自動解放される
}

TEST_F(VariableTest, SelfBufferVariableTest) {
    using T = float;
    constexpr std::size_t N = 3;
    
    auto device_buffers = makeCudaUnique<VariableTestBuffers<T>>();
    
    test_variable_self_buffer_kernel<T, N><<<1, 1>>>(device_buffers.get()->output);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    VariableTestBuffers<T> host_buffers;
    cudaMemcpy(&host_buffers, device_buffers.get(), sizeof(VariableTestBuffers<T>), cudaMemcpyDeviceToHost);
    
    // データ値をチェック
    for (std::size_t i = 0; i < N; ++i) {
        EXPECT_FLOAT_EQ(host_buffers.output[i], static_cast<T>(i + 10));
    }
    
    // 勾配値をチェック
    for (std::size_t i = 0; i < N; ++i) {
        EXPECT_FLOAT_EQ(host_buffers.output[N + i], static_cast<T>(i * 3));
    }
}

TEST_F(VariableTest, ConceptCheck) {
    // Concept チェック - Variable と VariableRef 両方
    static_assert(xyz_autodiff::VariableConcept<Variable<float, 4>>);
    static_assert(xyz_autodiff::DifferentiableVariableConcept<Variable<float, 4>>);
    static_assert(xyz_autodiff::VariableConcept<VariableRef<float, 4>>);
    static_assert(xyz_autodiff::DifferentiableVariableConcept<VariableRef<float, 4>>);
    
    // サイズチェック
    EXPECT_EQ((xyz_autodiff::Variable<float, 4>::size), 4);
    EXPECT_EQ((xyz_autodiff::Variable<double, 10>::size), 10);
    EXPECT_EQ((xyz_autodiff::VariableRef<float, 4>::size), 4);
    EXPECT_EQ((xyz_autodiff::VariableRef<double, 10>::size), 10);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}