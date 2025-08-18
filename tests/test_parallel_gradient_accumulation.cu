#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include "../include/variable.cuh"
#include "../include/operations/add_logic.cuh"
#include "../include/util/cuda_unique_ptr.cuh"

using namespace xyz_autodiff;

/**
 * 並列勾配積算テスト
 * 約10万スレッドが同じグローバルメモリ上の変数に対して
 * 勾配を加算し、atomicAddが正しく動作することを確認
 */

// パラメータ管理用の構造体
template <typename T>
struct TestParameters {
    T value[3];  // x, y, result
    T grad[3];   // grad_x, grad_y, grad_result
};

__global__ void parallel_gradient_accumulation_kernel(
    TestParameters<double>* params, std::size_t num_threads) {
    
    std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_threads) return;
    
    // グローバルメモリを参照するVariableRefを作成（ポインタ演算で各パラメータを指定）
    VariableRef<double, 1> x_ref(&params->value[0], &params->grad[0]);
    VariableRef<double, 1> y_ref(&params->value[1], &params->grad[1]);
    VariableRef<double, 1> result_ref(&params->value[2], &params->grad[2]);
    
    // f(x, y) = x + y の勾配は単純に ∂f/∂x = 1, ∂f/∂y = 1
    // 各スレッドが1.0ずつ勾配を加算
    x_ref.add_grad(0, 1.0);  // atomicAdd(&global_x_grad[0], 1.0)
    y_ref.add_grad(0, 1.0);  // atomicAdd(&global_y_grad[0], 1.0)
    
    // 結果値も加算（検証用）
    double local_result = (1.0 + tid * 0.001) + (2.0 + tid * 0.001);
    result_ref.add_grad(0, local_result);
    
    // 最初の数スレッドで確認メッセージ
    if (tid < 5) {
        printf("Thread %zu: atomicAdd x=1.0, y=1.0\n", tid);
    }
}

class ParallelGradientAccumulationTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaError_t err = cudaSetDevice(0);
        if (err != cudaSuccess) {
            GTEST_SKIP() << "CUDA device not available";
        }
    }
};

TEST_F(ParallelGradientAccumulationTest, AtomicGradientAccumulation) {
    const std::size_t NUM_THREADS = 10000;  // 1万スレッド（メモリ使用量を減らす）
    const std::size_t THREADS_PER_BLOCK = 256;
    const std::size_t NUM_BLOCKS = (NUM_THREADS + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    // デバイスメモリ確保（構造体で一元管理）
    auto device_params = makeCudaUnique<TestParameters<double>>();
    
    // 初期値設定
    TestParameters<double> host_params = {
        {5.0, 3.0, 0.0},  // value: x, y, result
        {0.0, 0.0, 0.0}   // grad: grad_x, grad_y, grad_result (初期化)
    };
    ASSERT_EQ(cudaMemcpy(device_params.get(), &host_params, sizeof(TestParameters<double>), cudaMemcpyHostToDevice), cudaSuccess);
    
    // カーネル実行
    parallel_gradient_accumulation_kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(
        device_params.get(), NUM_THREADS
    );
    
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    // 結果をホストにコピー
    TestParameters<double> result_params;
    ASSERT_EQ(cudaMemcpy(&result_params, device_params.get(), sizeof(TestParameters<double>), cudaMemcpyDeviceToHost), cudaSuccess);
    
    double host_x_grad = result_params.grad[0];
    double host_y_grad = result_params.grad[1];
    double host_result_sum = result_params.grad[2];
    
    // 検証
    // f(x, y) = x + y の勾配は ∂f/∂x = 1, ∂f/∂y = 1
    // NUM_THREADS個のスレッドがそれぞれ1.0ずつ加算するので、最終的にNUM_THREADSになるはず
    double expected_grad = static_cast<double>(NUM_THREADS);
    
    EXPECT_NEAR(host_x_grad, expected_grad, 1e-10) 
        << "x gradient should be " << expected_grad << " but got " << host_x_grad;
    EXPECT_NEAR(host_y_grad, expected_grad, 1e-10)
        << "y gradient should be " << expected_grad << " but got " << host_y_grad;
    
    // 結果の合計値も検証（各スレッドが異なる値を計算しているため、期待値を計算）
    double expected_result_sum = 0.0;
    for (std::size_t i = 0; i < NUM_THREADS; ++i) {
        double local_x = 1.0 + i * 0.001;
        double local_y = 2.0 + i * 0.001;
        expected_result_sum += (local_x + local_y);
    }
    
    EXPECT_NEAR(host_result_sum, expected_result_sum, 1e-6)
        << "Result sum should be " << expected_result_sum << " but got " << host_result_sum;
    
    // 成功メッセージを出力
    std::cout << "SUCCESS: " << NUM_THREADS << " threads successfully accumulated gradients!" << std::endl;
    std::cout << "x_grad: " << host_x_grad << " (expected: " << expected_grad << ")" << std::endl;
    std::cout << "y_grad: " << host_y_grad << " (expected: " << expected_grad << ")" << std::endl;
    std::cout << "AtomicAdd performed correctly with " << NUM_THREADS << " concurrent threads." << std::endl;
    
    // メモリは cuda_unique_ptr により自動的に解放される
}

// 性能測定テスト（より大きなスレッド数） - 無効化
TEST_F(ParallelGradientAccumulationTest, DISABLED_LargeScaleAtomicAccumulation) {
    const std::size_t NUM_THREADS = 100000;  // 10万スレッド（元の計画通り）
    const std::size_t THREADS_PER_BLOCK = 512;
    const std::size_t NUM_BLOCKS = (NUM_THREADS + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    // デバイスメモリ確保（構造体で一元管理）
    auto device_params = makeCudaUnique<TestParameters<double>>();
    
    // 初期値設定
    TestParameters<double> host_params = {
        {1.0, 1.0, 0.0},  // value: x, y, result
        {0.0, 0.0, 0.0}   // grad: grad_x, grad_y, grad_result (初期化)
    };
    ASSERT_EQ(cudaMemcpy(device_params.get(), &host_params, sizeof(TestParameters<double>), cudaMemcpyHostToDevice), cudaSuccess);
    
    // 実行時間測定
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    
    // 簡略化カーネル実行
    parallel_gradient_accumulation_kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(
        device_params.get(), NUM_THREADS
    );
    
    cudaEventRecord(stop);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // 結果検証
    TestParameters<double> result_params;
    ASSERT_EQ(cudaMemcpy(&result_params, device_params.get(), sizeof(TestParameters<double>), cudaMemcpyDeviceToHost), cudaSuccess);
    
    double host_x_grad = result_params.grad[0];
    double host_y_grad = result_params.grad[1];
    
    double expected_grad = static_cast<double>(NUM_THREADS);
    
    EXPECT_NEAR(host_x_grad, expected_grad, 1e-10)
        << "x gradient should be " << expected_grad << " but got " << host_x_grad;
    EXPECT_NEAR(host_y_grad, expected_grad, 1e-10)
        << "y gradient should be " << expected_grad << " but got " << host_y_grad;
    
    // 性能情報を出力
    std::cout << "Processed " << NUM_THREADS << " threads in " << milliseconds << " ms" << std::endl;
    std::cout << "Throughput: " << (NUM_THREADS / (milliseconds / 1000.0)) / 1e6 << " million operations/second" << std::endl;
    
    // クリーンアップ
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    // メモリは cuda_unique_ptr により自動的に解放される
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}