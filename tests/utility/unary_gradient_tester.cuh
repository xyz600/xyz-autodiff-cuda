#pragma once

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <random>
#include <vector>
#include <cmath>
#include <algorithm>
#include "../../include/variable.cuh"
#include "../../include/operations/operation.cuh"
#include "../../include/util/cuda_unique_ptr.cuh"

namespace xyz_autodiff {
namespace test {

// ユナリ操作テスト用統合バッファ構造体
template <typename T, std::size_t InputDim, std::size_t OutputDim>
struct UnaryGradientTestBuffers {
    T input_data[InputDim];
    T input_grad[InputDim];
    T output_grad[OutputDim];
    T analytical_grad[InputDim];
    T numerical_grad[InputDim];
};

// 相対誤差と絶対誤差の最小値を計算するヘルパー関数
template <typename T>
__host__ T compute_error_min(T analytical, T numerical) {
    T abs_error = std::abs(analytical - numerical);
    T rel_error = std::abs(abs_error / (std::abs(analytical) + T(1e-15))); // ゼロ除算回避
    return std::min(abs_error, rel_error);
}

// UnaryOperation用カーネル関数
template <typename LogicType, std::size_t InDim, std::size_t OutDim>
__global__ void test_unary_gradient_kernel(
    UnaryGradientTestBuffers<double, InDim, OutDim>* buffers, double delta) {
    
    using T = double;
    
    // Variable作成
    VariableRef<T, InDim> input_var(buffers->input_data, buffers->input_grad);
    
    LogicType logic;
    
    // 解析的勾配計算
    {
        // 入力勾配をクリア
        for (std::size_t i = 0; i < InDim; ++i) {
            buffers->input_grad[i] = T(0);
        }
        
        auto op = UnaryOperation<OutDim, LogicType, VariableRef<T, InDim>>(logic, input_var);
        op.forward();
        
        // 上流勾配設定
        op.zero_grad();
        for (std::size_t i = 0; i < OutDim; ++i) {
            op.add_grad(i, buffers->output_grad[i]);
        }
        
        op.backward();
        
        // 結果保存
        for (std::size_t i = 0; i < InDim; ++i) {
            buffers->analytical_grad[i] = input_var.grad(i);
        }
    }
    
    // 数値勾配計算
    {
        // 入力勾配をクリア
        for (std::size_t i = 0; i < InDim; ++i) {
            buffers->input_grad[i] = T(0);
        }
        
        auto op = UnaryOperation<OutDim, LogicType, VariableRef<T, InDim>>(logic, input_var);
        op.forward();
        
        // 上流勾配設定
        op.zero_grad();
        for (std::size_t i = 0; i < OutDim; ++i) {
            op.add_grad(i, buffers->output_grad[i]);
        }
        
        op.backward_numerical(delta);
        
        // 結果保存
        for (std::size_t i = 0; i < InDim; ++i) {
            buffers->numerical_grad[i] = input_var.grad(i);
        }
    }
}

// UnaryOperation のランダムテスト
template <typename Logic, std::size_t InputDim, std::size_t OutputDim>
class UnaryGradientTester {
private:
    static constexpr std::size_t NUM_TESTS = 100;
    static constexpr double TOLERANCE = 1e-5;
    static constexpr double DELTA = 1e-5;
    
public:
    static void test(const std::string& operation_name) {
        using T = double;
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<T> dist(-2.0, 2.0);
        
        // デバイスメモリ確保（単一確保）
        auto device_buffers = makeCudaUnique<UnaryGradientTestBuffers<T, InputDim, OutputDim>>();
        ASSERT_NE(device_buffers, nullptr);
        
        // NUM_TESTSのランダムテストケース
        for (int test_case = 0; test_case < NUM_TESTS; ++test_case) {
            UnaryGradientTestBuffers<T, InputDim, OutputDim> host_buffers = {};
            
            // ランダム入力生成
            for (std::size_t i = 0; i < InputDim; ++i) {
                host_buffers.input_data[i] = dist(gen);
            }
            
            // ランダム上流勾配生成
            for (std::size_t i = 0; i < OutputDim; ++i) {
                host_buffers.output_grad[i] = dist(gen);
            }
            
            // デバイスにコピー
            ASSERT_EQ(cudaMemcpy(device_buffers.get(), &host_buffers, 
                               sizeof(UnaryGradientTestBuffers<T, InputDim, OutputDim>), cudaMemcpyHostToDevice), cudaSuccess);
            
            // テストカーネル実行
            test_unary_gradient_kernel<Logic, InputDim, OutputDim><<<1, 1>>>(
                device_buffers.get(), static_cast<T>(DELTA));
            
            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
            
            // 結果をホストにコピー
            ASSERT_EQ(cudaMemcpy(&host_buffers, device_buffers.get(), 
                               sizeof(UnaryGradientTestBuffers<T, InputDim, OutputDim>), cudaMemcpyDeviceToHost), cudaSuccess);
            
            // 誤差チェック
            for (std::size_t i = 0; i < InputDim; ++i) {
                T error_min = compute_error_min(host_buffers.analytical_grad[i], host_buffers.numerical_grad[i]);
                
                EXPECT_LE(error_min, TOLERANCE) 
                    << operation_name << " test case " << test_case 
                    << ", input[" << i << "]: analytical=" << host_buffers.analytical_grad[i]
                    << ", numerical=" << host_buffers.numerical_grad[i]
                    << ", error_min=" << error_min;
            }
        }
    }
    
    // カスタム設定でのテスト
    static void test_custom(const std::string& operation_name, 
                           std::size_t num_tests, 
                           double tolerance, 
                           double delta,
                           double input_min = -2.0,
                           double input_max = 2.0) {
        using T = double;
        
        // 許容誤差の制約チェック（禁止）
        if (tolerance < 0.0 || tolerance > 1e-5) {
            FAIL() << "FORBIDDEN: Using tolerance " << tolerance 
                   << " which is smaller than 1e-5. Minimum tolerance for double precision tests is 1e-5.";
        }
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<T> dist(input_min, input_max);
        
        // デバイスメモリ確保（単一確保）
        auto device_buffers = makeCudaUnique<UnaryGradientTestBuffers<T, InputDim, OutputDim>>();
        ASSERT_NE(device_buffers, nullptr);
        
        T max_error = 0.0;
        std::size_t max_error_test_case = 0;
        std::size_t max_error_input_index = 0;
        T max_error_analytical = 0.0;
        T max_error_numerical = 0.0;
        
        for (std::size_t test_case = 0; test_case < num_tests; ++test_case) {
            UnaryGradientTestBuffers<T, InputDim, OutputDim> host_buffers = {};
            
            // ランダム入力生成
            for (std::size_t i = 0; i < InputDim; ++i) {
                host_buffers.input_data[i] = dist(gen);
            }
            
            // ランダム上流勾配生成
            for (std::size_t i = 0; i < OutputDim; ++i) {
                host_buffers.output_grad[i] = dist(gen);
            }
            
            // デバイスにコピー
            ASSERT_EQ(cudaMemcpy(device_buffers.get(), &host_buffers, 
                               sizeof(UnaryGradientTestBuffers<T, InputDim, OutputDim>), cudaMemcpyHostToDevice), cudaSuccess);
            
            // テストカーネル実行
            test_unary_gradient_kernel<Logic, InputDim, OutputDim><<<1, 1>>>(
                device_buffers.get(), static_cast<T>(delta));
            
            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
            
            // 結果をホストにコピー
            ASSERT_EQ(cudaMemcpy(&host_buffers, device_buffers.get(), 
                               sizeof(UnaryGradientTestBuffers<T, InputDim, OutputDim>), cudaMemcpyDeviceToHost), cudaSuccess);
            
            // 誤差チェック
            for (std::size_t i = 0; i < InputDim; ++i) {
                T error_min = compute_error_min(host_buffers.analytical_grad[i], host_buffers.numerical_grad[i]);
                
                // 最大誤差を記録
                if (error_min > max_error) {
                    max_error = error_min;
                    max_error_test_case = test_case;
                    max_error_input_index = i;
                    max_error_analytical = host_buffers.analytical_grad[i];
                    max_error_numerical = host_buffers.numerical_grad[i];
                }
                
                EXPECT_LE(error_min, tolerance) 
                    << operation_name << " test case " << test_case 
                    << ", input[" << i << "]: analytical=" << host_buffers.analytical_grad[i]
                    << ", numerical=" << host_buffers.numerical_grad[i]
                    << ", error_min=" << error_min;
            }
        }
        
        // 最大誤差を出力
        std::cout << "=== GRADIENT TEST SUMMARY for " << operation_name << " ===" << std::endl;
        std::cout << "Number of tests: " << num_tests << std::endl;
        std::cout << "Tolerance: " << tolerance << std::endl;
        std::cout << "Delta: " << delta << std::endl;
        std::cout << "Maximum error: " << max_error << std::endl;
        std::cout << "Max error location: test case " << max_error_test_case 
                  << ", input[" << max_error_input_index << "]" << std::endl;
        std::cout << "Max error values: analytical=" << max_error_analytical 
                  << ", numerical=" << max_error_numerical << std::endl;
        if (max_error > tolerance) {
            std::cout << "RECOMMENDATION: Use tolerance >= " << max_error * 1.1 << " for this operation" << std::endl;
        }
        std::cout << "=========================================" << std::endl;
    }
};

// 便利なマクロ定義
#define TEST_UNARY_GRADIENT(LogicType, InputDim, OutputDim, TestName) \
    TEST(GradientTest, TestName) { \
        xyz_autodiff::test::UnaryGradientTester<LogicType, InputDim, OutputDim>::test(#TestName); \
    }

#define TEST_UNARY_GRADIENT_CUSTOM(LogicType, InputDim, OutputDim, TestName, NumTests, Tolerance, Delta) \
    TEST(GradientTest, TestName) { \
        xyz_autodiff::test::UnaryGradientTester<LogicType, InputDim, OutputDim>::test_custom(#TestName, NumTests, Tolerance, Delta); \
    }

} // namespace test
} // namespace xyz_autodiff