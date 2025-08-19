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

// バイナリ操作テスト用統合バッファ構造体
template <typename T, std::size_t Input1Dim, std::size_t Input2Dim, std::size_t OutputDim>
struct BinaryGradientTestBuffers {
    T input1_data[Input1Dim];
    T input1_grad[Input1Dim];
    T input2_data[Input2Dim];  
    T input2_grad[Input2Dim];
    T output_grad[OutputDim];
    T analytical_grad1[Input1Dim];
    T numerical_grad1[Input1Dim];
    T analytical_grad2[Input2Dim];
    T numerical_grad2[Input2Dim];
};

// BinaryOperation用カーネル関数
template <typename LogicType, std::size_t In1Dim, std::size_t In2Dim, std::size_t OutDim>
__global__ void test_binary_gradient_kernel(
    BinaryGradientTestBuffers<double, In1Dim, In2Dim, OutDim>* buffers, double delta) {
    
    using T = double;
    
    // Variable作成
    VariableRef<T, In1Dim> input1_var(buffers->input1_data, buffers->input1_grad);
    VariableRef<T, In2Dim> input2_var(buffers->input2_data, buffers->input2_grad);
    
    LogicType logic;
    
    // 解析的勾配計算
    {
        // 入力勾配をクリア
        for (std::size_t i = 0; i < In1Dim; ++i) {
            buffers->input1_grad[i] = T(0);
        }
        for (std::size_t i = 0; i < In2Dim; ++i) {
            buffers->input2_grad[i] = T(0);
        }
        
        auto op = BinaryOperation<OutDim, LogicType, VariableRef<T, In1Dim>, VariableRef<T, In2Dim>>(
            logic, input1_var, input2_var);

        op.forward();
        
        // 上流勾配設定（数値勾配と同じ）
        op.zero_grad();
        for (std::size_t i = 0; i < OutDim; ++i) {
            op.add_grad(i, buffers->output_grad[i]);
        }
        
        op.backward();
        
        // 結果保存
        for (std::size_t i = 0; i < In1Dim; ++i) {
            buffers->analytical_grad1[i] = input1_var.grad(i);
        }
        for (std::size_t i = 0; i < In2Dim; ++i) {
            buffers->analytical_grad2[i] = input2_var.grad(i);
        }
    }
    
    // 数値勾配計算
    {
        // 入力勾配をクリア
        for (std::size_t i = 0; i < In1Dim; ++i) {
            buffers->input1_grad[i] = T(0);
        }
        for (std::size_t i = 0; i < In2Dim; ++i) {
            buffers->input2_grad[i] = T(0);
        }
        
        auto op = BinaryOperation<OutDim, LogicType, VariableRef<T, In1Dim>, VariableRef<T, In2Dim>>(
            logic, input1_var, input2_var);
        
        op.forward();
        
        // 上流勾配設定
        op.zero_grad();
        for (std::size_t i = 0; i < OutDim; ++i) {
            op.add_grad(i, buffers->output_grad[i]);
        }
        
        op.backward_numerical(delta);
        
        // 結果保存
        for (std::size_t i = 0; i < In1Dim; ++i) {
            buffers->numerical_grad1[i] = input1_var.grad(i);
        }
        for (std::size_t i = 0; i < In2Dim; ++i) {
            buffers->numerical_grad2[i] = input2_var.grad(i);
        }
    }
}

// BinaryOperation のランダムテスト
template <typename Logic, std::size_t Input1Dim, std::size_t Input2Dim, std::size_t OutputDim>
class BinaryGradientTester {
private:
    static constexpr std::size_t NUM_TESTS = 100;
    static constexpr double TOLERANCE = 1e-5;
    static constexpr double DELTA = 1e-5;
    
    // 相対誤差と絶対誤差の最小値を計算するヘルパー関数
    static double compute_error_min(double analytical, double numerical) {
        double abs_error = std::abs(analytical - numerical);
        double rel_error = std::abs(abs_error / (std::abs(analytical) + 1e-15)); // ゼロ除算回避
        return std::min(abs_error, rel_error);
    }
    
public:
    static void test(const std::string& operation_name) {
        using T = double;
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<T> dist(-2.0, 2.0);
        
        // デバイスメモリ確保（単一確保）
        auto device_buffers = makeCudaUnique<BinaryGradientTestBuffers<T, Input1Dim, Input2Dim, OutputDim>>();
        
        // NUM_TESTSのランダムテストケース
        for (int test_case = 0; test_case < NUM_TESTS; ++test_case) {
            BinaryGradientTestBuffers<T, Input1Dim, Input2Dim, OutputDim> host_buffers = {};
            
            // ランダム入力生成
            for (std::size_t i = 0; i < Input1Dim; ++i) {
                host_buffers.input1_data[i] = dist(gen);
            }
            for (std::size_t i = 0; i < Input2Dim; ++i) {
                host_buffers.input2_data[i] = dist(gen);
            }
            
            // ランダム上流勾配生成
            for (std::size_t i = 0; i < OutputDim; ++i) {
                host_buffers.output_grad[i] = dist(gen);
            }
            
            // デバイスにコピー
            ASSERT_EQ(cudaMemcpy(device_buffers.get(), &host_buffers, 
                               sizeof(BinaryGradientTestBuffers<T, Input1Dim, Input2Dim, OutputDim>), cudaMemcpyHostToDevice), cudaSuccess);
            
            // テストカーネル実行
            test_binary_gradient_kernel<Logic, Input1Dim, Input2Dim, OutputDim><<<1, 1>>>(
                device_buffers.get(), static_cast<T>(DELTA));
            
            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
            
            // 結果をホストにコピー
            ASSERT_EQ(cudaMemcpy(&host_buffers, device_buffers.get(), 
                               sizeof(BinaryGradientTestBuffers<T, Input1Dim, Input2Dim, OutputDim>), cudaMemcpyDeviceToHost), cudaSuccess);
            
            // Input1の誤差チェック
            for (std::size_t i = 0; i < Input1Dim; ++i) {
                T error_min = compute_error_min(host_buffers.analytical_grad1[i], host_buffers.numerical_grad1[i]);
                
                EXPECT_LE(error_min, TOLERANCE) 
                    << operation_name << " test case " << test_case 
                    << ", input1[" << i << "]: analytical=" << host_buffers.analytical_grad1[i]
                    << ", numerical=" << host_buffers.numerical_grad1[i]
                    << ", error_min=" << error_min;
            }
            
            // Input2の誤差チェック
            for (std::size_t i = 0; i < Input2Dim; ++i) {
                T error_min = compute_error_min(host_buffers.analytical_grad2[i], host_buffers.numerical_grad2[i]);
                
                EXPECT_LE(error_min, TOLERANCE) 
                    << operation_name << " test case " << test_case 
                    << ", input2[" << i << "]: analytical=" << host_buffers.analytical_grad2[i]
                    << ", numerical=" << host_buffers.numerical_grad2[i]
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
        auto device_buffers = makeCudaUnique<BinaryGradientTestBuffers<T, Input1Dim, Input2Dim, OutputDim>>();
        
        T max_error = 0.0;
        std::size_t max_error_test_case = 0;
        std::size_t max_error_input_index = 0;
        bool max_error_is_input1 = true;
        T max_error_analytical = 0.0;
        T max_error_numerical = 0.0;
        
        for (std::size_t test_case = 0; test_case < num_tests; ++test_case) {
            BinaryGradientTestBuffers<T, Input1Dim, Input2Dim, OutputDim> host_buffers = {};
            
            // ランダム入力生成
            for (std::size_t i = 0; i < Input1Dim; ++i) {
                host_buffers.input1_data[i] = dist(gen);
            }
            for (std::size_t i = 0; i < Input2Dim; ++i) {
                host_buffers.input2_data[i] = dist(gen);
            }
            
            // ランダム上流勾配生成
            for (std::size_t i = 0; i < OutputDim; ++i) {
                host_buffers.output_grad[i] = dist(gen);
            }
            
            // デバイスにコピー
            ASSERT_EQ(cudaMemcpy(device_buffers.get(), &host_buffers, 
                               sizeof(BinaryGradientTestBuffers<T, Input1Dim, Input2Dim, OutputDim>), cudaMemcpyHostToDevice), cudaSuccess);
            
            // テストカーネル実行
            test_binary_gradient_kernel<Logic, Input1Dim, Input2Dim, OutputDim><<<1, 1>>>(
                device_buffers.get(), static_cast<T>(delta));
            
            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
            
            // 結果をホストにコピー
            ASSERT_EQ(cudaMemcpy(&host_buffers, device_buffers.get(), 
                               sizeof(BinaryGradientTestBuffers<T, Input1Dim, Input2Dim, OutputDim>), cudaMemcpyDeviceToHost), cudaSuccess);
            
            // Input1の誤差チェック
            for (std::size_t i = 0; i < Input1Dim; ++i) {
                T error_min = compute_error_min(host_buffers.analytical_grad1[i], host_buffers.numerical_grad1[i]);
                
                // 最大誤差を記録
                if (error_min > max_error) {
                    max_error = error_min;
                    max_error_test_case = test_case;
                    max_error_input_index = i;
                    max_error_is_input1 = true;
                    max_error_analytical = host_buffers.analytical_grad1[i];
                    max_error_numerical = host_buffers.numerical_grad1[i];
                }
                
                EXPECT_LE(error_min, tolerance) 
                    << operation_name << " test case " << test_case 
                    << ", input1[" << i << "]: analytical=" << host_buffers.analytical_grad1[i]
                    << ", numerical=" << host_buffers.numerical_grad1[i]
                    << ", error_min=" << error_min;
            }
            
            // Input2の誤差チェック
            for (std::size_t i = 0; i < Input2Dim; ++i) {
                T error_min = compute_error_min(host_buffers.analytical_grad2[i], host_buffers.numerical_grad2[i]);
                
                // 最大誤差を記録
                if (error_min > max_error) {
                    max_error = error_min;
                    max_error_test_case = test_case;
                    max_error_input_index = i;
                    max_error_is_input1 = false;
                    max_error_analytical = host_buffers.analytical_grad2[i];
                    max_error_numerical = host_buffers.numerical_grad2[i];
                }
                
                EXPECT_LE(error_min, tolerance) 
                    << operation_name << " test case " << test_case 
                    << ", input2[" << i << "]: analytical=" << host_buffers.analytical_grad2[i]
                    << ", numerical=" << host_buffers.numerical_grad2[i]
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
                  << ", " << (max_error_is_input1 ? "input1" : "input2") 
                  << "[" << max_error_input_index << "]" << std::endl;
        std::cout << "Max error values: analytical=" << max_error_analytical 
                  << ", numerical=" << max_error_numerical << std::endl;
        if (max_error > tolerance) {
            std::cout << "RECOMMENDATION: Use tolerance >= " << max_error * 1.1 << " for this operation" << std::endl;
        }
        std::cout << "==========================================" << std::endl;
    }
};

// 便利なマクロ定義
#define TEST_BINARY_GRADIENT(LogicType, Input1Dim, Input2Dim, OutputDim, TestName) \
    TEST(GradientTest, TestName) { \
        xyz_autodiff::test::BinaryGradientTester<LogicType, Input1Dim, Input2Dim, OutputDim>::test(#TestName); \
    }

#define TEST_BINARY_GRADIENT_CUSTOM(LogicType, Input1Dim, Input2Dim, OutputDim, TestName, NumTests, Tolerance, Delta) \
    TEST(GradientTest, TestName) { \
        xyz_autodiff::test::BinaryGradientTester<LogicType, Input1Dim, Input2Dim, OutputDim>::test_custom(#TestName, NumTests, Tolerance, Delta); \
    }

} // namespace test
} // namespace xyz_autodiff