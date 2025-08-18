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
    double* input_data, double* input_grad, double* output_grad,
    double* analytical_grad, double* numerical_grad, double delta) {
    
    using T = double;
    
    // Variable作成
    VariableRef<T, InDim> input_var(input_data, input_grad);
    
    LogicType logic;
    
    // 解析的勾配計算
    {
        auto op = UnaryOperation<OutDim, LogicType, VariableRef<T, InDim>>(logic, input_var);
        op.forward();
        
        // 上流勾配設定
        op.zero_grad();
        for (std::size_t i = 0; i < OutDim; ++i) {
            op.add_grad(i, output_grad[i]);
        }
        
        input_var.zero_grad();
        op.backward();
        
        // 結果保存
        for (std::size_t i = 0; i < InDim; ++i) {
            analytical_grad[i] = input_var.grad(i);
        }
    }
    
    // 数値勾配計算
    {
        auto op = UnaryOperation<OutDim, LogicType, VariableRef<T, InDim>>(logic, input_var);
        op.forward();
        
        // 上流勾配設定
        op.zero_grad();
        for (std::size_t i = 0; i < OutDim; ++i) {
            op.add_grad(i, output_grad[i]);
        }
        
        input_var.zero_grad();
        op.backward_numerical(delta);
        
        // 結果保存
        for (std::size_t i = 0; i < InDim; ++i) {
            numerical_grad[i] = input_var.grad(i);
        }
    }
}

// UnaryOperation のランダムテスト
template <typename Logic, std::size_t InputDim, std::size_t OutputDim>
class UnaryGradientTester {
private:
    static constexpr std::size_t NUM_TESTS = 100;
    static constexpr double TOLERANCE = 1e-5;
    static constexpr double DELTA = 1e-7;
    
public:
    static void test(const std::string& operation_name) {
        using T = double;
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<T> dist(-2.0, 2.0);
        
        // ホストメモリ
        std::vector<T> host_input_data(InputDim);
        std::vector<T> host_input_grad(InputDim);
        std::vector<T> host_output_grad(OutputDim);
        std::vector<T> host_analytical_grad(InputDim);
        std::vector<T> host_numerical_grad(InputDim);
        
        // デバイスメモリ確保
        auto device_input_data = makeCudaUniqueArray<T>(InputDim);
        auto device_input_grad = makeCudaUniqueArray<T>(InputDim);
        auto device_output_grad = makeCudaUniqueArray<T>(OutputDim);
        auto device_analytical_grad = makeCudaUniqueArray<T>(InputDim);
        auto device_numerical_grad = makeCudaUniqueArray<T>(InputDim);
        
        ASSERT_NE(device_input_data, nullptr);
        ASSERT_NE(device_input_grad, nullptr);
        ASSERT_NE(device_output_grad, nullptr);
        ASSERT_NE(device_analytical_grad, nullptr);
        ASSERT_NE(device_numerical_grad, nullptr);
        
        // NUM_TESTSのランダムテストケース
        for (int test_case = 0; test_case < NUM_TESTS; ++test_case) {
            // ランダム入力生成
            for (std::size_t i = 0; i < InputDim; ++i) {
                host_input_data[i] = dist(gen);
            }
            
            // ランダム上流勾配生成
            for (std::size_t i = 0; i < OutputDim; ++i) {
                host_output_grad[i] = dist(gen);
            }
            
            // デバイスにコピー
            ASSERT_EQ(cudaMemcpy(device_input_data.get(), host_input_data.data(), 
                               InputDim * sizeof(T), cudaMemcpyHostToDevice), cudaSuccess);
            ASSERT_EQ(cudaMemcpy(device_output_grad.get(), host_output_grad.data(), 
                               OutputDim * sizeof(T), cudaMemcpyHostToDevice), cudaSuccess);
            
            // テストカーネル実行
            test_unary_gradient_kernel<Logic, InputDim, OutputDim><<<1, 1>>>(
                device_input_data.get(), device_input_grad.get(),
                device_output_grad.get(), device_analytical_grad.get(),
                device_numerical_grad.get(), static_cast<T>(DELTA));
            
            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
            
            // 結果をホストにコピー
            ASSERT_EQ(cudaMemcpy(host_analytical_grad.data(), device_analytical_grad.get(), 
                               InputDim * sizeof(T), cudaMemcpyDeviceToHost), cudaSuccess);
            ASSERT_EQ(cudaMemcpy(host_numerical_grad.data(), device_numerical_grad.get(), 
                               InputDim * sizeof(T), cudaMemcpyDeviceToHost), cudaSuccess);
            
            // 誤差チェック
            for (std::size_t i = 0; i < InputDim; ++i) {
                T error_min = compute_error_min(host_analytical_grad[i], host_numerical_grad[i]);
                
                EXPECT_LE(error_min, TOLERANCE) 
                    << operation_name << " test case " << test_case 
                    << ", input[" << i << "]: analytical=" << host_analytical_grad[i]
                    << ", numerical=" << host_numerical_grad[i]
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
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<T> dist(input_min, input_max);
        
        // ホストメモリ
        std::vector<T> host_input_data(InputDim);
        std::vector<T> host_input_grad(InputDim);
        std::vector<T> host_output_grad(OutputDim);
        std::vector<T> host_analytical_grad(InputDim);
        std::vector<T> host_numerical_grad(InputDim);
        
        // デバイスメモリ確保
        auto device_input_data = makeCudaUniqueArray<T>(InputDim);
        auto device_input_grad = makeCudaUniqueArray<T>(InputDim);
        auto device_output_grad = makeCudaUniqueArray<T>(OutputDim);
        auto device_analytical_grad = makeCudaUniqueArray<T>(InputDim);
        auto device_numerical_grad = makeCudaUniqueArray<T>(InputDim);
        
        ASSERT_NE(device_input_data, nullptr);
        ASSERT_NE(device_input_grad, nullptr);
        ASSERT_NE(device_output_grad, nullptr);
        ASSERT_NE(device_analytical_grad, nullptr);
        ASSERT_NE(device_numerical_grad, nullptr);
        
        for (std::size_t test_case = 0; test_case < num_tests; ++test_case) {
            // ランダム入力生成
            for (std::size_t i = 0; i < InputDim; ++i) {
                host_input_data[i] = dist(gen);
            }
            
            // ランダム上流勾配生成
            for (std::size_t i = 0; i < OutputDim; ++i) {
                host_output_grad[i] = dist(gen);
            }
            
            // デバイスにコピー
            ASSERT_EQ(cudaMemcpy(device_input_data.get(), host_input_data.data(), 
                               InputDim * sizeof(T), cudaMemcpyHostToDevice), cudaSuccess);
            ASSERT_EQ(cudaMemcpy(device_output_grad.get(), host_output_grad.data(), 
                               OutputDim * sizeof(T), cudaMemcpyHostToDevice), cudaSuccess);
            
            // テストカーネル実行
            test_unary_gradient_kernel<Logic, InputDim, OutputDim><<<1, 1>>>(
                device_input_data.get(), device_input_grad.get(),
                device_output_grad.get(), device_analytical_grad.get(),
                device_numerical_grad.get(), static_cast<T>(delta));
            
            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
            
            // 結果をホストにコピー
            ASSERT_EQ(cudaMemcpy(host_analytical_grad.data(), device_analytical_grad.get(), 
                               InputDim * sizeof(T), cudaMemcpyDeviceToHost), cudaSuccess);
            ASSERT_EQ(cudaMemcpy(host_numerical_grad.data(), device_numerical_grad.get(), 
                               InputDim * sizeof(T), cudaMemcpyDeviceToHost), cudaSuccess);
            
            // 誤差チェック
            for (std::size_t i = 0; i < InputDim; ++i) {
                T error_min = compute_error_min(host_analytical_grad[i], host_numerical_grad[i]);
                
                EXPECT_LE(error_min, tolerance) 
                    << operation_name << " test case " << test_case 
                    << ", input[" << i << "]: analytical=" << host_analytical_grad[i]
                    << ", numerical=" << host_numerical_grad[i]
                    << ", error_min=" << error_min;
            }
        }
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