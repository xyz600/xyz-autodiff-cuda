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

// BinaryOperation用カーネル関数
template <typename LogicType, std::size_t In1Dim, std::size_t In2Dim, std::size_t OutDim>
__global__ void test_binary_gradient_kernel(
    double* input1_data, double* input1_grad, double* input2_data, double* input2_grad,
    double* output_grad, double* analytical_grad1, double* numerical_grad1,
    double* analytical_grad2, double* numerical_grad2, double delta) {
    
    using T = double;
    
    // Variable作成
    VariableRef<T, In1Dim> input1_var(input1_data, input1_grad);
    VariableRef<T, In2Dim> input2_var(input2_data, input2_grad);
    
    LogicType logic;
    
    // 解析的勾配計算
    {
        auto op = BinaryOperation<OutDim, LogicType, VariableRef<T, In1Dim>, VariableRef<T, In2Dim>>(
            logic, input1_var, input2_var);
        op.forward();
        
        // 上流勾配設定
        for (std::size_t i = 0; i < OutDim; ++i) {
            op.grad(i) = output_grad[i];
        }
        
        input1_var.zero_grad();
        input2_var.zero_grad();
        op.backward();
        
        // 結果保存
        for (std::size_t i = 0; i < In1Dim; ++i) {
            analytical_grad1[i] = input1_var.grad(i);
        }
        for (std::size_t i = 0; i < In2Dim; ++i) {
            analytical_grad2[i] = input2_var.grad(i);
        }
    }
    
    // 数値勾配計算
    {
        auto op = BinaryOperation<OutDim, LogicType, VariableRef<T, In1Dim>, VariableRef<T, In2Dim>>(
            logic, input1_var, input2_var);
        op.forward();
        
        // 上流勾配設定
        for (std::size_t i = 0; i < OutDim; ++i) {
            op.grad(i) = output_grad[i];
        }
        
        input1_var.zero_grad();
        input2_var.zero_grad();
        op.backward_numerical(delta);
        
        // 結果保存
        for (std::size_t i = 0; i < In1Dim; ++i) {
            numerical_grad1[i] = input1_var.grad(i);
        }
        for (std::size_t i = 0; i < In2Dim; ++i) {
            numerical_grad2[i] = input2_var.grad(i);
        }
    }
}

// BinaryOperation のランダムテスト
template <typename Logic, std::size_t Input1Dim, std::size_t Input2Dim, std::size_t OutputDim>
class BinaryGradientTester {
private:
    static constexpr std::size_t NUM_TESTS = 100;
    static constexpr double TOLERANCE = 1e-5;
    static constexpr double DELTA = 1e-7;
    
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
        
        // ホストメモリ
        std::vector<T> host_input1_data(Input1Dim);
        std::vector<T> host_input1_grad(Input1Dim);
        std::vector<T> host_input2_data(Input2Dim);
        std::vector<T> host_input2_grad(Input2Dim);
        std::vector<T> host_output_grad(OutputDim);
        std::vector<T> host_analytical_grad1(Input1Dim);
        std::vector<T> host_numerical_grad1(Input1Dim);
        std::vector<T> host_analytical_grad2(Input2Dim);
        std::vector<T> host_numerical_grad2(Input2Dim);
        
        // デバイスメモリ確保
        auto device_input1_data = makeCudaUniqueArray<T>(Input1Dim);
        auto device_input1_grad = makeCudaUniqueArray<T>(Input1Dim);
        auto device_input2_data = makeCudaUniqueArray<T>(Input2Dim);
        auto device_input2_grad = makeCudaUniqueArray<T>(Input2Dim);
        auto device_output_grad = makeCudaUniqueArray<T>(OutputDim);
        auto device_analytical_grad1 = makeCudaUniqueArray<T>(Input1Dim);
        auto device_numerical_grad1 = makeCudaUniqueArray<T>(Input1Dim);
        auto device_analytical_grad2 = makeCudaUniqueArray<T>(Input2Dim);
        auto device_numerical_grad2 = makeCudaUniqueArray<T>(Input2Dim);
        
        // NUM_TESTSのランダムテストケース
        for (int test_case = 0; test_case < NUM_TESTS; ++test_case) {
            // ランダム入力生成
            for (std::size_t i = 0; i < Input1Dim; ++i) {
                host_input1_data[i] = dist(gen);
            }
            for (std::size_t i = 0; i < Input2Dim; ++i) {
                host_input2_data[i] = dist(gen);
            }
            
            // ランダム上流勾配生成
            for (std::size_t i = 0; i < OutputDim; ++i) {
                host_output_grad[i] = dist(gen);
            }
            
            // デバイスにコピー
            ASSERT_EQ(cudaMemcpy(device_input1_data.get(), host_input1_data.data(), 
                               Input1Dim * sizeof(T), cudaMemcpyHostToDevice), cudaSuccess);
            ASSERT_EQ(cudaMemcpy(device_input2_data.get(), host_input2_data.data(), 
                               Input2Dim * sizeof(T), cudaMemcpyHostToDevice), cudaSuccess);
            ASSERT_EQ(cudaMemcpy(device_output_grad.get(), host_output_grad.data(), 
                               OutputDim * sizeof(T), cudaMemcpyHostToDevice), cudaSuccess);
            
            // テストカーネル実行
            test_binary_gradient_kernel<Logic, Input1Dim, Input2Dim, OutputDim><<<1, 1>>>(
                device_input1_data.get(), device_input1_grad.get(),
                device_input2_data.get(), device_input2_grad.get(),
                device_output_grad.get(), 
                device_analytical_grad1.get(), device_numerical_grad1.get(),
                device_analytical_grad2.get(), device_numerical_grad2.get(),
                static_cast<T>(DELTA));
            
            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
            
            // 結果をホストにコピー
            ASSERT_EQ(cudaMemcpy(host_analytical_grad1.data(), device_analytical_grad1.get(), 
                               Input1Dim * sizeof(T), cudaMemcpyDeviceToHost), cudaSuccess);
            ASSERT_EQ(cudaMemcpy(host_numerical_grad1.data(), device_numerical_grad1.get(), 
                               Input1Dim * sizeof(T), cudaMemcpyDeviceToHost), cudaSuccess);
            ASSERT_EQ(cudaMemcpy(host_analytical_grad2.data(), device_analytical_grad2.get(), 
                               Input2Dim * sizeof(T), cudaMemcpyDeviceToHost), cudaSuccess);
            ASSERT_EQ(cudaMemcpy(host_numerical_grad2.data(), device_numerical_grad2.get(), 
                               Input2Dim * sizeof(T), cudaMemcpyDeviceToHost), cudaSuccess);
            
            // Input1の誤差チェック
            for (std::size_t i = 0; i < Input1Dim; ++i) {
                T error_min = compute_error_min(host_analytical_grad1[i], host_numerical_grad1[i]);
                
                EXPECT_LE(error_min, TOLERANCE) 
                    << operation_name << " test case " << test_case 
                    << ", input1[" << i << "]: analytical=" << host_analytical_grad1[i]
                    << ", numerical=" << host_numerical_grad1[i]
                    << ", error_min=" << error_min;
            }
            
            // Input2の誤差チェック
            for (std::size_t i = 0; i < Input2Dim; ++i) {
                T error_min = compute_error_min(host_analytical_grad2[i], host_numerical_grad2[i]);
                
                EXPECT_LE(error_min, TOLERANCE) 
                    << operation_name << " test case " << test_case 
                    << ", input2[" << i << "]: analytical=" << host_analytical_grad2[i]
                    << ", numerical=" << host_numerical_grad2[i]
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
        std::vector<T> host_input1_data(Input1Dim);
        std::vector<T> host_input1_grad(Input1Dim);
        std::vector<T> host_input2_data(Input2Dim);
        std::vector<T> host_input2_grad(Input2Dim);
        std::vector<T> host_output_grad(OutputDim);
        std::vector<T> host_analytical_grad1(Input1Dim);
        std::vector<T> host_numerical_grad1(Input1Dim);
        std::vector<T> host_analytical_grad2(Input2Dim);
        std::vector<T> host_numerical_grad2(Input2Dim);
        
        // デバイスメモリ確保
        auto device_input1_data = makeCudaUniqueArray<T>(Input1Dim);
        auto device_input1_grad = makeCudaUniqueArray<T>(Input1Dim);
        auto device_input2_data = makeCudaUniqueArray<T>(Input2Dim);
        auto device_input2_grad = makeCudaUniqueArray<T>(Input2Dim);
        auto device_output_grad = makeCudaUniqueArray<T>(OutputDim);
        auto device_analytical_grad1 = makeCudaUniqueArray<T>(Input1Dim);
        auto device_numerical_grad1 = makeCudaUniqueArray<T>(Input1Dim);
        auto device_analytical_grad2 = makeCudaUniqueArray<T>(Input2Dim);
        auto device_numerical_grad2 = makeCudaUniqueArray<T>(Input2Dim);
        
        for (std::size_t test_case = 0; test_case < num_tests; ++test_case) {
            // ランダム入力生成
            for (std::size_t i = 0; i < Input1Dim; ++i) {
                host_input1_data[i] = dist(gen);
            }
            for (std::size_t i = 0; i < Input2Dim; ++i) {
                host_input2_data[i] = dist(gen);
            }
            
            // ランダム上流勾配生成
            for (std::size_t i = 0; i < OutputDim; ++i) {
                host_output_grad[i] = dist(gen);
            }
            
            // デバイスにコピー
            ASSERT_EQ(cudaMemcpy(device_input1_data.get(), host_input1_data.data(), 
                               Input1Dim * sizeof(T), cudaMemcpyHostToDevice), cudaSuccess);
            ASSERT_EQ(cudaMemcpy(device_input2_data.get(), host_input2_data.data(), 
                               Input2Dim * sizeof(T), cudaMemcpyHostToDevice), cudaSuccess);
            ASSERT_EQ(cudaMemcpy(device_output_grad.get(), host_output_grad.data(), 
                               OutputDim * sizeof(T), cudaMemcpyHostToDevice), cudaSuccess);
            
            // テストカーネル実行
            test_binary_gradient_kernel<Logic, Input1Dim, Input2Dim, OutputDim><<<1, 1>>>(
                device_input1_data.get(), device_input1_grad.get(),
                device_input2_data.get(), device_input2_grad.get(),
                device_output_grad.get(), 
                device_analytical_grad1.get(), device_numerical_grad1.get(),
                device_analytical_grad2.get(), device_numerical_grad2.get(),
                static_cast<T>(delta));
            
            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
            
            // 結果をホストにコピー
            ASSERT_EQ(cudaMemcpy(host_analytical_grad1.data(), device_analytical_grad1.get(), 
                               Input1Dim * sizeof(T), cudaMemcpyDeviceToHost), cudaSuccess);
            ASSERT_EQ(cudaMemcpy(host_numerical_grad1.data(), device_numerical_grad1.get(), 
                               Input1Dim * sizeof(T), cudaMemcpyDeviceToHost), cudaSuccess);
            ASSERT_EQ(cudaMemcpy(host_analytical_grad2.data(), device_analytical_grad2.get(), 
                               Input2Dim * sizeof(T), cudaMemcpyDeviceToHost), cudaSuccess);
            ASSERT_EQ(cudaMemcpy(host_numerical_grad2.data(), device_numerical_grad2.get(), 
                               Input2Dim * sizeof(T), cudaMemcpyDeviceToHost), cudaSuccess);
            
            // Input1の誤差チェック
            for (std::size_t i = 0; i < Input1Dim; ++i) {
                T error_min = compute_error_min(host_analytical_grad1[i], host_numerical_grad1[i]);
                
                EXPECT_LE(error_min, tolerance) 
                    << operation_name << " test case " << test_case 
                    << ", input1[" << i << "]: analytical=" << host_analytical_grad1[i]
                    << ", numerical=" << host_numerical_grad1[i]
                    << ", error_min=" << error_min;
            }
            
            // Input2の誤差チェック
            for (std::size_t i = 0; i < Input2Dim; ++i) {
                T error_min = compute_error_min(host_analytical_grad2[i], host_numerical_grad2[i]);
                
                EXPECT_LE(error_min, tolerance) 
                    << operation_name << " test case " << test_case 
                    << ", input2[" << i << "]: analytical=" << host_analytical_grad2[i]
                    << ", numerical=" << host_numerical_grad2[i]
                    << ", error_min=" << error_min;
            }
        }
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