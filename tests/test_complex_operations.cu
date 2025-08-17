#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <array>
#include <cmath>
#include <random>
#include "../include/variable.cuh"
#include "../include/operations/sigmoid_logic.cuh"
#include "../include/operations/exp_logic.cuh"
#include "../include/operations/add_logic.cuh"
#include "../include/operations/unary_functions.cuh"
#include "../include/dense_matrix.cuh"
#include "../include/diagonal_matrix_view.cuh"
#include "../include/symmetric_matrix_view.cuh"
#include "../include/util/cuda_unique_ptr.cuh"

using namespace xyz_autodiff;

// 数値微分のためのパラメータ
constexpr double NUMERICAL_DELTA = 1e-7;
constexpr double TOLERANCE = 1e-5;

// ================================================================================
// Unary Operations の数値微分検証
// ================================================================================

// Sigmoid の数値微分検証カーネル
template <typename T, std::size_t Dim>
__global__ void verify_sigmoid_gradient_kernel(
    T* input_data, T* analytical_grad, T* numerical_grad, bool* passed) {
    
    SigmoidLogic<Dim> logic;
    
    // 解析的勾配の計算
    T input_grad_data[Dim] = {};
    T output_data[Dim];
    T output_grad_data[Dim];
    
    // 出力勾配を単位ベクトルに設定（各次元独立にテスト）
    for (std::size_t out_idx = 0; out_idx < Dim; ++out_idx) {
        // リセット
        for (std::size_t i = 0; i < Dim; ++i) {
            input_grad_data[i] = T{0};
            output_grad_data[i] = (i == out_idx) ? T{1} : T{0};
        }
        
        Variable<T, Dim> input(input_data, input_grad_data);
        Variable<T, Dim> output(output_data, output_grad_data);
        
        logic.forward(output, input);
        logic.backward(output, input);
        
        // 各入力次元の勾配を保存
        for (std::size_t in_idx = 0; in_idx < Dim; ++in_idx) {
            analytical_grad[out_idx * Dim + in_idx] = input.grad(in_idx);
        }
    }
    
    // 数値微分の計算
    for (std::size_t out_idx = 0; out_idx < Dim; ++out_idx) {
        for (std::size_t in_idx = 0; in_idx < Dim; ++in_idx) {
            T original = input_data[in_idx];
            
            // f(x + delta)
            input_data[in_idx] = original + static_cast<T>(NUMERICAL_DELTA);
            Variable<T, Dim> input_plus(input_data, input_grad_data);
            Variable<T, Dim> output_plus(output_data, output_grad_data);
            logic.forward(output_plus, input_plus);
            T f_plus = output_plus[out_idx];
            
            // f(x - delta)
            input_data[in_idx] = original - static_cast<T>(NUMERICAL_DELTA);
            Variable<T, Dim> input_minus(input_data, input_grad_data);
            Variable<T, Dim> output_minus(output_data, output_grad_data);
            logic.forward(output_minus, input_minus);
            T f_minus = output_minus[out_idx];
            
            // 数値勾配
            numerical_grad[out_idx * Dim + in_idx] = 
                (f_plus - f_minus) / (2 * static_cast<T>(NUMERICAL_DELTA));
            
            // 元に戻す
            input_data[in_idx] = original;
        }
    }
    
    // 勾配の比較
    *passed = true;
    for (std::size_t i = 0; i < Dim * Dim; ++i) {
        T diff = fabs(analytical_grad[i] - numerical_grad[i]);
        T scale = fmax(fabs(analytical_grad[i]), fabs(numerical_grad[i])) + static_cast<T>(1e-10);
        T rel_error = diff / scale;
        
        if (rel_error > static_cast<T>(TOLERANCE)) {
            *passed = false;
            break;
        }
    }
}

// Exp の数値微分検証カーネル
template <typename T, std::size_t Dim>
__global__ void verify_exp_gradient_kernel(
    T* input_data, T* analytical_grad, T* numerical_grad, bool* passed) {
    
    ExpLogic<Dim> logic;
    
    // 解析的勾配の計算
    T input_grad_data[Dim] = {};
    T output_data[Dim];
    T output_grad_data[Dim];
    
    for (std::size_t out_idx = 0; out_idx < Dim; ++out_idx) {
        for (std::size_t i = 0; i < Dim; ++i) {
            input_grad_data[i] = T{0};
            output_grad_data[i] = (i == out_idx) ? T{1} : T{0};
        }
        
        Variable<T, Dim> input(input_data, input_grad_data);
        Variable<T, Dim> output(output_data, output_grad_data);
        
        logic.forward(output, input);
        logic.backward(output, input);
        
        for (std::size_t in_idx = 0; in_idx < Dim; ++in_idx) {
            analytical_grad[out_idx * Dim + in_idx] = input.grad(in_idx);
        }
    }
    
    // 数値微分の計算
    for (std::size_t out_idx = 0; out_idx < Dim; ++out_idx) {
        for (std::size_t in_idx = 0; in_idx < Dim; ++in_idx) {
            T original = input_data[in_idx];
            
            input_data[in_idx] = original + static_cast<T>(NUMERICAL_DELTA);
            Variable<T, Dim> input_plus(input_data, input_grad_data);
            Variable<T, Dim> output_plus(output_data, output_grad_data);
            logic.forward(output_plus, input_plus);
            T f_plus = output_plus[out_idx];
            
            input_data[in_idx] = original - static_cast<T>(NUMERICAL_DELTA);
            Variable<T, Dim> input_minus(input_data, input_grad_data);
            Variable<T, Dim> output_minus(output_data, output_grad_data);
            logic.forward(output_minus, input_minus);
            T f_minus = output_minus[out_idx];
            
            numerical_grad[out_idx * Dim + in_idx] = 
                (f_plus - f_minus) / (2 * static_cast<T>(NUMERICAL_DELTA));
            
            input_data[in_idx] = original;
        }
    }
    
    // 勾配の比較
    *passed = true;
    for (std::size_t i = 0; i < Dim * Dim; ++i) {
        T diff = fabs(analytical_grad[i] - numerical_grad[i]);
        T scale = fmax(fabs(analytical_grad[i]), fabs(numerical_grad[i])) + static_cast<T>(1e-10);
        T rel_error = diff / scale;
        
        if (rel_error > static_cast<T>(TOLERANCE)) {
            *passed = false;
            break;
        }
    }
}

// ================================================================================
// Binary Operations の数値微分検証
// ================================================================================

// Add の数値微分検証カーネル
template <typename T>
__global__ void verify_add_gradient_kernel(
    T* input1_data, T* input2_data, 
    T* analytical_grad1, T* analytical_grad2,
    T* numerical_grad1, T* numerical_grad2,
    bool* passed) {
    
    using V = Variable<T, 1>;
    op::AddLogic<V, V> logic;
    
    // 解析的勾配の計算
    T input1_grad_data[1] = {T{0}};
    T input2_grad_data[1] = {T{0}};
    T output_data[1];
    T output_grad_data[1] = {T{1}};
    
    V input1(input1_data, input1_grad_data);
    V input2(input2_data, input2_grad_data);
    V output(output_data, output_grad_data);
    
    logic.forward(output, input1, input2);
    logic.backward(output, input1, input2);
    
    analytical_grad1[0] = input1.grad(0);
    analytical_grad2[0] = input2.grad(0);
    
    // 数値微分 - input1
    T original1 = input1_data[0];
    
    input1_data[0] = original1 + static_cast<T>(NUMERICAL_DELTA);
    V input1_plus(input1_data, input1_grad_data);
    V output_plus(output_data, output_grad_data);
    logic.forward(output_plus, input1_plus, input2);
    T f1_plus = output_plus[0];
    
    input1_data[0] = original1 - static_cast<T>(NUMERICAL_DELTA);
    V input1_minus(input1_data, input1_grad_data);
    V output_minus(output_data, output_grad_data);
    logic.forward(output_minus, input1_minus, input2);
    T f1_minus = output_minus[0];
    
    numerical_grad1[0] = (f1_plus - f1_minus) / (2 * static_cast<T>(NUMERICAL_DELTA));
    input1_data[0] = original1;
    
    // 数値微分 - input2
    T original2 = input2_data[0];
    
    input2_data[0] = original2 + static_cast<T>(NUMERICAL_DELTA);
    V input2_plus(input2_data, input2_grad_data);
    logic.forward(output_plus, input1, input2_plus);
    T f2_plus = output_plus[0];
    
    input2_data[0] = original2 - static_cast<T>(NUMERICAL_DELTA);
    V input2_minus(input2_data, input2_grad_data);
    logic.forward(output_minus, input1, input2_minus);
    T f2_minus = output_minus[0];
    
    numerical_grad2[0] = (f2_plus - f2_minus) / (2 * static_cast<T>(NUMERICAL_DELTA));
    input2_data[0] = original2;
    
    // 勾配の比較
    *passed = true;
    
    T diff1 = fabs(analytical_grad1[0] - numerical_grad1[0]);
    T scale1 = fmax(fabs(analytical_grad1[0]), fabs(numerical_grad1[0])) + static_cast<T>(1e-10);
    T rel_error1 = diff1 / scale1;
    
    T diff2 = fabs(analytical_grad2[0] - numerical_grad2[0]);
    T scale2 = fmax(fabs(analytical_grad2[0]), fabs(numerical_grad2[0])) + static_cast<T>(1e-10);
    T rel_error2 = diff2 / scale2;
    
    if (rel_error1 > static_cast<T>(TOLERANCE) || rel_error2 > static_cast<T>(TOLERANCE)) {
        *passed = false;
    }
}

// ================================================================================
// Combined Operations の数値微分検証
// ================================================================================

// Combined operation (sigmoid ∘ exp) の数値微分検証カーネル
template <typename T, std::size_t Dim>
__global__ void verify_combined_gradient_kernel(
    T* input_data, T* analytical_grad, T* numerical_grad, bool* passed) {
    
    // 解析的勾配の計算: sigmoid(exp(x))
    T input_grad_data[Dim] = {};
    T exp_output_data[Dim];
    T exp_output_grad_data[Dim];
    T sigmoid_output_data[Dim];
    T sigmoid_output_grad_data[Dim];
    
    for (std::size_t out_idx = 0; out_idx < Dim; ++out_idx) {
        // リセット
        for (std::size_t i = 0; i < Dim; ++i) {
            input_grad_data[i] = T{0};
            exp_output_grad_data[i] = T{0};
            sigmoid_output_grad_data[i] = (i == out_idx) ? T{1} : T{0};
        }
        
        Variable<T, Dim> input(input_data, input_grad_data);
        Variable<T, Dim> exp_output(exp_output_data, exp_output_grad_data);
        Variable<T, Dim> sigmoid_output(sigmoid_output_data, sigmoid_output_grad_data);
        
        ExpLogic<Dim> exp_logic;
        SigmoidLogic<Dim> sigmoid_logic;
        
        // Forward pass: input -> exp -> sigmoid
        exp_logic.forward(exp_output, input);
        sigmoid_logic.forward(sigmoid_output, exp_output);
        
        // Backward pass: sigmoid -> exp -> input
        sigmoid_logic.backward(sigmoid_output, exp_output);
        exp_logic.backward(exp_output, input);
        
        for (std::size_t in_idx = 0; in_idx < Dim; ++in_idx) {
            analytical_grad[out_idx * Dim + in_idx] = input.grad(in_idx);
        }
    }
    
    // 数値微分の計算: sigmoid(exp(x))
    for (std::size_t out_idx = 0; out_idx < Dim; ++out_idx) {
        for (std::size_t in_idx = 0; in_idx < Dim; ++in_idx) {
            T original = input_data[in_idx];
            
            // f(x + delta) = sigmoid(exp(x + delta))
            input_data[in_idx] = original + static_cast<T>(NUMERICAL_DELTA);
            Variable<T, Dim> input_plus(input_data, input_grad_data);
            Variable<T, Dim> exp_output_plus(exp_output_data, exp_output_grad_data);
            Variable<T, Dim> sigmoid_output_plus(sigmoid_output_data, sigmoid_output_grad_data);
            
            ExpLogic<Dim> exp_logic;
            SigmoidLogic<Dim> sigmoid_logic;
            exp_logic.forward(exp_output_plus, input_plus);
            sigmoid_logic.forward(sigmoid_output_plus, exp_output_plus);
            T f_plus = sigmoid_output_plus[out_idx];
            
            // f(x - delta) = sigmoid(exp(x - delta))
            input_data[in_idx] = original - static_cast<T>(NUMERICAL_DELTA);
            Variable<T, Dim> input_minus(input_data, input_grad_data);
            Variable<T, Dim> exp_output_minus(exp_output_data, exp_output_grad_data);
            Variable<T, Dim> sigmoid_output_minus(sigmoid_output_data, sigmoid_output_grad_data);
            
            exp_logic.forward(exp_output_minus, input_minus);
            sigmoid_logic.forward(sigmoid_output_minus, exp_output_minus);
            T f_minus = sigmoid_output_minus[out_idx];
            
            numerical_grad[out_idx * Dim + in_idx] = 
                (f_plus - f_minus) / (2 * static_cast<T>(NUMERICAL_DELTA));
            
            input_data[in_idx] = original;
        }
    }
    
    // 勾配の比較
    *passed = true;
    for (std::size_t i = 0; i < Dim * Dim; ++i) {
        T diff = fabs(analytical_grad[i] - numerical_grad[i]);
        T scale = fmax(fabs(analytical_grad[i]), fabs(numerical_grad[i])) + static_cast<T>(1e-10);
        T rel_error = diff / scale;
        
        if (rel_error > static_cast<T>(TOLERANCE)) {
            *passed = false;
            break;
        }
    }
}

// ================================================================================
// Test Suite
// ================================================================================

class NumericalGradientVerificationTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaError_t err = cudaSetDevice(0);
        if (err != cudaSuccess) {
            GTEST_SKIP() << "CUDA device not available";
        }
    }
};

// Sigmoid gradient verification
TEST_F(NumericalGradientVerificationTest, SigmoidGradient) {
    using T = double;
    constexpr std::size_t Dim = 3;
    
    // ランダムな入力を生成
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> dis(-2.0, 2.0);
    
    T host_input[Dim];
    for (std::size_t i = 0; i < Dim; ++i) {
        host_input[i] = dis(gen);
    }
    
    auto device_input = makeCudaUniqueArray<T>(Dim);
    auto device_analytical = makeCudaUniqueArray<T>(Dim * Dim);
    auto device_numerical = makeCudaUniqueArray<T>(Dim * Dim);
    auto device_passed = makeCudaUniqueArray<bool>(1);
    
    cudaMemcpy(device_input.get(), host_input, Dim * sizeof(T), cudaMemcpyHostToDevice);
    
    verify_sigmoid_gradient_kernel<T, Dim><<<1, 1>>>(
        device_input.get(),
        device_analytical.get(),
        device_numerical.get(),
        device_passed.get()
    );
    
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    bool passed;
    cudaMemcpy(&passed, device_passed.get(), sizeof(bool), cudaMemcpyDeviceToHost);
    
    if (!passed) {
        T analytical[Dim * Dim], numerical[Dim * Dim];
        cudaMemcpy(analytical, device_analytical.get(), Dim * Dim * sizeof(T), cudaMemcpyDeviceToHost);
        cudaMemcpy(numerical, device_numerical.get(), Dim * Dim * sizeof(T), cudaMemcpyDeviceToHost);
        
        for (std::size_t i = 0; i < Dim * Dim; ++i) {
            T diff = std::abs(analytical[i] - numerical[i]);
            T scale = std::max(std::abs(analytical[i]), std::abs(numerical[i])) + 1e-10;
            T rel_error = diff / scale;
            if (rel_error > TOLERANCE) {
                ADD_FAILURE() << "Gradient mismatch at index " << i 
                            << ": analytical=" << analytical[i] 
                            << ", numerical=" << numerical[i]
                            << ", rel_error=" << rel_error;
            }
        }
    }
    
    EXPECT_TRUE(passed);
}

// Exp gradient verification
TEST_F(NumericalGradientVerificationTest, ExpGradient) {
    using T = double;
    constexpr std::size_t Dim = 3;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> dis(-1.0, 1.0);
    
    T host_input[Dim];
    for (std::size_t i = 0; i < Dim; ++i) {
        host_input[i] = dis(gen);
    }
    
    auto device_input = makeCudaUniqueArray<T>(Dim);
    auto device_analytical = makeCudaUniqueArray<T>(Dim * Dim);
    auto device_numerical = makeCudaUniqueArray<T>(Dim * Dim);
    auto device_passed = makeCudaUniqueArray<bool>(1);
    
    cudaMemcpy(device_input.get(), host_input, Dim * sizeof(T), cudaMemcpyHostToDevice);
    
    verify_exp_gradient_kernel<T, Dim><<<1, 1>>>(
        device_input.get(),
        device_analytical.get(),
        device_numerical.get(),
        device_passed.get()
    );
    
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    bool passed;
    cudaMemcpy(&passed, device_passed.get(), sizeof(bool), cudaMemcpyDeviceToHost);
    
    EXPECT_TRUE(passed);
}

// Add gradient verification
TEST_F(NumericalGradientVerificationTest, AddGradient) {
    using T = double;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> dis(-2.0, 2.0);
    
    T host_input1[1] = {dis(gen)};
    T host_input2[1] = {dis(gen)};
    
    auto device_input1 = makeCudaUniqueArray<T>(1);
    auto device_input2 = makeCudaUniqueArray<T>(1);
    auto device_analytical1 = makeCudaUniqueArray<T>(1);
    auto device_analytical2 = makeCudaUniqueArray<T>(1);
    auto device_numerical1 = makeCudaUniqueArray<T>(1);
    auto device_numerical2 = makeCudaUniqueArray<T>(1);
    auto device_passed = makeCudaUniqueArray<bool>(1);
    
    cudaMemcpy(device_input1.get(), host_input1, sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(device_input2.get(), host_input2, sizeof(T), cudaMemcpyHostToDevice);
    
    verify_add_gradient_kernel<T><<<1, 1>>>(
        device_input1.get(),
        device_input2.get(),
        device_analytical1.get(),
        device_analytical2.get(),
        device_numerical1.get(),
        device_numerical2.get(),
        device_passed.get()
    );
    
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    bool passed;
    cudaMemcpy(&passed, device_passed.get(), sizeof(bool), cudaMemcpyDeviceToHost);
    
    EXPECT_TRUE(passed);
}

// Combined operations gradient verification (sigmoid ∘ exp)
TEST_F(NumericalGradientVerificationTest, CombinedOperationsGradient) {
    using T = double;
    constexpr std::size_t Dim = 3;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> dis(-1.0, 1.0);  // 小さい範囲でテスト
    
    T host_input[Dim];
    for (std::size_t i = 0; i < Dim; ++i) {
        host_input[i] = dis(gen);
    }
    
    auto device_input = makeCudaUniqueArray<T>(Dim);
    auto device_analytical = makeCudaUniqueArray<T>(Dim * Dim);
    auto device_numerical = makeCudaUniqueArray<T>(Dim * Dim);
    auto device_passed = makeCudaUniqueArray<bool>(1);
    
    cudaMemcpy(device_input.get(), host_input, Dim * sizeof(T), cudaMemcpyHostToDevice);
    
    verify_combined_gradient_kernel<T, Dim><<<1, 1>>>(
        device_input.get(),
        device_analytical.get(),
        device_numerical.get(),
        device_passed.get()
    );
    
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    bool passed;
    cudaMemcpy(&passed, device_passed.get(), sizeof(bool), cudaMemcpyDeviceToHost);
    
    if (!passed) {
        T analytical[Dim * Dim], numerical[Dim * Dim];
        cudaMemcpy(analytical, device_analytical.get(), Dim * Dim * sizeof(T), cudaMemcpyDeviceToHost);
        cudaMemcpy(numerical, device_numerical.get(), Dim * Dim * sizeof(T), cudaMemcpyDeviceToHost);
        
        for (std::size_t i = 0; i < Dim * Dim; ++i) {
            T diff = std::abs(analytical[i] - numerical[i]);
            T scale = std::max(std::abs(analytical[i]), std::abs(numerical[i])) + 1e-10;
            T rel_error = diff / scale;
            if (rel_error > TOLERANCE) {
                ADD_FAILURE() << "Combined gradient mismatch at index " << i 
                            << ": analytical=" << analytical[i] 
                            << ", numerical=" << numerical[i]
                            << ", rel_error=" << rel_error;
            }
        }
    }
    
    EXPECT_TRUE(passed);
}

// Large-scale random tests for robustness
TEST_F(NumericalGradientVerificationTest, LargeScaleRandomTests) {
    using T = double;
    
    const int num_tests = 100;
    int sigmoid_passed = 0;
    int exp_passed = 0;
    int add_passed = 0;
    int combined_passed = 0;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> dis_sigmoid(-2.0, 2.0);
    std::uniform_real_distribution<T> dis_exp(-1.0, 1.0);
    
    // Sigmoid tests
    for (int test = 0; test < num_tests; ++test) {
        constexpr std::size_t Dim = 3;
        T host_input[Dim];
        for (std::size_t i = 0; i < Dim; ++i) {
            host_input[i] = dis_sigmoid(gen);
        }
        
        auto device_input = makeCudaUniqueArray<T>(Dim);
        auto device_analytical = makeCudaUniqueArray<T>(Dim * Dim);
        auto device_numerical = makeCudaUniqueArray<T>(Dim * Dim);
        auto device_passed = makeCudaUniqueArray<bool>(1);
        
        cudaMemcpy(device_input.get(), host_input, Dim * sizeof(T), cudaMemcpyHostToDevice);
        
        verify_sigmoid_gradient_kernel<T, Dim><<<1, 1>>>(
            device_input.get(),
            device_analytical.get(),
            device_numerical.get(),
            device_passed.get()
        );
        
        cudaDeviceSynchronize();
        
        bool passed;
        cudaMemcpy(&passed, device_passed.get(), sizeof(bool), cudaMemcpyDeviceToHost);
        if (passed) sigmoid_passed++;
    }
    
    // Exp tests
    for (int test = 0; test < num_tests; ++test) {
        constexpr std::size_t Dim = 3;
        T host_input[Dim];
        for (std::size_t i = 0; i < Dim; ++i) {
            host_input[i] = dis_exp(gen);
        }
        
        auto device_input = makeCudaUniqueArray<T>(Dim);
        auto device_analytical = makeCudaUniqueArray<T>(Dim * Dim);
        auto device_numerical = makeCudaUniqueArray<T>(Dim * Dim);
        auto device_passed = makeCudaUniqueArray<bool>(1);
        
        cudaMemcpy(device_input.get(), host_input, Dim * sizeof(T), cudaMemcpyHostToDevice);
        
        verify_exp_gradient_kernel<T, Dim><<<1, 1>>>(
            device_input.get(),
            device_analytical.get(),
            device_numerical.get(),
            device_passed.get()
        );
        
        cudaDeviceSynchronize();
        
        bool passed;
        cudaMemcpy(&passed, device_passed.get(), sizeof(bool), cudaMemcpyDeviceToHost);
        if (passed) exp_passed++;
    }
    
    // Add tests
    for (int test = 0; test < num_tests; ++test) {
        T host_input1[1] = {dis_sigmoid(gen)};
        T host_input2[1] = {dis_sigmoid(gen)};
        
        auto device_input1 = makeCudaUniqueArray<T>(1);
        auto device_input2 = makeCudaUniqueArray<T>(1);
        auto device_analytical1 = makeCudaUniqueArray<T>(1);
        auto device_analytical2 = makeCudaUniqueArray<T>(1);
        auto device_numerical1 = makeCudaUniqueArray<T>(1);
        auto device_numerical2 = makeCudaUniqueArray<T>(1);
        auto device_passed = makeCudaUniqueArray<bool>(1);
        
        cudaMemcpy(device_input1.get(), host_input1, sizeof(T), cudaMemcpyHostToDevice);
        cudaMemcpy(device_input2.get(), host_input2, sizeof(T), cudaMemcpyHostToDevice);
        
        verify_add_gradient_kernel<T><<<1, 1>>>(
            device_input1.get(),
            device_input2.get(),
            device_analytical1.get(),
            device_analytical2.get(),
            device_numerical1.get(),
            device_numerical2.get(),
            device_passed.get()
        );
        
        cudaDeviceSynchronize();
        
        bool passed;
        cudaMemcpy(&passed, device_passed.get(), sizeof(bool), cudaMemcpyDeviceToHost);
        if (passed) add_passed++;
    }
    
    // Combined operation tests
    std::uniform_real_distribution<T> dis_combined(-1.0, 1.0);
    for (int test = 0; test < num_tests; ++test) {
        constexpr std::size_t Dim = 3;
        T host_input[Dim];
        for (std::size_t i = 0; i < Dim; ++i) {
            host_input[i] = dis_combined(gen);
        }
        
        auto device_input = makeCudaUniqueArray<T>(Dim);
        auto device_analytical = makeCudaUniqueArray<T>(Dim * Dim);
        auto device_numerical = makeCudaUniqueArray<T>(Dim * Dim);
        auto device_passed = makeCudaUniqueArray<bool>(1);
        
        cudaMemcpy(device_input.get(), host_input, Dim * sizeof(T), cudaMemcpyHostToDevice);
        
        verify_combined_gradient_kernel<T, Dim><<<1, 1>>>(
            device_input.get(),
            device_analytical.get(),
            device_numerical.get(),
            device_passed.get()
        );
        
        cudaDeviceSynchronize();
        
        bool passed;
        cudaMemcpy(&passed, device_passed.get(), sizeof(bool), cudaMemcpyDeviceToHost);
        if (passed) combined_passed++;
    }
    
    EXPECT_GE(sigmoid_passed, num_tests * 95 / 100) 
        << "Sigmoid: " << sigmoid_passed << "/" << num_tests << " passed";
    EXPECT_GE(exp_passed, num_tests * 95 / 100) 
        << "Exp: " << exp_passed << "/" << num_tests << " passed";
    EXPECT_EQ(add_passed, num_tests) 
        << "Add: " << add_passed << "/" << num_tests << " passed";
    EXPECT_GE(combined_passed, num_tests * 95 / 100) 
        << "Combined (sigmoid∘exp): " << combined_passed << "/" << num_tests << " passed";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}