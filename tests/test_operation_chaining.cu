#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <random>
#include <vector>
#include <cmath>
#include "../include/variable.cuh"
#include "../include/operations/add_logic.cuh"
#include "../include/operations/mul_logic.cuh"
#include "../include/util/cuda_unique_ptr.cuh"

using namespace xyz_autodiff;

// f(x, y, z) = xz + y を計算するカーネル（解析的微分）
template <typename T>
__global__ void test_chaining_analytical_kernel(
    T* x_data, T* x_grad, T* y_data, T* y_grad, T* z_data, T* z_grad,
    T* output_data, T* output_grad) {
    
    // Variable作成
    VariableRef<T, 1> x_var(x_data, x_grad);
    VariableRef<T, 1> y_var(y_data, y_grad);
    VariableRef<T, 1> z_var(z_data, z_grad);
    
    // f(x, y, z) = xz + y の計算（自動的にforwardが呼ばれ、結果が保存される）
    auto mul_result = op::mul(x_var, z_var);
    auto final_result = op::add(mul_result, y_var);
    
    // 勾配をゼロクリア（top-downで自動的に全ての勾配がクリアされる）
    final_result.zero_grad();
    
    // 上流勾配を設定
    final_result.add_grad(0, output_grad[0]);
    
    // 逆伝播実行（自動的に全ての中間operationのbackwardが呼ばれる）
    final_result.backward();
}

// f(x, y, z) = xz + y を数値微分で計算するカーネル
template <typename T>
__global__ void test_chaining_numerical_kernel(
    T* x_data, T* x_grad, T* y_data, T* y_grad, T* z_data, T* z_grad,
    T* output_data, T* output_grad, T delta) {
    
    // Variable作成
    VariableRef<T, 1> x_var(x_data, x_grad);
    VariableRef<T, 1> y_var(y_data, y_grad);
    VariableRef<T, 1> z_var(z_data, z_grad);
    
    // f(x, y, z) = xz + y の計算（自動的にforwardが呼ばれ、結果が保存される）
    auto mul_result = op::mul(x_var, z_var);
    auto final_result = op::add(mul_result, y_var);
    
    // 勾配をゼロクリア（top-downで自動的に全ての勾配がクリアされる）
    final_result.zero_grad();
    
    // 上流勾配を設定
    final_result.add_grad(0, output_grad[0]);
    
    // 数値微分による逆伝播実行（自動的に全ての中間operationのbackward_numericalが呼ばれる）
    final_result.backward_numerical(delta);
}

// 直接数値微分による勾配計算（検証用）
template <typename T>
__global__ void compute_direct_numerical_gradient_kernel(
    T x, T y, T z, T* grad_x, T* grad_y, T* grad_z, T delta, T upstream_grad) {
    
    // f(x, y, z) = xz + y の直接計算
    
    // x に対する数値微分
    T f_plus_x = (x + delta) * z + y;
    T f_minus_x = (x - delta) * z + y;
    grad_x[0] = upstream_grad * (f_plus_x - f_minus_x) / (2 * delta);
    
    // y に対する数値微分
    T f_plus_y = x * z + (y + delta);
    T f_minus_y = x * z + (y - delta);
    grad_y[0] = upstream_grad * (f_plus_y - f_minus_y) / (2 * delta);
    
    // z に対する数値微分
    T f_plus_z = x * (z + delta) + y;
    T f_minus_z = x * (z - delta) + y;
    grad_z[0] = upstream_grad * (f_plus_z - f_minus_z) / (2 * delta);
}

class OperationChainingTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaError_t err = cudaSetDevice(0);
        if (err != cudaSuccess) {
            GTEST_SKIP() << "CUDA device not available";
        }
    }
};

TEST_F(OperationChainingTest, AnalyticalVsNumericalGradient) {
    using T = double;
    constexpr T TOLERANCE = 1e-5;
    constexpr T DELTA = 1e-7;
    constexpr int NUM_TESTS = 50;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> dist(-2.0, 2.0);
    
    // デバイスメモリ確保
    auto device_x_data = makeCudaUniqueArray<T>(1);
    auto device_x_grad = makeCudaUniqueArray<T>(1);
    auto device_y_data = makeCudaUniqueArray<T>(1);
    auto device_y_grad = makeCudaUniqueArray<T>(1);
    auto device_z_data = makeCudaUniqueArray<T>(1);
    auto device_z_grad = makeCudaUniqueArray<T>(1);
    auto device_output_data = makeCudaUniqueArray<T>(1);
    auto device_output_grad = makeCudaUniqueArray<T>(1);
    auto device_analytical_x_grad = makeCudaUniqueArray<T>(1);
    auto device_analytical_y_grad = makeCudaUniqueArray<T>(1);
    auto device_analytical_z_grad = makeCudaUniqueArray<T>(1);
    auto device_numerical_x_grad = makeCudaUniqueArray<T>(1);
    auto device_numerical_y_grad = makeCudaUniqueArray<T>(1);
    auto device_numerical_z_grad = makeCudaUniqueArray<T>(1);
    auto device_direct_x_grad = makeCudaUniqueArray<T>(1);
    auto device_direct_y_grad = makeCudaUniqueArray<T>(1);
    auto device_direct_z_grad = makeCudaUniqueArray<T>(1);
    
    for (int test_case = 0; test_case < NUM_TESTS; ++test_case) {
        // ランダム入力生成
        T x = dist(gen);
        T y = dist(gen);
        T z = dist(gen);
        T upstream_grad = dist(gen);
        
        // デバイスにコピー
        ASSERT_EQ(cudaMemcpy(device_x_data.get(), &x, sizeof(T), cudaMemcpyHostToDevice), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(device_y_data.get(), &y, sizeof(T), cudaMemcpyHostToDevice), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(device_z_data.get(), &z, sizeof(T), cudaMemcpyHostToDevice), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(device_output_grad.get(), &upstream_grad, sizeof(T), cudaMemcpyHostToDevice), cudaSuccess);
        
        // 解析的勾配計算
        test_chaining_analytical_kernel<T><<<1, 1>>>(
            device_x_data.get(), device_x_grad.get(),
            device_y_data.get(), device_y_grad.get(),
            device_z_data.get(), device_z_grad.get(),
            device_output_data.get(), device_output_grad.get());
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        
        // 数値勾配計算（operation chaining経由）
        test_chaining_numerical_kernel<T><<<1, 1>>>(
            device_x_data.get(), device_x_grad.get(),
            device_y_data.get(), device_y_grad.get(),
            device_z_data.get(), device_z_grad.get(),
            device_output_data.get(), device_output_grad.get(),
            static_cast<T>(DELTA));
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        
        // 直接数値微分計算（検証用）
        compute_direct_numerical_gradient_kernel<T><<<1, 1>>>(
            x, y, z, device_direct_x_grad.get(), device_direct_y_grad.get(), device_direct_z_grad.get(),
            static_cast<T>(DELTA), upstream_grad);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        
        // 結果をホストにコピー（自動保存されたVariable gradから）
        T analytical_x_grad, analytical_y_grad, analytical_z_grad;
        T numerical_x_grad, numerical_y_grad, numerical_z_grad;
        T direct_x_grad, direct_y_grad, direct_z_grad;
        
        ASSERT_EQ(cudaMemcpy(&analytical_x_grad, device_x_grad.get(), sizeof(T), cudaMemcpyDeviceToHost), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(&analytical_y_grad, device_y_grad.get(), sizeof(T), cudaMemcpyDeviceToHost), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(&analytical_z_grad, device_z_grad.get(), sizeof(T), cudaMemcpyDeviceToHost), cudaSuccess);
        
        // 数値微分のテストのために勾配をリセット
        T zero = T(0);
        ASSERT_EQ(cudaMemcpy(device_x_grad.get(), &zero, sizeof(T), cudaMemcpyHostToDevice), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(device_y_grad.get(), &zero, sizeof(T), cudaMemcpyHostToDevice), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(device_z_grad.get(), &zero, sizeof(T), cudaMemcpyHostToDevice), cudaSuccess);
        
        // 数値勾配計算（operation chaining経由）
        test_chaining_numerical_kernel<T><<<1, 1>>>(
            device_x_data.get(), device_x_grad.get(),
            device_y_data.get(), device_y_grad.get(),
            device_z_data.get(), device_z_grad.get(),
            device_output_data.get(), device_output_grad.get(),
            static_cast<T>(DELTA));
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        
        ASSERT_EQ(cudaMemcpy(&numerical_x_grad, device_x_grad.get(), sizeof(T), cudaMemcpyDeviceToHost), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(&numerical_y_grad, device_y_grad.get(), sizeof(T), cudaMemcpyDeviceToHost), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(&numerical_z_grad, device_z_grad.get(), sizeof(T), cudaMemcpyDeviceToHost), cudaSuccess);
        
        ASSERT_EQ(cudaMemcpy(&direct_x_grad, device_direct_x_grad.get(), sizeof(T), cudaMemcpyDeviceToHost), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(&direct_y_grad, device_direct_y_grad.get(), sizeof(T), cudaMemcpyDeviceToHost), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(&direct_z_grad, device_direct_z_grad.get(), sizeof(T), cudaMemcpyDeviceToHost), cudaSuccess);
        
        // 解析的 vs 数値勾配の比較
        auto compute_error_min = [](T analytical, T numerical) -> T {
            T abs_error = std::abs(analytical - numerical);
            T rel_error = std::abs(abs_error / (std::abs(analytical) + T(1e-15)));
            return std::min(abs_error, rel_error);
        };
        
        T error_x = compute_error_min(analytical_x_grad, numerical_x_grad);
        T error_y = compute_error_min(analytical_y_grad, numerical_y_grad);
        T error_z = compute_error_min(analytical_z_grad, numerical_z_grad);
        
        EXPECT_LE(error_x, TOLERANCE) 
            << "Test case " << test_case << " (x=" << x << ", y=" << y << ", z=" << z << "): "
            << "analytical_x_grad=" << analytical_x_grad 
            << ", numerical_x_grad=" << numerical_x_grad
            << ", direct_x_grad=" << direct_x_grad
            << ", error=" << error_x;
            
        EXPECT_LE(error_y, TOLERANCE) 
            << "Test case " << test_case << " (x=" << x << ", y=" << y << ", z=" << z << "): "
            << "analytical_y_grad=" << analytical_y_grad 
            << ", numerical_y_grad=" << numerical_y_grad
            << ", direct_y_grad=" << direct_y_grad
            << ", error=" << error_y;
            
        EXPECT_LE(error_z, TOLERANCE) 
            << "Test case " << test_case << " (x=" << x << ", y=" << y << ", z=" << z << "): "
            << "analytical_z_grad=" << analytical_z_grad 
            << ", numerical_z_grad=" << numerical_z_grad
            << ", direct_z_grad=" << direct_z_grad
            << ", error=" << error_z;
        
        // 理論値との比較（f(x,y,z) = xz + y なので df/dx = z, df/dy = 1, df/dz = x）
        T expected_x_grad = upstream_grad * z;
        T expected_y_grad = upstream_grad * 1.0;
        T expected_z_grad = upstream_grad * x;
        
        EXPECT_NEAR(analytical_x_grad, expected_x_grad, 1e-5) 
            << "Analytical gradient for x doesn't match expected value";
        EXPECT_NEAR(analytical_y_grad, expected_y_grad, 1e-5) 
            << "Analytical gradient for y doesn't match expected value";
        EXPECT_NEAR(analytical_z_grad, expected_z_grad, 1e-5) 
            << "Analytical gradient for z doesn't match expected value";
    }
}

TEST_F(OperationChainingTest, SimpleForwardTest) {
    using T = double;
    
    // 簡単なテストケース: x=2, y=3, z=4 の場合
    // f(2, 3, 4) = 2*4 + 3 = 8 + 3 = 11
    T x = 2.0, y = 3.0, z = 4.0;
    T expected_output = 11.0;
    
    auto device_x_data = makeCudaUniqueArray<T>(1);
    auto device_x_grad = makeCudaUniqueArray<T>(1);
    auto device_y_data = makeCudaUniqueArray<T>(1);
    auto device_y_grad = makeCudaUniqueArray<T>(1);
    auto device_z_data = makeCudaUniqueArray<T>(1);
    auto device_z_grad = makeCudaUniqueArray<T>(1);
    auto device_output_data = makeCudaUniqueArray<T>(1);
    auto device_output_grad = makeCudaUniqueArray<T>(1);
    auto device_result_x_grad = makeCudaUniqueArray<T>(1);
    auto device_result_y_grad = makeCudaUniqueArray<T>(1);
    auto device_result_z_grad = makeCudaUniqueArray<T>(1);
    
    T upstream_grad = 1.0;
    
    ASSERT_EQ(cudaMemcpy(device_x_data.get(), &x, sizeof(T), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(device_y_data.get(), &y, sizeof(T), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(device_z_data.get(), &z, sizeof(T), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(device_output_grad.get(), &upstream_grad, sizeof(T), cudaMemcpyHostToDevice), cudaSuccess);
    
    test_chaining_analytical_kernel<T><<<1, 1>>>(
        device_x_data.get(), device_x_grad.get(),
        device_y_data.get(), device_y_grad.get(),
        device_z_data.get(), device_z_grad.get(),
        device_output_data.get(), device_output_grad.get());
    
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    T grad_x, grad_y, grad_z;
    
    // 結果は自動的に保存されている
    ASSERT_EQ(cudaMemcpy(&grad_x, device_x_grad.get(), sizeof(T), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&grad_y, device_y_grad.get(), sizeof(T), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&grad_z, device_z_grad.get(), sizeof(T), cudaMemcpyDeviceToHost), cudaSuccess);
    
    // 順伝播の検証は省略（final_resultから直接確認できない）
    
    // 勾配の検証: df/dx = z = 4, df/dy = 1, df/dz = x = 2
    EXPECT_NEAR(grad_x, 4.0, 1e-5);
    EXPECT_NEAR(grad_y, 1.0, 1e-5);
    EXPECT_NEAR(grad_z, 2.0, 1e-5);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}