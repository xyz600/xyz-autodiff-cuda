#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <random>
#include <vector>
#include <cmath>
#include <xyz_autodiff/variable.cuh>
#include <xyz_autodiff/operations/binary/add_logic.cuh>
#include <xyz_autodiff/operations/binary/mul_logic.cuh>
#include <xyz_autodiff/util/cuda_unique_ptr.cuh>
#include <xyz_autodiff/variable_operators.cuh>

using namespace xyz_autodiff;

// パラメータ管理用の構造体
template <typename T>
struct TestParameters {
    T value[3];  // x, y, z
    T grad[3];   // grad_x, grad_y, grad_z
};

// 全体のテストバッファ構造体（単一メモリ確保用）
template <typename T>
struct OperationChainingBuffers {
    TestParameters<T> params;
    T output_data[1];
    T output_grad[1];
    T direct_x_grad[1];
    T direct_y_grad[1];
    T direct_z_grad[1];
};

// f(x, y, z) = xz + y を計算するカーネル（解析的微分）
template <typename T>
__global__ void test_chaining_analytical_kernel(
    TestParameters<T>* params, T* output_data, T* output_grad) {
    
    // Variable作成（ポインタ演算で各パラメータを指定）
    VariableRef<1, T> x_var(&params->value[0], &params->grad[0]);  // x
    VariableRef<1, T> y_var(&params->value[1], &params->grad[1]);  // y
    VariableRef<1, T> z_var(&params->value[2], &params->grad[2]);  // z
    
    // f(x, y, z) = xz + y の計算
    auto mul_result = op::mul(x_var, z_var);
    auto final_result = op::add(mul_result, y_var);
    
    final_result.forward();
    final_result.zero_grad();
    final_result.add_grad(0, output_grad[0]); // 指定された上流勾配を使用
    final_result.backward();
}

// f(x, y, z) = xz + y を数値微分で計算するカーネル
template <typename T>
__global__ void test_chaining_numerical_kernel(
    TestParameters<T>* params, T* output_data, T* output_grad, T delta) {
    
    // Variable作成（ポインタ演算で各パラメータを指定）
    VariableRef<1, T> x_var(&params->value[0], &params->grad[0]);  // x
    VariableRef<1, T> y_var(&params->value[1], &params->grad[1]);  // y
    VariableRef<1, T> z_var(&params->value[2], &params->grad[2]);  // z
    
    // f(x, y, z) = xz + y の計算
    auto mul_result = op::mul(x_var, z_var);
    auto final_result = op::add(mul_result, y_var);
    
    final_result.forward();
    final_result.zero_grad();
    final_result.add_grad(0, output_grad[0]); // 指定された上流勾配を使用
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
    
    // デバイスメモリ確保（単一確保）
    auto device_buffers = makeCudaUnique<OperationChainingBuffers<T>>();
    
    for (int test_case = 0; test_case < NUM_TESTS; ++test_case) {
        // ランダム入力生成
        T x = dist(gen);
        T y = dist(gen);
        T z = dist(gen);
        T upstream_grad = dist(gen);
        
        // バッファ構造体にデータをセット
        OperationChainingBuffers<T> host_buffers = {};
        host_buffers.params.value[0] = x;
        host_buffers.params.value[1] = y;
        host_buffers.params.value[2] = z;
        host_buffers.output_grad[0] = upstream_grad;
        
        ASSERT_EQ(cudaMemcpy(device_buffers.get(), &host_buffers, sizeof(OperationChainingBuffers<T>), cudaMemcpyHostToDevice), cudaSuccess);
        
        // 解析的勾配計算
        test_chaining_analytical_kernel<T><<<1, 1>>>(
            &device_buffers.get()->params, device_buffers.get()->output_data, device_buffers.get()->output_grad);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        
        // 数値勾配計算用にパラメータをリセット
        host_buffers.params.grad[0] = host_buffers.params.grad[1] = host_buffers.params.grad[2] = 0;
        ASSERT_EQ(cudaMemcpy(device_buffers.get(), &host_buffers, sizeof(OperationChainingBuffers<T>), cudaMemcpyHostToDevice), cudaSuccess);
        
        test_chaining_numerical_kernel<T><<<1, 1>>>(
            &device_buffers.get()->params, device_buffers.get()->output_data, device_buffers.get()->output_grad,
            static_cast<T>(DELTA));
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        
        // 直接数値微分計算（検証用）
        compute_direct_numerical_gradient_kernel<T><<<1, 1>>>(
            x, y, z, device_buffers.get()->direct_x_grad, device_buffers.get()->direct_y_grad, device_buffers.get()->direct_z_grad,
            static_cast<T>(DELTA), upstream_grad);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        
        // 解析的勾配結果をホストにコピー
        OperationChainingBuffers<T> analytical_result;
        ASSERT_EQ(cudaMemcpy(&analytical_result, device_buffers.get(), sizeof(OperationChainingBuffers<T>), cudaMemcpyDeviceToHost), cudaSuccess);
        
        T analytical_x_grad = analytical_result.params.grad[0];
        T analytical_y_grad = analytical_result.params.grad[1];
        T analytical_z_grad = analytical_result.params.grad[2];
        
        // 数値微分のテストのためにパラメータをリセット
        host_buffers.params.grad[0] = host_buffers.params.grad[1] = host_buffers.params.grad[2] = 0;
        ASSERT_EQ(cudaMemcpy(device_buffers.get(), &host_buffers, sizeof(OperationChainingBuffers<T>), cudaMemcpyHostToDevice), cudaSuccess);
        
        // 数値勾配計算（operation chaining経由）
        test_chaining_numerical_kernel<T><<<1, 1>>>(
            &device_buffers.get()->params, device_buffers.get()->output_data, device_buffers.get()->output_grad,
            static_cast<T>(DELTA));
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        
        // 数値勾配結果をホストにコピー
        OperationChainingBuffers<T> numerical_result;
        ASSERT_EQ(cudaMemcpy(&numerical_result, device_buffers.get(), sizeof(OperationChainingBuffers<T>), cudaMemcpyDeviceToHost), cudaSuccess);
        
        T numerical_x_grad = numerical_result.params.grad[0];
        T numerical_y_grad = numerical_result.params.grad[1];
        T numerical_z_grad = numerical_result.params.grad[2];
        
        T direct_x_grad = analytical_result.direct_x_grad[0];
        T direct_y_grad = analytical_result.direct_y_grad[0];
        T direct_z_grad = analytical_result.direct_z_grad[0];
        
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

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}