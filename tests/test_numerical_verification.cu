#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <array>
#include <random>
#include "../include/operations/sigmoid_logic.cuh"
#include "../include/operations/exp_logic.cuh"
#include "../include/operations/add_logic.cuh"
#include "../include/variable.cuh"
#include "../include/util/cuda_unique_ptr.cuh"

using namespace xyz_autodiff;

// テスト用バッファ構造体
template <typename T, std::size_t NumElements>
class TestNumericalBuffer {
public:
    std::array<T, NumElements> host_data;
    std::array<bool, NumElements> host_result;
    cuda_unique_ptr<T[]> device_data;
    cuda_unique_ptr<bool[]> device_result;
    
    TestNumericalBuffer() {
        host_data.fill(T{});
        host_result.fill(false);
        device_data = makeCudaUniqueArray<T>(NumElements);
        device_result = makeCudaUniqueArray<bool>(NumElements);
    }
    
    void toGpu() {
        cudaMemcpy(device_data.get(), host_data.data(), NumElements * sizeof(T), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
    }
    
    void toHost() {
        cudaMemcpy(host_result.data(), device_result.get(), NumElements * sizeof(bool), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
    }
    
    void setData(std::size_t idx, T value) {
        if (idx < NumElements) {
            host_data[idx] = value;
        }
    }
    
    bool getResult(std::size_t idx) const {
        return idx < NumElements ? host_result[idx] : false;
    }
    
    T* getDeviceData() { return device_data.get(); }
    bool* getDeviceResult() { return device_result.get(); }
};

// SigmoidLogic数値微分テスト用カーネル
template <typename T>
__global__ void test_sigmoid_numerical_kernel(T* analytical_grad, T* numerical_grad, T* input_values, int num_tests) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id >= num_tests) return;
    
    SigmoidLogic<1> logic;
    
    T input_val = input_values[thread_id];
    T input_grad_data[1] = {T{0}};
    T output_data[1];
    T output_grad_data[1] = {T{1}};  // 単位勾配
    
    Variable<T, 1> input(&input_val, input_grad_data);
    Variable<T, 1> output(output_data, output_grad_data);
    
    // 解析的勾配
    logic.forward(output, input);
    logic.backward(output, input);
    analytical_grad[thread_id] = input.grad(0);
    
    // 数値勾配
    constexpr T delta = static_cast<T>(1e-8);
    T original = input_val;
    
    T input_plus_val = original + delta;
    T input_plus_grad[1] = {T{0}};
    Variable<T, 1> input_plus(&input_plus_val, input_plus_grad);
    Variable<T, 1> output_plus(output_data, output_grad_data);
    logic.forward(output_plus, input_plus);
    T f_plus = output_plus[0];
    
    T input_minus_val = original - delta;
    T input_minus_grad[1] = {T{0}};
    Variable<T, 1> input_minus(&input_minus_val, input_minus_grad);
    Variable<T, 1> output_minus(output_data, output_grad_data);
    logic.forward(output_minus, input_minus);
    T f_minus = output_minus[0];
    
    numerical_grad[thread_id] = (f_plus - f_minus) / (2 * delta);
}

// ExpLogic数値微分テスト用カーネル  
template <typename T>
__global__ void test_exp_numerical_kernel(T* analytical_grad, T* numerical_grad, T* input_values, int num_tests) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id >= num_tests) return;
    
    ExpLogic<1> logic;
    
    T input_val = input_values[thread_id];
    T input_grad_data[1] = {T{0}};
    T output_data[1];
    T output_grad_data[1] = {T{1}};
    
    Variable<T, 1> input(&input_val, input_grad_data);
    Variable<T, 1> output(output_data, output_grad_data);
    
    // 解析的勾配
    logic.forward(output, input);
    logic.backward(output, input);
    analytical_grad[thread_id] = input.grad(0);
    
    // 数値勾配
    constexpr T delta = static_cast<T>(1e-8);
    T original = input_val;
    
    T input_plus_val = original + delta;
    T input_plus_grad[1] = {T{0}};
    Variable<T, 1> input_plus(&input_plus_val, input_plus_grad);
    Variable<T, 1> output_plus(output_data, output_grad_data);
    logic.forward(output_plus, input_plus);
    T f_plus = output_plus[0];
    
    T input_minus_val = original - delta;
    T input_minus_grad[1] = {T{0}};
    Variable<T, 1> input_minus(&input_minus_val, input_minus_grad);
    Variable<T, 1> output_minus(output_data, output_grad_data);
    logic.forward(output_minus, input_minus);
    T f_minus = output_minus[0];
    
    numerical_grad[thread_id] = (f_plus - f_minus) / (2 * delta);
}

// AddLogic数値微分テスト用カーネル
template <typename T>
__global__ void test_add_numerical_kernel(T* analytical_grad1, T* analytical_grad2, T* numerical_grad1, T* numerical_grad2, 
                                         T* input1_values, T* input2_values, int num_tests) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id >= num_tests) return;
    
    using TestInput = Variable<T, 1>;
    xyz_autodiff::op::AddLogic<TestInput, TestInput> logic;
    
    T input1_val = input1_values[thread_id];
    T input2_val = input2_values[thread_id];
    T input1_grad_data[1] = {T{0}};
    T input2_grad_data[1] = {T{0}};
    T output_data[1];
    T output_grad_data[1] = {T{1}};
    
    Variable<T, 1> input1(&input1_val, input1_grad_data);
    Variable<T, 1> input2(&input2_val, input2_grad_data);
    Variable<T, 1> output(output_data, output_grad_data);
    
    // 解析的勾配
    logic.forward(output, input1, input2);
    logic.backward(output, input1, input2);
    analytical_grad1[thread_id] = input1.grad(0);
    analytical_grad2[thread_id] = input2.grad(0);
    
    // 数値勾配 - input1について
    constexpr T delta = static_cast<T>(1e-8);
    T original1 = input1_val;
    T original2 = input2_val;
    
    T input1_plus_val = original1 + delta;
    T input1_plus_grad[1] = {T{0}};
    Variable<T, 1> input1_plus(&input1_plus_val, input1_plus_grad);
    Variable<T, 1> output_plus(output_data, output_grad_data);
    logic.forward(output_plus, input1_plus, input2);
    T f_plus = output_plus[0];
    
    T input1_minus_val = original1 - delta;
    T input1_minus_grad[1] = {T{0}};
    Variable<T, 1> input1_minus(&input1_minus_val, input1_minus_grad);
    Variable<T, 1> output_minus(output_data, output_grad_data);
    logic.forward(output_minus, input1_minus, input2);
    T f_minus = output_minus[0];
    
    numerical_grad1[thread_id] = (f_plus - f_minus) / (2 * delta);
    
    // 数値勾配 - input2について
    T input2_plus_val = original2 + delta;
    T input2_plus_grad[1] = {T{0}};
    Variable<T, 1> input2_plus(&input2_plus_val, input2_plus_grad);
    logic.forward(output_plus, input1, input2_plus);
    f_plus = output_plus[0];
    
    T input2_minus_val = original2 - delta;
    T input2_minus_grad[1] = {T{0}};
    Variable<T, 1> input2_minus(&input2_minus_val, input2_minus_grad);
    logic.forward(output_minus, input1, input2_minus);
    f_minus = output_minus[0];
    
    numerical_grad2[thread_id] = (f_plus - f_minus) / (2 * delta);
}

// 手動でのシンプルな検証用カーネル（デバッグ用）
template <typename T>
__global__ void test_simple_sigmoid_kernel(T* input_data, T* analytical_grad, T* numerical_grad) {
    SigmoidLogic<1> logic;
    
    T input_val = input_data[0];
    T input_grad_data[1] = {T{0}};
    T output_data[1];
    T output_grad_data[1] = {T{1}};  // 単位勾配
    
    Variable<T, 1> input(&input_val, input_grad_data);
    Variable<T, 1> output(output_data, output_grad_data);
    
    // 解析的勾配
    logic.forward(output, input);
    logic.backward(output, input);
    analytical_grad[0] = input.grad(0);
    
    // 数値勾配
    constexpr T delta = static_cast<T>(1e-8);
    T original = input_val;
    
    T input_plus_val = original + delta;
    T input_plus_grad[1] = {T{0}};
    Variable<T, 1> input_plus(&input_plus_val, input_plus_grad);
    Variable<T, 1> output_plus(output_data, output_grad_data);
    logic.forward(output_plus, input_plus);
    T f_plus = output_plus[0];
    
    T input_minus_val = original - delta;
    T input_minus_grad[1] = {T{0}};
    Variable<T, 1> input_minus(&input_minus_val, input_minus_grad);
    Variable<T, 1> output_minus(output_data, output_grad_data);
    logic.forward(output_minus, input_minus);
    T f_minus = output_minus[0];
    
    numerical_grad[0] = (f_plus - f_minus) / (2 * delta);
}

class NumericalVerificationTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaError_t err = cudaSetDevice(0);
        if (err != cudaSuccess) {
            GTEST_SKIP() << "CUDA device not available";
        }
    }
};

TEST_F(NumericalVerificationTest, SimpleSigmoidVerification) {
    using T = double;
    
    auto device_input = makeCudaUniqueArray<T>(1);
    auto device_analytical = makeCudaUniqueArray<T>(1);
    auto device_numerical = makeCudaUniqueArray<T>(1);
    
    // テスト値
    T test_input = 0.5;
    cudaMemcpy(device_input.get(), &test_input, sizeof(T), cudaMemcpyHostToDevice);
    
    test_simple_sigmoid_kernel<T><<<1, 1>>>(
        device_input.get(),
        device_analytical.get(),
        device_numerical.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    T analytical_result, numerical_result;
    cudaMemcpy(&analytical_result, device_analytical.get(), sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(&numerical_result, device_numerical.get(), sizeof(T), cudaMemcpyDeviceToHost);
    
    // 相対誤差の確認
    double rel_error = std::abs(analytical_result - numerical_result) / 
                      (std::abs(analytical_result) + 1e-12);
    
    EXPECT_LT(rel_error, 1e-6) << "Analytical: " << analytical_result 
                              << ", Numerical: " << numerical_result 
                              << ", Relative error: " << rel_error;
}

TEST_F(NumericalVerificationTest, SigmoidLogicRandomTests) {
    using T = double;
    
    constexpr int num_tests = 100;
    
    // ランダム入力値を生成
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> dis(-2.0, 2.0);  // sigmoid用の適当な範囲
    
    std::vector<T> input_values(num_tests);
    for (int i = 0; i < num_tests; ++i) {
        input_values[i] = dis(gen);
    }
    
    auto device_inputs = makeCudaUniqueArray<T>(num_tests);
    auto device_analytical = makeCudaUniqueArray<T>(num_tests);
    auto device_numerical = makeCudaUniqueArray<T>(num_tests);
    
    cudaMemcpy(device_inputs.get(), input_values.data(), num_tests * sizeof(T), cudaMemcpyHostToDevice);
    
    test_sigmoid_numerical_kernel<T><<<(num_tests + 31) / 32, 32>>>(
        device_analytical.get(), device_numerical.get(), device_inputs.get(), num_tests);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    std::vector<T> analytical_results(num_tests);
    std::vector<T> numerical_results(num_tests);
    cudaMemcpy(analytical_results.data(), device_analytical.get(), num_tests * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(numerical_results.data(), device_numerical.get(), num_tests * sizeof(T), cudaMemcpyDeviceToHost);
    
    // 相対誤差をチェック
    int passed_tests = 0;
    for (int i = 0; i < num_tests; ++i) {
        double rel_error = std::abs(analytical_results[i] - numerical_results[i]) / 
                          (std::abs(analytical_results[i]) + 1e-12);
        if (rel_error < 1e-6) {
            passed_tests++;
        }
    }
    
    EXPECT_GE(passed_tests, static_cast<int>(num_tests * 0.95)) 
        << "Only " << passed_tests << "/" << num_tests << " tests passed";
}

TEST_F(NumericalVerificationTest, ExpLogicRandomTests) {
    using T = double;
    
    constexpr int num_tests = 100;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> dis(-1.0, 1.0);  // exp用の範囲
    
    std::vector<T> input_values(num_tests);
    for (int i = 0; i < num_tests; ++i) {
        input_values[i] = dis(gen);
    }
    
    auto device_inputs = makeCudaUniqueArray<T>(num_tests);
    auto device_analytical = makeCudaUniqueArray<T>(num_tests);
    auto device_numerical = makeCudaUniqueArray<T>(num_tests);
    
    cudaMemcpy(device_inputs.get(), input_values.data(), num_tests * sizeof(T), cudaMemcpyHostToDevice);
    
    test_exp_numerical_kernel<T><<<(num_tests + 31) / 32, 32>>>(
        device_analytical.get(), device_numerical.get(), device_inputs.get(), num_tests);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    std::vector<T> analytical_results(num_tests);
    std::vector<T> numerical_results(num_tests);
    cudaMemcpy(analytical_results.data(), device_analytical.get(), num_tests * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(numerical_results.data(), device_numerical.get(), num_tests * sizeof(T), cudaMemcpyDeviceToHost);
    
    int passed_tests = 0;
    for (int i = 0; i < num_tests; ++i) {
        double rel_error = std::abs(analytical_results[i] - numerical_results[i]) / 
                          (std::abs(analytical_results[i]) + 1e-12);
        if (rel_error < 1e-6) {
            passed_tests++;
        }
    }
    
    EXPECT_GE(passed_tests, static_cast<int>(num_tests * 0.95))
        << "Only " << passed_tests << "/" << num_tests << " tests passed";
}

TEST_F(NumericalVerificationTest, AddLogicRandomTests) {
    using T = double;
    
    constexpr int num_tests = 100;
    
    // ランダム入力値を生成
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> dis(-2.0, 2.0);
    
    std::vector<T> input1_values(num_tests);
    std::vector<T> input2_values(num_tests);
    for (int i = 0; i < num_tests; ++i) {
        input1_values[i] = dis(gen);
        input2_values[i] = dis(gen);
    }
    
    auto device_input1 = makeCudaUniqueArray<T>(num_tests);
    auto device_input2 = makeCudaUniqueArray<T>(num_tests);
    auto device_analytical1 = makeCudaUniqueArray<T>(num_tests);
    auto device_analytical2 = makeCudaUniqueArray<T>(num_tests);
    auto device_numerical1 = makeCudaUniqueArray<T>(num_tests);
    auto device_numerical2 = makeCudaUniqueArray<T>(num_tests);
    
    cudaMemcpy(device_input1.get(), input1_values.data(), num_tests * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(device_input2.get(), input2_values.data(), num_tests * sizeof(T), cudaMemcpyHostToDevice);
    
    test_add_numerical_kernel<T><<<(num_tests + 31) / 32, 32>>>(
        device_analytical1.get(), device_analytical2.get(), 
        device_numerical1.get(), device_numerical2.get(),
        device_input1.get(), device_input2.get(), num_tests);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    std::vector<T> analytical1_results(num_tests);
    std::vector<T> analytical2_results(num_tests);
    std::vector<T> numerical1_results(num_tests);
    std::vector<T> numerical2_results(num_tests);
    
    cudaMemcpy(analytical1_results.data(), device_analytical1.get(), num_tests * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(analytical2_results.data(), device_analytical2.get(), num_tests * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(numerical1_results.data(), device_numerical1.get(), num_tests * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(numerical2_results.data(), device_numerical2.get(), num_tests * sizeof(T), cudaMemcpyDeviceToHost);
    
    int passed_tests = 0;
    for (int i = 0; i < num_tests; ++i) {
        double rel_error1 = std::abs(analytical1_results[i] - numerical1_results[i]) / 
                           (std::abs(analytical1_results[i]) + 1e-12);
        double rel_error2 = std::abs(analytical2_results[i] - numerical2_results[i]) / 
                           (std::abs(analytical2_results[i]) + 1e-12);
        
        if (rel_error1 < 1e-6 && rel_error2 < 1e-6) {
            passed_tests++;
        }
    }
    
    // AddLogicは簡単なので100%成功を期待
    EXPECT_EQ(passed_tests, static_cast<int>(num_tests))
        << "Only " << passed_tests << "/" << num_tests << " tests passed";
}

// 大規模なランダムテスト（ストレステスト）
TEST_F(NumericalVerificationTest, LargeScaleRandomTests) {
    using T = double;
    
    constexpr int num_tests = 1000;
    
    // Sigmoid大規模テスト
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<T> dis(-2.0, 2.0);
        
        std::vector<T> input_values(num_tests);
        for (int i = 0; i < num_tests; ++i) {
            input_values[i] = dis(gen);
        }
        
        auto device_inputs = makeCudaUniqueArray<T>(num_tests);
        auto device_analytical = makeCudaUniqueArray<T>(num_tests);
        auto device_numerical = makeCudaUniqueArray<T>(num_tests);
        
        cudaMemcpy(device_inputs.get(), input_values.data(), num_tests * sizeof(T), cudaMemcpyHostToDevice);
        
        test_sigmoid_numerical_kernel<T><<<(num_tests + 255) / 256, 256>>>(
            device_analytical.get(), device_numerical.get(), device_inputs.get(), num_tests);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        
        std::vector<T> analytical_results(num_tests);
        std::vector<T> numerical_results(num_tests);
        cudaMemcpy(analytical_results.data(), device_analytical.get(), num_tests * sizeof(T), cudaMemcpyDeviceToHost);
        cudaMemcpy(numerical_results.data(), device_numerical.get(), num_tests * sizeof(T), cudaMemcpyDeviceToHost);
        
        int passed = 0;
        for (int i = 0; i < num_tests; ++i) {
            double rel_error = std::abs(analytical_results[i] - numerical_results[i]) / 
                              (std::abs(analytical_results[i]) + 1e-12);
            if (rel_error < 1e-6) {
                passed++;
            }
        }
        EXPECT_GE(passed, static_cast<int>(num_tests * 0.95))
            << "Sigmoid large scale: " << passed << "/" << num_tests << " passed";
    }
    
    // Exp大規模テスト
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<T> dis(-1.0, 1.0);
        
        std::vector<T> input_values(num_tests);
        for (int i = 0; i < num_tests; ++i) {
            input_values[i] = dis(gen);
        }
        
        auto device_inputs = makeCudaUniqueArray<T>(num_tests);
        auto device_analytical = makeCudaUniqueArray<T>(num_tests);
        auto device_numerical = makeCudaUniqueArray<T>(num_tests);
        
        cudaMemcpy(device_inputs.get(), input_values.data(), num_tests * sizeof(T), cudaMemcpyHostToDevice);
        
        test_exp_numerical_kernel<T><<<(num_tests + 255) / 256, 256>>>(
            device_analytical.get(), device_numerical.get(), device_inputs.get(), num_tests);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        
        std::vector<T> analytical_results(num_tests);
        std::vector<T> numerical_results(num_tests);
        cudaMemcpy(analytical_results.data(), device_analytical.get(), num_tests * sizeof(T), cudaMemcpyDeviceToHost);
        cudaMemcpy(numerical_results.data(), device_numerical.get(), num_tests * sizeof(T), cudaMemcpyDeviceToHost);
        
        int passed = 0;
        for (int i = 0; i < num_tests; ++i) {
            double rel_error = std::abs(analytical_results[i] - numerical_results[i]) / 
                              (std::abs(analytical_results[i]) + 1e-12);
            if (rel_error < 1e-6) {
                passed++;
            }
        }
        EXPECT_GE(passed, static_cast<int>(num_tests * 0.95))
            << "Exp large scale: " << passed << "/" << num_tests << " passed";
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}