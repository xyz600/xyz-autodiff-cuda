#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <array>
#include <cmath>
#include "../include/operations/unary_functions.cuh"
#include "../include/variable.cuh"
#include "../include/util/cuda_unique_ptr.cuh"

using namespace xyz_autodiff;

// テスト用汎用バッファ構造体
template <typename T, std::size_t NumElements>
class TestUnaryBuffer {
public:
    std::array<T, NumElements> host_data;
    std::array<T, NumElements> host_result;
    cuda_unique_ptr<T[]> device_data;
    cuda_unique_ptr<T[]> device_result;
    
    TestUnaryBuffer() {
        host_data.fill(T{});
        host_result.fill(T{});
        device_data = makeCudaUniqueArray<T>(NumElements);
        device_result = makeCudaUniqueArray<T>(NumElements);
    }
    
    void toGpu() {
        cudaMemcpy(device_data.get(), host_data.data(), NumElements * sizeof(T), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
    }
    
    void toHost() {
        cudaMemcpy(host_result.data(), device_result.get(), NumElements * sizeof(T), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
    }
    
    void setData(std::size_t idx, T value) {
        if (idx < NumElements) {
            host_data[idx] = value;
        }
    }
    
    T getResult(std::size_t idx) const {
        return idx < NumElements ? host_result[idx] : T{};
    }
    
    T* getDeviceData() { return device_data.get(); }
    T* getDeviceResult() { return device_result.get(); }
};

// Sigmoid関数テスト用CUDAカーネル
template <typename T>
__global__ void test_sigmoid_kernel(T* data, T* grad, T* result) {
    // 3要素のVariableを作成
    Variable<T, 3> var(data, grad);
    
    // 入力値設定: [-2, 0, 2]
    var[0] = static_cast<T>(-2.0);
    var[1] = static_cast<T>(0.0);
    var[2] = static_cast<T>(2.0);
    
    // sigmoid関数適用
    auto sigmoid_result = sigmoid(var);
    
    // 結果保存
    result[0] = sigmoid_result[0];  // sigmoid(-2)
    result[1] = sigmoid_result[1];  // sigmoid(0) = 0.5
    result[2] = sigmoid_result[2];  // sigmoid(2)
    
    // 勾配テスト用の上流勾配設定
    sigmoid_result.grad(0) = static_cast<T>(1.0);
    sigmoid_result.grad(1) = static_cast<T>(1.0);
    sigmoid_result.grad(2) = static_cast<T>(1.0);
    
    // backward pass
    sigmoid_result.backward();
    
    // 入力の勾配を結果に保存
    result[3] = var.grad(0);  // d/dx sigmoid(-2)
    result[4] = var.grad(1);  // d/dx sigmoid(0) = 0.25
    result[5] = var.grad(2);  // d/dx sigmoid(2)
}

// Exponential関数テスト用CUDAカーネル
template <typename T>
__global__ void test_exp_kernel(T* data, T* grad, T* result) {
    // 3要素のVariableを作成
    Variable<T, 3> var(data, grad);
    
    // 入力値設定: [0, 1, 2]
    var[0] = static_cast<T>(0.0);
    var[1] = static_cast<T>(1.0);
    var[2] = static_cast<T>(2.0);
    
    // exp関数適用
    auto exp_result = exp(var);
    
    // 結果保存
    result[0] = exp_result[0];  // exp(0) = 1
    result[1] = exp_result[1];  // exp(1) ≈ 2.718
    result[2] = exp_result[2];  // exp(2) ≈ 7.389
    
    // 勾配テスト用の上流勾配設定
    exp_result.grad(0) = static_cast<T>(1.0);
    exp_result.grad(1) = static_cast<T>(1.0);
    exp_result.grad(2) = static_cast<T>(1.0);
    
    // backward pass
    exp_result.backward();
    
    // 入力の勾配を結果に保存
    result[3] = var.grad(0);  // d/dx exp(0) = 1
    result[4] = var.grad(1);  // d/dx exp(1) ≈ 2.718
    result[5] = var.grad(2);  // d/dx exp(2) ≈ 7.389
}

// 明示的な次元指定テスト用CUDAカーネル
template <typename T>
__global__ void test_explicit_dim_kernel(T* data, T* grad, T* result) {
    // 2要素のVariableを作成
    Variable<T, 2> var(data, grad);
    
    // 入力値設定
    var[0] = static_cast<T>(1.0);
    var[1] = static_cast<T>(-1.0);
    
    // 明示的に次元を指定してsigmoid適用
    auto sigmoid_result = sigmoid<2>(var);
    
    // 結果保存
    result[0] = sigmoid_result[0];  // sigmoid(1)
    result[1] = sigmoid_result[1];  // sigmoid(-1)
    
    // 明示的に次元を指定してexp適用
    auto exp_result = exp<2>(var);
    
    // 結果保存
    result[2] = exp_result[0];  // exp(1)
    result[3] = exp_result[1];  // exp(-1)
}

class UnaryFunctionsTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaError_t err = cudaSetDevice(0);
        if (err != cudaSuccess) {
            GTEST_SKIP() << "CUDA device not available";
        }
    }
};

TEST_F(UnaryFunctionsTest, SigmoidFunction) {
    using T = float;
    
    TestUnaryBuffer<T, 16> buffer;  // 3要素データ + 勾配 + 結果6個
    buffer.toGpu();
    
    test_sigmoid_kernel<T><<<1, 1>>>(
        buffer.getDeviceData(),
        buffer.getDeviceData() + 3,  // 勾配用
        buffer.getDeviceResult());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    buffer.toHost();
    
    // sigmoid値の検証
    // sigmoid(-2) ≈ 0.119
    EXPECT_NEAR(buffer.getResult(0), 0.119f, 0.001f);
    // sigmoid(0) = 0.5
    EXPECT_FLOAT_EQ(buffer.getResult(1), 0.5f);
    // sigmoid(2) ≈ 0.881
    EXPECT_NEAR(buffer.getResult(2), 0.881f, 0.001f);
    
    // sigmoid勾配の検証: sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
    // sigmoid'(-2) ≈ 0.119 * 0.881 ≈ 0.105
    EXPECT_NEAR(buffer.getResult(3), 0.105f, 0.001f);
    // sigmoid'(0) = 0.5 * 0.5 = 0.25
    EXPECT_FLOAT_EQ(buffer.getResult(4), 0.25f);
    // sigmoid'(2) ≈ 0.881 * 0.119 ≈ 0.105
    EXPECT_NEAR(buffer.getResult(5), 0.105f, 0.001f);
}

TEST_F(UnaryFunctionsTest, ExpFunction) {
    using T = float;
    
    TestUnaryBuffer<T, 16> buffer;  // 3要素データ + 勾配 + 結果6個
    buffer.toGpu();
    
    test_exp_kernel<T><<<1, 1>>>(
        buffer.getDeviceData(),
        buffer.getDeviceData() + 3,  // 勾配用
        buffer.getDeviceResult());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    buffer.toHost();
    
    // exp値の検証
    // exp(0) = 1
    EXPECT_FLOAT_EQ(buffer.getResult(0), 1.0f);
    // exp(1) ≈ 2.718
    EXPECT_NEAR(buffer.getResult(1), 2.718f, 0.001f);
    // exp(2) ≈ 7.389
    EXPECT_NEAR(buffer.getResult(2), 7.389f, 0.001f);
    
    // exp勾配の検証: exp'(x) = exp(x)
    // exp'(0) = 1
    EXPECT_FLOAT_EQ(buffer.getResult(3), 1.0f);
    // exp'(1) ≈ 2.718
    EXPECT_NEAR(buffer.getResult(4), 2.718f, 0.001f);
    // exp'(2) ≈ 7.389
    EXPECT_NEAR(buffer.getResult(5), 7.389f, 0.001f);
}

TEST_F(UnaryFunctionsTest, ExplicitDimensionSpecification) {
    using T = float;
    
    TestUnaryBuffer<T, 16> buffer;
    buffer.toGpu();
    
    test_explicit_dim_kernel<T><<<1, 1>>>(
        buffer.getDeviceData(),
        buffer.getDeviceData() + 2,  // 勾配用
        buffer.getDeviceResult());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    buffer.toHost();
    
    // 明示的次元指定でのsigmoid検証
    // sigmoid(1) ≈ 0.731
    EXPECT_NEAR(buffer.getResult(0), 0.731f, 0.001f);
    // sigmoid(-1) ≈ 0.269
    EXPECT_NEAR(buffer.getResult(1), 0.269f, 0.001f);
    
    // 明示的次元指定でのexp検証
    // exp(1) ≈ 2.718
    EXPECT_NEAR(buffer.getResult(2), 2.718f, 0.001f);
    // exp(-1) ≈ 0.368
    EXPECT_NEAR(buffer.getResult(3), 0.368f, 0.001f);
}

// Concept チェックテスト
TEST_F(UnaryFunctionsTest, ConceptCheck) {
    // SigmoidLogicとExpLogicが適切に定義されていることを確認
    static_assert(SigmoidLogic<3>::outputDim == 3);
    static_assert(ExpLogic<2>::outputDim == 2);
    
    // 基本的なサイズ計算
    constexpr std::size_t sigmoid_output_size = SigmoidLogic<4>::outputDim;
    constexpr std::size_t exp_output_size = ExpLogic<5>::outputDim;
    static_assert(sigmoid_output_size == 4);
    static_assert(exp_output_size == 5);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}