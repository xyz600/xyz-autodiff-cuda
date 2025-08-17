#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <type_traits>
#include "../include/variable.cuh"
#include "../include/operation.cuh"
#include "../include/operations/add_logic.cuh"
#include "../include/util/cuda_unique_ptr.cuh"

using namespace xyz_autodiff;

// Operation テスト用のCUDAカーネル
template <typename T>
__global__ void test_operation_kernel(T* data1, T* grad1, T* data2, T* grad2, T* result) {
    // Variable作成
    Variable<T, 1> var1(data1, grad1);
    Variable<T, 1> var2(data2, grad2);
    
    // 値設定
    var1[0] = static_cast<T>(3.0);
    var2[0] = static_cast<T>(4.0);
    
    // make_add関数を使用してBinaryOperationを作成（内部でbufferを保持）
    auto op = make_add<1>(var1, var2);
    
    // forward計算
    op.forward();
    result[0] = op.output()[0];
    
    // 出力に単位勾配を設定してbackward計算
    op.output().grad(0) = static_cast<T>(1.0);
    
    // backward計算
    op.backward();
    
    // 勾配結果を保存
    result[1] = var1.grad(0);  // dL/dvar1
    result[2] = var2.grad(0);  // dL/dvar2
}

class OperationTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaError_t err = cudaSetDevice(0);
        if (err != cudaSuccess) {
            GTEST_SKIP() << "CUDA device not available";
        }
    }
};

TEST_F(OperationTest, BasicAddition) {
    using T = float;
    
    // ホストメモリ
    std::vector<T> host_result(3, 0);
    
    // デバイスメモリ確保
    auto device_data1 = makeCudaUniqueArray<T>(1);
    auto device_grad1 = makeCudaUniqueArray<T>(1);
    auto device_data2 = makeCudaUniqueArray<T>(1);
    auto device_grad2 = makeCudaUniqueArray<T>(1);
    auto device_result = makeCudaUniqueArray<T>(3);
    
    ASSERT_NE(device_data1, nullptr);
    ASSERT_NE(device_grad1, nullptr);
    ASSERT_NE(device_data2, nullptr);
    ASSERT_NE(device_grad2, nullptr);
    ASSERT_NE(device_result, nullptr);
    
    // 勾配初期化
    T zero = 0.0f;
    ASSERT_EQ(cudaMemcpy(device_grad1.get(), &zero, sizeof(T), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(device_grad2.get(), &zero, sizeof(T), cudaMemcpyHostToDevice), cudaSuccess);
    
    // カーネル実行
    test_operation_kernel<T><<<1, 1>>>(
        device_data1.get(), device_grad1.get(),
        device_data2.get(), device_grad2.get(),
        device_result.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    // 結果をホストにコピー
    ASSERT_EQ(cudaMemcpy(host_result.data(), device_result.get(), 3 * sizeof(T), cudaMemcpyDeviceToHost), cudaSuccess);
    
    // 検証
    EXPECT_FLOAT_EQ(host_result[0], 7.0f);  // 3 + 4 = 7
    EXPECT_FLOAT_EQ(host_result[1], 1.0f);  // d(3+4)/d3 = 1
    EXPECT_FLOAT_EQ(host_result[2], 1.0f);  // d(3+4)/d4 = 1
}

TEST_F(OperationTest, ConceptCheck) {
    using Var1 = Variable<float, 1>;
    using AddLogicType = AddLogic<Var1, Var1, 1>;
    using AddOp = BinaryOperation<AddLogicType, Var1, Var1, 1>;
    
    // 型要件のチェック
    static_assert(std::is_same_v<AddOp::input1_type, Var1>);
    static_assert(std::is_same_v<AddOp::input2_type, Var1>);
    static_assert(std::is_same_v<AddOp::value_type, float>);
    static_assert(AddOp::output_size == 1);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}