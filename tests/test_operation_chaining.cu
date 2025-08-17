#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <type_traits>
#include <array>
#include "../include/variable.cuh"
#include "../include/operations/operation.cuh"
#include "../include/operations/add_logic.cuh"
#include "../include/util/cuda_unique_ptr.cuh"

using namespace xyz_autodiff;

// テスト用汎用バッファ構造体
template <typename T, std::size_t NumVariables, std::size_t VarSize = 1>
class TestBuffer {
public:
    // ホスト側データ
    std::array<std::array<T, VarSize>, NumVariables> host_data;
    std::array<std::array<T, VarSize>, NumVariables> host_grad;
    std::vector<T> host_result;
    
    // デバイス側データ
    std::array<cuda_unique_ptr<T[]>, NumVariables> device_data;
    std::array<cuda_unique_ptr<T[]>, NumVariables> device_grad;
    cuda_unique_ptr<T[]> device_result;
    
    // 結果格納用変数の数
    std::size_t result_size;
    
    TestBuffer(std::size_t result_count = 0) : result_size(result_count) {
        // ホストデータ初期化
        for (auto& data : host_data) {
            data.fill(T{});
        }
        for (auto& grad : host_grad) {
            grad.fill(T{});
        }
        
        if (result_size > 0) {
            host_result.resize(result_size, T{});
        }
        
        // デバイスメモリ確保
        for (std::size_t i = 0; i < NumVariables; ++i) {
            device_data[i] = makeCudaUniqueArray<T>(VarSize);
            device_grad[i] = makeCudaUniqueArray<T>(VarSize);
        }
        
        if (result_size > 0) {
            device_result = makeCudaUniqueArray<T>(result_size);
        }
    }
    
    // ホストからデバイスへデータ転送
    void toGpu() {
        for (std::size_t i = 0; i < NumVariables; ++i) {
            cudaMemcpy(device_data[i].get(), host_data[i].data(), VarSize * sizeof(T), cudaMemcpyHostToDevice);
            cudaMemcpy(device_grad[i].get(), host_grad[i].data(), VarSize * sizeof(T), cudaMemcpyHostToDevice);
        }
        cudaDeviceSynchronize();
    }
    
    // デバイスからホストへ結果転送
    void toHost() {
        if (result_size > 0) {
            cudaMemcpy(host_result.data(), device_result.get(), result_size * sizeof(T), cudaMemcpyDeviceToHost);
        }
        
        // 勾配も取得（テスト検証用）
        for (std::size_t i = 0; i < NumVariables; ++i) {
            cudaMemcpy(host_grad[i].data(), device_grad[i].get(), VarSize * sizeof(T), cudaMemcpyDeviceToHost);
        }
        cudaDeviceSynchronize();
    }
    
    // 指定したインデックスの変数データを設定
    void setData(std::size_t var_idx, std::size_t element_idx, T value) {
        if (var_idx < NumVariables && element_idx < VarSize) {
            host_data[var_idx][element_idx] = value;
        }
    }
    
    // 指定したインデックスの勾配データを設定
    void setGrad(std::size_t var_idx, std::size_t element_idx, T value) {
        if (var_idx < NumVariables && element_idx < VarSize) {
            host_grad[var_idx][element_idx] = value;
        }
    }
    
    // デバイスポインタ取得
    T* getDeviceData(std::size_t var_idx) {
        return var_idx < NumVariables ? device_data[var_idx].get() : nullptr;
    }
    
    T* getDeviceGrad(std::size_t var_idx) {
        return var_idx < NumVariables ? device_grad[var_idx].get() : nullptr;
    }
    
    T* getDeviceResult() {
        return device_result.get();
    }
    
    // 結果値取得
    T getResult(std::size_t idx) const {
        return idx < result_size ? host_result[idx] : T{};
    }
    
    // 勾配値取得
    T getGrad(std::size_t var_idx, std::size_t element_idx) const {
        return (var_idx < NumVariables && element_idx < VarSize) ? host_grad[var_idx][element_idx] : T{};
    }
};

// Variable Concept テスト用のCUDAカーネル
template <typename T>
__global__ void test_variable_concept_kernel(T* data1, T* grad1, T* data2, T* grad2, T* result) {
    // Variable作成
    Variable<T, 1> var1(data1, grad1);
    Variable<T, 1> var2(data2, grad2);
    
    // 値設定
    var1[0] = static_cast<T>(3.0);
    var2[0] = static_cast<T>(4.0);
    
    // Operation作成してVariable Conceptのメソッドをテスト
    auto node1 = op::add(var1, var2);  // 3 + 4 = 7
    
    // Variable Conceptのメソッドを使用
    result[0] = node1[0];          // operator[] アクセス
    result[1] = *node1.data();     // data() アクセス
    
    // 勾配設定とアクセス
    node1.grad(0) = static_cast<T>(2.0);
    result[2] = node1.grad(0);     // grad() アクセス
    result[3] = *node1.grad();     // grad() ポインタアクセス
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

TEST_F(OperationChainingTest, VariableConceptTest) {
    using T = float;
    
    // テストバッファ作成（2変数、結果4個）
    TestBuffer<T, 2> buffer(4);
    
    // 勾配初期化（データは自動的にゼロ初期化される）
    buffer.toGpu();
    
    // カーネル実行
    test_variable_concept_kernel<T><<<1, 1>>>(
        buffer.getDeviceData(0), buffer.getDeviceGrad(0),
        buffer.getDeviceData(1), buffer.getDeviceGrad(1),
        buffer.getDeviceResult());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    // 結果をホストにコピー
    buffer.toHost();
    
    // 検証
    EXPECT_FLOAT_EQ(buffer.getResult(0), 7.0f);  // 3 + 4 = 7 (operator[])
    EXPECT_FLOAT_EQ(buffer.getResult(1), 7.0f);  // 3 + 4 = 7 (data())
    EXPECT_FLOAT_EQ(buffer.getResult(2), 2.0f);  // grad(0) = 2.0
    EXPECT_FLOAT_EQ(buffer.getResult(3), 2.0f);  // grad() = 2.0
}

// Operation チェーン テスト用のCUDAカーネル
template <typename T>
__global__ void test_operation_chaining_kernel(T* data1, T* grad1, T* data2, T* grad2, T* data3, T* grad3, T* result) {
    // Variable作成
    Variable<T, 1> var1(data1, grad1);
    Variable<T, 1> var2(data2, grad2);
    Variable<T, 1> var3(data3, grad3);
    
    // 値設定
    var1[0] = static_cast<T>(3.0);
    var2[0] = static_cast<T>(4.0);
    var3[0] = static_cast<T>(5.0);
    
    // Operation チェーン: node1 = add(var1, var2), node2 = add(var3, node1)
    auto node1 = op::add(var1, var2);  // 3 + 4 = 7
    auto node2 = op::add(var3, node1); // 5 + 7 = 12
    
    // 結果を取得
    result[0] = node2[0];  // node2はVariable Conceptを満たすので直接アクセス可能
    
    // 出力に単位勾配を設定してbackward計算
    node2.grad(0) = static_cast<T>(1.0);
    
    // backward計算 (チェーンルール)
    node2.backward();  // dL/dnode1 = 1.0, dL/dvar3 = 1.0
    node1.backward();  // dL/dvar1 = 1.0, dL/dvar2 = 1.0
    
    // 勾配結果を保存
    result[1] = var1.grad(0);  // dL/dvar1
    result[2] = var2.grad(0);  // dL/dvar2
    result[3] = var3.grad(0);  // dL/dvar3
}

TEST_F(OperationChainingTest, BasicChaining) {
    using T = float;
    
    // テストバッファ作成（3変数、結果4個）
    TestBuffer<T, 3> buffer(4);
    
    // 勾配初期化（データは自動的にゼロ初期化される）
    buffer.toGpu();
    
    // カーネル実行
    test_operation_chaining_kernel<T><<<1, 1>>>(
        buffer.getDeviceData(0), buffer.getDeviceGrad(0),
        buffer.getDeviceData(1), buffer.getDeviceGrad(1),
        buffer.getDeviceData(2), buffer.getDeviceGrad(2),
        buffer.getDeviceResult());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    // 結果をホストにコピー
    buffer.toHost();
    
    // 検証
    EXPECT_FLOAT_EQ(buffer.getResult(0), 12.0f); // (3 + 4) + 5 = 12
    EXPECT_FLOAT_EQ(buffer.getResult(1), 1.0f);  // d(12)/d3 = 1
    EXPECT_FLOAT_EQ(buffer.getResult(2), 1.0f);  // d(12)/d4 = 1
    EXPECT_FLOAT_EQ(buffer.getResult(3), 1.0f);  // d(12)/d5 = 1
}

// Variable Concept チェックテスト
TEST_F(OperationChainingTest, ConceptCheck) {
    using namespace op;
    using Var1 = Variable<float, 1>;
    using AddLogicType = AddLogic<Var1, Var1>;
    using AddOp = BinaryOperation<AddLogicType::outputDim, AddLogicType, Var1, Var1>;
    
    // Variable Concept の要件チェック
    static_assert(VariableConcept<AddOp>);
    static_assert(DifferentiableVariableConcept<AddOp>);
    
    // サイズ情報のチェック
    static_assert(AddOp::size == 1);
    static_assert(AddOp::output_size == 1);
    
    // 型情報のチェック
    static_assert(std::is_same_v<AddOp::value_type, float>);
    static_assert(std::is_same_v<AddOp::output_type, Variable<float, 1>>);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}