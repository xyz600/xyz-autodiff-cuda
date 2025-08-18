#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include "../include/dense_matrix.cuh"
#include "../include/util/cuda_unique_ptr.cuh"
#include "concept/matrix.cuh"
#include "concept/variable.cuh"

using namespace xyz_autodiff;

// テスト用バッファ構造体
template <typename T>
struct DenseMatrixTestBuffers {
    T output[50];  // 最大50要素までサポート
};

// DenseMatrix テスト用のCUDAカーネル
template <typename T, std::size_t Rows, std::size_t Cols>
__global__ void test_dense_matrix_basic_kernel(T* output) {
    DenseMatrix<T, Rows, Cols> matrix;
    
    // MatrixView機能テスト - Variable conceptで値設定
    for (std::size_t i = 0; i < Rows; ++i) {
        for (std::size_t j = 0; j < Cols; ++j) {
            std::size_t linear_idx = i * Cols + j;
            matrix[linear_idx] = static_cast<T>(linear_idx + 1);  // 1, 2, 3, ...
        }
    }
    
    // 結果をoutputに保存（検証用）
    // 前半：データ値、後半：勾配値
    for (std::size_t i = 0; i < matrix.size; ++i) {
        output[i] = matrix[i];                    // データ値（1次元アクセス）
        output[matrix.size + i] = matrix.grad(i); // 勾配値
    }
}

template <typename T, std::size_t Rows, std::size_t Cols>
__global__ void test_dense_matrix_operations_kernel(T* output) {
    DenseMatrix<T, Rows, Cols> matrix;
    
    // データ初期化
    matrix.zero_grad();
    for (std::size_t i = 0; i < Rows; ++i) {
        for (std::size_t j = 0; j < Cols; ++j) {
            std::size_t linear_idx = i * Cols + j;
            matrix[linear_idx] = static_cast<T>((i + 1) * (j + 1));
            matrix.add_grad(linear_idx, static_cast<T>(linear_idx + 1)); // 初期勾配設定
        }
    }
    
    // zero_grad操作テスト
    matrix.zero_grad();
    
    // 結果保存
    for (std::size_t i = 0; i < matrix.size; ++i) {
        output[i] = matrix[i];                    // データ値
        output[matrix.size + i] = matrix.grad(i); // 勾配値
    }
}

template <typename T, std::size_t Rows, std::size_t Cols>
__global__ void test_matrix_accessors_kernel(T* output) {
    DenseMatrix<T, Rows, Cols> matrix;
    
    // データポインタテスト
    T* data_ptr = matrix.data();
    T* grad_ptr = matrix.grad();
    
    // 2次元と1次元アクセスの一致をテスト
    bool access_match = true;
    for (std::size_t i = 0; i < Rows; ++i) {
        for (std::size_t j = 0; j < Cols; ++j) {
            std::size_t linear_idx = i * Cols + j;
            matrix[linear_idx] = static_cast<T>(linear_idx + 10);
            if (matrix[linear_idx] != matrix(i, j)) {
                access_match = false;
            }
        }
    }
    
    output[0] = access_match ? 1.0f : 0.0f;
    output[1] = (data_ptr != nullptr) ? 1.0f : 0.0f;
    output[2] = (grad_ptr != nullptr) ? 1.0f : 0.0f;
}

class DenseMatrixTest : public ::testing::Test {
protected:
    void SetUp() override {
        // CUDA初期化チェック
        cudaError_t err = cudaSetDevice(0);
        if (err != cudaSuccess) {
            GTEST_SKIP() << "CUDA device not available";
        }
    }
};

TEST_F(DenseMatrixTest, BasicConstruction) {
    constexpr std::size_t Rows = 2;
    constexpr std::size_t Cols = 3;
    constexpr std::size_t Size = Rows * Cols;
    using T = float;
    
    // デバイスメモリ確保 (cuda_unique_ptr使用)
    auto device_buffers = makeCudaUnique<DenseMatrixTestBuffers<T>>();
    ASSERT_NE(device_buffers, nullptr);
    
    // カーネル実行
    test_dense_matrix_basic_kernel<T, Rows, Cols><<<1, 1>>>(device_buffers.get()->output);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    // 結果をホストにコピー
    DenseMatrixTestBuffers<T> host_buffers;
    ASSERT_EQ(cudaMemcpy(&host_buffers, device_buffers.get(), sizeof(DenseMatrixTestBuffers<T>), cudaMemcpyDeviceToHost), cudaSuccess);
    
    // データ値の検証
    for (std::size_t i = 0; i < Size; ++i) {
        EXPECT_FLOAT_EQ(host_buffers.output[i], static_cast<T>(i + 1));
    }
    
    // 勾配値の検証（初期化時は0）
    for (std::size_t i = 0; i < Size; ++i) {
        EXPECT_FLOAT_EQ(host_buffers.output[Size + i], static_cast<T>(0));
    }
    
    // メモリは自動解放される
}

TEST_F(DenseMatrixTest, ZeroGradOperation) {
    constexpr std::size_t Rows = 3;
    constexpr std::size_t Cols = 2;
    constexpr std::size_t Size = Rows * Cols;
    using T = float;
    
    // デバイスメモリ確保 (cuda_unique_ptr使用)
    auto device_buffers = makeCudaUnique<DenseMatrixTestBuffers<T>>();
    ASSERT_NE(device_buffers, nullptr);
    
    // カーネル実行
    test_dense_matrix_operations_kernel<T, Rows, Cols><<<1, 1>>>(device_buffers.get()->output);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    // 結果をホストにコピー
    DenseMatrixTestBuffers<T> host_buffers;
    ASSERT_EQ(cudaMemcpy(&host_buffers, device_buffers.get(), sizeof(DenseMatrixTestBuffers<T>), cudaMemcpyDeviceToHost), cudaSuccess);
    
    // データ値の検証 ((i+1) * (j+1))
    std::vector<T> expected_data = {1.0f, 2.0f, 2.0f, 4.0f, 3.0f, 6.0f};
    for (std::size_t i = 0; i < Size; ++i) {
        EXPECT_FLOAT_EQ(host_buffers.output[i], expected_data[i]);
    }
    
    // 勾配値の検証（zero_gradにより全て0になっているはず）
    for (std::size_t i = 0; i < Size; ++i) {
        EXPECT_FLOAT_EQ(host_buffers.output[Size + i], 0.0f);
    }
    
    // メモリは自動解放される
}

TEST_F(DenseMatrixTest, AccessorConsistency) {
    constexpr std::size_t Rows = 2;
    constexpr std::size_t Cols = 4;
    using T = float;
    
    // デバイスメモリ確保 (cuda_unique_ptr使用)
    auto device_buffers = makeCudaUnique<DenseMatrixTestBuffers<T>>();
    ASSERT_NE(device_buffers, nullptr);
    
    // カーネル実行
    test_matrix_accessors_kernel<T, Rows, Cols><<<1, 1>>>(device_buffers.get()->output);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    // 結果をホストにコピー
    DenseMatrixTestBuffers<T> host_buffers;
    ASSERT_EQ(cudaMemcpy(&host_buffers, device_buffers.get(), sizeof(DenseMatrixTestBuffers<T>), cudaMemcpyDeviceToHost), cudaSuccess);
    
    // 検証
    EXPECT_FLOAT_EQ(host_buffers.output[0], 1.0f); // 2次元と1次元アクセスの一致
    EXPECT_FLOAT_EQ(host_buffers.output[1], 1.0f); // dataポインタの有効性
    EXPECT_FLOAT_EQ(host_buffers.output[2], 1.0f); // gradポインタの有効性
    
    // メモリは自動解放される
}

TEST_F(DenseMatrixTest, ConceptCheck) {

    static_assert(xyz_autodiff::VariableConcept<DenseMatrix<float, 3, 4>>);
    static_assert(xyz_autodiff::DifferentiableVariableConcept<DenseMatrix<float, 3, 4>>);
    static_assert(xyz_autodiff::MatrixViewConcept<DenseMatrix<float, 3, 4>>);

    // サイズチェック
    EXPECT_EQ((xyz_autodiff::DenseMatrix<float, 3, 4>::rows), 3);
    EXPECT_EQ((xyz_autodiff::DenseMatrix<float, 3, 4>::cols), 4);
    EXPECT_EQ((xyz_autodiff::DenseMatrix<float, 3, 4>::size), 12);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}