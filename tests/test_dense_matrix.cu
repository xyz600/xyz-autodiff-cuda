#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include "../include/dense_matrix.cuh"
// #include "../include/concept/variable.cuh"  // CUDA compiler concept limitations
// #include "../include/concept/matrix.cuh"   // CUDA compiler concept limitations

using namespace xyz_autodiff;

// DenseMatrix テスト用のCUDAカーネル
template <typename T, std::size_t Rows, std::size_t Cols>
__global__ void test_dense_matrix_basic_kernel(T* output) {
    DenseMatrix<T, Rows, Cols> matrix;
    
    // MatrixView機能テスト - 2次元アクセス
    for (std::size_t i = 0; i < Rows; ++i) {
        for (std::size_t j = 0; j < Cols; ++j) {
            matrix(i, j) = static_cast<T>(i * Cols + j + 1);  // 1, 2, 3, ...
        }
    }
    
    // Variable機能テスト - 1次元アクセス
    for (std::size_t i = 0; i < matrix.size; ++i) {
        matrix.grad(i) = static_cast<T>(i * 2);  // 0, 2, 4, ...
    }
    
    // 結果をoutputに保存（検証用）
    // 前半：データ値、後半：勾配値
    for (std::size_t i = 0; i < matrix.size; ++i) {
        output[i] = matrix[i];                    // データ値（1次元アクセス）
        output[matrix.size + i] = matrix.grad(i); // 勾配値
    }
}

template <typename T, std::size_t Rows, std::size_t Cols>
__global__ void test_dense_matrix_operations_kernel(T* grad_values, T* output) {
    DenseMatrix<T, Rows, Cols> matrix;
    
    // データ初期化
    for (std::size_t i = 0; i < Rows; ++i) {
        for (std::size_t j = 0; j < Cols; ++j) {
            matrix(i, j) = static_cast<T>((i + 1) * (j + 1));
        }
    }
    
    // 勾配操作テスト
    matrix.zero_grad();
    matrix.accumulate_grad(grad_values);
    
    // 疎行列サポートテスト
    bool active_test1 = matrix.is_active_in_col(0, 0);
    bool active_test2 = matrix.is_active_in_row(0, 0);
    
    // 結果保存
    for (std::size_t i = 0; i < matrix.size; ++i) {
        output[i] = matrix[i];                    // データ値
        output[matrix.size + i] = matrix.grad(i); // 勾配値
    }
    
    // アクティブ行・列の検証（密行列なので全てtrue）
    output[2 * matrix.size] = active_test1 ? 1.0f : 0.0f;
    output[2 * matrix.size + 1] = active_test2 ? 1.0f : 0.0f;
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
            matrix(i, j) = static_cast<T>(i * Cols + j + 10);
            std::size_t linear_idx = i * Cols + j;
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
    
    // ホストメモリ
    std::vector<T> host_output(2 * Size, 0);
    
    // デバイスメモリ確保
    T* device_output;
    ASSERT_EQ(cudaMalloc(&device_output, 2 * Size * sizeof(T)), cudaSuccess);
    
    // カーネル実行
    test_dense_matrix_basic_kernel<T, Rows, Cols><<<1, 1>>>(device_output);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    // 結果をホストにコピー
    ASSERT_EQ(cudaMemcpy(host_output.data(), device_output, 2 * Size * sizeof(T), cudaMemcpyDeviceToHost), cudaSuccess);
    
    // データ値の検証
    for (std::size_t i = 0; i < Size; ++i) {
        EXPECT_FLOAT_EQ(host_output[i], static_cast<T>(i + 1));
    }
    
    // 勾配値の検証
    for (std::size_t i = 0; i < Size; ++i) {
        EXPECT_FLOAT_EQ(host_output[Size + i], static_cast<T>(i * 2));
    }
    
    // メモリ解放
    cudaFree(device_output);
}

TEST_F(DenseMatrixTest, MatrixOperations) {
    constexpr std::size_t Rows = 3;
    constexpr std::size_t Cols = 2;
    constexpr std::size_t Size = Rows * Cols;
    using T = float;
    
    // ホストメモリ
    std::vector<T> host_grad_values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<T> host_output(2 * Size + 2, 0);
    
    // デバイスメモリ確保
    T* device_grad_values;
    T* device_output;
    
    ASSERT_EQ(cudaMalloc(&device_grad_values, Size * sizeof(T)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&device_output, (2 * Size + 2) * sizeof(T)), cudaSuccess);
    
    // 勾配値をデバイスにコピー
    ASSERT_EQ(cudaMemcpy(device_grad_values, host_grad_values.data(), Size * sizeof(T), cudaMemcpyHostToDevice), cudaSuccess);
    
    // カーネル実行
    test_dense_matrix_operations_kernel<T, Rows, Cols><<<1, 1>>>(device_grad_values, device_output);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    // 結果をホストにコピー
    ASSERT_EQ(cudaMemcpy(host_output.data(), device_output, (2 * Size + 2) * sizeof(T), cudaMemcpyDeviceToHost), cudaSuccess);
    
    // データ値の検証 ((i+1) * (j+1))
    std::vector<T> expected_data = {1.0f, 2.0f, 2.0f, 4.0f, 3.0f, 6.0f};
    for (std::size_t i = 0; i < Size; ++i) {
        EXPECT_FLOAT_EQ(host_output[i], expected_data[i]);
    }
    
    // 勾配値の検証（zero_grad + accumulate_grad）
    for (std::size_t i = 0; i < Size; ++i) {
        EXPECT_FLOAT_EQ(host_output[Size + i], host_grad_values[i]);
    }
    
    // アクティブ行・列の検証
    EXPECT_FLOAT_EQ(host_output[2 * Size], 1.0f);     // active_rows check
    EXPECT_FLOAT_EQ(host_output[2 * Size + 1], 1.0f); // active_cols check
    
    // メモリ解放
    cudaFree(device_grad_values);
    cudaFree(device_output);
}

TEST_F(DenseMatrixTest, AccessorConsistency) {
    constexpr std::size_t Rows = 2;
    constexpr std::size_t Cols = 4;
    using T = float;
    
    // ホストメモリ
    std::vector<T> host_output(3, 0);
    
    // デバイスメモリ確保
    T* device_output;
    ASSERT_EQ(cudaMalloc(&device_output, 3 * sizeof(T)), cudaSuccess);
    
    // カーネル実行
    test_matrix_accessors_kernel<T, Rows, Cols><<<1, 1>>>(device_output);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    // 結果をホストにコピー
    ASSERT_EQ(cudaMemcpy(host_output.data(), device_output, 3 * sizeof(T), cudaMemcpyDeviceToHost), cudaSuccess);
    
    // 検証
    EXPECT_FLOAT_EQ(host_output[0], 1.0f); // 2次元と1次元アクセスの一致
    EXPECT_FLOAT_EQ(host_output[1], 1.0f); // dataポインタの有効性
    EXPECT_FLOAT_EQ(host_output[2], 1.0f); // gradポインタの有効性
    
    // メモリ解放
    cudaFree(device_output);
}

TEST_F(DenseMatrixTest, ConceptCheck) {
    // コンパイル時概念チェック (CUDA compiler limitations)
    // static_assert(concept::Variable<DenseMatrix<float, 3, 4>>);
    // static_assert(concept::DifferentiableVariable<DenseMatrix<float, 3, 4>>);
    // static_assert(concept::MatrixView<DenseMatrix<float, 3, 4>>);
    // static_assert(concept::DifferentiableMatrixView<DenseMatrix<float, 3, 4>>);
    
    // サイズチェック
    EXPECT_EQ((xyz_autodiff::DenseMatrix<float, 3, 4>::rows), 3);
    EXPECT_EQ((xyz_autodiff::DenseMatrix<float, 3, 4>::cols), 4);
    EXPECT_EQ((xyz_autodiff::DenseMatrix<float, 3, 4>::size), 12);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}