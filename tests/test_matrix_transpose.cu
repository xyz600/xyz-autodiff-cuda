#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <array>
#include <xyz_autodiff/dense_matrix.cuh>
#include <xyz_autodiff/diagonal_matrix_view.cuh>
#include <xyz_autodiff/variable.cuh>
#include <xyz_autodiff/util/cuda_unique_ptr.cuh>

using namespace xyz_autodiff;

// テスト用バッファ構造体（シンプル版）
template <typename T>
struct MatrixTransposeTestBuffers {
    T data[10];    // 入力データ用
    T result[20];  // 結果格納用
};

// DenseMatrix transpose テスト用CUDAカーネル
template <typename T>
__global__ void test_matrix_view_transpose_kernel(T* result) {
    // 2x3行列を作成
    DenseMatrix<T, 2, 3> matrix;
    
    // データ設定
    matrix(0, 0) = static_cast<T>(1.0);  // [1, 2, 3]
    matrix(0, 1) = static_cast<T>(2.0);  // [4, 5, 6]
    matrix(0, 2) = static_cast<T>(3.0);
    matrix(1, 0) = static_cast<T>(4.0);
    matrix(1, 1) = static_cast<T>(5.0);
    matrix(1, 2) = static_cast<T>(6.0);
    
    // transpose操作
    auto transposed = matrix.transpose();
    
    // transposed は 3x2 行列
    // [1, 4]
    // [2, 5] 
    // [3, 6]
    result[0] = transposed(0, 0);  // 1
    result[1] = transposed(0, 1);  // 4
    result[2] = transposed(1, 0);  // 2
    result[3] = transposed(1, 1);  // 5
    result[4] = transposed(2, 0);  // 3
    result[5] = transposed(2, 1);  // 6
    
    // 二重transpose（元に戻る）
    auto double_transposed = transposed.transpose();
    result[6] = double_transposed(0, 0);  // 1
    result[7] = double_transposed(1, 2);  // 6
}

// DenseMatrix transpose テスト用CUDAカーネル
template <typename T>
__global__ void test_dense_matrix_transpose_kernel(T* result) {
    DenseMatrix<T, 2, 3> matrix;
    
    // データ設定
    matrix(0, 0) = static_cast<T>(1.0);
    matrix(0, 1) = static_cast<T>(2.0);
    matrix(0, 2) = static_cast<T>(3.0);
    matrix(1, 0) = static_cast<T>(4.0);
    matrix(1, 1) = static_cast<T>(5.0);
    matrix(1, 2) = static_cast<T>(6.0);
    
    // transpose操作
    auto transposed = matrix.transpose();
    
    // 結果確認
    result[0] = transposed(0, 0);  // 1
    result[1] = transposed(0, 1);  // 4
    result[2] = transposed(1, 0);  // 2
    result[3] = transposed(1, 1);  // 5
    result[4] = transposed(2, 0);  // 3
    result[5] = transposed(2, 1);  // 6
}

// DiagonalMatrix transpose テスト用CUDAカーネル
template <typename T>
__global__ void test_diagonal_matrix_transpose_kernel(T* data, T* result) {
    VariableRef<3, T> var(data, data + 3);  // data後半を勾配用に使用
    
    // 対角要素設定
    var[0] = static_cast<T>(1.0);
    var[1] = static_cast<T>(2.0);
    var[2] = static_cast<T>(3.0);
    
    DiagonalMatrixView<T, 3, VariableRef<3, T>> diag_view(var);
    
    // transpose操作（対角行列なので変わらない）
    auto transposed = diag_view.transpose();
    
    // 結果確認
    result[0] = transposed(0, 0);  // 1
    result[1] = transposed(1, 1);  // 2  
    result[2] = transposed(2, 2);  // 3
    result[3] = transposed(0, 1);  // 0（非対角要素）
    result[4] = transposed(1, 0);  // 0（非対角要素）
}

class MatrixTransposeTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaError_t err = cudaSetDevice(0);
        if (err != cudaSuccess) {
            GTEST_SKIP() << "CUDA device not available";
        }
    }
};

TEST_F(MatrixTransposeTest, MatrixViewTranspose) {
    using T = float;
    
    auto device_buffers = makeCudaUnique<MatrixTransposeTestBuffers<T>>();
    ASSERT_NE(device_buffers, nullptr);
    
    test_matrix_view_transpose_kernel<T><<<1, 1>>>(device_buffers.get()->result);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    MatrixTransposeTestBuffers<T> host_buffers;
    ASSERT_EQ(cudaMemcpy(&host_buffers, device_buffers.get(), sizeof(MatrixTransposeTestBuffers<T>), cudaMemcpyDeviceToHost), cudaSuccess);
    
    // transpose結果の検証
    EXPECT_FLOAT_EQ(host_buffers.result[0], 1.0f);  // (0,0) = 1
    EXPECT_FLOAT_EQ(host_buffers.result[1], 4.0f);  // (0,1) = 4
    EXPECT_FLOAT_EQ(host_buffers.result[2], 2.0f);  // (1,0) = 2
    EXPECT_FLOAT_EQ(host_buffers.result[3], 5.0f);  // (1,1) = 5
    EXPECT_FLOAT_EQ(host_buffers.result[4], 3.0f);  // (2,0) = 3
    EXPECT_FLOAT_EQ(host_buffers.result[5], 6.0f);  // (2,1) = 6
    
    // 二重transpose結果の検証
    EXPECT_FLOAT_EQ(host_buffers.result[6], 1.0f);  // (0,0) = 1
    EXPECT_FLOAT_EQ(host_buffers.result[7], 6.0f);  // (1,2) = 6
}

TEST_F(MatrixTransposeTest, DenseMatrixTranspose) {
    using T = float;
    
    auto device_buffers = makeCudaUnique<MatrixTransposeTestBuffers<T>>();
    ASSERT_NE(device_buffers, nullptr);
    
    test_dense_matrix_transpose_kernel<T><<<1, 1>>>(device_buffers.get()->result);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    MatrixTransposeTestBuffers<T> host_buffers;
    ASSERT_EQ(cudaMemcpy(&host_buffers, device_buffers.get(), sizeof(MatrixTransposeTestBuffers<T>), cudaMemcpyDeviceToHost), cudaSuccess);
    
    // transpose結果の検証
    EXPECT_FLOAT_EQ(host_buffers.result[0], 1.0f);  // (0,0) = 1
    EXPECT_FLOAT_EQ(host_buffers.result[1], 4.0f);  // (0,1) = 4
    EXPECT_FLOAT_EQ(host_buffers.result[2], 2.0f);  // (1,0) = 2
    EXPECT_FLOAT_EQ(host_buffers.result[3], 5.0f);  // (1,1) = 5
    EXPECT_FLOAT_EQ(host_buffers.result[4], 3.0f);  // (2,0) = 3
    EXPECT_FLOAT_EQ(host_buffers.result[5], 6.0f);  // (2,1) = 6
}

TEST_F(MatrixTransposeTest, DiagonalMatrixTranspose) {
    using T = float;
    
    auto device_buffers = makeCudaUnique<MatrixTransposeTestBuffers<T>>();
    ASSERT_NE(device_buffers, nullptr);
    
    test_diagonal_matrix_transpose_kernel<T><<<1, 1>>>(
        device_buffers.get()->data, device_buffers.get()->result);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    MatrixTransposeTestBuffers<T> host_buffers;
    ASSERT_EQ(cudaMemcpy(&host_buffers, device_buffers.get(), sizeof(MatrixTransposeTestBuffers<T>), cudaMemcpyDeviceToHost), cudaSuccess);
    
    // transpose結果の検証（対角行列なので変わらない）
    EXPECT_FLOAT_EQ(host_buffers.result[0], 1.0f);  // (0,0) = 1
    EXPECT_FLOAT_EQ(host_buffers.result[1], 2.0f);  // (1,1) = 2
    EXPECT_FLOAT_EQ(host_buffers.result[2], 3.0f);  // (2,2) = 3
    EXPECT_FLOAT_EQ(host_buffers.result[3], 0.0f);  // (0,1) = 0
    EXPECT_FLOAT_EQ(host_buffers.result[4], 0.0f);  // (1,0) = 0
}

// Concept チェックテスト
TEST_F(MatrixTransposeTest, ConceptCheck) {
    using DenseMat23 = DenseMatrix<float, 2, 3>;
    using DenseMat32 = DenseMatrix<float, 3, 2>;
    using DiagView3 = DiagonalMatrixView<float, 3, VariableRef<3, float>>;
    
    // MatrixViewConcept の要件チェック
    static_assert(MatrixViewConcept<DenseMat23>);
    static_assert(MatrixViewConcept<DenseMat32>);
    static_assert(MatrixViewConcept<DiagView3>);
    
    // サイズ情報のチェック
    static_assert(DenseMat23::rows == 2 && DenseMat23::cols == 3);
    static_assert(DenseMat32::rows == 3 && DenseMat32::cols == 2);
    static_assert(DiagView3::rows == 3 && DiagView3::cols == 3);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}