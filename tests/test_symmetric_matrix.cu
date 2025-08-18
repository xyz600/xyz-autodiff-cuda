#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <array>
#include "../include/symmetric_matrix_view.cuh"
#include "../include/variable.cuh"
#include "../include/util/cuda_unique_ptr.cuh"

using namespace xyz_autodiff;

// テスト用バッファ構造体（シンプル版）
template <typename T>
struct SymmetricMatrixTestBuffers {
    T data[6];     // 3x3対称行列データ用（6要素）
    T grad[6];     // 勾配用
    T result[16];  // 結果格納用
};

// SymmetricMatrix基本テスト用CUDAカーネル
template <typename T>
__global__ void test_symmetric_matrix_basic_kernel(T* data, T* grad, T* result) {
    // 3x3対称行列（6要素）を作成
    VariableRef<T, 6> var(data, grad);
    auto sym_matrix = make_symmetric_matrix_view<3>(var);
    
    // 上三角要素の設定
    // [1, 2, 3]
    // [2, 4, 5] 
    // [3, 5, 6]
    sym_matrix(0, 0) = static_cast<T>(1.0);  // 対角
    sym_matrix(0, 1) = static_cast<T>(2.0);  // 上三角
    sym_matrix(0, 2) = static_cast<T>(3.0);  // 上三角
    sym_matrix(1, 1) = static_cast<T>(4.0);  // 対角
    sym_matrix(1, 2) = static_cast<T>(5.0);  // 上三角
    sym_matrix(2, 2) = static_cast<T>(6.0);  // 対角
    
    // 対称性の確認
    result[0] = sym_matrix(0, 0);  // 1
    result[1] = sym_matrix(0, 1);  // 2
    result[2] = sym_matrix(1, 0);  // 2 (対称)
    result[3] = sym_matrix(1, 1);  // 4
    result[4] = sym_matrix(2, 0);  // 3 (対称)
    result[5] = sym_matrix(2, 2);  // 6
    
    // 下三角要素の読み取りテスト
    result[6] = sym_matrix(1, 0);  // should be 2
    result[7] = sym_matrix(2, 1);  // should be 5
}

// SymmetricMatrix transpose テスト用CUDAカーネル
template <typename T>
__global__ void test_symmetric_matrix_transpose_kernel(T* data, T* grad, T* result) {
    VariableRef<T, 6> var(data, grad);
    auto sym_matrix = make_symmetric_matrix_view<3>(var);
    
    // データ設定
    sym_matrix(0, 0) = static_cast<T>(1.0);
    sym_matrix(0, 1) = static_cast<T>(2.0);
    sym_matrix(0, 2) = static_cast<T>(3.0);
    sym_matrix(1, 1) = static_cast<T>(4.0);
    sym_matrix(1, 2) = static_cast<T>(5.0);
    sym_matrix(2, 2) = static_cast<T>(6.0);
    
    // transpose操作（自分自身を返すはず）
    auto transposed = sym_matrix.transpose();
    
    // transpose後も同じ値を持つはず
    result[0] = transposed(0, 0);  // 1
    result[1] = transposed(0, 1);  // 2
    result[2] = transposed(1, 0);  // 2
    result[3] = transposed(1, 1);  // 4
    result[4] = transposed(2, 1);  // 5
    result[5] = transposed(1, 2);  // 5
}

// SymmetricMatrix ストレージテスト用CUDAカーネル
template <typename T>
__global__ void test_symmetric_matrix_storage_kernel(T* data, T* grad, T* result) {
    VariableRef<T, 6> var(data, grad);
    auto sym_matrix = make_symmetric_matrix_view<3>(var);
    
    // 直接的にストレージにアクセス
    sym_matrix[0] = static_cast<T>(1.0);  // (0,0)
    sym_matrix[1] = static_cast<T>(2.0);  // (0,1)
    sym_matrix[2] = static_cast<T>(3.0);  // (0,2)
    sym_matrix[3] = static_cast<T>(4.0);  // (1,1)
    sym_matrix[4] = static_cast<T>(5.0);  // (1,2)
    sym_matrix[5] = static_cast<T>(6.0);  // (2,2)
    
    // 2次元アクセスで確認
    result[0] = sym_matrix(0, 0);  // 1
    result[1] = sym_matrix(0, 1);  // 2
    result[2] = sym_matrix(0, 2);  // 3
    result[3] = sym_matrix(1, 1);  // 4
    result[4] = sym_matrix(1, 2);  // 5
    result[5] = sym_matrix(2, 2);  // 6
    
    // 対称性チェック
    result[6] = sym_matrix(1, 0);  // 2
    result[7] = sym_matrix(2, 0);  // 3
    result[8] = sym_matrix(2, 1);  // 5
}

class SymmetricMatrixTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaError_t err = cudaSetDevice(0);
        if (err != cudaSuccess) {
            GTEST_SKIP() << "CUDA device not available";
        }
    }
};

TEST_F(SymmetricMatrixTest, BasicSymmetricMatrix) {
    using T = float;
    
    auto device_buffers = makeCudaUnique<SymmetricMatrixTestBuffers<T>>();
    ASSERT_NE(device_buffers, nullptr);
    
    test_symmetric_matrix_basic_kernel<T><<<1, 1>>>(
        device_buffers.get()->data,
        device_buffers.get()->grad,
        device_buffers.get()->result);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    SymmetricMatrixTestBuffers<T> host_buffers;
    ASSERT_EQ(cudaMemcpy(&host_buffers, device_buffers.get(), sizeof(SymmetricMatrixTestBuffers<T>), cudaMemcpyDeviceToHost), cudaSuccess);
    
    // 基本的な対称性の検証
    EXPECT_FLOAT_EQ(host_buffers.result[0], 1.0f);  // (0,0) = 1
    EXPECT_FLOAT_EQ(host_buffers.result[1], 2.0f);  // (0,1) = 2
    EXPECT_FLOAT_EQ(host_buffers.result[2], 2.0f);  // (1,0) = 2 (対称)
    EXPECT_FLOAT_EQ(host_buffers.result[3], 4.0f);  // (1,1) = 4
    EXPECT_FLOAT_EQ(host_buffers.result[4], 3.0f);  // (2,0) = 3 (対称)
    EXPECT_FLOAT_EQ(host_buffers.result[5], 6.0f);  // (2,2) = 6
    
    // 下三角要素のアクセステスト
    EXPECT_FLOAT_EQ(host_buffers.result[6], 2.0f);  // (1,0) = 2
    EXPECT_FLOAT_EQ(host_buffers.result[7], 5.0f);  // (2,1) = 5
}

TEST_F(SymmetricMatrixTest, SymmetricMatrixTranspose) {
    using T = float;
    
    auto device_buffers = makeCudaUnique<SymmetricMatrixTestBuffers<T>>();
    ASSERT_NE(device_buffers, nullptr);
    
    test_symmetric_matrix_transpose_kernel<T><<<1, 1>>>(
        device_buffers.get()->data,
        device_buffers.get()->grad,
        device_buffers.get()->result);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    SymmetricMatrixTestBuffers<T> host_buffers;
    ASSERT_EQ(cudaMemcpy(&host_buffers, device_buffers.get(), sizeof(SymmetricMatrixTestBuffers<T>), cudaMemcpyDeviceToHost), cudaSuccess);
    
    // transpose後も同じ値（対称行列なので）
    EXPECT_FLOAT_EQ(host_buffers.result[0], 1.0f);  // (0,0) = 1
    EXPECT_FLOAT_EQ(host_buffers.result[1], 2.0f);  // (0,1) = 2
    EXPECT_FLOAT_EQ(host_buffers.result[2], 2.0f);  // (1,0) = 2
    EXPECT_FLOAT_EQ(host_buffers.result[3], 4.0f);  // (1,1) = 4
    EXPECT_FLOAT_EQ(host_buffers.result[4], 5.0f);  // (2,1) = 5
    EXPECT_FLOAT_EQ(host_buffers.result[5], 5.0f);  // (1,2) = 5
}

TEST_F(SymmetricMatrixTest, SymmetricMatrixStorage) {
    using T = float;
    
    auto device_buffers = makeCudaUnique<SymmetricMatrixTestBuffers<T>>();
    ASSERT_NE(device_buffers, nullptr);
    
    test_symmetric_matrix_storage_kernel<T><<<1, 1>>>(
        device_buffers.get()->data,
        device_buffers.get()->grad,
        device_buffers.get()->result);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    SymmetricMatrixTestBuffers<T> host_buffers;
    ASSERT_EQ(cudaMemcpy(&host_buffers, device_buffers.get(), sizeof(SymmetricMatrixTestBuffers<T>), cudaMemcpyDeviceToHost), cudaSuccess);
    
    // ストレージ順序でのアクセス結果確認
    EXPECT_FLOAT_EQ(host_buffers.result[0], 1.0f);  // (0,0) = 1
    EXPECT_FLOAT_EQ(host_buffers.result[1], 2.0f);  // (0,1) = 2
    EXPECT_FLOAT_EQ(host_buffers.result[2], 3.0f);  // (0,2) = 3
    EXPECT_FLOAT_EQ(host_buffers.result[3], 4.0f);  // (1,1) = 4
    EXPECT_FLOAT_EQ(host_buffers.result[4], 5.0f);  // (1,2) = 5
    EXPECT_FLOAT_EQ(host_buffers.result[5], 6.0f);  // (2,2) = 6
    
    // 対称性の確認
    EXPECT_FLOAT_EQ(host_buffers.result[6], 2.0f);  // (1,0) = 2
    EXPECT_FLOAT_EQ(host_buffers.result[7], 3.0f);  // (2,0) = 3
    EXPECT_FLOAT_EQ(host_buffers.result[8], 5.0f);  // (2,1) = 5
}

TEST_F(SymmetricMatrixTest, ConceptCheck) {
    // Concept チェック
    static_assert(xyz_autodiff::VariableConcept<SymmetricMatrixView<float, 3, Variable<float, 6>>>);
    static_assert(xyz_autodiff::DifferentiableVariableConcept<SymmetricMatrixView<float, 3, Variable<float, 6>>>);
    static_assert(xyz_autodiff::MatrixViewConcept<SymmetricMatrixView<float, 3, Variable<float, 6>>>);
    static_assert(xyz_autodiff::MatrixViewConcept<SymmetricMatrixView<double, 4, Variable<double, 10>>>);
    
    // サイズチェック
    EXPECT_EQ((SymmetricMatrixView<float, 3, Variable<float, 6>>::rows), 3);
    EXPECT_EQ((SymmetricMatrixView<float, 3, Variable<float, 6>>::cols), 3);
    EXPECT_EQ((SymmetricMatrixView<float, 3, Variable<float, 6>>::storage_size), 6);  // 3*(3+1)/2 = 6
    EXPECT_EQ((SymmetricMatrixView<double, 4, Variable<double, 10>>::storage_size), 10);  // 4*(4+1)/2 = 10
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}