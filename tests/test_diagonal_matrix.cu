#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include "../include/diagonal_matrix_view.cuh"
#include "../include/dense_matrix.cuh"
#include "../include/util/cuda_unique_ptr.cuh"

using namespace xyz_autodiff;

// DiagonalMatrixView テスト用のCUDAカーネル
template <typename T, std::size_t N>
__global__ void test_diagonal_basic_kernel(T* data, T* grad, T* output) {
    Variable<T, N> var(data, grad);
    DiagonalMatrixView<T, N> diag_view(var);
    
    // 対角要素を設定
    for (std::size_t i = 0; i < N; ++i) {
        diag_view[i] = static_cast<T>(i + 1);  // 1, 2, 3, ...
    }
    
    // 2次元アクセステスト
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            output[i * N + j] = diag_view(i, j);
        }
    }
}

// DiagonalMatrixView と DenseMatrix の行列積テスト
template <typename T, std::size_t N>
__global__ void test_diagonal_dense_multiply_kernel(T* diag_data, T* diag_grad, T* result_output) {
    // 対角行列作成: diag([1, 2, 3])
    Variable<T, N> diag_var(diag_data, diag_grad);
    DiagonalMatrixView<T, N> diag_matrix(diag_var);
    
    for (std::size_t i = 0; i < N; ++i) {
        diag_matrix[i] = static_cast<T>(i + 1);
    }
    
    // 密行列作成: [[1, 2], [3, 4], [5, 6]] (3x2)
    DenseMatrix<T, N, 2> dense_matrix;
    dense_matrix(0, 0) = 1; dense_matrix(0, 1) = 2;
    dense_matrix(1, 0) = 3; dense_matrix(1, 1) = 4;
    dense_matrix(2, 0) = 5; dense_matrix(2, 1) = 6;
    
    // 行列積: diag * dense (3x3 * 3x2 = 3x2)
    auto result = diag_matrix * dense_matrix;
    
    // // 結果を保存
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < 2; ++j) {
            result_output[i * 2 + j] = result(i, j);
        }
    }
}

// 密行列と対角行列の行列積テスト
template <typename T, std::size_t N>
__global__ void test_dense_diagonal_multiply_kernel(T* diag_data, T* diag_grad, T* result_output) {
    // 密行列作成: [[1, 2, 3], [4, 5, 6]] (2x3)
    DenseMatrix<T, 2, N> dense_matrix;
    dense_matrix(0, 0) = 1; dense_matrix(0, 1) = 2; dense_matrix(0, 2) = 3;
    dense_matrix(1, 0) = 4; dense_matrix(1, 1) = 5; dense_matrix(1, 2) = 6;
    
    // 対角行列作成: diag([1, 2, 3])
    Variable<T, N> diag_var(diag_data, diag_grad);
    DiagonalMatrixView<T, N> diag_matrix(diag_var);
    
    for (std::size_t i = 0; i < N; ++i) {
        diag_matrix[i] = static_cast<T>(i + 1);
    }
    
    // 行列積: dense * diag (2x3 * 3x3 = 2x3)
    auto result = dense_matrix * diag_matrix;
    
    // // 結果を保存
    for (std::size_t i = 0; i < 2; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            result_output[i * N + j] = result(i, j);
        }
    }
}

class DiagonalMatrixTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaError_t err = cudaSetDevice(0);
        if (err != cudaSuccess) {
            GTEST_SKIP() << "CUDA device not available";
        }
    }
};

TEST_F(DiagonalMatrixTest, BasicConstruction) {
    constexpr std::size_t N = 3;
    using T = float;
    
    std::vector<T> host_output(N * N, 0);
    
    auto device_data = makeCudaUniqueArray<T>(N);
    auto device_grad = makeCudaUniqueArray<T>(N);
    auto device_output = makeCudaUniqueArray<T>(N * N);
    
    ASSERT_NE(device_data, nullptr);
    ASSERT_NE(device_grad, nullptr);
    ASSERT_NE(device_output, nullptr);
    
    test_diagonal_basic_kernel<T, N><<<1, 1>>>(device_data.get(), device_grad.get(), device_output.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    ASSERT_EQ(cudaMemcpy(host_output.data(), device_output.get(), N * N * sizeof(T), cudaMemcpyDeviceToHost), cudaSuccess);
    
    // 対角行列の検証: diag([1, 2, 3])
    std::vector<T> expected = {
        1, 0, 0,
        0, 2, 0,
        0, 0, 3
    };
    
    for (std::size_t i = 0; i < N * N; ++i) {
        EXPECT_FLOAT_EQ(host_output[i], expected[i]);
    }
}

TEST_F(DiagonalMatrixTest, DiagonalDenseMultiply) {
    constexpr std::size_t N = 3;
    using T = float;
    
    std::vector<T> host_output(N * 2, 0);
    
    auto device_diag_data = makeCudaUniqueArray<T>(N);
    auto device_diag_grad = makeCudaUniqueArray<T>(N);
    auto device_output = makeCudaUniqueArray<T>(N * 2);
    
    ASSERT_NE(device_diag_data, nullptr);
    ASSERT_NE(device_diag_grad, nullptr);
    ASSERT_NE(device_output, nullptr);
    
    test_diagonal_dense_multiply_kernel<T, N><<<1, 1>>>(
        device_diag_data.get(), device_diag_grad.get(), device_output.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    ASSERT_EQ(cudaMemcpy(host_output.data(), device_output.get(), N * 2 * sizeof(T), cudaMemcpyDeviceToHost), cudaSuccess);
    
    // 期待値: diag([1,2,3]) * [[1,2],[3,4],[5,6]] = [[1,2],[6,8],[15,18]]
    std::vector<T> expected = {1, 2, 6, 8, 15, 18};
    
    for (std::size_t i = 0; i < N * 2; ++i) {
        EXPECT_FLOAT_EQ(host_output[i], expected[i]);
    }
}

TEST_F(DiagonalMatrixTest, DenseDiagonalMultiply) {
    constexpr std::size_t N = 3;
    using T = float;
    
    std::vector<T> host_output(2 * N, 0);
    
    auto device_diag_data = makeCudaUniqueArray<T>(N);
    auto device_diag_grad = makeCudaUniqueArray<T>(N);
    auto device_output = makeCudaUniqueArray<T>(2 * N);
    
    ASSERT_NE(device_diag_data, nullptr);
    ASSERT_NE(device_diag_grad, nullptr);
    ASSERT_NE(device_output, nullptr);
    
    test_dense_diagonal_multiply_kernel<T, N><<<1, 1>>>(
        device_diag_data.get(), device_diag_grad.get(), device_output.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    ASSERT_EQ(cudaMemcpy(host_output.data(), device_output.get(), 2 * N * sizeof(T), cudaMemcpyDeviceToHost), cudaSuccess);
    
    // 期待値: [[1,2,3],[4,5,6]] * diag([1,2,3]) = [[1,4,9],[4,10,18]]
    std::vector<T> expected = {1, 4, 9, 4, 10, 18};
    
    for (std::size_t i = 0; i < 2 * N; ++i) {
        EXPECT_FLOAT_EQ(host_output[i], expected[i]);
    }
}

TEST_F(DiagonalMatrixTest, ConceptCheck) {
    // サイズチェック
    EXPECT_EQ((DiagonalMatrixView<float, 4>::rows), 4);
    EXPECT_EQ((DiagonalMatrixView<float, 4>::cols), 4);
    EXPECT_EQ((DiagonalMatrixView<float, 4>::size), 4);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}