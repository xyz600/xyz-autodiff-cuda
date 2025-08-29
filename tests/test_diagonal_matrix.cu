#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <xyz_autodiff/diagonal_matrix_view.cuh>
#include <xyz_autodiff/dense_matrix.cuh>
#include <xyz_autodiff/util/cuda_unique_ptr.cuh>
#include <xyz_autodiff/concept/variable.cuh>
#include <xyz_autodiff/concept/matrix.cuh>

using namespace xyz_autodiff;

// テスト用バッファ構造体
template <typename T>
struct DiagonalTestBuffers {
    T data[10];     // 最大10要素までサポート
    T grad[10];     // 最大10要素までサポート
    T output[100];  // 最大10x10行列結果用
};

// DiagonalMatrixView テスト用のCUDAカーネル
template <typename T, std::size_t N>
__global__ void test_diagonal_basic_kernel(T* data, T* grad, T* output) {
    VariableRef<N, T> var(data, grad);
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
    VariableRef<N, T> diag_var(diag_data, diag_grad);
    DiagonalMatrixView<T, N> diag_matrix(diag_var);
    
    for (std::size_t i = 0; i < N; ++i) {
        diag_matrix[i] = static_cast<T>(i + 1);
    }
    
    // 密行列作成: [[1, 2], [3, 4], [5, 6]] (3x2)
    DenseMatrix<T, N, 2> dense_matrix;
    dense_matrix[0] = 1; dense_matrix[1] = 2;  // 行0
    dense_matrix[2] = 3; dense_matrix[3] = 4;  // 行1
    dense_matrix[4] = 5; dense_matrix[5] = 6;  // 行2
    
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
    dense_matrix[0] = 1; dense_matrix[1] = 2; dense_matrix[2] = 3;  // 行0
    dense_matrix[3] = 4; dense_matrix[4] = 5; dense_matrix[5] = 6;  // 行1
    
    // 対角行列作成: diag([1, 2, 3])
    VariableRef<N, T> diag_var(diag_data, diag_grad);
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
    
    auto device_buffers = makeCudaUnique<DiagonalTestBuffers<T>>();
    ASSERT_NE(device_buffers, nullptr);
    
    test_diagonal_basic_kernel<T, N><<<1, 1>>>(device_buffers.get()->data, device_buffers.get()->grad, device_buffers.get()->output);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    DiagonalTestBuffers<T> host_buffers;
    ASSERT_EQ(cudaMemcpy(&host_buffers, device_buffers.get(), sizeof(DiagonalTestBuffers<T>), cudaMemcpyDeviceToHost), cudaSuccess);
    
    // 対角行列の検証: diag([1, 2, 3])
    std::vector<T> expected = {
        1, 0, 0,
        0, 2, 0,
        0, 0, 3
    };
    
    for (std::size_t i = 0; i < N * N; ++i) {
        EXPECT_FLOAT_EQ(host_buffers.output[i], expected[i]);
    }
}

TEST_F(DiagonalMatrixTest, DiagonalDenseMultiply) {
    constexpr std::size_t N = 3;
    using T = float;
    
    auto device_buffers = makeCudaUnique<DiagonalTestBuffers<T>>();
    ASSERT_NE(device_buffers, nullptr);
    
    test_diagonal_dense_multiply_kernel<T, N><<<1, 1>>>(
        device_buffers.get()->data, device_buffers.get()->grad, device_buffers.get()->output);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    DiagonalTestBuffers<T> host_buffers;
    ASSERT_EQ(cudaMemcpy(&host_buffers, device_buffers.get(), sizeof(DiagonalTestBuffers<T>), cudaMemcpyDeviceToHost), cudaSuccess);
    
    // 期待値: diag([1,2,3]) * [[1,2],[3,4],[5,6]] = [[1,2],[6,8],[15,18]]
    std::vector<T> expected = {1, 2, 6, 8, 15, 18};
    
    for (std::size_t i = 0; i < N * 2; ++i) {
        EXPECT_FLOAT_EQ(host_buffers.output[i], expected[i]);
    }
}

TEST_F(DiagonalMatrixTest, DenseDiagonalMultiply) {
    constexpr std::size_t N = 3;
    using T = float;
    
    auto device_buffers = makeCudaUnique<DiagonalTestBuffers<T>>();
    ASSERT_NE(device_buffers, nullptr);
    
    test_dense_diagonal_multiply_kernel<T, N><<<1, 1>>>(
        device_buffers.get()->data, device_buffers.get()->grad, device_buffers.get()->output);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    DiagonalTestBuffers<T> host_buffers;
    ASSERT_EQ(cudaMemcpy(&host_buffers, device_buffers.get(), sizeof(DiagonalTestBuffers<T>), cudaMemcpyDeviceToHost), cudaSuccess);
    
    // 期待値: [[1,2,3],[4,5,6]] * diag([1,2,3]) = [[1,4,9],[4,10,18]]
    std::vector<T> expected = {1, 4, 9, 4, 10, 18};
    
    for (std::size_t i = 0; i < 2 * N; ++i) {
        EXPECT_FLOAT_EQ(host_buffers.output[i], expected[i]);
    }
}

TEST_F(DiagonalMatrixTest, ConceptCheck) {
    // Concept チェック
    static_assert(xyz_autodiff::VariableConcept<DiagonalMatrixView<float, 4>>);
    static_assert(xyz_autodiff::DifferentiableVariableConcept<DiagonalMatrixView<float, 4>>);
    static_assert(xyz_autodiff::MatrixViewConcept<DiagonalMatrixView<float, 4>>);
    static_assert(xyz_autodiff::MatrixViewConcept<DiagonalMatrixView<double, 3>>);
    
    // サイズチェック
    EXPECT_EQ((DiagonalMatrixView<float, 4>::rows), 4);
    EXPECT_EQ((DiagonalMatrixView<float, 4>::cols), 4);
    EXPECT_EQ((DiagonalMatrixView<float, 4>::size), 4);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}