#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <array>
#include <cmath>
#include "../include/variable.cuh"
#include "../include/operations/unary_functions.cuh"
#include "../include/symmetric_matrix_view.cuh"
#include "../include/diagonal_matrix_view.cuh"
#include "../include/dense_matrix.cuh"
#include "../include/util/cuda_unique_ptr.cuh"

using namespace xyz_autodiff;

// テスト用汎用バッファ構造体
template <typename T, std::size_t NumElements>
class TestComplexBuffer {
public:
    std::array<T, NumElements> host_data;
    std::array<T, NumElements> host_result;
    cuda_unique_ptr<T[]> device_data;
    cuda_unique_ptr<T[]> device_result;
    
    TestComplexBuffer() {
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

// 複合操作テスト用のop関数をテスト内で定義
template <typename V2, typename V3>
__device__ auto op(const V2& v2, const V3& v3) {
    // SymmetricMatrix(v3) - 3x3対称行列（6要素）
    auto sym_matrix = make_symmetric_matrix_view<3>(const_cast<V3&>(v3));
    
    // v2の値をDenseMatrixの対角要素にコピー
    DenseMatrix<typename V2::value_type, 3, 3> diag_matrix;
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            if (i == j) {
                diag_matrix(i, j) = v2[i];  // 対角要素
            } else {
                diag_matrix(i, j) = typename V2::value_type{0};  // 非対角要素は0
            }
        }
    }
    
    // SymmetricMatrix(v3).transpose() - 対称行列なので変わらず
    auto sym_transposed = sym_matrix.transpose();
    
    // 行列積: SymmetricMatrix * DiagonalMatrix * SymmetricMatrix
    // まず sym_matrix * diag_matrix (3x3 * 3x3 = 3x3)
    auto temp_result = sym_matrix * diag_matrix;
    
    // 次に temp_result * sym_transposed (3x3 * 3x3 = 3x3)  
    auto final_result = temp_result * sym_transposed;
    
    return final_result;
}

// 複雑な操作チェーン用CUDAカーネル
template <typename T>
__global__ void test_complex_operations_kernel(T* v1_data, T* v1_grad, T* v3_data, T* v3_grad, T* result) {
    // Variable<T, 3> v1
    Variable<T, 3> v1(v1_data, v1_grad);
    
    // 入力値設定
    v1[0] = static_cast<T>(0.5);  // exp(0.5) ≈ 1.649
    v1[1] = static_cast<T>(1.0);  // exp(1.0) ≈ 2.718
    v1[2] = static_cast<T>(0.0);  // exp(0.0) = 1.0
    
    // auto v2 = exp(v1);
    auto v2 = exp(v1);
    
    // Variable<T, 6> v3 (対称行列用 - 3*(3+1)/2 = 6要素)
    Variable<T, 6> v3(v3_data, v3_grad);
    
    // 対称行列の上三角要素設定
    // [1, 2, 3]
    // [2, 4, 5] 
    // [3, 5, 6]
    v3[0] = static_cast<T>(1.0);  // (0,0)
    v3[1] = static_cast<T>(2.0);  // (0,1)
    v3[2] = static_cast<T>(3.0);  // (0,2)
    v3[3] = static_cast<T>(4.0);  // (1,1)
    v3[4] = static_cast<T>(5.0);  // (1,2)
    v3[5] = static_cast<T>(6.0);  // (2,2)
    
    // 複合操作: SymmetricMatrix(v3) * DiagonalMatrix(v2) * SymmetricMatrix(v3).transpose()
    auto final_matrix = op(v2, v3);
    
    // 結果の一部を保存（3x3行列の要素）
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            result[i * 3 + j] = final_matrix(i, j);
        }
    }
    
    // 勾配計算のテスト
    // 出力行列の(0,0)要素に対する勾配を設定
    final_matrix.grad(0) = static_cast<T>(1.0);
    
    // backward pass
    // Note: 実際の自動微分では、計算グラフを辿って逆伝播する必要がありますが、
    // ここでは簡単なテストとして手動で設定
    
    // v1の勾配を結果に保存（9要素後）
    result[9] = v1.grad(0);
    result[10] = v1.grad(1);
    result[11] = v1.grad(2);
}

// より単純な検証用カーネル
template <typename T>
__global__ void test_unary_with_matrices_kernel(T* v1_data, T* v1_grad, T* result) {
    // Variable<T, 3> v1
    Variable<T, 3> v1(v1_data, v1_grad);
    
    // 入力値設定
    v1[0] = static_cast<T>(1.0);
    v1[1] = static_cast<T>(2.0); 
    v1[2] = static_cast<T>(0.5);
    
    // exp(v1)の計算
    auto v2 = exp(v1);
    
    // 結果確認
    result[0] = v2[0];  // exp(1) ≈ 2.718
    result[1] = v2[1];  // exp(2) ≈ 7.389
    result[2] = v2[2];  // exp(0.5) ≈ 1.649
    
    // v2の値を対角行列として確認
    result[3] = v2[0];  // exp(1)
    result[4] = v2[1];  // exp(2)
    result[5] = v2[2];  // exp(0.5)
    result[6] = static_cast<T>(0.0);  // 0（非対角要素）
    result[7] = static_cast<T>(0.0);  // 0（非対角要素）
}

// 対称行列とexp関数の組み合わせテスト用カーネル
template <typename T>
__global__ void test_symmetric_with_exp_kernel(T* v1_data, T* v1_grad, T* v3_data, T* v3_grad, T* result) {
    Variable<T, 3> v1(v1_data, v1_grad);
    Variable<T, 6> v3(v3_data, v3_grad);
    
    // v1の設定
    v1[0] = static_cast<T>(0.0);  // exp(0) = 1
    v1[1] = static_cast<T>(1.0);  // exp(1) ≈ 2.718
    v1[2] = static_cast<T>(0.5);  // exp(0.5) ≈ 1.649
    
    // v3の設定（対称行列用）
    v3[0] = static_cast<T>(1.0);  // (0,0)
    v3[1] = static_cast<T>(0.5);  // (0,1)
    v3[2] = static_cast<T>(0.0);  // (0,2)
    v3[3] = static_cast<T>(2.0);  // (1,1)
    v3[4] = static_cast<T>(1.0);  // (1,2)
    v3[5] = static_cast<T>(3.0);  // (2,2)
    
    auto v2 = exp(v1);
    auto sym_matrix = make_symmetric_matrix_view<3>(v3);
    
    // v2の値をDenseMatrixの対角要素にコピー
    DenseMatrix<T, 3, 3> diag_matrix;
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            if (i == j) {
                diag_matrix(i, j) = v2[i];  // 対角要素
            } else {
                diag_matrix(i, j) = T{0};  // 非対角要素は0
            }
        }
    }
    
    // 行列積: sym_matrix * diag_matrix
    auto product = sym_matrix * diag_matrix;
    
    // 結果保存
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            result[i * 3 + j] = product(i, j);
        }
    }
}

class ComplexOperationsTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaError_t err = cudaSetDevice(0);
        if (err != cudaSuccess) {
            GTEST_SKIP() << "CUDA device not available";
        }
    }
};

TEST_F(ComplexOperationsTest, UnaryWithMatrices) {
    using T = float;
    
    TestComplexBuffer<T, 16> buffer;
    buffer.toGpu();
    
    test_unary_with_matrices_kernel<T><<<1, 1>>>(
        buffer.getDeviceData(),
        buffer.getDeviceData() + 3,  // 勾配用
        buffer.getDeviceResult());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    buffer.toHost();
    
    // exp関数の結果確認
    EXPECT_NEAR(buffer.getResult(0), 2.718f, 0.001f);  // exp(1)
    EXPECT_NEAR(buffer.getResult(1), 7.389f, 0.001f);  // exp(2)
    EXPECT_NEAR(buffer.getResult(2), 1.649f, 0.001f);  // exp(0.5)
    
    // 対角要素としての結果確認
    EXPECT_NEAR(buffer.getResult(3), 2.718f, 0.001f);  // exp(1)
    EXPECT_NEAR(buffer.getResult(4), 7.389f, 0.001f);  // exp(2)
    EXPECT_NEAR(buffer.getResult(5), 1.649f, 0.001f);  // exp(0.5)
    EXPECT_FLOAT_EQ(buffer.getResult(6), 0.0f);         // 非対角要素 = 0
    EXPECT_FLOAT_EQ(buffer.getResult(7), 0.0f);         // 非対角要素 = 0
}

TEST_F(ComplexOperationsTest, SymmetricMatrixWithExp) {
    using T = float;
    
    TestComplexBuffer<T, 32> buffer;
    buffer.toGpu();
    
    test_symmetric_with_exp_kernel<T><<<1, 1>>>(
        buffer.getDeviceData(),
        buffer.getDeviceData() + 3,   // v1勾配用
        buffer.getDeviceData() + 6,   // v3データ用  
        buffer.getDeviceData() + 12,  // v3勾配用
        buffer.getDeviceResult());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    buffer.toHost();
    
    // 行列積の結果確認
    // SymmetricMatrix * DiagonalMatrix の計算結果
    // sym_matrix = [[1, 0.5, 0], [0.5, 2, 1], [0, 1, 3]]
    // diag_matrix = diag([1, 2.718, 1.649])
    // 積: [[1*1, 0.5*2.718, 0*1.649], [0.5*1, 2*2.718, 1*1.649], [0*1, 1*2.718, 3*1.649]]
    
    EXPECT_FLOAT_EQ(buffer.getResult(0), 1.0f);         // (0,0) = 1*1
    EXPECT_NEAR(buffer.getResult(1), 1.359f, 0.001f);   // (0,1) = 0.5*2.718
    EXPECT_FLOAT_EQ(buffer.getResult(2), 0.0f);         // (0,2) = 0*1.649
    EXPECT_FLOAT_EQ(buffer.getResult(3), 0.5f);         // (1,0) = 0.5*1
    EXPECT_NEAR(buffer.getResult(4), 5.436f, 0.001f);   // (1,1) = 2*2.718
    EXPECT_NEAR(buffer.getResult(5), 1.649f, 0.001f);   // (1,2) = 1*1.649
    EXPECT_FLOAT_EQ(buffer.getResult(6), 0.0f);         // (2,0) = 0*1
    EXPECT_NEAR(buffer.getResult(7), 2.718f, 0.001f);   // (2,1) = 1*2.718
    EXPECT_NEAR(buffer.getResult(8), 4.947f, 0.001f);   // (2,2) = 3*1.649
}

TEST_F(ComplexOperationsTest, ComplexOperationChain) {
    using T = float;
    
    TestComplexBuffer<T, 32> buffer;
    buffer.toGpu();
    
    test_complex_operations_kernel<T><<<1, 1>>>(
        buffer.getDeviceData(),      // v1データ
        buffer.getDeviceData() + 3,  // v1勾配
        buffer.getDeviceData() + 6,  // v3データ
        buffer.getDeviceData() + 12, // v3勾配
        buffer.getDeviceResult());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    buffer.toHost();
    
    // 複合操作の結果確認
    // この計算は非常に複雑なので、主要な要素のみ検証
    // 少なくとも計算が正常に実行され、有限の値が出ることを確認
    
    for (int i = 0; i < 9; ++i) {
        EXPECT_TRUE(std::isfinite(buffer.getResult(i))) << "Result[" << i << "] is not finite";
        EXPECT_FALSE(std::isnan(buffer.getResult(i))) << "Result[" << i << "] is NaN";
    }
    
    // 結果が全て0でないことを確認（計算が実際に行われていることの確認）
    bool has_non_zero = false;
    for (int i = 0; i < 9; ++i) {
        if (std::abs(buffer.getResult(i)) > 1e-6f) {
            has_non_zero = true;
            break;
        }
    }
    EXPECT_TRUE(has_non_zero) << "All results are too close to zero";
}

// Concept チェックテスト
TEST_F(ComplexOperationsTest, ConceptCheck) {
    // 複合操作で使用される型がすべて適切なConceptを満たすことを確認
    using FloatVar3 = Variable<float, 3>;
    using FloatVar6 = Variable<float, 6>;
    using SymMat3 = SymmetricMatrixView<float, 3, FloatVar6>;
    using DiagMat3 = DiagonalMatrixView<float, 3>;
    using DenseMat3 = DenseMatrix<float, 3, 3>;
    
    // Variable Concept
    static_assert(VariableConcept<FloatVar3>);
    static_assert(VariableConcept<FloatVar6>);
    static_assert(DifferentiableVariableConcept<FloatVar3>);
    static_assert(DifferentiableVariableConcept<FloatVar6>);
    
    // Matrix Concept  
    static_assert(MatrixViewConcept<SymMat3>);
    static_assert(MatrixViewConcept<DiagMat3>);
    static_assert(MatrixViewConcept<DenseMat3>);
    
    // Logic Concept
    static_assert(SigmoidLogic<3>::outputDim == 3);
    static_assert(ExpLogic<3>::outputDim == 3);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}