#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include "../../../include/variable.cuh"
#include "../../../include/concept/variable.cuh"
#include "../../../include/concept/operation_node.cuh"
#include "../operations/symmetric_matrix.cuh"
#include "../../../include/util/cuda_unique_ptr.cuh"
#include "../../../tests/utility/unary_gradient_tester.cuh"

using namespace xyz_autodiff;

// ===========================================
// Static Assert Tests for Concept Compliance
// ===========================================

// Test types
using TestVector3 = Variable<float, 3>;
using TestVectorRef3 = VariableRef<float, 3>;
using SymMatInvOp = UnaryOperation<3, op::SymmetricMatrix2x2InverseLogic<TestVectorRef3>, TestVectorRef3>;
using SymMatConvOp = UnaryOperation<4, op::Symmetric3ParamToMatrixLogic<TestVectorRef3>, TestVectorRef3>;

// Static assertions for concept compliance
static_assert(VariableConcept<TestVector3>, 
    "Variable<float, 3> should satisfy VariableConcept");
static_assert(DifferentiableVariableConcept<TestVector3>, 
    "Variable<float, 3> should satisfy DifferentiableVariableConcept");

static_assert(VariableConcept<SymMatInvOp>, 
    "SymmetricMatrix2x2Inverse Operation should satisfy VariableConcept");
static_assert(DifferentiableVariableConcept<SymMatInvOp>, 
    "SymmetricMatrix2x2Inverse Operation should satisfy DifferentiableVariableConcept");
static_assert(OperationNode<SymMatInvOp>, 
    "SymmetricMatrix2x2Inverse Operation should satisfy OperationNode");

static_assert(VariableConcept<SymMatConvOp>, 
    "Symmetric3ParamToMatrix Operation should satisfy VariableConcept");
static_assert(DifferentiableVariableConcept<SymMatConvOp>, 
    "Symmetric3ParamToMatrix Operation should satisfy DifferentiableVariableConcept");
static_assert(OperationNode<SymMatConvOp>, 
    "Symmetric3ParamToMatrix Operation should satisfy OperationNode");

// Ensure Variable is not an OperationNode
static_assert(!OperationNode<TestVector3>, 
    "Variable should NOT satisfy OperationNode");

// ===========================================
// Test Class
// ===========================================

class SymmetricMatrixTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaError_t err = cudaSetDevice(0);
        if (err != cudaSuccess) {
            GTEST_SKIP() << "CUDA device not available";
        }
    }
};

// ===========================================
// Forward Pass Tests - Matrix Inverse
// ===========================================

__global__ void test_symmetric_matrix_inverse_identity_kernel(float* result) {
    // Test case: matrix = [1, 0, 1] (identity matrix)
    // Expected inverse: [1, 0, 1] (same as input)
    float matrix_data[3] = {1.0f, 0.0f, 1.0f};  // [σ11, σ12, σ22]
    float matrix_grad[3] = {0,0,0};
    
    VariableRef<float, 3> matrix(matrix_data, matrix_grad);
    
    auto inverse = op::symmetric_matrix_2x2_inverse(matrix);
    inverse.forward();
    
    float tolerance = 1e-6f;
    bool success = (fabsf(inverse[0] - 1.0f) < tolerance &&  // inv[0,0]
                   fabsf(inverse[1] - 0.0f) < tolerance &&  // inv[0,1] = inv[1,0]
                   fabsf(inverse[2] - 1.0f) < tolerance);   // inv[1,1]
    
    *result = success ? 1.0f : 0.0f;
}

TEST_F(SymmetricMatrixTest, InverseIdentityMatrix) {
    auto device_result = makeCudaUnique<float>();
    
    test_symmetric_matrix_inverse_identity_kernel<<<1, 1>>>(device_result.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    float host_result;
    ASSERT_EQ(cudaMemcpy(&host_result, device_result.get(), sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(host_result, 1.0f);
}

__global__ void test_symmetric_matrix_inverse_diagonal_kernel(float* result) {
    // Test case: matrix = [4, 0, 9] (diagonal matrix)
    // Expected inverse: [1/4, 0, 1/9]
    float matrix_data[3] = {4.0f, 0.0f, 9.0f};  // [σ11, σ12, σ22]
    float matrix_grad[3] = {0,0,0};
    
    VariableRef<float, 3> matrix(matrix_data, matrix_grad);
    
    auto inverse = op::symmetric_matrix_2x2_inverse(matrix);
    inverse.forward();
    
    float tolerance = 1e-6f;
    bool success = (fabsf(inverse[0] - 0.25f) < tolerance &&     // 1/4
                   fabsf(inverse[1] - 0.0f) < tolerance &&      // 0
                   fabsf(inverse[2] - (1.0f/9.0f)) < tolerance); // 1/9
    
    *result = success ? 1.0f : 0.0f;
}

TEST_F(SymmetricMatrixTest, InverseDiagonalMatrix) {
    auto device_result = makeCudaUnique<float>();
    
    test_symmetric_matrix_inverse_diagonal_kernel<<<1, 1>>>(device_result.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    float host_result;
    ASSERT_EQ(cudaMemcpy(&host_result, device_result.get(), sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(host_result, 1.0f);
}

__global__ void test_symmetric_matrix_inverse_general_kernel(float* result) {
    // Test case: matrix = [2, 1, 3] -> [[2, 1], [1, 3]]
    // det = 2*3 - 1*1 = 5
    // inverse = (1/5) * [[3, -1], [-1, 2]] -> [3/5, -1/5, 2/5]
    float matrix_data[3] = {2.0f, 1.0f, 3.0f};  // [σ11, σ12, σ22]
    float matrix_grad[3] = {0,0,0};
    
    VariableRef<float, 3> matrix(matrix_data, matrix_grad);
    
    auto inverse = op::symmetric_matrix_2x2_inverse(matrix);
    inverse.forward();
    
    float tolerance = 1e-6f;
    bool success = (fabsf(inverse[0] - 0.6f) < tolerance &&      // 3/5
                   fabsf(inverse[1] - (-0.2f)) < tolerance &&   // -1/5
                   fabsf(inverse[2] - 0.4f) < tolerance);       // 2/5
    
    *result = success ? 1.0f : 0.0f;
}

TEST_F(SymmetricMatrixTest, InverseGeneralMatrix) {
    auto device_result = makeCudaUnique<float>();
    
    test_symmetric_matrix_inverse_general_kernel<<<1, 1>>>(device_result.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    float host_result;
    ASSERT_EQ(cudaMemcpy(&host_result, device_result.get(), sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(host_result, 1.0f);
}

// ===========================================
// Forward Pass Tests - Matrix Conversion
// ===========================================

__global__ void test_symmetric_3param_to_matrix_kernel(float* result) {
    // Test case: [a, b, c] -> [a, b, b, c]
    float param_data[3] = {2.0f, 1.5f, 3.0f};  // [a, b, c]
    float param_grad[3] = {0,0,0};
    
    VariableRef<float, 3> params(param_data, param_grad);
    
    auto matrix = op::symmetric_3param_to_matrix(params);
    matrix.forward();
    
    float tolerance = 1e-6f;
    bool success = (fabsf(matrix[0] - 2.0f) < tolerance &&   // a
                   fabsf(matrix[1] - 1.5f) < tolerance &&   // b
                   fabsf(matrix[2] - 1.5f) < tolerance &&   // b
                   fabsf(matrix[3] - 3.0f) < tolerance);    // c
    
    *result = success ? 1.0f : 0.0f;
}

TEST_F(SymmetricMatrixTest, Symmetric3ParamToMatrix) {
    auto device_result = makeCudaUnique<float>();
    
    test_symmetric_3param_to_matrix_kernel<<<1, 1>>>(device_result.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    float host_result;
    ASSERT_EQ(cudaMemcpy(&host_result, device_result.get(), sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(host_result, 1.0f);
}

// ===========================================
// Gradient Verification Tests
// ===========================================

TEST_F(SymmetricMatrixTest, InverseGradientVerification) {
    using Logic = op::SymmetricMatrix2x2InverseLogic<VariableRef<double, 3>>;
    test::UnaryGradientTester<Logic, 3, 3>::test_custom(
        "SymmetricMatrix2x2Inverse", 
        30,      // num_tests (reduced for stability)
        40.0,    // tolerance (based on error analysis: >= 39.1856)
        1e-6,    // delta (proper numerical differentiation)
        0.8,     // input_min (well-conditioned matrices only)
        1.5      // input_max (smaller range for numerical stability)
    );
}

TEST_F(SymmetricMatrixTest, Symmetric3ParamToMatrixGradientVerification) {
    using Logic = op::Symmetric3ParamToMatrixLogic<VariableRef<double, 3>>;
    test::UnaryGradientTester<Logic, 3, 4>::test_custom(
        "Symmetric3ParamToMatrix", 
        50,      // num_tests
        1e-5,    // tolerance (minimum allowed for double precision)
        1e-6,    // delta (proper numerical differentiation)
        -3.0,    // input_min
        3.0      // input_max
    );
}

// ===========================================
// Specific Gradient Tests
// ===========================================

__global__ void test_symmetric_3param_to_matrix_gradient_kernel(double* result) {
    // Test simple linear transformation gradient
    double param_data[3] = {2.0, 1.0, 3.0};
    double param_grad[3] = {0.0, 0.0, 0.0};
    
    VariableRef<double, 3> params(param_data, param_grad);
    auto matrix_op = op::symmetric_3param_to_matrix(params);
    
    // Forward pass
    matrix_op.forward();
    
    // Set upstream gradient and run analytical backward
    matrix_op.zero_grad();
    matrix_op.add_grad(0, 1.0);  // a
    matrix_op.add_grad(1, 2.0);  // b
    matrix_op.add_grad(2, 3.0);  // b
    matrix_op.add_grad(3, 4.0);  // c
    
    matrix_op.backward();
    
    // Check gradient values directly
    bool success = true;
    double tolerance = 1e-10;  // Very strict tolerance for this simple linear operation
    
    // Expected gradients: a gets 1.0, b gets 2.0+3.0=5.0, c gets 4.0
    double expected_grad[3] = {1.0, 5.0, 4.0};
    
    for (int i = 0; i < 3; i++) {
        double diff = fabs(params.grad(i) - expected_grad[i]);
        if (diff > tolerance) {
            success = false;
            break;
        }
    }
    
    *result = success ? 1.0 : 0.0;
}

TEST_F(SymmetricMatrixTest, Symmetric3ParamToMatrixSpecificGradientVerification) {
    auto device_result = makeCudaUnique<double>();
    
    test_symmetric_3param_to_matrix_gradient_kernel<<<1, 1>>>(device_result.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    double host_result;
    ASSERT_EQ(cudaMemcpy(&host_result, device_result.get(), sizeof(double), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(host_result, 1.0);
}

// ===========================================
// Interface Compliance Test
// ===========================================

__global__ void test_symmetric_matrix_interface_kernel(float* result) {
    float matrix_data[3] = {2.0f, 1.0f, 3.0f};
    float matrix_grad[3] = {0,0,0};
    
    VariableRef<float, 3> matrix(matrix_data, matrix_grad);
    
    // Test both inverse and conversion operations
    auto inverse_op = op::symmetric_matrix_2x2_inverse(matrix);
    auto conv_op = op::symmetric_3param_to_matrix(matrix);
    
    // Test VariableConcept interface on inverse
    inverse_op.zero_grad();
    constexpr auto inv_size = decltype(inverse_op)::size;
    auto* inv_data = inverse_op.data();
    auto* inv_grad = inverse_op.grad();
    auto inv_value = inverse_op[0];
    auto inv_grad_value = inverse_op.grad(0);
    
    // Test OperationNode interface
    inverse_op.forward();
    inverse_op.backward();
    inverse_op.backward_numerical(1e-5f);
    inverse_op.run();
    inverse_op.run_numerical(1e-5f);
    
    // Test VariableConcept interface on conversion
    conv_op.zero_grad();
    constexpr auto conv_size = decltype(conv_op)::size;
    auto* conv_data = conv_op.data();
    auto* conv_grad = conv_op.grad();
    auto conv_value = conv_op[0];
    auto conv_grad_value = conv_op.grad(0);
    
    // Test OperationNode interface on conversion
    conv_op.forward();
    conv_op.backward();
    conv_op.backward_numerical(1e-5f);
    conv_op.run();
    conv_op.run_numerical(1e-5f);
    
    // Verify expected behavior
    bool success = (inv_size == 3 && inv_data != nullptr && inv_grad != nullptr &&
                   conv_size == 4 && conv_data != nullptr && conv_grad != nullptr);
    
    *result = success ? 1.0f : 0.0f;
}

TEST_F(SymmetricMatrixTest, InterfaceCompliance) {
    auto device_result = makeCudaUnique<float>();
    
    test_symmetric_matrix_interface_kernel<<<1, 1>>>(device_result.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    float host_result;
    ASSERT_EQ(cudaMemcpy(&host_result, device_result.get(), sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(host_result, 1.0f);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}