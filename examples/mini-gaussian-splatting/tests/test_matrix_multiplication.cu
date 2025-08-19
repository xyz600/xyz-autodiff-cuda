#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include "../../../include/variable.cuh"
#include "../../../include/concept/variable.cuh"
#include "../../../include/concept/operation_node.cuh"
#include "../operations/matrix_multiplication.cuh"
#include "../../../include/util/cuda_unique_ptr.cuh"
#include "../../../tests/utility/binary_gradient_tester.cuh"

using namespace xyz_autodiff;

// ===========================================
// Static Assert Tests for Concept Compliance
// ===========================================

// Test types
using TestMatrix9 = Variable<float, 9>;
using TestMatrixRef9 = VariableRef<float, 9>;
using MatMulOp = BinaryOperation<9, op::MatrixMultiplication3x3Logic<TestMatrixRef9, TestMatrixRef9>, TestMatrixRef9, TestMatrixRef9>;

// Static assertions for concept compliance
static_assert(VariableConcept<TestMatrix9>, 
    "Variable<float, 9> should satisfy VariableConcept");
static_assert(DifferentiableVariableConcept<TestMatrix9>, 
    "Variable<float, 9> should satisfy DifferentiableVariableConcept");

static_assert(VariableConcept<MatMulOp>, 
    "MatrixMultiplication Operation should satisfy VariableConcept");
static_assert(DifferentiableVariableConcept<MatMulOp>, 
    "MatrixMultiplication Operation should satisfy DifferentiableVariableConcept");
static_assert(OperationNode<MatMulOp>, 
    "MatrixMultiplication Operation should satisfy OperationNode");

// Ensure Variable is not an OperationNode
static_assert(!OperationNode<TestMatrix9>, 
    "Variable should NOT satisfy OperationNode");

// ===========================================
// Test Class
// ===========================================

class MatrixMultiplicationTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaError_t err = cudaSetDevice(0);
        if (err != cudaSuccess) {
            GTEST_SKIP() << "CUDA device not available";
        }
    }
};

// ===========================================
// Forward Pass Tests
// ===========================================

// Test kernel for 3x3 matrix multiplication forward pass
__global__ void test_matrix_multiply_3x3_forward_kernel(float* result) {
    // Test case: A = [[1,2,3],[4,5,6],[7,8,9]], B = [[9,8,7],[6,5,4],[3,2,1]]
    float a_data[9] = {1,2,3,4,5,6,7,8,9};
    float a_grad[9] = {0,0,0,0,0,0,0,0,0};
    float b_data[9] = {9,8,7,6,5,4,3,2,1};
    float b_grad[9] = {0,0,0,0,0,0,0,0,0};
    
    VariableRef<float, 9> A(a_data, a_grad);
    VariableRef<float, 9> B(b_data, b_grad);
    
    auto C = op::matrix_multiply_3x3(A, B);
    C.forward();
    
    // Expected result: C = [[30,24,18],[84,69,54],[138,114,90]]
    float expected[9] = {30,24,18,84,69,54,138,114,90};
    
    bool success = true;
    float tolerance = 1e-6f;
    
    for (int i = 0; i < 9; i++) {
        if (fabsf(C[i] - expected[i]) > tolerance) {
            success = false;
            break;
        }
    }
    
    *result = success ? 1.0f : 0.0f;
}

TEST_F(MatrixMultiplicationTest, ForwardPass) {
    auto device_result = makeCudaUnique<float>();
    
    test_matrix_multiply_3x3_forward_kernel<<<1, 1>>>(device_result.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    float host_result;
    ASSERT_EQ(cudaMemcpy(&host_result, device_result.get(), sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(host_result, 1.0f);
}

// Test identity matrix multiplication
__global__ void test_matrix_multiply_identity_kernel(float* result) {
    // Identity matrix * any matrix = same matrix
    float identity[9] = {1,0,0,0,1,0,0,0,1};
    float identity_grad[9] = {0,0,0,0,0,0,0,0,0};
    float matrix[9] = {2,3,4,5,6,7,8,9,10};
    float matrix_grad[9] = {0,0,0,0,0,0,0,0,0};
    
    VariableRef<float, 9> I(identity, identity_grad);
    VariableRef<float, 9> M(matrix, matrix_grad);
    
    auto R = op::matrix_multiply_3x3(I, M);
    R.forward();
    
    bool success = true;
    float tolerance = 1e-6f;
    
    // Result should be same as input matrix
    for (int i = 0; i < 9; i++) {
        if (fabsf(R[i] - matrix[i]) > tolerance) {
            success = false;
            break;
        }
    }
    
    *result = success ? 1.0f : 0.0f;
}

TEST_F(MatrixMultiplicationTest, IdentityMatrix) {
    auto device_result = makeCudaUnique<float>();
    
    test_matrix_multiply_identity_kernel<<<1, 1>>>(device_result.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    float host_result;
    ASSERT_EQ(cudaMemcpy(&host_result, device_result.get(), sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(host_result, 1.0f);
}

// ===========================================
// Gradient Verification Tests
// ===========================================

// Use the binary gradient tester utility
TEST_F(MatrixMultiplicationTest, GradientVerification) {
    using Logic = op::MatrixMultiplication3x3Logic<VariableRef<double, 9>, VariableRef<double, 9>>;
    test::BinaryGradientTester<Logic, 9, 9, 9>::test_custom(
        "MatrixMultiplication3x3", 
        50,      // num_tests
        1e-5,    // tolerance
        1e-6,    // delta
        -2.0,    // input_min
        2.0      // input_max
    );
}

// Test gradient with specific matrices
__global__ void test_matrix_multiply_gradient_kernel(double* result) {
    // Simple test case for gradient verification
    double a_data[9] = {1.0, 0.5, 0.2, 
                        0.3, 1.0, 0.4, 
                        0.1, 0.6, 1.0};
    double a_grad[9] = {0,0,0,0,0,0,0,0,0};
    double b_data[9] = {0.8, 0.3, 0.1,
                        0.2, 0.9, 0.2,
                        0.1, 0.3, 0.7};
    double b_grad[9] = {0,0,0,0,0,0,0,0,0};
    
    VariableRef<double, 9> A(a_data, a_grad);
    VariableRef<double, 9> B(b_data, b_grad);
    
    auto C = op::matrix_multiply_3x3(A, B);
    
    // Forward pass
    C.forward();
    
    // Set upstream gradient to all 1s
    C.zero_grad();
    for (int i = 0; i < 9; i++) {
        C.add_grad(i, 1.0);
    }
    
    // Analytical backward
    C.backward();
    
    // Save analytical gradients
    double analytical_grad_a[9];
    double analytical_grad_b[9];
    for (int i = 0; i < 9; i++) {
        analytical_grad_a[i] = A.grad(i);
        analytical_grad_b[i] = B.grad(i);
    }
    
    // Reset gradients
    for (int i = 0; i < 9; i++) {
        a_grad[i] = 0.0;
        b_grad[i] = 0.0;
    }
    
    // Numerical backward
    C.run_numerical(1e-7);
    
    // Check gradient consistency
    bool success = true;
    double tolerance = 1e-5;
    
    for (int i = 0; i < 9; i++) {
        double diff_a = fabs(analytical_grad_a[i] - A.grad(i));
        double diff_b = fabs(analytical_grad_b[i] - B.grad(i));
        
        if (diff_a > tolerance || diff_b > tolerance) {
            success = false;
            break;
        }
    }
    
    *result = success ? 1.0 : 0.0;
}

TEST_F(MatrixMultiplicationTest, SpecificGradientVerification) {
    auto device_result = makeCudaUnique<double>();
    
    test_matrix_multiply_gradient_kernel<<<1, 1>>>(device_result.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    double host_result;
    ASSERT_EQ(cudaMemcpy(&host_result, device_result.get(), sizeof(double), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(host_result, 1.0);
}

// ===========================================
// Interface Compliance Test
// ===========================================

__global__ void test_matrix_multiply_interface_kernel(float* result) {
    float a_data[9] = {1,2,3,4,5,6,7,8,9};
    float a_grad[9] = {0,0,0,0,0,0,0,0,0};
    float b_data[9] = {9,8,7,6,5,4,3,2,1};
    float b_grad[9] = {0,0,0,0,0,0,0,0,0};
    
    VariableRef<float, 9> A(a_data, a_grad);
    VariableRef<float, 9> B(b_data, b_grad);
    
    auto C = op::matrix_multiply_3x3(A, B);
    
    // Test VariableConcept interface
    C.zero_grad();  // zero_grad
    constexpr auto size = decltype(C)::size;  // size
    auto* data = C.data();  // data()
    auto* grad = C.grad();  // grad()
    auto value = C[0];  // operator[]
    auto grad_value = C.grad(0);  // grad(size_t)
    
    // Test OperationNode interface
    C.forward();  // forward
    C.backward();  // backward
    C.backward_numerical(1e-5);  // backward_numerical
    
    // Test convenience methods
    C.run();  // run (forward -> zero_grad -> add_grad -> backward)
    C.run_numerical(1e-5);  // run_numerical
    
    // Verify expected behavior
    bool success = (size == 9 && data != nullptr && grad != nullptr);
    
    *result = success ? 1.0f : 0.0f;
}

TEST_F(MatrixMultiplicationTest, InterfaceCompliance) {
    auto device_result = makeCudaUnique<float>();
    
    test_matrix_multiply_interface_kernel<<<1, 1>>>(device_result.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    float host_result;
    ASSERT_EQ(cudaMemcpy(&host_result, device_result.get(), sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(host_result, 1.0f);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}