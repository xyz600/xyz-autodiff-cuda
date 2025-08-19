#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include "../../../include/variable.cuh"
#include "../../../include/concept/variable.cuh"
#include "../../../include/concept/operation_node.cuh"
#include "../operations/element_wise_multiply.cuh"
#include "../../../include/util/cuda_unique_ptr.cuh"
#include "../../../tests/utility/binary_gradient_tester.cuh"

using namespace xyz_autodiff;

// ===========================================
// Static Assert Tests for Concept Compliance
// ===========================================

// Test types
using TestVector3 = Variable<float, 3>;
using TestVectorRef3 = VariableRef<float, 3>;
using ElemMulOp = BinaryOperation<3, op::ElementWiseMultiplyLogic<TestVectorRef3, TestVectorRef3>, TestVectorRef3, TestVectorRef3>;

// Static assertions for concept compliance
static_assert(VariableConcept<TestVector3>, 
    "Variable<float, 3> should satisfy VariableConcept");
static_assert(DifferentiableVariableConcept<TestVector3>, 
    "Variable<float, 3> should satisfy DifferentiableVariableConcept");

static_assert(VariableConcept<ElemMulOp>, 
    "ElementWiseMultiply Operation should satisfy VariableConcept");
static_assert(DifferentiableVariableConcept<ElemMulOp>, 
    "ElementWiseMultiply Operation should satisfy DifferentiableVariableConcept");
static_assert(OperationNode<ElemMulOp>, 
    "ElementWiseMultiply Operation should satisfy OperationNode");

// Ensure Variable is not an OperationNode
static_assert(!OperationNode<TestVector3>, 
    "Variable should NOT satisfy OperationNode");

// ===========================================
// Test Class
// ===========================================

class ElementWiseMultiplyTest : public ::testing::Test {
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

__global__ void test_element_wise_multiply_forward_kernel(float* result) {
    // Test vectors [2, 3, 4] * [1, 2, 3] = [2, 6, 12]
    float a_data[3] = {2.0f, 3.0f, 4.0f};
    float a_grad[3] = {0,0,0};
    float b_data[3] = {1.0f, 2.0f, 3.0f};
    float b_grad[3] = {0,0,0};
    
    VariableRef<float, 3> a(a_data, a_grad);
    VariableRef<float, 3> b(b_data, b_grad);
    
    auto result_vec = op::element_wise_multiply(a, b);
    result_vec.forward();
    
    float tolerance = 1e-6f;
    bool success = (fabsf(result_vec[0] - 2.0f) < tolerance &&
                   fabsf(result_vec[1] - 6.0f) < tolerance &&
                   fabsf(result_vec[2] - 12.0f) < tolerance);
    
    *result = success ? 1.0f : 0.0f;
}

TEST_F(ElementWiseMultiplyTest, ForwardPass) {
    auto device_result = makeCudaUnique<float>();
    
    test_element_wise_multiply_forward_kernel<<<1, 1>>>(device_result.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    float host_result;
    ASSERT_EQ(cudaMemcpy(&host_result, device_result.get(), sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(host_result, 1.0f);
}

__global__ void test_element_wise_multiply_identity_kernel(float* result) {
    // Test multiplication by ones: [5, 6, 7] * [1, 1, 1] = [5, 6, 7]
    float a_data[3] = {5.0f, 6.0f, 7.0f};
    float a_grad[3] = {0,0,0};
    float b_data[3] = {1.0f, 1.0f, 1.0f};
    float b_grad[3] = {0,0,0};
    
    VariableRef<float, 3> a(a_data, a_grad);
    VariableRef<float, 3> b(b_data, b_grad);
    
    auto result_vec = op::element_wise_multiply(a, b);
    result_vec.forward();
    
    float tolerance = 1e-6f;
    bool success = (fabsf(result_vec[0] - 5.0f) < tolerance &&
                   fabsf(result_vec[1] - 6.0f) < tolerance &&
                   fabsf(result_vec[2] - 7.0f) < tolerance);
    
    *result = success ? 1.0f : 0.0f;
}

TEST_F(ElementWiseMultiplyTest, IdentityMultiplication) {
    auto device_result = makeCudaUnique<float>();
    
    test_element_wise_multiply_identity_kernel<<<1, 1>>>(device_result.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    float host_result;
    ASSERT_EQ(cudaMemcpy(&host_result, device_result.get(), sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(host_result, 1.0f);
}

// ===========================================
// Gradient Verification Tests
// ===========================================

TEST_F(ElementWiseMultiplyTest, GradientVerification) {
    using Logic = op::ElementWiseMultiplyLogic<VariableRef<double, 3>, VariableRef<double, 3>>;
    test::BinaryGradientTester<Logic, 3, 3, 3>::test_custom(
        "ElementWiseMultiply", 
        50,      // num_tests
        1e-5,    // tolerance
        1e-7,    // delta
        -2.0,    // input_min
        2.0      // input_max
    );
}

// ===========================================
// Specific Gradient Tests
// ===========================================

__global__ void test_element_wise_multiply_gradient_kernel(double* result) {
    // Test element-wise multiply gradient: d(a*b)/da = b, d(a*b)/db = a
    double a_data[3] = {2.0, 3.0, 4.0};
    double a_grad[3] = {0.0, 0.0, 0.0};
    double b_data[3] = {1.5, 2.5, 3.5};
    double b_grad[3] = {0.0, 0.0, 0.0};
    
    VariableRef<double, 3> a(a_data, a_grad);
    VariableRef<double, 3> b(b_data, b_grad);
    
    auto mul_op = op::element_wise_multiply(a, b);
    
    // Forward pass
    mul_op.forward();
    
    // Set upstream gradient
    mul_op.zero_grad();
    for (int i = 0; i < 3; i++) {
        mul_op.add_grad(i, 1.0);
    }
    
    // Analytical backward
    mul_op.backward();
    
    // Save analytical gradients
    double analytical_grad_a[3];
    double analytical_grad_b[3];
    for (int i = 0; i < 3; i++) {
        analytical_grad_a[i] = a.grad(i);
        analytical_grad_b[i] = b.grad(i);
    }
    
    // Reset gradients
    for (int i = 0; i < 3; i++) {
        a_grad[i] = 0.0;
        b_grad[i] = 0.0;
    }
    
    // Numerical backward
    mul_op.run_numerical(1e-8);
    
    // Check gradient consistency
    bool success = true;
    double tolerance = 1e-5;
    
    for (int i = 0; i < 3; i++) {
        // Expected gradients: da = b, db = a
        double expected_grad_a = b_data[i];
        double expected_grad_b = a_data[i];
        
        double diff_analytical_a = fabs(analytical_grad_a[i] - expected_grad_a);
        double diff_numerical_a = fabs(a.grad(i) - expected_grad_a);
        double diff_analytical_b = fabs(analytical_grad_b[i] - expected_grad_b);
        double diff_numerical_b = fabs(b.grad(i) - expected_grad_b);
        
        if (diff_analytical_a > tolerance || diff_numerical_a > tolerance ||
            diff_analytical_b > tolerance || diff_numerical_b > tolerance) {
            success = false;
            break;
        }
    }
    
    *result = success ? 1.0 : 0.0;
}

TEST_F(ElementWiseMultiplyTest, SpecificGradientVerification) {
    auto device_result = makeCudaUnique<double>();
    
    test_element_wise_multiply_gradient_kernel<<<1, 1>>>(device_result.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    double host_result;
    ASSERT_EQ(cudaMemcpy(&host_result, device_result.get(), sizeof(double), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(host_result, 1.0);
}

// ===========================================
// Interface Compliance Test
// ===========================================

__global__ void test_element_wise_multiply_interface_kernel(float* result) {
    float a_data[3] = {1.0f, 2.0f, 3.0f};
    float a_grad[3] = {0,0,0};
    float b_data[3] = {4.0f, 5.0f, 6.0f};
    float b_grad[3] = {0,0,0};
    
    VariableRef<float, 3> a(a_data, a_grad);
    VariableRef<float, 3> b(b_data, b_grad);
    
    auto mul_op = op::element_wise_multiply(a, b);
    
    // Test VariableConcept interface
    mul_op.zero_grad();
    constexpr auto size = decltype(mul_op)::size;
    auto* data = mul_op.data();
    auto* grad = mul_op.grad();
    auto value = mul_op[0];
    auto grad_value = mul_op.grad(0);
    
    // Test OperationNode interface
    mul_op.forward();
    mul_op.backward();
    mul_op.backward_numerical(1e-5f);
    mul_op.run();
    mul_op.run_numerical(1e-5f);
    
    // Verify expected behavior
    bool success = (size == 3 && data != nullptr && grad != nullptr);
    
    *result = success ? 1.0f : 0.0f;
}

TEST_F(ElementWiseMultiplyTest, InterfaceCompliance) {
    auto device_result = makeCudaUnique<float>();
    
    test_element_wise_multiply_interface_kernel<<<1, 1>>>(device_result.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    float host_result;
    ASSERT_EQ(cudaMemcpy(&host_result, device_result.get(), sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(host_result, 1.0f);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}