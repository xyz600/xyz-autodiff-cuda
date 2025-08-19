#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include "../../../include/variable.cuh"
#include "../../../include/concept/variable.cuh"
#include "../../../include/concept/operation_node.cuh"
#include "../../../include/operations/unary/sin_logic.cuh"
#include "../../../include/operations/unary/cos_logic.cuh"
#include "../../../include/util/cuda_unique_ptr.cuh"
#include "../../utility/unary_gradient_tester.cuh"

using namespace xyz_autodiff;

// ===========================================
// Static Assert Tests for Concept Compliance
// ===========================================

// Test types
using TestVector3 = Variable<3, float>;
using TestVectorRef3 = VariableRef<3, float>;

// Trigonometric operation types
using SinOp = UnaryOperation<3, op::SinLogic<3>, TestVectorRef3>;
using CosOp = UnaryOperation<3, op::CosLogic<3>, TestVectorRef3>;

// Static assertions for concept compliance
static_assert(VariableConcept<SinOp>, "SinOperation should satisfy VariableConcept");
static_assert(DifferentiableVariableConcept<SinOp>, "SinOperation should satisfy DifferentiableVariableConcept");
static_assert(OperationNode<SinOp>, "SinOperation should satisfy OperationNode");

static_assert(VariableConcept<CosOp>, "CosOperation should satisfy VariableConcept");
static_assert(DifferentiableVariableConcept<CosOp>, "CosOperation should satisfy DifferentiableVariableConcept");
static_assert(OperationNode<CosOp>, "CosOperation should satisfy OperationNode");

// Ensure Variable is NOT an OperationNode
static_assert(!OperationNode<TestVector3>, "Variable should NOT be OperationNode");

// ===========================================
// Test Class
// ===========================================

class TrigonometricOperationsGradientTest : public ::testing::Test {
protected:
    void SetUp() override {
        int device_count;
        cudaGetDeviceCount(&device_count);
        if (device_count == 0) {
            GTEST_SKIP() << "No CUDA devices available";
        }
    }
};

// ===========================================
// Forward Pass Tests
// ===========================================

__global__ void test_sin_forward_kernel(float* result) {
    float input_data[3] = {0.0f, 1.5707963f, 3.1415927f}; // 0, π/2, π
    float input_grad[3] = {0.0f, 0.0f, 0.0f};
    
    VariableRef<3, float> input(input_data, input_grad);
    
    auto sin_op = op::sin(input);
    sin_op.forward();
    
    // Expected: [0.0, 1.0, 0.0] (approximately)
    float tolerance = 1e-5f;
    bool success = (fabsf(sin_op[0] - 0.0f) < tolerance &&
                   fabsf(sin_op[1] - 1.0f) < tolerance &&
                   fabsf(sin_op[2] - 0.0f) < tolerance);
    
    result[0] = success ? 1.0f : 0.0f;
}

__global__ void test_cos_forward_kernel(float* result) {
    float input_data[3] = {0.0f, 1.5707963f, 3.1415927f}; // 0, π/2, π
    float input_grad[3] = {0.0f, 0.0f, 0.0f};
    
    VariableRef<3, float> input(input_data, input_grad);
    
    auto cos_op = op::cos(input);
    cos_op.forward();
    
    // Expected: [1.0, 0.0, -1.0] (approximately)
    float tolerance = 1e-5f;
    bool success = (fabsf(cos_op[0] - 1.0f) < tolerance &&
                   fabsf(cos_op[1] - 0.0f) < tolerance &&
                   fabsf(cos_op[2] - (-1.0f)) < tolerance);
    
    result[0] = success ? 1.0f : 0.0f;
}

// ===========================================
// Test Cases
// ===========================================

TEST_F(TrigonometricOperationsGradientTest, SinForwardPass) {
    auto device_result = makeCudaUnique<float>();
    test_sin_forward_kernel<<<1, 1>>>(device_result.get());
    cudaDeviceSynchronize();
    
    float host_result;
    cudaMemcpy(&host_result, device_result.get(), sizeof(float), cudaMemcpyDeviceToHost);
    EXPECT_EQ(host_result, 1.0f);
}

TEST_F(TrigonometricOperationsGradientTest, CosForwardPass) {
    auto device_result = makeCudaUnique<float>();
    test_cos_forward_kernel<<<1, 1>>>(device_result.get());
    cudaDeviceSynchronize();
    
    float host_result;
    cudaMemcpy(&host_result, device_result.get(), sizeof(float), cudaMemcpyDeviceToHost);
    EXPECT_EQ(host_result, 1.0f);
}

// ===========================================
// Gradient Verification Tests
// ===========================================

TEST_F(TrigonometricOperationsGradientTest, SinGradientVerification) {
    using Logic = op::SinLogic<3>;
    test::UnaryGradientTester<Logic, 3, 3>::test_custom(
        "SinLogic", 
        50,      // num_tests
        1e-5,    // tolerance
        1e-7,    // delta
        -3.14159, // input_min (-π)
        3.14159   // input_max (π)
    );
}

TEST_F(TrigonometricOperationsGradientTest, CosGradientVerification) {
    using Logic = op::CosLogic<3>;
    test::UnaryGradientTester<Logic, 3, 3>::test_custom(
        "CosLogic", 
        50,      // num_tests
        1e-5,    // tolerance
        1e-7,    // delta
        -3.14159, // input_min (-π)
        3.14159   // input_max (π)
    );
}

// Test with different dimensions
TEST_F(TrigonometricOperationsGradientTest, SinGradientVerification2D) {
    using Logic = op::SinLogic<2>;
    test::UnaryGradientTester<Logic, 2, 2>::test_custom(
        "SinLogic2D", 
        30,      // num_tests
        1e-5,    // tolerance
        1e-7,    // delta
        -6.28318, // input_min (-2π)
        6.28318   // input_max (2π)
    );
}

TEST_F(TrigonometricOperationsGradientTest, CosGradientVerification1D) {
    using Logic = op::CosLogic<1>;
    test::UnaryGradientTester<Logic, 1, 1>::test_custom(
        "CosLogic1D", 
        30,      // num_tests
        1e-5,    // tolerance
        1e-7,    // delta
        -1.5708,  // input_min (-π/2)
        1.5708    // input_max (π/2)
    );
}

// Test gradient properties
__global__ void test_sin_gradient_properties_kernel(double* result) {
    // Test that gradient of sin(x) is cos(x)
    double input_data[3] = {0.0, 1.5707963267948966, 3.141592653589793}; // 0, π/2, π
    double input_grad[3] = {0.0, 0.0, 0.0};
    
    VariableRef<3, double> input(input_data, input_grad);
    
    auto sin_op = op::sin(input);
    
    // Forward pass
    sin_op.forward();
    
    // Set upstream gradient and run backward
    sin_op.zero_grad();
    sin_op.add_grad(0, 1.0);  // d/dx1
    sin_op.add_grad(1, 1.0);  // d/dx2
    sin_op.add_grad(2, 1.0);  // d/dx3
    sin_op.backward();
    
    // For sin, gradient should be cos(x)
    // cos(0) = 1.0, cos(π/2) ≈ 0.0, cos(π) = -1.0
    double tolerance = 1e-10;
    bool success = (abs(input_grad[0] - 1.0) < tolerance &&
                   abs(input_grad[1] - 0.0) < tolerance &&
                   abs(input_grad[2] - (-1.0)) < tolerance);
    result[0] = success ? 1.0 : 0.0;
}

__global__ void test_cos_gradient_properties_kernel(double* result) {
    // Test that gradient of cos(x) is -sin(x)
    double input_data[3] = {0.0, 1.5707963267948966, 3.141592653589793}; // 0, π/2, π
    double input_grad[3] = {0.0, 0.0, 0.0};
    
    VariableRef<3, double> input(input_data, input_grad);
    
    auto cos_op = op::cos(input);
    
    // Forward pass
    cos_op.forward();
    
    // Set upstream gradient and run backward
    cos_op.zero_grad();
    cos_op.add_grad(0, 1.0);  // d/dx1
    cos_op.add_grad(1, 1.0);  // d/dx2
    cos_op.add_grad(2, 1.0);  // d/dx3
    cos_op.backward();
    
    // For cos, gradient should be -sin(x)
    // -sin(0) = 0.0, -sin(π/2) = -1.0, -sin(π) ≈ 0.0
    double tolerance = 1e-10;
    bool success = (abs(input_grad[0] - 0.0) < tolerance &&
                   abs(input_grad[1] - (-1.0)) < tolerance &&
                   abs(input_grad[2] - 0.0) < tolerance);
    result[0] = success ? 1.0 : 0.0;
}

TEST_F(TrigonometricOperationsGradientTest, SinGradientProperties) {
    auto device_result = makeCudaUnique<double>();
    test_sin_gradient_properties_kernel<<<1, 1>>>(device_result.get());
    cudaDeviceSynchronize();
    
    double host_result;
    cudaMemcpy(&host_result, device_result.get(), sizeof(double), cudaMemcpyDeviceToHost);
    EXPECT_EQ(host_result, 1.0);
}

TEST_F(TrigonometricOperationsGradientTest, CosGradientProperties) {
    auto device_result = makeCudaUnique<double>();
    test_cos_gradient_properties_kernel<<<1, 1>>>(device_result.get());
    cudaDeviceSynchronize();
    
    double host_result;
    cudaMemcpy(&host_result, device_result.get(), sizeof(double), cudaMemcpyDeviceToHost);
    EXPECT_EQ(host_result, 1.0);
}

// Test edge cases - near critical points
TEST_F(TrigonometricOperationsGradientTest, SinNearZero) {
    using Logic = op::SinLogic<3>;
    test::UnaryGradientTester<Logic, 3, 3>::test_custom(
        "SinLogicNearZero", 
        30,      // num_tests
        1e-5,    // tolerance
        1e-8,    // delta (smaller for near-zero)
        -0.1,    // input_min (near zero)
        0.1      // input_max
    );
}

TEST_F(TrigonometricOperationsGradientTest, CosNearPiHalf) {
    using Logic = op::CosLogic<2>;
    test::UnaryGradientTester<Logic, 2, 2>::test_custom(
        "CosLogicNearPiHalf", 
        30,      // num_tests
        1e-5,    // tolerance
        1e-7,    // delta
        1.4708,   // input_min (near π/2)
        1.6708    // input_max
    );
}

// Test larger range
TEST_F(TrigonometricOperationsGradientTest, SinLargeRange) {
    using Logic = op::SinLogic<2>;
    test::UnaryGradientTester<Logic, 2, 2>::test_custom(
        "SinLogicLargeRange", 
        40,      // num_tests
        1e-5,    // tolerance
        1e-7,    // delta
        -10.0,   // input_min
        10.0     // input_max
    );
}

TEST_F(TrigonometricOperationsGradientTest, CosLargeRange) {
    using Logic = op::CosLogic<2>;
    test::UnaryGradientTester<Logic, 2, 2>::test_custom(
        "CosLogicLargeRange", 
        40,      // num_tests
        1e-5,    // tolerance
        1e-7,    // delta
        -10.0,   // input_min
        10.0     // input_max
    );
}