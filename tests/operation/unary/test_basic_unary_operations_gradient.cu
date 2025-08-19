#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include "../../../include/variable.cuh"
#include "../../../include/concept/variable.cuh"
#include "../../../include/concept/operation_node.cuh"
#include "../../../include/operations/unary/squared_logic.cuh"
#include "../../../include/operations/unary/neg_logic.cuh"
#include "../../../include/util/cuda_unique_ptr.cuh"
#include "../../utility/unary_gradient_tester.cuh"

using namespace xyz_autodiff;

// ===========================================
// Static Assert Tests for Concept Compliance
// ===========================================

// Test types
using TestVector3 = Variable<3, float>;
using TestVectorRef3 = VariableRef<3, float>;

// Unary operation types
using SquaredOp = UnaryOperation<3, op::SquaredLogic<3>, TestVectorRef3>;
using NegOp = UnaryOperation<3, op::NegLogic<3>, TestVectorRef3>;

// Static assertions for concept compliance
static_assert(VariableConcept<SquaredOp>, "SquaredOperation should satisfy VariableConcept");
static_assert(DifferentiableVariableConcept<SquaredOp>, "SquaredOperation should satisfy DifferentiableVariableConcept");
static_assert(OperationNode<SquaredOp>, "SquaredOperation should satisfy OperationNode");

static_assert(VariableConcept<NegOp>, "NegOperation should satisfy VariableConcept");
static_assert(DifferentiableVariableConcept<NegOp>, "NegOperation should satisfy DifferentiableVariableConcept");
static_assert(OperationNode<NegOp>, "NegOperation should satisfy OperationNode");

// Ensure Variable is NOT an OperationNode
static_assert(!OperationNode<TestVector3>, "Variable should NOT be OperationNode");

// ===========================================
// Test Class
// ===========================================

class BasicUnaryOperationsGradientTest : public ::testing::Test {
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

__global__ void test_squared_forward_kernel(float* result) {
    float input_data[3] = {2.0f, -3.0f, 1.5f};
    float input_grad[3] = {0.0f, 0.0f, 0.0f};
    
    VariableRef<3, float> input(input_data, input_grad);
    
    auto squared_op = op::squared(input);
    squared_op.forward();
    
    // Expected: [4.0, 9.0, 2.25]
    float tolerance = 1e-6f;
    bool success = (fabsf(squared_op[0] - 4.0f) < tolerance &&
                   fabsf(squared_op[1] - 9.0f) < tolerance &&
                   fabsf(squared_op[2] - 2.25f) < tolerance);
    
    result[0] = success ? 1.0f : 0.0f;
}

__global__ void test_neg_forward_kernel(float* result) {
    float input_data[3] = {2.0f, -3.5f, 0.0f};
    float input_grad[3] = {0.0f, 0.0f, 0.0f};
    
    VariableRef<3, float> input(input_data, input_grad);
    
    auto neg_op = op::neg(input);
    neg_op.forward();
    
    // Expected: [-2.0, 3.5, 0.0]
    float tolerance = 1e-6f;
    bool success = (fabsf(neg_op[0] - (-2.0f)) < tolerance &&
                   fabsf(neg_op[1] - 3.5f) < tolerance &&
                   fabsf(neg_op[2] - 0.0f) < tolerance);
    
    result[0] = success ? 1.0f : 0.0f;
}

// ===========================================
// Test Cases
// ===========================================

TEST_F(BasicUnaryOperationsGradientTest, SquaredForwardPass) {
    auto device_result = makeCudaUnique<float>();
    test_squared_forward_kernel<<<1, 1>>>(device_result.get());
    cudaDeviceSynchronize();
    
    float host_result;
    cudaMemcpy(&host_result, device_result.get(), sizeof(float), cudaMemcpyDeviceToHost);
    EXPECT_EQ(host_result, 1.0f);
}

TEST_F(BasicUnaryOperationsGradientTest, NegForwardPass) {
    auto device_result = makeCudaUnique<float>();
    test_neg_forward_kernel<<<1, 1>>>(device_result.get());
    cudaDeviceSynchronize();
    
    float host_result;
    cudaMemcpy(&host_result, device_result.get(), sizeof(float), cudaMemcpyDeviceToHost);
    EXPECT_EQ(host_result, 1.0f);
}

// ===========================================
// Gradient Verification Tests
// ===========================================

TEST_F(BasicUnaryOperationsGradientTest, SquaredGradientVerification) {
    using Logic = op::SquaredLogic<3>;
    test::UnaryGradientTester<Logic, 3, 3>::test_custom(
        "SquaredLogic", 
        50,      // num_tests
        1e-5,    // tolerance
        1e-7,    // delta
        -5.0,    // input_min
        5.0      // input_max
    );
}

TEST_F(BasicUnaryOperationsGradientTest, NegGradientVerification) {
    using Logic = op::NegLogic<3>;
    test::UnaryGradientTester<Logic, 3, 3>::test_custom(
        "NegLogic", 
        50,      // num_tests
        1e-5,    // tolerance
        1e-7,    // delta
        -10.0,   // input_min
        10.0     // input_max
    );
}

// Test with different dimensions
TEST_F(BasicUnaryOperationsGradientTest, SquaredGradientVerification2D) {
    using Logic = op::SquaredLogic<2>;
    test::UnaryGradientTester<Logic, 2, 2>::test_custom(
        "SquaredLogic2D", 
        30,      // num_tests
        1e-5,    // tolerance
        1e-7,    // delta
        -3.0,    // input_min
        3.0      // input_max
    );
}

TEST_F(BasicUnaryOperationsGradientTest, NegGradientVerification1D) {
    using Logic = op::NegLogic<1>;
    test::UnaryGradientTester<Logic, 1, 1>::test_custom(
        "NegLogic1D", 
        30,      // num_tests
        1e-5,    // tolerance
        1e-7,    // delta
        -15.0,   // input_min
        15.0     // input_max
    );
}

// Test gradient properties
__global__ void test_squared_gradient_properties_kernel(double* result) {
    // Test that gradient of x^2 is 2x
    double input_data[2] = {3.0, -2.5};
    double input_grad[2] = {0.0, 0.0};
    
    VariableRef<2, double> input(input_data, input_grad);
    
    auto squared_op = op::squared(input);
    
    // Forward pass
    squared_op.forward();
    
    // Set upstream gradient and run backward
    squared_op.zero_grad();
    squared_op.add_grad(0, 1.0);  // d/dx1
    squared_op.add_grad(1, 1.0);  // d/dx2
    squared_op.backward();
    
    // For squared, gradient should be 2*x
    // input[0] = 3.0 -> gradient should be 6.0
    // input[1] = -2.5 -> gradient should be -5.0
    bool success = (abs(input_grad[0] - 6.0) < 1e-10 && 
                   abs(input_grad[1] - (-5.0)) < 1e-10);
    result[0] = success ? 1.0 : 0.0;
}

__global__ void test_neg_gradient_properties_kernel(double* result) {
    // Test that gradient of -x is -1
    double input_data[2] = {1.5, -2.0};
    double input_grad[2] = {0.0, 0.0};
    
    VariableRef<2, double> input(input_data, input_grad);
    
    auto neg_op = op::neg(input);
    
    // Forward pass
    neg_op.forward();
    
    // Set upstream gradient and run backward
    neg_op.zero_grad();
    neg_op.add_grad(0, 1.0);  // d/dx1
    neg_op.add_grad(1, 1.0);  // d/dx2
    neg_op.backward();
    
    // For negation, gradient should be -1.0 for all elements
    bool success = (abs(input_grad[0] - (-1.0)) < 1e-10 && 
                   abs(input_grad[1] - (-1.0)) < 1e-10);
    result[0] = success ? 1.0 : 0.0;
}

TEST_F(BasicUnaryOperationsGradientTest, SquaredGradientProperties) {
    auto device_result = makeCudaUnique<double>();
    test_squared_gradient_properties_kernel<<<1, 1>>>(device_result.get());
    cudaDeviceSynchronize();
    
    double host_result;
    cudaMemcpy(&host_result, device_result.get(), sizeof(double), cudaMemcpyDeviceToHost);
    EXPECT_EQ(host_result, 1.0);
}

TEST_F(BasicUnaryOperationsGradientTest, NegGradientProperties) {
    auto device_result = makeCudaUnique<double>();
    test_neg_gradient_properties_kernel<<<1, 1>>>(device_result.get());
    cudaDeviceSynchronize();
    
    double host_result;
    cudaMemcpy(&host_result, device_result.get(), sizeof(double), cudaMemcpyDeviceToHost);
    EXPECT_EQ(host_result, 1.0);
}

// Edge case tests
TEST_F(BasicUnaryOperationsGradientTest, SquaredNearZero) {
    using Logic = op::SquaredLogic<3>;
    test::UnaryGradientTester<Logic, 3, 3>::test_custom(
        "SquaredLogicNearZero", 
        30,      // num_tests
        1e-5,    // tolerance
        1e-8,    // delta (smaller for near-zero)
        -0.1,    // input_min (near zero)
        0.1      // input_max
    );
}

TEST_F(BasicUnaryOperationsGradientTest, SquaredLargeValues) {
    using Logic = op::SquaredLogic<2>;
    test::UnaryGradientTester<Logic, 2, 2>::test_custom(
        "SquaredLogicLargeValues", 
        20,      // num_tests
        1e-5,    // tolerance (minimum allowed for double precision)
        1e-6,    // delta
        -10.0,   // input_min
        10.0     // input_max
    );
}