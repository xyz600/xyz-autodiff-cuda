#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include "../../../include/variable.cuh"
#include "../../../include/concept/variable.cuh"
#include "../../../include/concept/operation_node.cuh"
#include "../../../include/operations/unary/add_constant_logic.cuh"
#include "../../../include/operations/unary/sub_constant_logic.cuh"
#include "../../../include/operations/unary/mul_constant_logic.cuh"
#include "../../../include/operations/unary/div_constant_logic.cuh"
#include "../../../include/util/cuda_unique_ptr.cuh"
#include "../../utility/unary_gradient_tester.cuh"

using namespace xyz_autodiff;

// ===========================================
// Static Assert Tests for Concept Compliance
// ===========================================

// Test types
using TestVector3 = Variable<3, float>;
using TestVectorRef3 = VariableRef<3, float>;

// Unary constant operation types
using AddConstOp = UnaryOperation<3, op::AddConstantLogic<TestVectorRef3>, TestVectorRef3>;
using SubConstOp = UnaryOperation<3, op::SubConstantLogic<TestVectorRef3>, TestVectorRef3>;
using MulConstOp = UnaryOperation<3, op::MulConstantLogic<TestVectorRef3>, TestVectorRef3>;
using DivConstOp = UnaryOperation<3, op::DivConstantLogic<TestVectorRef3>, TestVectorRef3>;

// Static assertions for concept compliance
static_assert(VariableConcept<AddConstOp>, "AddConstantOperation should satisfy VariableConcept");
static_assert(DifferentiableVariableConcept<AddConstOp>, "AddConstantOperation should satisfy DifferentiableVariableConcept");
static_assert(OperationNode<AddConstOp>, "AddConstantOperation should satisfy OperationNode");

static_assert(VariableConcept<SubConstOp>, "SubConstantOperation should satisfy VariableConcept");
static_assert(DifferentiableVariableConcept<SubConstOp>, "SubConstantOperation should satisfy DifferentiableVariableConcept");
static_assert(OperationNode<SubConstOp>, "SubConstantOperation should satisfy OperationNode");

static_assert(VariableConcept<MulConstOp>, "MulConstantOperation should satisfy VariableConcept");
static_assert(DifferentiableVariableConcept<MulConstOp>, "MulConstantOperation should satisfy DifferentiableVariableConcept");
static_assert(OperationNode<MulConstOp>, "MulConstantOperation should satisfy OperationNode");

static_assert(VariableConcept<DivConstOp>, "DivConstantOperation should satisfy VariableConcept");
static_assert(DifferentiableVariableConcept<DivConstOp>, "DivConstantOperation should satisfy DifferentiableVariableConcept");
static_assert(OperationNode<DivConstOp>, "DivConstantOperation should satisfy OperationNode");

// Ensure Variable is NOT an OperationNode
static_assert(!OperationNode<TestVector3>, "Variable should NOT be OperationNode");

// ===========================================
// Test Class
// ===========================================

class ConstantOperationsGradientTest : public ::testing::Test {
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

__global__ void test_add_constant_forward_kernel(float* result) {
    float input_data[3] = {1.0f, 2.5f, -1.0f};
    float input_grad[3] = {0.0f, 0.0f, 0.0f};
    
    VariableRef<3, float> input(input_data, input_grad);
    
    auto add_const_op = op::add_constant(input, 3.0f);
    add_const_op.forward();
    
    // Expected: [4.0, 5.5, 2.0]
    float tolerance = 1e-6f;
    bool success = (fabsf(add_const_op[0] - 4.0f) < tolerance &&
                   fabsf(add_const_op[1] - 5.5f) < tolerance &&
                   fabsf(add_const_op[2] - 2.0f) < tolerance);
    
    result[0] = success ? 1.0f : 0.0f;
}

__global__ void test_sub_constant_forward_kernel(float* result) {
    float input_data[3] = {5.0f, 2.5f, -1.0f};
    float input_grad[3] = {0.0f, 0.0f, 0.0f};
    
    VariableRef<3, float> input(input_data, input_grad);
    
    auto sub_const_op = op::sub_constant(input, 2.0f);
    sub_const_op.forward();
    
    // Expected: [3.0, 0.5, -3.0]
    float tolerance = 1e-6f;
    bool success = (fabsf(sub_const_op[0] - 3.0f) < tolerance &&
                   fabsf(sub_const_op[1] - 0.5f) < tolerance &&
                   fabsf(sub_const_op[2] - (-3.0f)) < tolerance);
    
    result[0] = success ? 1.0f : 0.0f;
}

__global__ void test_mul_constant_forward_kernel(float* result) {
    float input_data[3] = {2.0f, -1.5f, 0.5f};
    float input_grad[3] = {0.0f, 0.0f, 0.0f};
    
    VariableRef<3, float> input(input_data, input_grad);
    
    auto mul_const_op = op::mul_constant(input, 3.0f);
    mul_const_op.forward();
    
    // Expected: [6.0, -4.5, 1.5]
    float tolerance = 1e-6f;
    bool success = (fabsf(mul_const_op[0] - 6.0f) < tolerance &&
                   fabsf(mul_const_op[1] - (-4.5f)) < tolerance &&
                   fabsf(mul_const_op[2] - 1.5f) < tolerance);
    
    result[0] = success ? 1.0f : 0.0f;
}

__global__ void test_div_constant_forward_kernel(float* result) {
    float input_data[3] = {6.0f, -4.0f, 10.0f};
    float input_grad[3] = {0.0f, 0.0f, 0.0f};
    
    VariableRef<3, float> input(input_data, input_grad);
    
    auto div_const_op = op::div_constant(input, 2.0f);
    div_const_op.forward();
    
    // Expected: [3.0, -2.0, 5.0]
    float tolerance = 1e-6f;
    bool success = (fabsf(div_const_op[0] - 3.0f) < tolerance &&
                   fabsf(div_const_op[1] - (-2.0f)) < tolerance &&
                   fabsf(div_const_op[2] - 5.0f) < tolerance);
    
    result[0] = success ? 1.0f : 0.0f;
}

// ===========================================
// Test Cases
// ===========================================

TEST_F(ConstantOperationsGradientTest, AddConstantForwardPass) {
    auto device_result = makeCudaUnique<float>();
    test_add_constant_forward_kernel<<<1, 1>>>(device_result.get());
    cudaDeviceSynchronize();
    
    float host_result;
    cudaMemcpy(&host_result, device_result.get(), sizeof(float), cudaMemcpyDeviceToHost);
    EXPECT_EQ(host_result, 1.0f);
}

TEST_F(ConstantOperationsGradientTest, SubConstantForwardPass) {
    auto device_result = makeCudaUnique<float>();
    test_sub_constant_forward_kernel<<<1, 1>>>(device_result.get());
    cudaDeviceSynchronize();
    
    float host_result;
    cudaMemcpy(&host_result, device_result.get(), sizeof(float), cudaMemcpyDeviceToHost);
    EXPECT_EQ(host_result, 1.0f);
}

TEST_F(ConstantOperationsGradientTest, MulConstantForwardPass) {
    auto device_result = makeCudaUnique<float>();
    test_mul_constant_forward_kernel<<<1, 1>>>(device_result.get());
    cudaDeviceSynchronize();
    
    float host_result;
    cudaMemcpy(&host_result, device_result.get(), sizeof(float), cudaMemcpyDeviceToHost);
    EXPECT_EQ(host_result, 1.0f);
}

TEST_F(ConstantOperationsGradientTest, DivConstantForwardPass) {
    auto device_result = makeCudaUnique<float>();
    test_div_constant_forward_kernel<<<1, 1>>>(device_result.get());
    cudaDeviceSynchronize();
    
    float host_result;
    cudaMemcpy(&host_result, device_result.get(), sizeof(float), cudaMemcpyDeviceToHost);
    EXPECT_EQ(host_result, 1.0f);
}

// ===========================================
// Gradient Verification Tests (Manual Implementation)
// ===========================================
// Note: Constant operations cannot use UnaryGradientTester because they require
// constructor parameters (the constant value). We test gradients manually.

__global__ void test_constant_operations_gradients_kernel(double* result) {
    double input_data[3] = {1.0, 2.0, 3.0};
    double input_grad[3] = {0.0, 0.0, 0.0};
    
    VariableRef<3, double> input(input_data, input_grad);
    
    // Test add_constant gradient (should be 1.0)
    auto add_op = op::add_constant(input, 5.0);
    add_op.forward();
    add_op.zero_grad();
    for (int i = 0; i < 3; ++i) {
        add_op.add_grad(i, 1.0);
    }
    add_op.backward();
    
    bool add_gradients_correct = true;
    for (int i = 0; i < 3; ++i) {
        if (abs(input_grad[i] - 1.0) > 1e-10) {
            add_gradients_correct = false;
            break;
        }
    }
    
    // Reset gradients
    for (int i = 0; i < 3; ++i) {
        input_grad[i] = 0.0;
    }
    
    // Test mul_constant gradient (should be the constant value)
    auto mul_op = op::mul_constant(input, 3.0);
    mul_op.forward();
    mul_op.zero_grad();
    for (int i = 0; i < 3; ++i) {
        mul_op.add_grad(i, 1.0);
    }
    mul_op.backward();
    
    bool mul_gradients_correct = true;
    for (int i = 0; i < 3; ++i) {
        if (abs(input_grad[i] - 3.0) > 1e-10) {
            mul_gradients_correct = false;
            break;
        }
    }
    
    result[0] = (add_gradients_correct && mul_gradients_correct) ? 1.0 : 0.0;
}

TEST_F(ConstantOperationsGradientTest, ConstantOperationsGradientVerification) {
    auto device_result = makeCudaUnique<double>();
    test_constant_operations_gradients_kernel<<<1, 1>>>(device_result.get());
    cudaDeviceSynchronize();
    
    double host_result;
    cudaMemcpy(&host_result, device_result.get(), sizeof(double), cudaMemcpyDeviceToHost);
    EXPECT_EQ(host_result, 1.0);
}

// Test edge cases with specific constants
__global__ void test_gradient_properties_kernel(double* result) {
    // Test that gradient of add_constant is always 1
    double input_data[2] = {1.5, -2.0};
    double input_grad[2] = {0.0, 0.0};
    
    VariableRef<2, double> input(input_data, input_grad);
    
    auto add_const_op = op::add_constant(input, 5.0);
    
    // Forward pass
    add_const_op.forward();
    
    // Set upstream gradient and run backward
    add_const_op.zero_grad();
    add_const_op.add_grad(0, 1.0);  // d/dx1
    add_const_op.add_grad(1, 1.0);  // d/dx2
    add_const_op.backward();
    
    // For add_constant, gradient should be 1.0 for all elements
    bool success = (abs(input_grad[0] - 1.0) < 1e-10 && abs(input_grad[1] - 1.0) < 1e-10);
    result[0] = success ? 1.0 : 0.0;
}

TEST_F(ConstantOperationsGradientTest, AddConstantGradientProperties) {
    auto device_result = makeCudaUnique<double>();
    test_gradient_properties_kernel<<<1, 1>>>(device_result.get());
    cudaDeviceSynchronize();
    
    double host_result;
    cudaMemcpy(&host_result, device_result.get(), sizeof(double), cudaMemcpyDeviceToHost);
    EXPECT_EQ(host_result, 1.0);
}