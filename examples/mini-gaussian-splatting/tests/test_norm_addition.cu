#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include "../../../include/variable.cuh"
#include "../../../include/concept/variable.cuh"
#include "../../../include/concept/operation_node.cuh"
#include "../operations/norm_addition.cuh"
#include "../../../include/util/cuda_unique_ptr.cuh"
#include "../../../tests/utility/binary_gradient_tester.cuh"

using namespace xyz_autodiff;

// ===========================================
// Static Assert Tests for Concept Compliance
// ===========================================

// Test types
using TestScalar = Variable<float, 1>;
using TestScalarRef = VariableRef<float, 1>;
using NormAddOp = BinaryOperation<1, op::ScalarAdditionLogic<TestScalarRef, TestScalarRef>, TestScalarRef, TestScalarRef>;

// Static assertions for concept compliance
static_assert(VariableConcept<TestScalar>, 
    "Variable<float, 1> should satisfy VariableConcept");
static_assert(DifferentiableVariableConcept<TestScalar>, 
    "Variable<float, 1> should satisfy DifferentiableVariableConcept");

static_assert(VariableConcept<NormAddOp>, 
    "ScalarAddition Operation should satisfy VariableConcept");
static_assert(DifferentiableVariableConcept<NormAddOp>, 
    "ScalarAddition Operation should satisfy DifferentiableVariableConcept");
static_assert(OperationNode<NormAddOp>, 
    "ScalarAddition Operation should satisfy OperationNode");

// Ensure Variable is not an OperationNode
static_assert(!OperationNode<TestScalar>, 
    "Variable should NOT satisfy OperationNode");

// ===========================================
// Test Class
// ===========================================

class NormAdditionTest : public ::testing::Test {
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

__global__ void test_norm_addition_forward_kernel(float* result) {
    // Test case: norm1 = 3.0, norm2 = 4.0 -> result = 3.0 + 4.0 = 7.0
    float norm1_data[1] = {3.0f};
    float norm1_grad[1] = {0.0f};
    float norm2_data[1] = {4.0f};
    float norm2_grad[1] = {0.0f};
    
    VariableRef<float, 1> norm1(norm1_data, norm1_grad);
    VariableRef<float, 1> norm2(norm2_data, norm2_grad);
    
    auto sum_result = op::scalar_add(norm1, norm2);
    sum_result.forward();
    
    float expected = 7.0f;
    float tolerance = 1e-6f;
    bool success = (fabsf(sum_result[0] - expected) < tolerance);
    
    *result = success ? 1.0f : 0.0f;
}

TEST_F(NormAdditionTest, ForwardPass) {
    auto device_result = makeCudaUnique<float>();
    
    test_norm_addition_forward_kernel<<<1, 1>>>(device_result.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    float host_result;
    ASSERT_EQ(cudaMemcpy(&host_result, device_result.get(), sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(host_result, 1.0f);
}

__global__ void test_norm_addition_zero_kernel(float* result) {
    // Test case: norm1 = 5.0, norm2 = 0.0 -> result = 5.0
    float norm1_data[1] = {5.0f};
    float norm1_grad[1] = {0.0f};
    float norm2_data[1] = {0.0f};
    float norm2_grad[1] = {0.0f};
    
    VariableRef<float, 1> norm1(norm1_data, norm1_grad);
    VariableRef<float, 1> norm2(norm2_data, norm2_grad);
    
    auto sum_result = op::scalar_add(norm1, norm2);
    sum_result.forward();
    
    float expected = 5.0f;
    float tolerance = 1e-6f;
    bool success = (fabsf(sum_result[0] - expected) < tolerance);
    
    *result = success ? 1.0f : 0.0f;
}

TEST_F(NormAdditionTest, ZeroAddition) {
    auto device_result = makeCudaUnique<float>();
    
    test_norm_addition_zero_kernel<<<1, 1>>>(device_result.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    float host_result;
    ASSERT_EQ(cudaMemcpy(&host_result, device_result.get(), sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(host_result, 1.0f);
}

__global__ void test_norm_addition_negative_kernel(float* result) {
    // Test case: norm1 = 2.5, norm2 = -1.5 -> result = 1.0
    float norm1_data[1] = {2.5f};
    float norm1_grad[1] = {0.0f};
    float norm2_data[1] = {-1.5f};
    float norm2_grad[1] = {0.0f};
    
    VariableRef<float, 1> norm1(norm1_data, norm1_grad);
    VariableRef<float, 1> norm2(norm2_data, norm2_grad);
    
    auto sum_result = op::scalar_add(norm1, norm2);
    sum_result.forward();
    
    float expected = 1.0f;
    float tolerance = 1e-6f;
    bool success = (fabsf(sum_result[0] - expected) < tolerance);
    
    *result = success ? 1.0f : 0.0f;
}

TEST_F(NormAdditionTest, NegativeAddition) {
    auto device_result = makeCudaUnique<float>();
    
    test_norm_addition_negative_kernel<<<1, 1>>>(device_result.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    float host_result;
    ASSERT_EQ(cudaMemcpy(&host_result, device_result.get(), sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(host_result, 1.0f);
}

// ===========================================
// Gradient Verification Tests
// ===========================================

TEST_F(NormAdditionTest, GradientVerification) {
    using Logic = op::ScalarAdditionLogic<VariableRef<double, 1>, VariableRef<double, 1>>;
    test::BinaryGradientTester<Logic, 1, 1, 1>::test_custom(
        "NormAddition", 
        100,     // num_tests
        1e-7,    // tolerance (relaxed for numerical precision)
        1e-8,    // delta
        -5.0,    // input_min
        5.0      // input_max
    );
}

// ===========================================
// Specific Gradient Tests
// ===========================================

__global__ void test_norm_addition_gradient_kernel(double* result) {
    // Test norm addition gradient: d(a + b)/da = 1, d(a + b)/db = 1
    double norm1_data[1] = {2.5};
    double norm1_grad[1] = {0.0};
    double norm2_data[1] = {3.7};
    double norm2_grad[1] = {0.0};
    
    VariableRef<double, 1> norm1(norm1_data, norm1_grad);
    VariableRef<double, 1> norm2(norm2_data, norm2_grad);
    
    auto add_op = op::scalar_add(norm1, norm2);
    
    // Forward pass
    add_op.forward();
    
    // Set upstream gradient
    add_op.zero_grad();
    add_op.add_grad(0, 1.0);
    
    // Analytical backward
    add_op.backward();
    
    // Save analytical gradients
    double analytical_grad1 = norm1.grad(0);
    double analytical_grad2 = norm2.grad(0);
    
    // Reset gradients
    norm1_grad[0] = 0.0;
    norm2_grad[0] = 0.0;
    
    // Numerical backward
    add_op.run_numerical(1e-8);
    
    // Check gradient consistency
    bool success = true;
    double tolerance = 1e-6;
    
    // Expected gradients: both should be 1.0
    double expected = 1.0;
    double diff_analytical1 = fabs(analytical_grad1 - expected);
    double diff_numerical1 = fabs(norm1.grad(0) - expected);
    double diff_analytical2 = fabs(analytical_grad2 - expected);
    double diff_numerical2 = fabs(norm2.grad(0) - expected);
    
    if (diff_analytical1 > tolerance || diff_numerical1 > tolerance ||
        diff_analytical2 > tolerance || diff_numerical2 > tolerance) {
        success = false;
    }
    
    *result = success ? 1.0 : 0.0;
}

TEST_F(NormAdditionTest, SpecificGradientVerification) {
    auto device_result = makeCudaUnique<double>();
    
    test_norm_addition_gradient_kernel<<<1, 1>>>(device_result.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    double host_result;
    ASSERT_EQ(cudaMemcpy(&host_result, device_result.get(), sizeof(double), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(host_result, 1.0);
}

__global__ void test_norm_addition_chain_rule_kernel(double* result) {
    // Test chain rule with upstream gradient != 1
    double norm1_data[1] = {1.0};
    double norm1_grad[1] = {0.0};
    double norm2_data[1] = {2.0};
    double norm2_grad[1] = {0.0};
    
    VariableRef<double, 1> norm1(norm1_data, norm1_grad);
    VariableRef<double, 1> norm2(norm2_data, norm2_grad);
    
    auto add_op = op::scalar_add(norm1, norm2);
    
    // Forward pass
    add_op.forward();
    
    // Set upstream gradient to 2.5
    double upstream_grad = 2.5;
    add_op.zero_grad();
    add_op.add_grad(0, upstream_grad);
    
    // Analytical backward
    add_op.backward();
    
    // Check that gradients are properly scaled
    bool success = true;
    double tolerance = 1e-10;
    
    // Expected gradients: both should be upstream_grad * 1.0 = 2.5
    double expected = upstream_grad;
    double diff1 = fabs(norm1.grad(0) - expected);
    double diff2 = fabs(norm2.grad(0) - expected);
    
    if (diff1 > tolerance || diff2 > tolerance) {
        success = false;
    }
    
    *result = success ? 1.0 : 0.0;
}

TEST_F(NormAdditionTest, ChainRuleVerification) {
    auto device_result = makeCudaUnique<double>();
    
    test_norm_addition_chain_rule_kernel<<<1, 1>>>(device_result.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    double host_result;
    ASSERT_EQ(cudaMemcpy(&host_result, device_result.get(), sizeof(double), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(host_result, 1.0);
}

// ===========================================
// Interface Compliance Test
// ===========================================

__global__ void test_norm_addition_interface_kernel(float* result) {
    float norm1_data[1] = {1.5f};
    float norm1_grad[1] = {0.0f};
    float norm2_data[1] = {2.3f};
    float norm2_grad[1] = {0.0f};
    
    VariableRef<float, 1> norm1(norm1_data, norm1_grad);
    VariableRef<float, 1> norm2(norm2_data, norm2_grad);
    
    auto add_op = op::scalar_add(norm1, norm2);
    
    // Test VariableConcept interface
    add_op.zero_grad();
    constexpr auto size = decltype(add_op)::size;
    auto* data = add_op.data();
    auto* grad = add_op.grad();
    auto value = add_op[0];
    auto grad_value = add_op.grad(0);
    
    // Test OperationNode interface
    add_op.forward();
    add_op.backward();
    add_op.backward_numerical(1e-5f);
    add_op.run();
    add_op.run_numerical(1e-5f);
    
    // Verify expected behavior (output size should be 1)
    bool success = (size == 1 && data != nullptr && grad != nullptr);
    
    *result = success ? 1.0f : 0.0f;
}

TEST_F(NormAdditionTest, InterfaceCompliance) {
    auto device_result = makeCudaUnique<float>();
    
    test_norm_addition_interface_kernel<<<1, 1>>>(device_result.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    float host_result;
    ASSERT_EQ(cudaMemcpy(&host_result, device_result.get(), sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(host_result, 1.0f);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}