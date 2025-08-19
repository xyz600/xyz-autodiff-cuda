#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include "../../../include/variable.cuh"
#include "../../../include/concept/variable.cuh"
#include "../../../include/concept/operation_node.cuh"
#include "../../../include/operations/unary/exp_logic.cuh"
#include "../operations/element_wise_exp.cuh"  // exp(-x)専用ロジックのため
#include "../../../include/util/cuda_unique_ptr.cuh"
#include "../../../tests/utility/unary_gradient_tester.cuh"

using namespace xyz_autodiff;

// ===========================================
// Static Assert Tests for Concept Compliance
// ===========================================

// Test types
using TestVector3 = Variable<float, 3>;
using TestVectorRef3 = VariableRef<float, 3>;
using ExpOp = UnaryOperation<3, ExpLogic<3>, TestVectorRef3>;
using ExpNegOp = UnaryOperation<3, op::ElementWiseExpNegLogic<TestVectorRef3>, TestVectorRef3>;

// Static assertions for concept compliance
static_assert(VariableConcept<TestVector3>, 
    "Variable<float, 3> should satisfy VariableConcept");
static_assert(DifferentiableVariableConcept<TestVector3>, 
    "Variable<float, 3> should satisfy DifferentiableVariableConcept");

static_assert(VariableConcept<ExpOp>, 
    "ElementWiseExp Operation should satisfy VariableConcept");
static_assert(DifferentiableVariableConcept<ExpOp>, 
    "ElementWiseExp Operation should satisfy DifferentiableVariableConcept");
static_assert(OperationNode<ExpOp>, 
    "ElementWiseExp Operation should satisfy OperationNode");

static_assert(VariableConcept<ExpNegOp>, 
    "ElementWiseExpNeg Operation should satisfy VariableConcept");
static_assert(DifferentiableVariableConcept<ExpNegOp>, 
    "ElementWiseExpNeg Operation should satisfy DifferentiableVariableConcept");
static_assert(OperationNode<ExpNegOp>, 
    "ElementWiseExpNeg Operation should satisfy OperationNode");

// Ensure Variable is not an OperationNode
static_assert(!OperationNode<TestVector3>, 
    "Variable should NOT satisfy OperationNode");

// ===========================================
// Test Class
// ===========================================

class ElementWiseExpTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaError_t err = cudaSetDevice(0);
        if (err != cudaSuccess) {
            GTEST_SKIP() << "CUDA device not available";
        }
    }
};

// ===========================================
// Forward Pass Tests - exp(x)
// ===========================================

__global__ void test_element_wise_exp_forward_kernel(float* result) {
    // Test vector [0, 1, 2] -> exp result = [1, e, e^2]
    float data[3] = {0.0f, 1.0f, 2.0f};
    float grad[3] = {0,0,0};
    
    VariableRef<float, 3> vec(data, grad);
    
    auto exp_result = exp(vec);
    exp_result.forward();
    
    float tolerance = 1e-5f;
    bool success = (fabsf(exp_result[0] - 1.0f) < tolerance &&
                   fabsf(exp_result[1] - expf(1.0f)) < tolerance &&
                   fabsf(exp_result[2] - expf(2.0f)) < tolerance);
    
    *result = success ? 1.0f : 0.0f;
}

TEST_F(ElementWiseExpTest, ExpForwardPass) {
    auto device_result = makeCudaUnique<float>();
    
    test_element_wise_exp_forward_kernel<<<1, 1>>>(device_result.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    float host_result;
    ASSERT_EQ(cudaMemcpy(&host_result, device_result.get(), sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(host_result, 1.0f);
}

// ===========================================
// Forward Pass Tests - exp(-x)
// ===========================================

__global__ void test_element_wise_exp_neg_forward_kernel(float* result) {
    // Test vector [0, 1, 2] -> exp(-x) result = [1, e^(-1), e^(-2)]
    float data[3] = {0.0f, 1.0f, 2.0f};
    float grad[3] = {0,0,0};
    
    VariableRef<float, 3> vec(data, grad);
    
    auto exp_neg_result = op::element_wise_exp_neg(vec);
    exp_neg_result.forward();
    
    float tolerance = 1e-5f;
    bool success = (fabsf(exp_neg_result[0] - 1.0f) < tolerance &&
                   fabsf(exp_neg_result[1] - expf(-1.0f)) < tolerance &&
                   fabsf(exp_neg_result[2] - expf(-2.0f)) < tolerance);
    
    *result = success ? 1.0f : 0.0f;
}

TEST_F(ElementWiseExpTest, ExpNegForwardPass) {
    auto device_result = makeCudaUnique<float>();
    
    test_element_wise_exp_neg_forward_kernel<<<1, 1>>>(device_result.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    float host_result;
    ASSERT_EQ(cudaMemcpy(&host_result, device_result.get(), sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(host_result, 1.0f);
}

// ===========================================
// Gradient Verification Tests - exp(x)
// ===========================================

TEST_F(ElementWiseExpTest, ExpGradientVerification) {
    using Logic = ExpLogic<3>;
    test::UnaryGradientTester<Logic, 3, 3>::test_custom(
        "ElementWiseExp", 
        50,      // num_tests
        1e-5,    // tolerance
        1e-7,    // delta
        -2.0,    // input_min (small range to avoid overflow)
        2.0      // input_max
    );
}

// ===========================================
// Gradient Verification Tests - exp(-x)
// ===========================================

TEST_F(ElementWiseExpTest, ExpNegGradientVerification) {
    using Logic = op::ElementWiseExpNegLogic<VariableRef<double, 3>>;
    test::UnaryGradientTester<Logic, 3, 3>::test_custom(
        "ElementWiseExpNeg", 
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

__global__ void test_exp_gradient_kernel(double* result) {
    // Test exp gradient: d(exp(x))/dx = exp(x)
    double data[3] = {0.0, 0.5, 1.0};
    double grad[3] = {0.0, 0.0, 0.0};
    
    VariableRef<double, 3> input(data, grad);
    auto exp_op = exp(input);
    
    // Forward pass
    exp_op.forward();
    
    // Set upstream gradient
    exp_op.zero_grad();
    for (int i = 0; i < 3; i++) {
        exp_op.add_grad(i, 1.0);
    }
    
    // Analytical backward
    exp_op.backward();
    
    // Save analytical gradients
    double analytical_grad[3];
    for (int i = 0; i < 3; i++) {
        analytical_grad[i] = input.grad(i);
    }
    
    // Reset gradients
    for (int i = 0; i < 3; i++) {
        grad[i] = 0.0;
    }
    
    // Numerical backward
    exp_op.run_numerical(1e-8);
    
    // Check gradient consistency
    bool success = true;
    double tolerance = 1e-5;
    
    for (int i = 0; i < 3; i++) {
        // Expected gradient: exp(x)
        double expected = exp(data[i]);
        double diff_analytical = fabs(analytical_grad[i] - expected);
        double diff_numerical = fabs(input.grad(i) - expected);
        
        if (diff_analytical > tolerance || diff_numerical > tolerance) {
            success = false;
            break;
        }
    }
    
    *result = success ? 1.0 : 0.0;
}

TEST_F(ElementWiseExpTest, ExpSpecificGradientVerification) {
    auto device_result = makeCudaUnique<double>();
    
    test_exp_gradient_kernel<<<1, 1>>>(device_result.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    double host_result;
    ASSERT_EQ(cudaMemcpy(&host_result, device_result.get(), sizeof(double), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(host_result, 1.0);
}

__global__ void test_exp_neg_gradient_kernel(double* result) {
    // Test exp(-x) gradient: d(exp(-x))/dx = -exp(-x)
    double data[3] = {0.0, 0.5, 1.0};
    double grad[3] = {0.0, 0.0, 0.0};
    
    VariableRef<double, 3> input(data, grad);
    auto exp_neg_op = op::element_wise_exp_neg(input);
    
    // Forward pass
    exp_neg_op.forward();
    
    // Set upstream gradient
    exp_neg_op.zero_grad();
    for (int i = 0; i < 3; i++) {
        exp_neg_op.add_grad(i, 1.0);
    }
    
    // Analytical backward
    exp_neg_op.backward();
    
    // Save analytical gradients
    double analytical_grad[3];
    for (int i = 0; i < 3; i++) {
        analytical_grad[i] = input.grad(i);
    }
    
    // Reset gradients
    for (int i = 0; i < 3; i++) {
        grad[i] = 0.0;
    }
    
    // Numerical backward
    exp_neg_op.run_numerical(1e-8);
    
    // Check gradient consistency
    bool success = true;
    double tolerance = 1e-5;
    
    for (int i = 0; i < 3; i++) {
        // Expected gradient: -exp(-x)
        double expected = -exp(-data[i]);
        double diff_analytical = fabs(analytical_grad[i] - expected);
        double diff_numerical = fabs(input.grad(i) - expected);
        
        if (diff_analytical > tolerance || diff_numerical > tolerance) {
            success = false;
            break;
        }
    }
    
    *result = success ? 1.0 : 0.0;
}

TEST_F(ElementWiseExpTest, ExpNegSpecificGradientVerification) {
    auto device_result = makeCudaUnique<double>();
    
    test_exp_neg_gradient_kernel<<<1, 1>>>(device_result.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    double host_result;
    ASSERT_EQ(cudaMemcpy(&host_result, device_result.get(), sizeof(double), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(host_result, 1.0);
}

// ===========================================
// Interface Compliance Test
// ===========================================

__global__ void test_exp_interface_kernel(float* result) {
    float data[3] = {0.0f, 1.0f, 2.0f};
    float grad[3] = {0,0,0};
    
    VariableRef<float, 3> input(data, grad);
    
    // Test both exp and exp_neg operations
    auto exp_op = exp(input);
    auto exp_neg_op = op::element_wise_exp_neg(input);
    
    // Test VariableConcept interface on exp
    exp_op.zero_grad();
    constexpr auto size = decltype(exp_op)::size;
    auto* exp_data = exp_op.data();
    auto* exp_grad = exp_op.grad();
    auto value = exp_op[0];
    auto grad_value = exp_op.grad(0);
    
    // Test OperationNode interface
    exp_op.forward();
    exp_op.backward();
    exp_op.backward_numerical(1e-5f);
    exp_op.run();
    exp_op.run_numerical(1e-5f);
    
    // Verify expected behavior
    bool success = (size == 3 && exp_data != nullptr && exp_grad != nullptr);
    
    *result = success ? 1.0f : 0.0f;
}

TEST_F(ElementWiseExpTest, InterfaceCompliance) {
    auto device_result = makeCudaUnique<float>();
    
    test_exp_interface_kernel<<<1, 1>>>(device_result.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    float host_result;
    ASSERT_EQ(cudaMemcpy(&host_result, device_result.get(), sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(host_result, 1.0f);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}