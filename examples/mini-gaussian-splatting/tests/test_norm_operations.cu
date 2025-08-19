#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include "../../../include/variable.cuh"
#include "../../../include/concept/variable.cuh"
#include "../../../include/concept/operation_node.cuh"
#include "../operations/norm_operations.cuh"
#include "../../../include/util/cuda_unique_ptr.cuh"
#include "../../../tests/utility/unary_gradient_tester.cuh"

using namespace xyz_autodiff;

// ===========================================
// Static Assert Tests for Concept Compliance
// ===========================================

// Test types
using TestVector3 = Variable<float, 3>;
using TestVectorRef3 = VariableRef<float, 3>;
using L1NormOp = UnaryOperation<1, op::L1NormLogic<TestVectorRef3>, TestVectorRef3>;
using L2NormOp = UnaryOperation<1, op::L2NormLogic<TestVectorRef3>, TestVectorRef3>;
using L2SquaredNormOp = UnaryOperation<1, op::L2SquaredNormLogic<TestVectorRef3>, TestVectorRef3>;

// Static assertions for concept compliance
static_assert(VariableConcept<TestVector3>, 
    "Variable<float, 3> should satisfy VariableConcept");
static_assert(DifferentiableVariableConcept<TestVector3>, 
    "Variable<float, 3> should satisfy DifferentiableVariableConcept");

static_assert(VariableConcept<L1NormOp>, 
    "L1Norm Operation should satisfy VariableConcept");
static_assert(DifferentiableVariableConcept<L1NormOp>, 
    "L1Norm Operation should satisfy DifferentiableVariableConcept");
static_assert(OperationNode<L1NormOp>, 
    "L1Norm Operation should satisfy OperationNode");

static_assert(VariableConcept<L2NormOp>, 
    "L2Norm Operation should satisfy VariableConcept");
static_assert(DifferentiableVariableConcept<L2NormOp>, 
    "L2Norm Operation should satisfy DifferentiableVariableConcept");
static_assert(OperationNode<L2NormOp>, 
    "L2Norm Operation should satisfy OperationNode");

static_assert(VariableConcept<L2SquaredNormOp>, 
    "L2SquaredNorm Operation should satisfy VariableConcept");
static_assert(DifferentiableVariableConcept<L2SquaredNormOp>, 
    "L2SquaredNorm Operation should satisfy DifferentiableVariableConcept");
static_assert(OperationNode<L2SquaredNormOp>, 
    "L2SquaredNorm Operation should satisfy OperationNode");

// Ensure Variable is not an OperationNode
static_assert(!OperationNode<TestVector3>, 
    "Variable should NOT satisfy OperationNode");

// ===========================================
// Test Class
// ===========================================

class NormOperationsTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaError_t err = cudaSetDevice(0);
        if (err != cudaSuccess) {
            GTEST_SKIP() << "CUDA device not available";
        }
    }
};

// ===========================================
// Forward Pass Tests - L1 Norm
// ===========================================

__global__ void test_l1_norm_forward_kernel(float* result) {
    // Test vector [3, -4, 5] -> L1 norm = 3 + 4 + 5 = 12
    float data[3] = {3.0f, -4.0f, 5.0f};
    float grad[3] = {0,0,0};
    
    VariableRef<float, 3> vec(data, grad);
    
    auto l1_result = op::l1_norm(vec);
    l1_result.forward();
    
    float expected = 12.0f;
    float tolerance = 1e-6f;
    bool success = (fabsf(l1_result[0] - expected) < tolerance);
    
    *result = success ? 1.0f : 0.0f;
}

TEST_F(NormOperationsTest, L1NormForwardPass) {
    auto device_result = makeCudaUnique<float>();
    
    test_l1_norm_forward_kernel<<<1, 1>>>(device_result.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    float host_result;
    ASSERT_EQ(cudaMemcpy(&host_result, device_result.get(), sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(host_result, 1.0f);
}

// ===========================================
// Forward Pass Tests - L2 Norm
// ===========================================

__global__ void test_l2_norm_forward_kernel(float* result) {
    // Test vector [3, 4] -> L2 norm = sqrt(9 + 16) = 5
    float data[2] = {3.0f, 4.0f};
    float grad[2] = {0,0};
    
    VariableRef<float, 2> vec(data, grad);
    
    auto l2_result = op::l2_norm(vec);
    l2_result.forward();
    
    float expected = 5.0f;
    float tolerance = 1e-6f;
    bool success = (fabsf(l2_result[0] - expected) < tolerance);
    
    *result = success ? 1.0f : 0.0f;
}

TEST_F(NormOperationsTest, L2NormForwardPass) {
    auto device_result = makeCudaUnique<float>();
    
    test_l2_norm_forward_kernel<<<1, 1>>>(device_result.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    float host_result;
    ASSERT_EQ(cudaMemcpy(&host_result, device_result.get(), sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(host_result, 1.0f);
}

// ===========================================
// Forward Pass Tests - L2 Squared Norm
// ===========================================

__global__ void test_l2_squared_norm_forward_kernel(float* result) {
    // Test vector [3, 4] -> L2 squared norm = 9 + 16 = 25
    float data[2] = {3.0f, 4.0f};
    float grad[2] = {0,0};
    
    VariableRef<float, 2> vec(data, grad);
    
    auto l2_sq_result = op::l2_squared_norm(vec);
    l2_sq_result.forward();
    
    float expected = 25.0f;
    float tolerance = 1e-6f;
    bool success = (fabsf(l2_sq_result[0] - expected) < tolerance);
    
    *result = success ? 1.0f : 0.0f;
}

TEST_F(NormOperationsTest, L2SquaredNormForwardPass) {
    auto device_result = makeCudaUnique<float>();
    
    test_l2_squared_norm_forward_kernel<<<1, 1>>>(device_result.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    float host_result;
    ASSERT_EQ(cudaMemcpy(&host_result, device_result.get(), sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(host_result, 1.0f);
}

// ===========================================
// Gradient Verification Tests - L1 Norm
// ===========================================

TEST_F(NormOperationsTest, L1NormGradientVerification) {
    using Logic = op::L1NormLogic<VariableRef<double, 3>>;
    test::UnaryGradientTester<Logic, 3, 1>::test_custom(
        "L1Norm", 
        50,      // num_tests
        1e-4,    // tolerance (L1 norm has non-smooth gradients at zero)
        1e-6,    // delta
        -2.0,    // input_min
        2.0      // input_max
    );
}

// ===========================================
// Gradient Verification Tests - L2 Norm
// ===========================================

TEST_F(NormOperationsTest, L2NormGradientVerification) {
    using Logic = op::L2NormLogic<VariableRef<double, 3>>;
    test::UnaryGradientTester<Logic, 3, 1>::test_custom(
        "L2Norm", 
        50,      // num_tests
        1e-5,    // tolerance
        1e-7,    // delta
        0.1,     // input_min (avoid zero for stability)
        2.0      // input_max
    );
}

// ===========================================
// Gradient Verification Tests - L2 Squared Norm
// ===========================================

TEST_F(NormOperationsTest, L2SquaredNormGradientVerification) {
    using Logic = op::L2SquaredNormLogic<VariableRef<double, 3>>;
    test::UnaryGradientTester<Logic, 3, 1>::test_custom(
        "L2SquaredNorm", 
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

__global__ void test_l2_squared_norm_gradient_kernel(double* result) {
    // Test L2 squared norm gradient: d(||x||²)/dx_i = 2*x_i
    double data[3] = {1.0, 2.0, 3.0};
    double grad[3] = {0.0, 0.0, 0.0};
    
    VariableRef<double, 3> input(data, grad);
    auto l2_sq_op = op::l2_squared_norm(input);
    
    // Forward pass
    l2_sq_op.forward();
    
    // Set upstream gradient
    l2_sq_op.zero_grad();
    l2_sq_op.add_grad(0, 1.0);
    
    // Analytical backward
    l2_sq_op.backward();
    
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
    l2_sq_op.run_numerical(1e-8);
    
    // Check gradient consistency
    bool success = true;
    double tolerance = 1e-5;
    
    for (int i = 0; i < 3; i++) {
        // Expected gradient: 2*x_i
        double expected = 2.0 * data[i];
        double diff_analytical = fabs(analytical_grad[i] - expected);
        double diff_numerical = fabs(input.grad(i) - expected);
        
        if (diff_analytical > tolerance || diff_numerical > tolerance) {
            success = false;
            break;
        }
    }
    
    *result = success ? 1.0 : 0.0;
}

TEST_F(NormOperationsTest, L2SquaredNormSpecificGradientVerification) {
    auto device_result = makeCudaUnique<double>();
    
    test_l2_squared_norm_gradient_kernel<<<1, 1>>>(device_result.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    double host_result;
    ASSERT_EQ(cudaMemcpy(&host_result, device_result.get(), sizeof(double), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(host_result, 1.0);
}

__global__ void test_l2_norm_gradient_kernel(double* result) {
    // Test L2 norm gradient: d(||x||)/dx_i = x_i / ||x||
    double data[2] = {3.0, 4.0};  // ||x|| = 5
    double grad[2] = {0.0, 0.0};
    
    VariableRef<double, 2> input(data, grad);
    auto l2_op = op::l2_norm(input);
    
    // Forward pass
    l2_op.forward();
    
    // Set upstream gradient
    l2_op.zero_grad();
    l2_op.add_grad(0, 1.0);
    
    // Analytical backward
    l2_op.backward();
    
    // Save analytical gradients
    double analytical_grad[2];
    for (int i = 0; i < 2; i++) {
        analytical_grad[i] = input.grad(i);
    }
    
    // Reset gradients
    for (int i = 0; i < 2; i++) {
        grad[i] = 0.0;
    }
    
    // Numerical backward
    l2_op.run_numerical(1e-8);
    
    // Check gradient consistency
    bool success = true;
    double tolerance = 1e-5;
    double norm = sqrt(data[0]*data[0] + data[1]*data[1]); // ||x|| = 5
    
    for (int i = 0; i < 2; i++) {
        // Expected gradient: x_i / ||x||
        double expected = data[i] / norm;
        double diff_analytical = fabs(analytical_grad[i] - expected);
        double diff_numerical = fabs(input.grad(i) - expected);
        
        if (diff_analytical > tolerance || diff_numerical > tolerance) {
            success = false;
            break;
        }
    }
    
    *result = success ? 1.0 : 0.0;
}

TEST_F(NormOperationsTest, L2NormSpecificGradientVerification) {
    auto device_result = makeCudaUnique<double>();
    
    test_l2_norm_gradient_kernel<<<1, 1>>>(device_result.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    double host_result;
    ASSERT_EQ(cudaMemcpy(&host_result, device_result.get(), sizeof(double), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(host_result, 1.0);
}

// ===========================================
// Interface Compliance Test
// ===========================================

__global__ void test_norm_interface_kernel(float* result) {
    float data[3] = {1.0f, 2.0f, 3.0f};
    float grad[3] = {0,0,0};
    
    VariableRef<float, 3> input(data, grad);
    
    // Test all norm operations
    auto l1_op = op::l1_norm(input);
    auto l2_op = op::l2_norm(input);
    auto l2_sq_op = op::l2_squared_norm(input);
    
    // Test VariableConcept interface on L1 norm
    l1_op.zero_grad();
    constexpr auto size = decltype(l1_op)::size;
    auto* l1_data = l1_op.data();
    auto* l1_grad = l1_op.grad();
    auto value = l1_op[0];
    auto grad_value = l1_op.grad(0);
    
    // Test OperationNode interface
    l1_op.forward();
    l1_op.backward();
    l1_op.backward_numerical(1e-5f);
    l1_op.run();
    l1_op.run_numerical(1e-5f);
    
    // Verify expected behavior (output size should be 1 for all norm operations)
    bool success = (size == 1 && l1_data != nullptr && l1_grad != nullptr);
    
    *result = success ? 1.0f : 0.0f;
}

TEST_F(NormOperationsTest, InterfaceCompliance) {
    auto device_result = makeCudaUnique<float>();
    
    test_norm_interface_kernel<<<1, 1>>>(device_result.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    float host_result;
    ASSERT_EQ(cudaMemcpy(&host_result, device_result.get(), sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(host_result, 1.0f);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}