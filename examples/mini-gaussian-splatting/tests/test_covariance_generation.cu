#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include "../../../include/variable.cuh"
#include "../../../include/concept/variable.cuh"
#include "../../../include/concept/operation_node.cuh"
#include "../operations/covariance_generation.cuh"
#include "../../../include/util/cuda_unique_ptr.cuh"
#include "../../../tests/utility/binary_gradient_tester.cuh"
#include "../../../tests/utility/unary_gradient_tester.cuh"

using namespace xyz_autodiff;

// ===========================================
// Static Assert Tests for Concept Compliance
// ===========================================

// Test types
using TestVector2 = Variable<float, 2>;
using TestVector1 = Variable<float, 1>;
using TestVectorRef2 = VariableRef<float, 2>;
using TestVectorRef1 = VariableRef<float, 1>;
using CovGenOp = BinaryOperation<4, op::CovarianceMatrixGenerationLogic<TestVectorRef2, TestVectorRef1>, TestVectorRef2, TestVectorRef1>;
using MatToCov3ParamOp = UnaryOperation<3, op::MatrixToCovariance3ParamLogic<VariableRef<float, 4>>, VariableRef<float, 4>>;
using ScaleRotToCov3ParamOp = BinaryOperation<3, op::ScaleRotationToCovariance3ParamLogic<TestVectorRef2, TestVectorRef1>, TestVectorRef2, TestVectorRef1>;

// Static assertions for concept compliance
static_assert(VariableConcept<TestVector2>, 
    "Variable<float, 2> should satisfy VariableConcept");
static_assert(DifferentiableVariableConcept<TestVector2>, 
    "Variable<float, 2> should satisfy DifferentiableVariableConcept");

static_assert(VariableConcept<CovGenOp>, 
    "CovarianceGeneration Operation should satisfy VariableConcept");
static_assert(DifferentiableVariableConcept<CovGenOp>, 
    "CovarianceGeneration Operation should satisfy DifferentiableVariableConcept");
static_assert(OperationNode<CovGenOp>, 
    "CovarianceGeneration Operation should satisfy OperationNode");

static_assert(VariableConcept<MatToCov3ParamOp>, 
    "MatrixToCovariance3Param Operation should satisfy VariableConcept");
static_assert(DifferentiableVariableConcept<MatToCov3ParamOp>, 
    "MatrixToCovariance3Param Operation should satisfy DifferentiableVariableConcept");
static_assert(OperationNode<MatToCov3ParamOp>, 
    "MatrixToCovariance3Param Operation should satisfy OperationNode");

static_assert(VariableConcept<ScaleRotToCov3ParamOp>, 
    "ScaleRotationToCovariance3Param Operation should satisfy VariableConcept");
static_assert(DifferentiableVariableConcept<ScaleRotToCov3ParamOp>, 
    "ScaleRotationToCovariance3Param Operation should satisfy DifferentiableVariableConcept");
static_assert(OperationNode<ScaleRotToCov3ParamOp>, 
    "ScaleRotationToCovariance3Param Operation should satisfy OperationNode");

// Ensure Variable is not an OperationNode
static_assert(!OperationNode<TestVector2>, 
    "Variable should NOT satisfy OperationNode");

// ===========================================
// Test Class
// ===========================================

class CovarianceGenerationTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaError_t err = cudaSetDevice(0);
        if (err != cudaSuccess) {
            GTEST_SKIP() << "CUDA device not available";
        }
    }
};

// ===========================================
// Forward Pass Tests - Covariance Matrix Generation
// ===========================================

__global__ void test_covariance_generation_identity_kernel(float* result) {
    // Test case: scale = [2, 1], rotation = 0 (no rotation)
    // Expected M = [[2, 0], [0, 1]]
    float scale_data[2] = {2.0f, 1.0f};
    float scale_grad[2] = {0,0};
    float rotation_data[1] = {0.0f};  // 0 radians
    float rotation_grad[1] = {0};
    
    VariableRef<float, 2> scale(scale_data, scale_grad);
    VariableRef<float, 1> rotation(rotation_data, rotation_grad);
    
    auto M = op::generate_covariance_matrix(scale, rotation);
    M.forward();
    
    // Expected: M = [[2*cos(0), -1*sin(0)], [2*sin(0), 1*cos(0)]] = [[2,0],[0,1]]
    float tolerance = 1e-6f;
    bool success = (fabsf(M[0] - 2.0f) < tolerance &&  // M[0,0]
                   fabsf(M[1] - 0.0f) < tolerance &&  // M[0,1]
                   fabsf(M[2] - 0.0f) < tolerance &&  // M[1,0]
                   fabsf(M[3] - 1.0f) < tolerance);   // M[1,1]
    
    *result = success ? 1.0f : 0.0f;
}

TEST_F(CovarianceGenerationTest, CovarianceGenerationIdentityRotation) {
    auto device_result = makeCudaUnique<float>();
    
    test_covariance_generation_identity_kernel<<<1, 1>>>(device_result.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    float host_result;
    ASSERT_EQ(cudaMemcpy(&host_result, device_result.get(), sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(host_result, 1.0f);
}

__global__ void test_covariance_generation_90deg_kernel(float* result) {
    // Test case: scale = [2, 1], rotation = π/2 (90 degrees)
    // Expected M = [[0,-1],[2,0]]
    float scale_data[2] = {2.0f, 1.0f};
    float scale_grad[2] = {0,0};
    float rotation_data[1] = {1.5707963f};  // π/2 radians
    float rotation_grad[1] = {0};
    
    VariableRef<float, 2> scale(scale_data, scale_grad);
    VariableRef<float, 1> rotation(rotation_data, rotation_grad);
    
    auto M = op::generate_covariance_matrix(scale, rotation);
    M.forward();
    
    float tolerance = 1e-5f;
    bool success = (fabsf(M[0] - 0.0f) < tolerance &&  // M[0,0]
                   fabsf(M[1] - (-1.0f)) < tolerance &&  // M[0,1]
                   fabsf(M[2] - 2.0f) < tolerance &&  // M[1,0]
                   fabsf(M[3] - 0.0f) < tolerance);   // M[1,1]
    
    *result = success ? 1.0f : 0.0f;
}

TEST_F(CovarianceGenerationTest, CovarianceGeneration90DegreeRotation) {
    auto device_result = makeCudaUnique<float>();
    
    test_covariance_generation_90deg_kernel<<<1, 1>>>(device_result.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    float host_result;
    ASSERT_EQ(cudaMemcpy(&host_result, device_result.get(), sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(host_result, 1.0f);
}

// ===========================================
// Forward Pass Tests - Matrix to Covariance 3-Parameter
// ===========================================

__global__ void test_matrix_to_covariance_3param_kernel(float* result) {
    // Test M = [[1,2],[3,4]] -> Σ = M*M^T = [[5,11],[11,25]] -> [5, 11, 25]
    float matrix_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float matrix_grad[4] = {0,0,0,0};
    
    VariableRef<float, 4> matrix(matrix_data, matrix_grad);
    
    auto cov_3param = op::matrix_to_covariance_3param(matrix);
    cov_3param.forward();
    
    // Expected: Σ = [[1*1+2*2, 1*3+2*4], [3*1+4*2, 3*3+4*4]] = [[5,11],[11,25]]
    float tolerance = 1e-6f;
    bool success = (fabsf(cov_3param[0] - 5.0f) < tolerance &&   // Σ[0,0]
                   fabsf(cov_3param[1] - 11.0f) < tolerance &&  // Σ[0,1]
                   fabsf(cov_3param[2] - 25.0f) < tolerance);   // Σ[1,1]
    
    *result = success ? 1.0f : 0.0f;
}

TEST_F(CovarianceGenerationTest, MatrixToCovariance3Param) {
    auto device_result = makeCudaUnique<float>();
    
    test_matrix_to_covariance_3param_kernel<<<1, 1>>>(device_result.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    float host_result;
    ASSERT_EQ(cudaMemcpy(&host_result, device_result.get(), sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(host_result, 1.0f);
}

// ===========================================
// Forward Pass Tests - Scale Rotation to Covariance 3-Parameter
// ===========================================

__global__ void test_scale_rotation_to_covariance_3param_kernel(float* result) {
    // Test case: scale = [1, 1], rotation = 0
    // M = [[1,0],[0,1]], Σ = M*M^T = [[1,0],[0,1]] -> [1, 0, 1]
    float scale_data[2] = {1.0f, 1.0f};
    float scale_grad[2] = {0,0};
    float rotation_data[1] = {0.0f};
    float rotation_grad[1] = {0};
    
    VariableRef<float, 2> scale(scale_data, scale_grad);
    VariableRef<float, 1> rotation(rotation_data, rotation_grad);
    
    auto cov_3param = op::scale_rotation_to_covariance_3param(scale, rotation);
    cov_3param.forward();
    
    float tolerance = 1e-6f;
    bool success = (fabsf(cov_3param[0] - 1.0f) < tolerance &&  // Σ[0,0]
                   fabsf(cov_3param[1] - 0.0f) < tolerance &&  // Σ[0,1]
                   fabsf(cov_3param[2] - 1.0f) < tolerance);   // Σ[1,1]
    
    *result = success ? 1.0f : 0.0f;
}

TEST_F(CovarianceGenerationTest, ScaleRotationToCovariance3Param) {
    auto device_result = makeCudaUnique<float>();
    
    test_scale_rotation_to_covariance_3param_kernel<<<1, 1>>>(device_result.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    float host_result;
    ASSERT_EQ(cudaMemcpy(&host_result, device_result.get(), sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(host_result, 1.0f);
}

// ===========================================
// Gradient Verification Tests
// ===========================================

TEST_F(CovarianceGenerationTest, CovarianceGenerationGradientVerification) {
    using Logic = op::CovarianceMatrixGenerationLogic<VariableRef<double, 2>, VariableRef<double, 1>>;
    test::BinaryGradientTester<Logic, 2, 1, 4>::test_custom(
        "CovarianceGeneration", 
        30,      // num_tests
        1e-5,    // tolerance
        1e-6,    // delta
        -2.0,    // input_min
        2.0      // input_max
    );
}

TEST_F(CovarianceGenerationTest, MatrixToCovariance3ParamGradientVerification) {
    using Logic = op::MatrixToCovariance3ParamLogic<VariableRef<double, 4>>;
    test::UnaryGradientTester<Logic, 4, 3>::test_custom(
        "MatrixToCovariance3Param", 
        30,      // num_tests
        1e-5,    // tolerance
        1e-7,    // delta
        -2.0,    // input_min
        2.0      // input_max
    );
}

TEST_F(CovarianceGenerationTest, ScaleRotationToCovariance3ParamGradientVerification) {
    using Logic = op::ScaleRotationToCovariance3ParamLogic<VariableRef<double, 2>, VariableRef<double, 1>>;
    test::BinaryGradientTester<Logic, 2, 1, 3>::test_custom(
        "ScaleRotationToCovariance3Param", 
        30,      // num_tests
        1e-5,    // tolerance
        1e-6,    // delta
        -2.0,    // input_min
        2.0      // input_max
    );
}

// ===========================================
// Specific Gradient Tests
// ===========================================

__global__ void test_covariance_generation_gradient_kernel(double* result) {
    // Test specific gradient properties
    double scale_data[2] = {1.5, 1.0};
    double scale_grad[2] = {0.0, 0.0};
    double rotation_data[1] = {0.2};  // small rotation
    double rotation_grad[1] = {0.0};
    
    VariableRef<double, 2> scale(scale_data, scale_grad);
    VariableRef<double, 1> rotation(rotation_data, rotation_grad);
    
    auto M = op::generate_covariance_matrix(scale, rotation);
    
    // Forward pass
    M.forward();
    
    // Set upstream gradient
    M.zero_grad();
    for (int i = 0; i < 4; i++) {
        M.add_grad(i, 1.0);
    }
    
    // Analytical backward
    M.backward();
    
    // Save analytical gradients
    double analytical_scale_grad[2];
    double analytical_rotation_grad[1];
    for (int i = 0; i < 2; i++) {
        analytical_scale_grad[i] = scale.grad(i);
    }
    analytical_rotation_grad[0] = rotation.grad(0);
    
    // Reset gradients
    for (int i = 0; i < 2; i++) {
        scale_grad[i] = 0.0;
    }
    rotation_grad[0] = 0.0;
    
    // Numerical backward
    M.run_numerical(1e-8);
    
    // Check gradient consistency
    bool success = true;
    double tolerance = 1e-5;
    
    for (int i = 0; i < 2; i++) {
        double diff = fabs(analytical_scale_grad[i] - scale.grad(i));
        if (diff > tolerance) {
            success = false;
            break;
        }
    }
    
    if (success) {
        double diff = fabs(analytical_rotation_grad[0] - rotation.grad(0));
        if (diff > tolerance) {
            success = false;
        }
    }
    
    *result = success ? 1.0 : 0.0;
}

TEST_F(CovarianceGenerationTest, CovarianceGenerationSpecificGradientVerification) {
    auto device_result = makeCudaUnique<double>();
    
    test_covariance_generation_gradient_kernel<<<1, 1>>>(device_result.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    double host_result;
    ASSERT_EQ(cudaMemcpy(&host_result, device_result.get(), sizeof(double), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(host_result, 1.0);
}

// ===========================================
// Interface Compliance Test
// ===========================================

__global__ void test_covariance_interface_kernel(float* result) {
    float scale_data[2] = {1.0f, 1.0f};
    float scale_grad[2] = {0,0};
    float rotation_data[1] = {0.0f};
    float rotation_grad[1] = {0};
    
    VariableRef<float, 2> scale(scale_data, scale_grad);
    VariableRef<float, 1> rotation(rotation_data, rotation_grad);
    
    // Test covariance generation operation
    auto M = op::generate_covariance_matrix(scale, rotation);
    
    // Test VariableConcept interface
    M.zero_grad();
    constexpr auto size = decltype(M)::size;
    auto* data = M.data();
    auto* grad = M.grad();
    auto value = M[0];
    auto grad_value = M.grad(0);
    
    // Test OperationNode interface
    M.forward();
    M.backward();
    M.backward_numerical(1e-5f);
    M.run();
    M.run_numerical(1e-5f);
    
    // Verify expected behavior
    bool success = (size == 4 && data != nullptr && grad != nullptr);
    
    *result = success ? 1.0f : 0.0f;
}

TEST_F(CovarianceGenerationTest, InterfaceCompliance) {
    auto device_result = makeCudaUnique<float>();
    
    test_covariance_interface_kernel<<<1, 1>>>(device_result.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    float host_result;
    ASSERT_EQ(cudaMemcpy(&host_result, device_result.get(), sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(host_result, 1.0f);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}