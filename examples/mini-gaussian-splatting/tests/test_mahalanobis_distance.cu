#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include "../../../include/variable.cuh"
#include "../../../include/concept/variable.cuh"
#include "../../../include/concept/operation_node.cuh"
#include "../operations/mahalanobis_distance.cuh"
#include "../../../include/util/cuda_unique_ptr.cuh"
#include "../../../tests/utility/binary_gradient_tester.cuh"

using namespace xyz_autodiff;

// ===========================================
// Static Assert Tests for Concept Compliance
// ===========================================

// Test types
using TestVector2 = Variable<float, 2>;
using TestVector3 = Variable<float, 3>;
using TestVectorRef2 = VariableRef<float, 2>;
using TestVectorRef3 = VariableRef<float, 3>;
using MahalDistOp = BinaryOperation<1, op::MahalanobisDistanceLogic<TestVectorRef2, TestVectorRef3>, TestVectorRef2, TestVectorRef3>;

// Static assertions for concept compliance
static_assert(VariableConcept<TestVector2>, 
    "Variable<float, 2> should satisfy VariableConcept");
static_assert(DifferentiableVariableConcept<TestVector2>, 
    "Variable<float, 2> should satisfy DifferentiableVariableConcept");

static_assert(VariableConcept<MahalDistOp>, 
    "MahalanobisDistance Operation should satisfy VariableConcept");
static_assert(DifferentiableVariableConcept<MahalDistOp>, 
    "MahalanobisDistance Operation should satisfy DifferentiableVariableConcept");
static_assert(OperationNode<MahalDistOp>, 
    "MahalanobisDistance Operation should satisfy OperationNode");

// Ensure Variable is not an OperationNode
static_assert(!OperationNode<TestVector2>, 
    "Variable should NOT satisfy OperationNode");

// ===========================================
// Test Class
// ===========================================

class MahalanobisDistanceTest : public ::testing::Test {
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

__global__ void test_mahalanobis_distance_identity_kernel(float* result) {
    // Test case: point = [1, 2], cov_3param = [1, 0, 1] (identity matrix)
    // Expected: distance^2 = 1^2 + 2^2 = 5
    float point_data[2] = {1.0f, 2.0f};
    float point_grad[2] = {0,0};
    float cov_data[3] = {1.0f, 0.0f, 1.0f};  // [σ11, σ12, σ22]
    float cov_grad[3] = {0,0,0};
    
    VariableRef<float, 2> point(point_data, point_grad);
    VariableRef<float, 3> cov_3param(cov_data, cov_grad);
    
    auto distance_sq = op::mahalanobis_distance(point, cov_3param);
    distance_sq.forward();
    
    float expected = 5.0f;  // 1^2 + 2^2
    float tolerance = 1e-5f;
    bool success = (fabsf(distance_sq[0] - expected) < tolerance);
    
    *result = success ? 1.0f : 0.0f;
}

TEST_F(MahalanobisDistanceTest, IdentityMatrixCase) {
    auto device_result = makeCudaUnique<float>();
    
    test_mahalanobis_distance_identity_kernel<<<1, 1>>>(device_result.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    float host_result;
    ASSERT_EQ(cudaMemcpy(&host_result, device_result.get(), sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(host_result, 1.0f);
}

__global__ void test_mahalanobis_distance_diagonal_kernel(float* result) {
    // Test case: point = [2, 3], inv_cov_3param = [1/4, 0, 1/9] (diagonal inverse covariance)
    // Expected: distance^2 = 2^2*(1/4) + 3^2*(1/9) = 1 + 1 = 2
    float point_data[2] = {2.0f, 3.0f};
    float point_grad[2] = {0,0};
    float inv_cov_data[3] = {0.25f, 0.0f, 1.0f/9.0f};  // [1/4, 0, 1/9]
    float inv_cov_grad[3] = {0,0,0};
    
    VariableRef<float, 2> point(point_data, point_grad);
    VariableRef<float, 3> inv_cov_3param(inv_cov_data, inv_cov_grad);
    
    auto distance_sq = op::mahalanobis_distance(point, inv_cov_3param);
    distance_sq.forward();
    
    float expected = 2.0f;  // 2^2*(1/4) + 3^2*(1/9) = 1 + 1
    float tolerance = 1e-5f;
    bool success = (fabsf(distance_sq[0] - expected) < tolerance);
    
    *result = success ? 1.0f : 0.0f;
}

TEST_F(MahalanobisDistanceTest, DiagonalMatrixCase) {
    auto device_result = makeCudaUnique<float>();
    
    test_mahalanobis_distance_diagonal_kernel<<<1, 1>>>(device_result.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    float host_result;
    ASSERT_EQ(cudaMemcpy(&host_result, device_result.get(), sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(host_result, 1.0f);
}

__global__ void test_mahalanobis_distance_general_kernel(float* result) {
    // Test case: point = [1, 1], inv_cov_3param = [2/3, -1/3, 2/3]
    // From Σ = [[2, 1], [1, 2]], det(Σ) = 3, Σ^-1 = (1/3) * [[2, -1], [-1, 2]] = [[2/3, -1/3], [-1/3, 2/3]]
    // distance^2 = [1, 1] * Σ^-1 * [1, 1]^T = (2/3)*1^2 + 2*(-1/3)*1*1 + (2/3)*1^2 = 2/3 - 2/3 + 2/3 = 2/3
    float point_data[2] = {1.0f, 1.0f};
    float point_grad[2] = {0,0};
    float inv_cov_data[3] = {2.0f/3.0f, -1.0f/3.0f, 2.0f/3.0f};  // [2/3, -1/3, 2/3]
    float inv_cov_grad[3] = {0,0,0};
    
    VariableRef<float, 2> point(point_data, point_grad);
    VariableRef<float, 3> inv_cov_3param(inv_cov_data, inv_cov_grad);
    
    auto distance_sq = op::mahalanobis_distance(point, inv_cov_3param);
    distance_sq.forward();
    
    float expected = 2.0f / 3.0f;
    float tolerance = 1e-5f;
    bool success = (fabsf(distance_sq[0] - expected) < tolerance);
    
    *result = success ? 1.0f : 0.0f;
}

TEST_F(MahalanobisDistanceTest, GeneralMatrixCase) {
    auto device_result = makeCudaUnique<float>();
    
    test_mahalanobis_distance_general_kernel<<<1, 1>>>(device_result.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    float host_result;
    ASSERT_EQ(cudaMemcpy(&host_result, device_result.get(), sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(host_result, 1.0f);
}

// ===========================================
// Gradient Verification Tests
// ===========================================

TEST_F(MahalanobisDistanceTest, GradientVerification) {
    using Logic = op::MahalanobisDistanceLogic<VariableRef<double, 2>, VariableRef<double, 3>>;
    test::BinaryGradientTester<Logic, 2, 3, 1>::test_custom(
        "MahalanobisDistance", 
        30,      // num_tests (reduced for stability)
        1e-4,    // tolerance (relaxed for matrix inversion stability)
        1e-6,    // delta
        0.1,     // input_min (avoid singular matrices)
        2.0      // input_max
    );
}

// ===========================================
// Specific Gradient Tests
// ===========================================

__global__ void test_mahalanobis_distance_gradient_kernel(double* result) {
    // Test specific gradient properties for well-conditioned case
    double point_data[2] = {1.0, 0.5};
    double point_grad[2] = {0.0, 0.0};
    double cov_data[3] = {2.0, 0.5, 3.0};  // Well-conditioned matrix
    double cov_grad[3] = {0.0, 0.0, 0.0};
    
    VariableRef<double, 2> point(point_data, point_grad);
    VariableRef<double, 3> cov_3param(cov_data, cov_grad);
    
    auto distance_op = op::mahalanobis_distance(point, cov_3param);
    
    // Forward pass
    distance_op.forward();
    
    // Set upstream gradient
    distance_op.zero_grad();
    distance_op.add_grad(0, 1.0);
    
    // Analytical backward
    distance_op.backward();
    
    // Save analytical gradients
    double analytical_point_grad[2];
    double analytical_cov_grad[3];
    for (int i = 0; i < 2; i++) {
        analytical_point_grad[i] = point.grad(i);
    }
    for (int i = 0; i < 3; i++) {
        analytical_cov_grad[i] = cov_3param.grad(i);
    }
    
    // Reset gradients
    for (int i = 0; i < 2; i++) {
        point_grad[i] = 0.0;
    }
    for (int i = 0; i < 3; i++) {
        cov_grad[i] = 0.0;
    }
    
    // Numerical backward
    distance_op.run_numerical(1e-8);
    
    // Check gradient consistency
    bool success = true;
    double tolerance = 1e-4;
    
    for (int i = 0; i < 2; i++) {
        double diff = fabs(analytical_point_grad[i] - point.grad(i));
        if (diff > tolerance) {
            success = false;
            break;
        }
    }
    
    if (success) {
        for (int i = 0; i < 3; i++) {
            double diff = fabs(analytical_cov_grad[i] - cov_3param.grad(i));
            if (diff > tolerance) {
                success = false;
                break;
            }
        }
    }
    
    *result = success ? 1.0 : 0.0;
}

TEST_F(MahalanobisDistanceTest, SpecificGradientVerification) {
    auto device_result = makeCudaUnique<double>();
    
    test_mahalanobis_distance_gradient_kernel<<<1, 1>>>(device_result.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    double host_result;
    ASSERT_EQ(cudaMemcpy(&host_result, device_result.get(), sizeof(double), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(host_result, 1.0);
}

// ===========================================
// Interface Compliance Test
// ===========================================

__global__ void test_mahalanobis_distance_interface_kernel(float* result) {
    float point_data[2] = {1.0f, 2.0f};
    float point_grad[2] = {0,0};
    float cov_data[3] = {1.0f, 0.0f, 1.0f};
    float cov_grad[3] = {0,0,0};
    
    VariableRef<float, 2> point(point_data, point_grad);
    VariableRef<float, 3> cov_3param(cov_data, cov_grad);
    
    auto distance_op = op::mahalanobis_distance(point, cov_3param);
    
    // Test VariableConcept interface
    distance_op.zero_grad();
    constexpr auto size = decltype(distance_op)::size;
    auto* data = distance_op.data();
    auto* grad = distance_op.grad();
    auto value = distance_op[0];
    auto grad_value = distance_op.grad(0);
    
    // Test OperationNode interface
    distance_op.forward();
    distance_op.backward();
    distance_op.backward_numerical(1e-5f);
    distance_op.run();
    distance_op.run_numerical(1e-5f);
    
    // Verify expected behavior (output size should be 1)
    bool success = (size == 1 && data != nullptr && grad != nullptr);
    
    *result = success ? 1.0f : 0.0f;
}

TEST_F(MahalanobisDistanceTest, InterfaceCompliance) {
    auto device_result = makeCudaUnique<float>();
    
    test_mahalanobis_distance_interface_kernel<<<1, 1>>>(device_result.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    float host_result;
    ASSERT_EQ(cudaMemcpy(&host_result, device_result.get(), sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(host_result, 1.0f);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}