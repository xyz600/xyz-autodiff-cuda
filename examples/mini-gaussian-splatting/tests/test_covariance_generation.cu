#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include <xyz_autodiff/variable.cuh>
#include <xyz_autodiff/concept/variable.cuh>
#include <xyz_autodiff/concept/operation_node.cuh>
#include "../operations/covariance_generation.cuh"
#include <xyz_autodiff/util/cuda_unique_ptr.cuh>
#include <xyz_autodiff/testing/binary_gradient_tester.cuh>
#include <xyz_autodiff/testing/unary_gradient_tester.cuh>

using namespace xyz_autodiff;

// ===========================================
// Static Assert Tests for Concept Compliance
// ===========================================

// Test types
using TestVector2 = Variable<2, float>;
using TestVector1 = Variable<1, float>;
using TestVectorRef2 = VariableRef<2, float>;
using TestVectorRef1 = VariableRef<1, float>;
using CovGenOp = BinaryOperation<4, op::CovarianceMatrixGenerationLogic<TestVectorRef2, TestVectorRef1>, TestVectorRef2, TestVectorRef1>;
using MatToCov3ParamOp = UnaryOperation<3, op::MatrixToCovariance3ParamLogic<VariableRef<4, float>>, VariableRef<4, float>>;
using ScaleRotToCov3ParamOp = BinaryOperation<3, op::ScaleRotationToCovariance3ParamLogic<TestVectorRef2, TestVectorRef1>, TestVectorRef2, TestVectorRef1>;

// Static assertions for concept compliance
static_assert(VariableConcept<TestVector2>, 
    "Variable<2, float> should satisfy VariableConcept");
static_assert(DifferentiableVariableConcept<TestVector2>, 
    "Variable<2, float> should satisfy DifferentiableVariableConcept");

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
// Gradient Verification Tests
// ===========================================

TEST_F(CovarianceGenerationTest, CovarianceGenerationGradientVerification) {
    using Logic = op::CovarianceMatrixGenerationLogic<VariableRef<2, double>, VariableRef<1, double>>;
    xyz_autodiff::testing::BinaryGradientTester<Logic, 2, 1, 4>::test_custom(
        "CovarianceGeneration", 
        30,      // num_tests
        1e-5,    // tolerance
        1e-6,    // delta
        -2.0,    // input_min
        2.0      // input_max
    );
}

TEST_F(CovarianceGenerationTest, MatrixToCovariance3ParamGradientVerification) {
    using Logic = op::MatrixToCovariance3ParamLogic<VariableRef<4, double>>;
    xyz_autodiff::testing::UnaryGradientTester<Logic, 4, 3>::test_custom(
        "MatrixToCovariance3Param", 
        30,      // num_tests
        1e-5,    // tolerance
        1e-7,    // delta
        -2.0,    // input_min
        2.0      // input_max
    );
}

TEST_F(CovarianceGenerationTest, ScaleRotationToCovariance3ParamGradientVerification) {
    using Logic = op::ScaleRotationToCovariance3ParamLogic<VariableRef<2, double>, VariableRef<1, double>>;
    xyz_autodiff::testing::BinaryGradientTester<Logic, 2, 1, 3>::test_custom(
        "ScaleRotationToCovariance3Param", 
        30,      // num_tests
        1e-5,    // tolerance
        1e-6,    // delta
        -2.0,    // input_min
        2.0      // input_max
    );
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}