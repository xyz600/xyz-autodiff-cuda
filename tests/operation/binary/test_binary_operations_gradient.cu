#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include "../../../include/variable.cuh"
#include "../../../include/concept/variable.cuh"
#include "../../../include/concept/operation_node.cuh"
#include "../../../include/operations/binary/add_logic.cuh"
#include "../../../include/operations/binary/sub_logic.cuh"
#include "../../../include/operations/binary/mul_logic.cuh"
#include "../../../include/operations/binary/div_logic.cuh"
#include "../../../include/util/cuda_unique_ptr.cuh"
#include "../../utility/binary_gradient_tester.cuh"

using namespace xyz_autodiff;

// ===========================================
// Static Assert Tests for Concept Compliance
// ===========================================

// Test types
using TestVector3 = Variable<3, float>;
using TestVectorRef3 = VariableRef<3, float>;
using TestVector2 = Variable<2, float>;
using TestVectorRef2 = VariableRef<2, float>;

// Binary operation types
using AddOp = BinaryOperation<3, op::AddLogic<TestVectorRef3, TestVectorRef3>, TestVectorRef3, TestVectorRef3>;
using SubOp = BinaryOperation<3, op::SubLogic<TestVectorRef3, TestVectorRef3>, TestVectorRef3, TestVectorRef3>;
using MulOp = BinaryOperation<3, op::MulLogic<TestVectorRef3, TestVectorRef3>, TestVectorRef3, TestVectorRef3>;
using DivOp = BinaryOperation<3, op::DivLogic<TestVectorRef3, TestVectorRef3>, TestVectorRef3, TestVectorRef3>;

// Static assertions for concept compliance
static_assert(VariableConcept<AddOp>, "AddOperation should satisfy VariableConcept");
static_assert(DifferentiableVariableConcept<AddOp>, "AddOperation should satisfy DifferentiableVariableConcept");
static_assert(OperationNode<AddOp>, "AddOperation should satisfy OperationNode");

static_assert(VariableConcept<SubOp>, "SubOperation should satisfy VariableConcept");
static_assert(DifferentiableVariableConcept<SubOp>, "SubOperation should satisfy DifferentiableVariableConcept");
static_assert(OperationNode<SubOp>, "SubOperation should satisfy OperationNode");

static_assert(VariableConcept<MulOp>, "MulOperation should satisfy VariableConcept");
static_assert(DifferentiableVariableConcept<MulOp>, "MulOperation should satisfy DifferentiableVariableConcept");
static_assert(OperationNode<MulOp>, "MulOperation should satisfy OperationNode");

static_assert(VariableConcept<DivOp>, "DivOperation should satisfy VariableConcept");
static_assert(DifferentiableVariableConcept<DivOp>, "DivOperation should satisfy DifferentiableVariableConcept");
static_assert(OperationNode<DivOp>, "DivOperation should satisfy OperationNode");

// Ensure Variable is NOT an OperationNode
static_assert(!OperationNode<TestVector3>, "Variable should NOT be OperationNode");

// ===========================================
// Test Class
// ===========================================

class BinaryOperationsGradientTest : public ::testing::Test {
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
// Gradient Verification Tests
// ===========================================

TEST_F(BinaryOperationsGradientTest, AddGradientVerification) {
    using Logic = op::AddLogic<VariableRef<3, double>, VariableRef<3, double>>;
    test::BinaryGradientTester<Logic, 3, 3, 3>::test_custom(
        "AddLogic", 
        50,      // num_tests
        1e-5,    // tolerance
        1e-7,    // delta
        -5.0,    // input_min
        5.0      // input_max
    );
}

TEST_F(BinaryOperationsGradientTest, SubGradientVerification) {
    using Logic = op::SubLogic<VariableRef<3, double>, VariableRef<3, double>>;
    test::BinaryGradientTester<Logic, 3, 3, 3>::test_custom(
        "SubLogic", 
        50,      // num_tests
        1e-5,    // tolerance
        1e-7,    // delta
        -5.0,    // input_min
        5.0      // input_max
    );
}

TEST_F(BinaryOperationsGradientTest, MulGradientVerification) {
    using Logic = op::MulLogic<VariableRef<3, double>, VariableRef<3, double>>;
    test::BinaryGradientTester<Logic, 3, 3, 3>::test_custom(
        "MulLogic", 
        50,      // num_tests
        1e-5,    // tolerance
        1e-7,    // delta
        -3.0,    // input_min (smaller range to avoid overflow)
        3.0      // input_max
    );
}

TEST_F(BinaryOperationsGradientTest, DivGradientVerification) {
    using Logic = op::DivLogic<VariableRef<3, double>, VariableRef<3, double>>;
    test::BinaryGradientTester<Logic, 3, 3, 3>::test_custom(
        "DivLogic", 
        50,      // num_tests
        1e-5,    // tolerance (minimum allowed for double precision)
        1e-6,    // delta
        0.1,     // input_min (avoid division by zero)
        5.0      // input_max
    );
}

// Test with different dimensions
TEST_F(BinaryOperationsGradientTest, AddGradientVerification2D) {
    using Logic = op::AddLogic<VariableRef<2, double>, VariableRef<2, double>>;
    test::BinaryGradientTester<Logic, 2, 2, 2>::test_custom(
        "AddLogic2D", 
        30,      // num_tests
        1e-5,    // tolerance
        1e-7,    // delta
        -10.0,   // input_min
        10.0     // input_max
    );
}

TEST_F(BinaryOperationsGradientTest, MulGradientVerification1D) {
    using Logic = op::MulLogic<VariableRef<1, double>, VariableRef<1, double>>;
    test::BinaryGradientTester<Logic, 1, 1, 1>::test_custom(
        "MulLogic1D", 
        30,      // num_tests
        1e-5,    // tolerance
        1e-7,    // delta
        -2.0,    // input_min
        2.0      // input_max
    );
}