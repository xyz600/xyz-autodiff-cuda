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

int main() {
    return 0;
}