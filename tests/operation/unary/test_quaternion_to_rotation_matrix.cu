#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include <xyz_autodiff/variable.cuh>
#include <xyz_autodiff/operations/unary/to_rotation_matrix_logic.cuh>
#include <xyz_autodiff/concept/variable.cuh>
#include <xyz_autodiff/concept/operation_node.cuh>
#include <xyz_autodiff/util/cuda_unique_ptr.cuh>
#include <xyz_autodiff/variable_operators.cuh>

using namespace xyz_autodiff;

class QuaternionToRotationMatrixTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaError_t err = cudaSetDevice(0);
        if (err != cudaSuccess) {
            GTEST_SKIP() << "CUDA device not available";
        }
    }
};

// Static assert tests for concept compliance
using TestQuaternion = Variable<4, float>;
using QuatToMatOp = UnaryOperation<9, op::QuaternionToRotationMatrixLogic<4>, TestQuaternion>;

static_assert(VariableConcept<TestQuaternion>, 
    "Quaternion Variable should satisfy VariableConcept");
static_assert(DifferentiableVariableConcept<TestQuaternion>, 
    "Quaternion Variable should satisfy DifferentiableVariableConcept");

static_assert(VariableConcept<QuatToMatOp>, 
    "QuaternionToRotationMatrix Operation should satisfy VariableConcept");
static_assert(DifferentiableVariableConcept<QuatToMatOp>, 
    "QuaternionToRotationMatrix Operation should satisfy DifferentiableVariableConcept");
static_assert(OperationNode<QuatToMatOp>, 
    "QuaternionToRotationMatrix Operation should satisfy OperationNode");

// Ensure quaternion is not an OperationNode
static_assert(!OperationNode<TestQuaternion>, 
    "Quaternion Variable should NOT satisfy OperationNode");

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}