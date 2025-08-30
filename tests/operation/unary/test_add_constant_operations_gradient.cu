#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include <xyz_autodiff/variable.cuh>
#include <xyz_autodiff/concept/variable.cuh>
#include <xyz_autodiff/concept/operation_node.cuh>
#include <xyz_autodiff/operations/unary/add_constant_logic.cuh>
#include <xyz_autodiff/util/cuda_unique_ptr.cuh>
#include <xyz_autodiff/variable_operators.cuh>
#include <xyz_autodiff/testing/unary_gradient_tester.cuh>

using namespace xyz_autodiff;

using TestVector3 = Variable<3, float>;
using TestVectorRef3 = VariableRef<3, float>;

using AddConstantOp = UnaryOperation<3, op::AddConstantLogic<TestVectorRef3>, TestVectorRef3>;

static_assert(VariableConcept<AddConstantOp>, "AddConstantOperation should satisfy VariableConcept");
static_assert(DifferentiableVariableConcept<AddConstantOp>, "AddConstantOperation should satisfy DifferentiableVariableConcept");
static_assert(OperationNode<AddConstantOp>, "AddConstantOperation should satisfy OperationNode");

static_assert(!OperationNode<TestVector3>, "Variable should NOT be OperationNode");

class AddConstantOperationsGradientTest : public ::testing::Test {
protected:
    void SetUp() override {
        int device_count;
        cudaGetDeviceCount(&device_count);
        if (device_count == 0) {
            GTEST_SKIP() << "No CUDA devices available";
        }
    }
};

__global__ void test_add_constant_forward_kernel(float* result) {
    Variable<3, float> input;
    input[0] = 1.0f;
    input[1] = 2.0f;
    input[2] = 3.0f;
    
    auto add_op = op::add_constant(input, 5.0f);
    add_op.forward();
    
    result[0] = add_op[0];
    result[1] = add_op[1]; 
    result[2] = add_op[2];
}

TEST_F(AddConstantOperationsGradientTest, ForwardPass) {
    auto device_result = makeCudaUniqueArray<float>(3);
    
    test_add_constant_forward_kernel<<<1, 1>>>(device_result.get());
    cudaDeviceSynchronize();
    
    float host_result[3];
    cudaMemcpy(host_result, device_result.get(), 3 * sizeof(float), cudaMemcpyDeviceToHost);
    
    EXPECT_FLOAT_EQ(host_result[0], 6.0f);
    EXPECT_FLOAT_EQ(host_result[1], 7.0f);
    EXPECT_FLOAT_EQ(host_result[2], 8.0f);
}

__global__ void test_add_constant_gradient_kernel(double* result) {
    Variable<3, double> input;
    input[0] = 1.5;
    input[1] = -2.3;
    input[2] = 0.7;
    
    auto add_op = op::add_constant(input, 3.14);
    add_op.forward();
    
    add_op.add_grad(0, 1.0);
    add_op.add_grad(1, 2.0);
    add_op.add_grad(2, -1.5);
    
    add_op.backward();
    
    result[0] = input.grad(0);
    result[1] = input.grad(1);
    result[2] = input.grad(2);
}

TEST_F(AddConstantOperationsGradientTest, BackwardPass) {
    auto device_result = makeCudaUniqueArray<double>(3);
    
    test_add_constant_gradient_kernel<<<1, 1>>>(device_result.get());
    cudaDeviceSynchronize();
    
    double host_result[3];
    cudaMemcpy(host_result, device_result.get(), 3 * sizeof(double), cudaMemcpyDeviceToHost);
    
    EXPECT_DOUBLE_EQ(host_result[0], 1.0);
    EXPECT_DOUBLE_EQ(host_result[1], 2.0);
    EXPECT_DOUBLE_EQ(host_result[2], -1.5);
}

template<typename T>
struct AddConstantTestLogic {
    using value_type = T;
    static constexpr std::size_t outputDim = 3;
    T constant_c;
    
    __host__ __device__ explicit AddConstantTestLogic(T c = static_cast<T>(2.5)) : constant_c(c) {}
    
    __device__ void forward(Variable<3, T>& output, const VariableRef<3, T>& input) const {
        for (std::size_t i = 0; i < 3; ++i) {
            output[i] = input[i] + constant_c;
        }
    }
    
    __device__ void backward(const Variable<3, T>& output, VariableRef<3, T>& input) const {
        for (std::size_t i = 0; i < 3; ++i) {
            const T& output_grad = output.grad(i);
            input.add_grad(i, output_grad);
        }
    }
};

TEST_F(AddConstantOperationsGradientTest, AddConstantGradientVerification) {
    using Logic = AddConstantTestLogic<double>;
    xyz_autodiff::testing::UnaryGradientTester<Logic, 3, 3>::test_custom(
        "AddConstantLogic", 
        50,      // num_tests
        1e-5,    // tolerance
        1e-6,    // delta
        -5.0,    // input_min
        5.0      // input_max
    );
}

struct AddConstant5DLogic {
    using value_type = double;
    static constexpr std::size_t outputDim = 5;
    double constant_c;
    
    __host__ __device__ explicit AddConstant5DLogic(double c = 1.7) : constant_c(c) {}
    
    __device__ void forward(Variable<5, double>& output, const VariableRef<5, double>& input) const {
        for (std::size_t i = 0; i < 5; ++i) {
            output[i] = input[i] + constant_c;
        }
    }
    
    __device__ void backward(const Variable<5, double>& output, VariableRef<5, double>& input) const {
        for (std::size_t i = 0; i < 5; ++i) {
            const double& output_grad = output.grad(i);
            input.add_grad(i, output_grad);
        }
    }
};

TEST_F(AddConstantOperationsGradientTest, AddConstantGradientVerification5D) {
    xyz_autodiff::testing::UnaryGradientTester<AddConstant5DLogic, 5, 5>::test_custom(
        "AddConstant5DLogic", 
        30,      // num_tests
        1e-5,    // tolerance
        1e-6,    // delta
        -3.0,    // input_min
        3.0      // input_max
    );
}

TEST_F(AddConstantOperationsGradientTest, AddConstantNearZero) {
    using Logic = AddConstantTestLogic<double>;
    xyz_autodiff::testing::UnaryGradientTester<Logic, 3, 3>::test_custom(
        "AddConstantLogicNearZero", 
        30,      // num_tests
        1e-5,    // tolerance
        1e-7,    // delta
        -0.5,    // input_min
        0.5      // input_max
    );
}

TEST_F(AddConstantOperationsGradientTest, AddConstantLargeValues) {
    using Logic = AddConstantTestLogic<double>;
    xyz_autodiff::testing::UnaryGradientTester<Logic, 3, 3>::test_custom(
        "AddConstantLogicLargeValues", 
        40,      // num_tests
        1e-5,    // tolerance
        1e-6,    // delta
        -100.0,  // input_min
        100.0    // input_max
    );
}