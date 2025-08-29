#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include "../../../include/variable.cuh"
#include "../../../include/concept/variable.cuh"
#include "../../../include/concept/operation_node.cuh"
#include "../../../include/operations/unary/sub_constant_logic.cuh"
#include "../../../include/util/cuda_unique_ptr.cuh"
#include "../../utility/unary_gradient_tester.cuh"

using namespace xyz_autodiff;

using TestVector3 = Variable<3, float>;
using TestVectorRef3 = VariableRef<3, float>;

using SubConstantOp = UnaryOperation<3, op::SubConstantLogic<TestVectorRef3>, TestVectorRef3>;

static_assert(VariableConcept<SubConstantOp>, "SubConstantOperation should satisfy VariableConcept");
static_assert(DifferentiableVariableConcept<SubConstantOp>, "SubConstantOperation should satisfy DifferentiableVariableConcept");
static_assert(OperationNode<SubConstantOp>, "SubConstantOperation should satisfy OperationNode");

static_assert(!OperationNode<TestVector3>, "Variable should NOT be OperationNode");

class SubConstantOperationsGradientTest : public ::testing::Test {
protected:
    void SetUp() override {
        int device_count;
        cudaGetDeviceCount(&device_count);
        if (device_count == 0) {
            GTEST_SKIP() << "No CUDA devices available";
        }
    }
};

__global__ void test_sub_constant_forward_kernel(float* result) {
    Variable<3, float> input;
    input[0] = 10.0f;
    input[1] = 8.0f;
    input[2] = 6.0f;
    
    auto sub_op = op::sub_constant(input, 3.0f);
    sub_op.forward();
    
    result[0] = sub_op[0];
    result[1] = sub_op[1]; 
    result[2] = sub_op[2];
}

TEST_F(SubConstantOperationsGradientTest, ForwardPass) {
    auto device_result = makeCudaUniqueArray<float>(3);
    
    test_sub_constant_forward_kernel<<<1, 1>>>(device_result.get());
    cudaDeviceSynchronize();
    
    float host_result[3];
    cudaMemcpy(host_result, device_result.get(), 3 * sizeof(float), cudaMemcpyDeviceToHost);
    
    EXPECT_FLOAT_EQ(host_result[0], 7.0f);
    EXPECT_FLOAT_EQ(host_result[1], 5.0f);
    EXPECT_FLOAT_EQ(host_result[2], 3.0f);
}

__global__ void test_sub_constant_gradient_kernel(double* result) {
    Variable<3, double> input;
    input[0] = 5.5;
    input[1] = -1.2;
    input[2] = 0.9;
    
    auto sub_op = op::sub_constant(input, 2.71);
    sub_op.forward();
    
    sub_op.add_grad(0, 1.0);
    sub_op.add_grad(1, -2.0);
    sub_op.add_grad(2, 3.5);
    
    sub_op.backward();
    
    result[0] = input.grad(0);
    result[1] = input.grad(1);
    result[2] = input.grad(2);
}

TEST_F(SubConstantOperationsGradientTest, BackwardPass) {
    auto device_result = makeCudaUniqueArray<double>(3);
    
    test_sub_constant_gradient_kernel<<<1, 1>>>(device_result.get());
    cudaDeviceSynchronize();
    
    double host_result[3];
    cudaMemcpy(host_result, device_result.get(), 3 * sizeof(double), cudaMemcpyDeviceToHost);
    
    EXPECT_DOUBLE_EQ(host_result[0], 1.0);
    EXPECT_DOUBLE_EQ(host_result[1], -2.0);
    EXPECT_DOUBLE_EQ(host_result[2], 3.5);
}

template<typename T>
struct SubConstantTestLogic {
    using value_type = T;
    static constexpr std::size_t outputDim = 3;
    T constant_c;
    
    __host__ __device__ explicit SubConstantTestLogic(T c = static_cast<T>(1.8)) : constant_c(c) {}
    
    __device__ void forward(Variable<3, T>& output, const VariableRef<3, T>& input) const {
        for (std::size_t i = 0; i < 3; ++i) {
            output[i] = input[i] - constant_c;
        }
    }
    
    __device__ void backward(const Variable<3, T>& output, VariableRef<3, T>& input) const {
        for (std::size_t i = 0; i < 3; ++i) {
            const T& output_grad = output.grad(i);
            input.add_grad(i, output_grad);
        }
    }
};

TEST_F(SubConstantOperationsGradientTest, SubConstantGradientVerification) {
    using Logic = SubConstantTestLogic<double>;
    test::UnaryGradientTester<Logic, 3, 3>::test_custom(
        "SubConstantLogic", 
        50,      // num_tests
        1e-5,    // tolerance
        1e-6,    // delta
        -5.0,    // input_min
        5.0      // input_max
    );
}

struct SubConstant5DLogic {
    using value_type = double;
    static constexpr std::size_t outputDim = 5;
    double constant_c;
    
    __host__ __device__ explicit SubConstant5DLogic(double c = 0.42) : constant_c(c) {}
    
    __device__ void forward(Variable<5, double>& output, const VariableRef<5, double>& input) const {
        for (std::size_t i = 0; i < 5; ++i) {
            output[i] = input[i] - constant_c;
        }
    }
    
    __device__ void backward(const Variable<5, double>& output, VariableRef<5, double>& input) const {
        for (std::size_t i = 0; i < 5; ++i) {
            const double& output_grad = output.grad(i);
            input.add_grad(i, output_grad);
        }
    }
};

TEST_F(SubConstantOperationsGradientTest, SubConstantGradientVerification5D) {
    test::UnaryGradientTester<SubConstant5DLogic, 5, 5>::test_custom(
        "SubConstant5DLogic", 
        30,      // num_tests
        1e-5,    // tolerance
        1e-6,    // delta
        -3.0,    // input_min
        3.0      // input_max
    );
}

TEST_F(SubConstantOperationsGradientTest, SubConstantNearZero) {
    using Logic = SubConstantTestLogic<double>;
    test::UnaryGradientTester<Logic, 3, 3>::test_custom(
        "SubConstantLogicNearZero", 
        30,      // num_tests
        1e-5,    // tolerance
        1e-7,    // delta
        -0.8,    // input_min
        0.8      // input_max
    );
}

TEST_F(SubConstantOperationsGradientTest, SubConstantLargeValues) {
    using Logic = SubConstantTestLogic<double>;
    test::UnaryGradientTester<Logic, 3, 3>::test_custom(
        "SubConstantLogicLargeValues", 
        40,      // num_tests
        1e-5,    // tolerance
        1e-6,    // delta
        -50.0,   // input_min
        50.0     // input_max
    );
}