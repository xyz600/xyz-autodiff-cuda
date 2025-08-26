#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include "../../../include/variable.cuh"
#include "../../../include/concept/variable.cuh"
#include "../../../include/concept/operation_node.cuh"
#include "../../../include/operations/unary/const_array_add_logic.cuh"
#include "../../../include/operations/unary/const_array_sub_logic.cuh"
#include "../../../include/util/cuda_unique_ptr.cuh"

using namespace xyz_autodiff;

// ===========================================
// Test Helper Structures
// ===========================================

template<typename T, std::size_t N>
struct TestArray {
    using value_type = T;
    T data[N];
    
    __host__ __device__ TestArray() = default;
    
    __host__ __device__ TestArray(std::initializer_list<T> init) {
        std::size_t i = 0;
        for (auto val : init) {
            if (i < N) data[i++] = val;
        }
    }
    
    __host__ __device__ const T& operator[](std::size_t i) const { return data[i]; }
    __host__ __device__ T& operator[](std::size_t i) { return data[i]; }
};

// ===========================================
// Static Assert Tests for Concept Compliance
// ===========================================

using TestVector3 = Variable<3, float>;
using TestVectorRef3 = VariableRef<3, float>;
using TestArrayType = TestArray<float, 3>;
using ConstArrayAddOp = UnaryOperation<3, op::ConstArrayAddLogic<3, TestArrayType>, TestVectorRef3>;

static_assert(VariableConcept<ConstArrayAddOp>, "ConstArrayAddOperation should satisfy VariableConcept");
static_assert(DifferentiableVariableConcept<ConstArrayAddOp>, "ConstArrayAddOperation should satisfy DifferentiableVariableConcept");
static_assert(OperationNode<ConstArrayAddOp>, "ConstArrayAddOperation should satisfy OperationNode");
static_assert(!OperationNode<TestVector3>, "Variable should NOT be OperationNode");
static_assert(op::ArrayLikeConcept<TestArray<float, 3>>, "TestArray should satisfy ArrayLikeConcept");

// ===========================================
// Test Class
// ===========================================

class ConstArrayOperationsSimpleTest : public ::testing::Test {
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
// Simple Forward Tests
// ===========================================

__global__ void test_const_add_kernel(float* result) {
    float input_data[3] = {1.0f, 2.0f, 3.0f};
    float input_grad[3] = {0.0f, 0.0f, 0.0f};
    
    VariableRef<3, float> input(input_data, input_grad);
    TestArray<float, 3> const_arr{0.5f, 1.5f, 2.5f};
    
    auto operation = op::const_add(input, const_arr);
    operation.forward();
    
    bool success = (abs(operation[0] - 1.5f) < 1e-6f) &&
                   (abs(operation[1] - 3.5f) < 1e-6f) &&
                   (abs(operation[2] - 5.5f) < 1e-6f);
    
    result[0] = success ? 1.0f : 0.0f;
}

TEST_F(ConstArrayOperationsSimpleTest, ConstAddForward) {
    auto device_result = makeCudaUnique<float>();
    
    test_const_add_kernel<<<1, 1>>>(device_result.get());
    cudaDeviceSynchronize();
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    
    float host_result;
    cudaMemcpy(&host_result, device_result.get(), sizeof(float), cudaMemcpyDeviceToHost);
    EXPECT_EQ(host_result, 1.0f);
}

__global__ void test_const_sub_kernel(float* result) {
    float input_data[3] = {5.0f, 4.0f, 3.0f};
    float input_grad[3] = {0.0f, 0.0f, 0.0f};
    
    VariableRef<3, float> input(input_data, input_grad);
    TestArray<float, 3> const_arr{1.5f, 2.5f, 0.5f};
    
    auto operation = op::const_sub(input, const_arr);
    operation.forward();
    
    bool success = (abs(operation[0] - 3.5f) < 1e-6f) &&
                   (abs(operation[1] - 1.5f) < 1e-6f) &&
                   (abs(operation[2] - 2.5f) < 1e-6f);
    
    result[0] = success ? 1.0f : 0.0f;
}

TEST_F(ConstArrayOperationsSimpleTest, ConstSubForward) {
    auto device_result = makeCudaUnique<float>();
    
    test_const_sub_kernel<<<1, 1>>>(device_result.get());
    cudaDeviceSynchronize();
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    
    float host_result;
    cudaMemcpy(&host_result, device_result.get(), sizeof(float), cudaMemcpyDeviceToHost);
    EXPECT_EQ(host_result, 1.0f);
}

// ===========================================
// Simple Gradient Tests
// ===========================================

__global__ void test_const_add_gradient_kernel(float* result) {
    float input_data[3] = {2.0f, 3.0f, 4.0f};
    float input_grad[3] = {0.0f, 0.0f, 0.0f};
    
    VariableRef<3, float> input(input_data, input_grad);
    TestArray<float, 3> const_arr{1.0f, 2.0f, 3.0f};
    
    auto operation = op::const_add(input, const_arr);
    operation.run();
    
    bool success = (abs(input.grad(0) - 1.0f) < 1e-6f) &&
                   (abs(input.grad(1) - 1.0f) < 1e-6f) &&
                   (abs(input.grad(2) - 1.0f) < 1e-6f);
    
    result[0] = success ? 1.0f : 0.0f;
}

TEST_F(ConstArrayOperationsSimpleTest, ConstAddGradient) {
    auto device_result = makeCudaUnique<float>();
    
    test_const_add_gradient_kernel<<<1, 1>>>(device_result.get());
    cudaDeviceSynchronize();
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    
    float host_result;
    cudaMemcpy(&host_result, device_result.get(), sizeof(float), cudaMemcpyDeviceToHost);
    EXPECT_EQ(host_result, 1.0f);
}

__global__ void test_const_sub_gradient_kernel(float* result) {
    float input_data[3] = {10.0f, 20.0f, 30.0f};
    float input_grad[3] = {0.0f, 0.0f, 0.0f};
    
    VariableRef<3, float> input(input_data, input_grad);
    TestArray<float, 3> const_arr{5.0f, 10.0f, 15.0f};
    
    auto operation = op::const_sub(input, const_arr);
    operation.run();
    
    bool success = (abs(input.grad(0) - 1.0f) < 1e-6f) &&
                   (abs(input.grad(1) - 1.0f) < 1e-6f) &&
                   (abs(input.grad(2) - 1.0f) < 1e-6f);
    
    result[0] = success ? 1.0f : 0.0f;
}

TEST_F(ConstArrayOperationsSimpleTest, ConstSubGradient) {
    auto device_result = makeCudaUnique<float>();
    
    test_const_sub_gradient_kernel<<<1, 1>>>(device_result.get());
    cudaDeviceSynchronize();
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    
    float host_result;
    cudaMemcpy(&host_result, device_result.get(), sizeof(float), cudaMemcpyDeviceToHost);
    EXPECT_EQ(host_result, 1.0f);
}