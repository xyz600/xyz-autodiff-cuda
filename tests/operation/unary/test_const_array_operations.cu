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

// Simple array-like structure for testing
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

// Test with raw array wrapper
template<typename T>
struct ArrayWrapper {
    using value_type = T;
    const T* ptr;
    
    __host__ __device__ ArrayWrapper(const T* p) : ptr(p) {}
    __host__ __device__ const T& operator[](std::size_t i) const { return ptr[i]; }
};

// ===========================================
// Static Assert Tests for Concept Compliance
// ===========================================

// Test types
using TestVector3 = Variable<3, float>;
using TestVectorRef3 = VariableRef<3, float>;
using TestArrayType = TestArray<float, 3>;

// Operation types
using ConstArrayAddOp = UnaryOperation<3, op::ConstArrayAddLogic<3, TestArrayType>, TestVectorRef3>;
using ConstArraySubOp = UnaryOperation<3, op::ConstArraySubLogic<3, TestArrayType>, TestVectorRef3>;

// Static assertions for concept compliance
static_assert(VariableConcept<ConstArrayAddOp>, "ConstArrayAddOperation should satisfy VariableConcept");
static_assert(DifferentiableVariableConcept<ConstArrayAddOp>, "ConstArrayAddOperation should satisfy DifferentiableVariableConcept");
static_assert(OperationNode<ConstArrayAddOp>, "ConstArrayAddOperation should satisfy OperationNode");

static_assert(VariableConcept<ConstArraySubOp>, "ConstArraySubOperation should satisfy VariableConcept");
static_assert(DifferentiableVariableConcept<ConstArraySubOp>, "ConstArraySubOperation should satisfy DifferentiableVariableConcept");
static_assert(OperationNode<ConstArraySubOp>, "ConstArraySubOperation should satisfy OperationNode");

// Ensure Variable is NOT an OperationNode
static_assert(!OperationNode<TestVector3>, "Variable should NOT be OperationNode");

// Test that ArrayLikeConcept works
static_assert(op::ArrayLikeConcept<TestArray<float, 3>>, "TestArray should satisfy ArrayLikeConcept");
static_assert(op::ArrayLikeConcept<ArrayWrapper<float>>, "ArrayWrapper should satisfy ArrayLikeConcept");

// ===========================================
// Test Class
// ===========================================

class ConstArrayOperationsTest : public ::testing::Test {
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
// Test Buffer Structure
// ===========================================

template <typename T>
struct TestBuffers {
    T input_data[3];
    T input_grad[3];
    T output_data[3];
    T output_grad[3];
    T expected_results[3];
    T constant_array[3];
};

// ===========================================
// Forward Pass Tests
// ===========================================

__global__ void test_const_array_add_forward_kernel(float* result) {
    // Test: [1, 2, 3] + [0.5, 1.5, 2.5] = [1.5, 3.5, 5.5]
    float input_data[3] = {1.0f, 2.0f, 3.0f};
    float input_grad[3] = {0.0f, 0.0f, 0.0f};
    float const_array[3] = {0.5f, 1.5f, 2.5f};
    
    VariableRef<3, float> input(input_data, input_grad);
    TestArray<float, 3> const_arr{0.5f, 1.5f, 2.5f};
    
    auto operation = op::const_add(input, const_arr);
    operation.forward();
    
    // Check results
    bool success = (abs(operation[0] - 1.5f) < 1e-6f) &&
                   (abs(operation[1] - 3.5f) < 1e-6f) &&
                   (abs(operation[2] - 5.5f) < 1e-6f);
    
    result[0] = success ? 1.0f : 0.0f;
}

TEST_F(ConstArrayOperationsTest, ConstArrayAddForwardPass) {
    auto device_result = makeCudaUnique<float>();
    
    test_const_array_add_forward_kernel<<<1, 1>>>(device_result.get());
    cudaDeviceSynchronize();
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    
    float host_result;
    cudaMemcpy(&host_result, device_result.get(), sizeof(float), cudaMemcpyDeviceToHost);
    EXPECT_EQ(host_result, 1.0f);
}

__global__ void test_const_array_sub_forward_kernel(float* result) {
    // Test: [5, 4, 3] - [1.5, 2.5, 0.5] = [3.5, 1.5, 2.5]
    float input_data[3] = {5.0f, 4.0f, 3.0f};
    float input_grad[3] = {0.0f, 0.0f, 0.0f};
    
    VariableRef<3, float> input(input_data, input_grad);
    TestArray<float, 3> const_arr{1.5f, 2.5f, 0.5f};
    
    auto operation = op::const_sub(input, const_arr);
    operation.forward();
    
    // Check results
    bool success = (abs(operation[0] - 3.5f) < 1e-6f) &&
                   (abs(operation[1] - 1.5f) < 1e-6f) &&
                   (abs(operation[2] - 2.5f) < 1e-6f);
    
    result[0] = success ? 1.0f : 0.0f;
}

TEST_F(ConstArrayOperationsTest, ConstArraySubForwardPass) {
    auto device_result = makeCudaUnique<float>();
    
    test_const_array_sub_forward_kernel<<<1, 1>>>(device_result.get());
    cudaDeviceSynchronize();
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    
    float host_result;
    cudaMemcpy(&host_result, device_result.get(), sizeof(float), cudaMemcpyDeviceToHost);
    EXPECT_EQ(host_result, 1.0f);
}

// ===========================================
// Gradient Tests
// ===========================================

__global__ void test_const_array_add_gradient_kernel(float* result) {
    float input_data[3] = {2.0f, 3.0f, 4.0f};
    float input_grad[3] = {0.0f, 0.0f, 0.0f};
    
    VariableRef<3, float> input(input_data, input_grad);
    TestArray<float, 3> const_arr{1.0f, 2.0f, 3.0f};
    
    auto operation = op::const_add(input, const_arr);
    
    // Use run() for proper forward + backward execution
    operation.run();
    
    // Check gradients: should be [1.0, 1.0, 1.0] (derivative of addition)
    bool success = (abs(input.grad(0) - 1.0f) < 1e-6f) &&
                   (abs(input.grad(1) - 1.0f) < 1e-6f) &&
                   (abs(input.grad(2) - 1.0f) < 1e-6f);
    
    result[0] = success ? 1.0f : 0.0f;
}

TEST_F(ConstArrayOperationsTest, ConstArrayAddGradient) {
    auto device_result = makeCudaUnique<float>();
    
    test_const_array_add_gradient_kernel<<<1, 1>>>(device_result.get());
    cudaDeviceSynchronize();
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    
    float host_result;
    cudaMemcpy(&host_result, device_result.get(), sizeof(float), cudaMemcpyDeviceToHost);
    EXPECT_EQ(host_result, 1.0f);
}

__global__ void test_const_array_sub_gradient_kernel(float* result) {
    float input_data[3] = {10.0f, 20.0f, 30.0f};
    float input_grad[3] = {0.0f, 0.0f, 0.0f};
    
    VariableRef<3, float> input(input_data, input_grad);
    TestArray<float, 3> const_arr{5.0f, 10.0f, 15.0f};
    
    auto operation = op::const_sub(input, const_arr);
    
    // Use run() for proper forward + backward execution
    operation.run();
    
    // Check gradients: should be [1.0, 1.0, 1.0] (derivative of subtraction)
    bool success = (abs(input.grad(0) - 1.0f) < 1e-6f) &&
                   (abs(input.grad(1) - 1.0f) < 1e-6f) &&
                   (abs(input.grad(2) - 1.0f) < 1e-6f);
    
    result[0] = success ? 1.0f : 0.0f;
}

TEST_F(ConstArrayOperationsTest, ConstArraySubGradient) {
    auto device_result = makeCudaUnique<float>();
    
    test_const_array_sub_gradient_kernel<<<1, 1>>>(device_result.get());
    cudaDeviceSynchronize();
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    
    float host_result;
    cudaMemcpy(&host_result, device_result.get(), sizeof(float), cudaMemcpyDeviceToHost);
    EXPECT_EQ(host_result, 1.0f);
}

// ===========================================
// Combined Test for Interface and Array Wrapper Support
// ===========================================

__global__ void test_combined_functionality_kernel(float* result) {
    // Test 1: Array wrapper support with 2D vectors
    {
        float input_data[2] = {10.0f, 20.0f};
        float input_grad[2] = {0.0f, 0.0f};
        float const_array[2] = {3.0f, 7.0f};
        
        VariableRef<2, float> input(input_data, input_grad);
        ArrayWrapper<float> wrapper(const_array);
        
        auto operation = op::const_add(input, wrapper);
        operation.forward();
        
        // Check results: [10, 20] + [3, 7] = [13, 27]
        bool wrapper_test = (abs(operation[0] - 13.0f) < 1e-6f) &&
                           (abs(operation[1] - 27.0f) < 1e-6f);
        
        if (!wrapper_test) {
            result[0] = 0.0f;
            return;
        }
    }
    
    // Test 2: Interface compliance with 3D vectors
    {
        float input_data[3] = {1.0f, 2.0f, 3.0f};
        float input_grad[3] = {0.0f, 0.0f, 0.0f};
        
        VariableRef<3, float> input(input_data, input_grad);
        TestArray<float, 3> const_arr{0.1f, 0.2f, 0.3f};
        
        auto operation = op::const_add(input, const_arr);
        operation.forward();
        
        // Test data() access
        float data_sum = operation.data()[0] + operation.data()[1] + operation.data()[2];
        float expected = (1.0f + 0.1f) + (2.0f + 0.2f) + (3.0f + 0.3f); // 6.6f
        
        bool interface_test = abs(data_sum - expected) < 1e-6f;
        
        if (!interface_test) {
            result[0] = 0.0f;
            return;
        }
    }
    
    result[0] = 1.0f; // All tests passed
}

TEST_F(ConstArrayOperationsTest, CombinedFunctionalityTest) {
    auto device_result = makeCudaUnique<float>();
    
    test_combined_functionality_kernel<<<1, 1>>>(device_result.get());
    cudaDeviceSynchronize();
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    
    float host_result;
    cudaMemcpy(&host_result, device_result.get(), sizeof(float), cudaMemcpyDeviceToHost);
    EXPECT_EQ(host_result, 1.0f);
}