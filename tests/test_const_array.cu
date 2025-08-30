#include <gtest/gtest.h>
#include <xyz_autodiff/const_array.cuh>
#include <xyz_autodiff/variable.cuh>
#include <xyz_autodiff/util/cuda_unique_ptr.cuh>
#include <xyz_autodiff/variable_operators.cuh>

using namespace xyz_autodiff;

// Test ConstArrayLike compatible structure (forward declare for static_assert)
template<typename T, int N>
struct SimpleArray {
    using value_type = T;
    static constexpr std::size_t size = N;
    T values[N];
    
    __device__ __host__ constexpr SimpleArray() = default;
    
    __device__ __host__ constexpr SimpleArray(T val) {
        for (int i = 0; i < N; ++i) {
            values[i] = val;
        }
    }
    
    __device__ __host__ constexpr T& operator[](std::size_t index) {
        return values[index];
    }
    
    __device__ __host__ constexpr const T& operator[](std::size_t index) const {
        return values[index];
    }
};

// Static assert tests for concept compliance
namespace concept_tests {
    // Test that ConstArray satisfies ConstArrayLike
    static_assert(ConstArrayLike<ConstArray<float, 3>>, "ConstArray should satisfy ConstArrayLike");
    static_assert(ConstArrayLike<ConstArray<double, 5>>, "ConstArray should satisfy ConstArrayLike");
    
    // Test that SimpleArray satisfies ConstArrayLike  
    static_assert(ConstArrayLike<SimpleArray<float, 3>>, "SimpleArray should satisfy ConstArrayLike");
    static_assert(ConstArrayLike<SimpleArray<double, 5>>, "SimpleArray should satisfy ConstArrayLike");
    
    // Test that Variable and VariableRef satisfy ConstArrayLike
    static_assert(ConstArrayLike<Variable<3, float>>, "Variable should satisfy ConstArrayLike");
    static_assert(ConstArrayLike<Variable<5, double>>, "Variable should satisfy ConstArrayLike");
    static_assert(ConstArrayLike<VariableRef<3, float>>, "VariableRef should satisfy ConstArrayLike");
    static_assert(ConstArrayLike<VariableRef<5, double>>, "VariableRef should satisfy ConstArrayLike");
    
    // Test ConstArrayCompatible between different types with same value_type
    static_assert(ConstArrayCompatible<ConstArray<float, 3>, SimpleArray<float, 3>>, 
                  "ConstArray and SimpleArray with same type should be compatible");
    static_assert(ConstArrayCompatible<ConstArray<float, 3>, Variable<3, float>>, 
                  "ConstArray and Variable with same type should be compatible");
    static_assert(ConstArrayCompatible<ConstArray<float, 3>, VariableRef<3, float>>, 
                  "ConstArray and VariableRef with same type should be compatible");
    static_assert(ConstArrayCompatible<Variable<3, float>, VariableRef<3, float>>, 
                  "Variable and VariableRef with same type should be compatible");
    
    // Test ConstArraySameSize
    static_assert(ConstArraySameSize<ConstArray<float, 3>, SimpleArray<float, 3>>, 
                  "ConstArray and SimpleArray with same size should have same size");
    static_assert(ConstArraySameSize<ConstArray<float, 3>, Variable<3, float>>, 
                  "ConstArray and Variable with same size should have same size");
    static_assert(ConstArraySameSize<ConstArray<float, 3>, VariableRef<3, float>>, 
                  "ConstArray and VariableRef with same size should have same size");
    
    // Test that incompatible types are rejected
    static_assert(!ConstArrayCompatible<ConstArray<float, 3>, ConstArray<double, 3>>, 
                  "ConstArrays with different value_types should not be compatible");
    static_assert(!ConstArraySameSize<ConstArray<float, 3>, ConstArray<float, 5>>, 
                  "ConstArrays with different sizes should not have same size");
}

class ConstArrayTest : public ::testing::Test {
protected:
    void SetUp() override {
        int deviceCount;
        cudaGetDeviceCount(&deviceCount);
        if (deviceCount == 0) {
            GTEST_SKIP() << "No CUDA device available";
        }
    }
};

__global__ void test_const_array_kernel(float* result) {
    ConstArray<float, 3> arr;
    arr[0] = 1.0f;
    arr[1] = 2.0f;
    arr[2] = 3.0f;
    
    result[0] = arr[0];
    result[1] = arr[1];
    result[2] = arr[2];
    result[3] = static_cast<float>(ConstArray<float, 3>::size);
}

__global__ void test_const_array_initialization_kernel(float* result) {
    const float init_data[3] = {10.0f, 20.0f, 30.0f};
    ConstArray<float, 3> arr(init_data);
    
    result[0] = arr[0];
    result[1] = arr[1];
    result[2] = arr[2];
}

__global__ void test_const_array_const_access_kernel(float* result) {
    const float init_data[3] = {100.0f, 200.0f, 300.0f};
    const ConstArray<float, 3> arr(init_data);
    
    result[0] = arr[0];
    result[1] = arr[1];
    result[2] = arr[2];
}

__global__ void test_const_array_operators_kernel(float* result) {
    const float init_data1[3] = {10.0f, 20.0f, 30.0f};
    const float init_data2[3] = {5.0f, 15.0f, 25.0f};
    
    ConstArray<float, 3> arr1(init_data1);
    ConstArray<float, 3> arr2(init_data2);
    
    // Test assignment operator
    ConstArray<float, 3> arr_copy;
    arr_copy = arr1;  // Copy arr1 to arr_copy
    result[0] = arr_copy[0];  // Should be 10.0f
    result[1] = arr_copy[1];  // Should be 20.0f
    result[2] = arr_copy[2];  // Should be 30.0f
    
    // Test compound assignment
    arr1 += arr2;  // arr1 becomes [15, 35, 55]
    result[3] = arr1[0];
    result[4] = arr1[1];
    result[5] = arr1[2];
    
    // Test binary subtraction
    auto arr3 = arr1 - arr2;  // [15, 35, 55] - [5, 15, 25] = [10, 20, 30]
    result[6] = arr3[0];
    result[7] = arr3[1];
    result[8] = arr3[2];
}

__global__ void test_const_array_like_kernel(float* result) {
    const float init_data[3] = {100.0f, 200.0f, 300.0f};
    ConstArray<float, 3> arr1(init_data);
    SimpleArray<float, 3> simple_arr(50.0f);  // All values set to 50.0f
    
    // Test ConstArray += ConstArrayLike
    arr1 += simple_arr;  // [100, 200, 300] += [50, 50, 50] = [150, 250, 350]
    result[0] = arr1[0];
    result[1] = arr1[1];
    result[2] = arr1[2];
    
    // Test ConstArray - ConstArrayLike
    auto diff = arr1 - simple_arr;  // [150, 250, 350] - [50, 50, 50] = [100, 200, 300]
    result[3] = diff[0];
    result[4] = diff[1];
    result[5] = diff[2];
    
    // Test ConstArrayLike + ConstArray
    SimpleArray<float, 3> simple_arr2(10.0f);
    auto sum = simple_arr2 + arr1;  // [10, 10, 10] + [150, 250, 350] = [160, 260, 360]
    result[6] = sum[0];
    result[7] = sum[1];
    result[8] = sum[2];
}

TEST_F(ConstArrayTest, BasicOperations) {
    struct TestBuffers {
        float result[4];
    };
    auto device_buffers = makeCudaUnique<TestBuffers>();
    
    test_const_array_kernel<<<1, 1>>>(device_buffers->result);
    cudaDeviceSynchronize();
    
    TestBuffers host_buffers;
    cudaMemcpy(&host_buffers, device_buffers.get(), sizeof(TestBuffers), cudaMemcpyDeviceToHost);
    
    EXPECT_FLOAT_EQ(host_buffers.result[0], 1.0f);
    EXPECT_FLOAT_EQ(host_buffers.result[1], 2.0f);
    EXPECT_FLOAT_EQ(host_buffers.result[2], 3.0f);
    EXPECT_FLOAT_EQ(host_buffers.result[3], 3.0f);
}

TEST_F(ConstArrayTest, InitializationConstructor) {
    struct TestBuffers {
        float result[3];
    };
    auto device_buffers = makeCudaUnique<TestBuffers>();
    
    test_const_array_initialization_kernel<<<1, 1>>>(device_buffers->result);
    cudaDeviceSynchronize();
    
    TestBuffers host_buffers;
    cudaMemcpy(&host_buffers, device_buffers.get(), sizeof(TestBuffers), cudaMemcpyDeviceToHost);
    
    EXPECT_FLOAT_EQ(host_buffers.result[0], 10.0f);
    EXPECT_FLOAT_EQ(host_buffers.result[1], 20.0f);
    EXPECT_FLOAT_EQ(host_buffers.result[2], 30.0f);
}

TEST_F(ConstArrayTest, ConstAccess) {
    struct TestBuffers {
        float result[3];
    };
    auto device_buffers = makeCudaUnique<TestBuffers>();
    
    test_const_array_const_access_kernel<<<1, 1>>>(device_buffers->result);
    cudaDeviceSynchronize();
    
    TestBuffers host_buffers;
    cudaMemcpy(&host_buffers, device_buffers.get(), sizeof(TestBuffers), cudaMemcpyDeviceToHost);
    
    EXPECT_FLOAT_EQ(host_buffers.result[0], 100.0f);
    EXPECT_FLOAT_EQ(host_buffers.result[1], 200.0f);
    EXPECT_FLOAT_EQ(host_buffers.result[2], 300.0f);
}

TEST_F(ConstArrayTest, HostOperations) {
    ConstArray<int, 4> arr;
    arr[0] = 10;
    arr[1] = 20;
    arr[2] = 30;
    arr[3] = 40;
    
    EXPECT_EQ(arr[0], 10);
    EXPECT_EQ(arr[1], 20);
    EXPECT_EQ(arr[2], 30);
    EXPECT_EQ(arr[3], 40);
    constexpr auto expected_size = ConstArray<int, 4>::size;
    EXPECT_EQ(expected_size, 4);
}

TEST_F(ConstArrayTest, HostInitialization) {
    const int init_data[5] = {1, 2, 3, 4, 5};
    ConstArray<int, 5> arr(init_data);
    
    EXPECT_EQ(arr[0], 1);
    EXPECT_EQ(arr[1], 2);
    EXPECT_EQ(arr[2], 3);
    EXPECT_EQ(arr[3], 4);
    EXPECT_EQ(arr[4], 5);
    constexpr auto expected_size_5 = ConstArray<int, 5>::size;
    EXPECT_EQ(expected_size_5, 5);
}

TEST_F(ConstArrayTest, ArithmeticOperators) {
    struct TestBuffers {
        float result[9];
    };
    auto device_buffers = makeCudaUnique<TestBuffers>();
    
    test_const_array_operators_kernel<<<1, 1>>>(device_buffers->result);
    cudaDeviceSynchronize();
    
    TestBuffers host_buffers;
    cudaMemcpy(&host_buffers, device_buffers.get(), sizeof(TestBuffers), cudaMemcpyDeviceToHost);
    
    // Test assignment results: arr_copy = arr1 (original [10, 20, 30])
    EXPECT_FLOAT_EQ(host_buffers.result[0], 10.0f);
    EXPECT_FLOAT_EQ(host_buffers.result[1], 20.0f);
    EXPECT_FLOAT_EQ(host_buffers.result[2], 30.0f);
    
    // Test compound assignment results: [10, 20, 30] += [5, 15, 25] = [15, 35, 55]
    EXPECT_FLOAT_EQ(host_buffers.result[3], 15.0f);
    EXPECT_FLOAT_EQ(host_buffers.result[4], 35.0f);
    EXPECT_FLOAT_EQ(host_buffers.result[5], 55.0f);
    
    // Test binary subtraction results: [15, 35, 55] - [5, 15, 25] = [10, 20, 30]
    EXPECT_FLOAT_EQ(host_buffers.result[6], 10.0f);
    EXPECT_FLOAT_EQ(host_buffers.result[7], 20.0f);
    EXPECT_FLOAT_EQ(host_buffers.result[8], 30.0f);
}

TEST_F(ConstArrayTest, HostArithmeticOperators) {
    ConstArray<float, 3> arr1;
    arr1[0] = 10.0f; arr1[1] = 20.0f; arr1[2] = 30.0f;
    
    ConstArray<float, 3> arr2;
    arr2[0] = 5.0f; arr2[1] = 15.0f; arr2[2] = 25.0f;
    
    // Test assignment operator
    ConstArray<float, 3> arr_copy;
    arr_copy = arr1;
    EXPECT_FLOAT_EQ(arr_copy[0], 10.0f);
    EXPECT_FLOAT_EQ(arr_copy[1], 20.0f);
    EXPECT_FLOAT_EQ(arr_copy[2], 30.0f);
    
    // Test compound assignment
    arr1 += arr2;
    EXPECT_FLOAT_EQ(arr1[0], 15.0f);
    EXPECT_FLOAT_EQ(arr1[1], 35.0f);
    EXPECT_FLOAT_EQ(arr1[2], 55.0f);
    
    // Test binary operations
    auto diff = arr1 - arr2;  // [15, 35, 55] - [5, 15, 25] = [10, 20, 30]
    EXPECT_FLOAT_EQ(diff[0], 10.0f);
    EXPECT_FLOAT_EQ(diff[1], 20.0f);
    EXPECT_FLOAT_EQ(diff[2], 30.0f);
    
    auto sum = arr1 + arr2;  // [15, 35, 55] + [5, 15, 25] = [20, 50, 80]
    EXPECT_FLOAT_EQ(sum[0], 20.0f);
    EXPECT_FLOAT_EQ(sum[1], 50.0f);
    EXPECT_FLOAT_EQ(sum[2], 80.0f);
}

TEST_F(ConstArrayTest, ConstArrayLikeOperators) {
    struct TestBuffers {
        float result[9];
    };
    auto device_buffers = makeCudaUnique<TestBuffers>();
    
    test_const_array_like_kernel<<<1, 1>>>(device_buffers->result);
    cudaDeviceSynchronize();
    
    TestBuffers host_buffers;
    cudaMemcpy(&host_buffers, device_buffers.get(), sizeof(TestBuffers), cudaMemcpyDeviceToHost);
    
    // Test ConstArray += ConstArrayLike: [100, 200, 300] += [50, 50, 50] = [150, 250, 350]
    EXPECT_FLOAT_EQ(host_buffers.result[0], 150.0f);
    EXPECT_FLOAT_EQ(host_buffers.result[1], 250.0f);
    EXPECT_FLOAT_EQ(host_buffers.result[2], 350.0f);
    
    // Test ConstArray - ConstArrayLike: [150, 250, 350] - [50, 50, 50] = [100, 200, 300]
    EXPECT_FLOAT_EQ(host_buffers.result[3], 100.0f);
    EXPECT_FLOAT_EQ(host_buffers.result[4], 200.0f);
    EXPECT_FLOAT_EQ(host_buffers.result[5], 300.0f);
    
    // Test ConstArrayLike + ConstArray: [10, 10, 10] + [150, 250, 350] = [160, 260, 360]
    EXPECT_FLOAT_EQ(host_buffers.result[6], 160.0f);
    EXPECT_FLOAT_EQ(host_buffers.result[7], 260.0f);
    EXPECT_FLOAT_EQ(host_buffers.result[8], 360.0f);
}

TEST_F(ConstArrayTest, HostConstArrayLikeOperators) {
    ConstArray<float, 3> arr1;
    arr1[0] = 100.0f; arr1[1] = 200.0f; arr1[2] = 300.0f;
    
    SimpleArray<float, 3> simple_arr(25.0f);
    
    // Test compound assignment with ConstArrayLike
    arr1 += simple_arr;  // [100, 200, 300] += [25, 25, 25] = [125, 225, 325]
    EXPECT_FLOAT_EQ(arr1[0], 125.0f);
    EXPECT_FLOAT_EQ(arr1[1], 225.0f);
    EXPECT_FLOAT_EQ(arr1[2], 325.0f);
    
    // Test binary operations between different ConstArrayLike types
    auto diff = arr1 - simple_arr;  // [125, 225, 325] - [25, 25, 25] = [100, 200, 300]
    EXPECT_FLOAT_EQ(diff[0], 100.0f);
    EXPECT_FLOAT_EQ(diff[1], 200.0f);
    EXPECT_FLOAT_EQ(diff[2], 300.0f);
    
    SimpleArray<float, 3> simple_arr2(5.0f);
    auto sum = simple_arr2 + arr1;  // [5, 5, 5] + [125, 225, 325] = [130, 230, 330]
    EXPECT_FLOAT_EQ(sum[0], 130.0f);
    EXPECT_FLOAT_EQ(sum[1], 230.0f);
    EXPECT_FLOAT_EQ(sum[2], 330.0f);
}