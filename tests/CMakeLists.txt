# Google Test を探す
find_package(GTest QUIET)
if(NOT GTest_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/03597a01ee50f33f9142cc5c5fd0f5e48f8f92a0.zip
    )
    FetchContent_MakeAvailable(googletest)
endif()

# Variable テスト
add_executable(test_variable test_variable.cu)
target_link_libraries(test_variable xyz_autodiff gtest gtest_main)
target_include_directories(test_variable PRIVATE ${CMAKE_SOURCE_DIR}/include)
set_target_properties(test_variable PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "89"
)

# DenseMatrix テスト
add_executable(test_dense_matrix test_dense_matrix.cu)
target_link_libraries(test_dense_matrix xyz_autodiff gtest gtest_main)
target_include_directories(test_dense_matrix PRIVATE ${CMAKE_SOURCE_DIR}/include)
set_target_properties(test_dense_matrix PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "89"
)

# DiagonalMatrix テスト
add_executable(test_diagonal_matrix test_diagonal_matrix.cu)
target_link_libraries(test_diagonal_matrix xyz_autodiff gtest gtest_main)
target_include_directories(test_diagonal_matrix PRIVATE ${CMAKE_SOURCE_DIR}/include)
set_target_properties(test_diagonal_matrix PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "89"
)

# Matrix Transpose テスト
add_executable(test_matrix_transpose test_matrix_transpose.cu)
target_link_libraries(test_matrix_transpose xyz_autodiff gtest gtest_main)
target_include_directories(test_matrix_transpose PRIVATE ${CMAKE_SOURCE_DIR}/include)
set_target_properties(test_matrix_transpose PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "89"
)

# SymmetricMatrix テスト
add_executable(test_symmetric_matrix test_symmetric_matrix.cu)
target_link_libraries(test_symmetric_matrix xyz_autodiff gtest gtest_main)
target_include_directories(test_symmetric_matrix PRIVATE ${CMAKE_SOURCE_DIR}/include)
set_target_properties(test_symmetric_matrix PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "89"
)

# UnaryOperations テスト
add_executable(test_unary_operations operation/unary/test_unary_operations.cu)
target_link_libraries(test_unary_operations xyz_autodiff gtest gtest_main)
target_include_directories(test_unary_operations PRIVATE ${CMAKE_SOURCE_DIR}/include)
set_target_properties(test_unary_operations PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "89"
)

# MathDispatcher テスト
add_executable(test_math_dispatcher test_math_dispatcher.cu)
target_link_libraries(test_math_dispatcher xyz_autodiff gtest gtest_main)
target_include_directories(test_math_dispatcher PRIVATE ${CMAKE_SOURCE_DIR}/include)
set_target_properties(test_math_dispatcher PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "89"
)

# GradientVerification テスト
add_executable(test_gradient_verification operation/base/test_gradient_verification.cu)
target_link_libraries(test_gradient_verification xyz_autodiff gtest gtest_main)
target_include_directories(test_gradient_verification PRIVATE ${CMAKE_SOURCE_DIR}/include)
set_target_properties(test_gradient_verification PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "89"
)

# OperationChaining テスト
add_executable(test_operation_chaining operation/base/test_operation_chaining.cu)
target_link_libraries(test_operation_chaining xyz_autodiff gtest gtest_main)
target_include_directories(test_operation_chaining PRIVATE ${CMAKE_SOURCE_DIR}/include)
set_target_properties(test_operation_chaining PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "89"
)

# OperationConcepts テスト
add_executable(test_operation_concepts operation/base/test_operation_concepts.cu)
target_link_libraries(test_operation_concepts xyz_autodiff gtest gtest_main)
target_include_directories(test_operation_concepts PRIVATE ${CMAKE_SOURCE_DIR}/include)
set_target_properties(test_operation_concepts PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "89"
)

# QuaternionToRotationMatrix テスト
add_executable(test_quaternion_to_rotation_matrix operation/unary/test_quaternion_to_rotation_matrix.cu)
target_link_libraries(test_quaternion_to_rotation_matrix xyz_autodiff gtest gtest_main)
target_include_directories(test_quaternion_to_rotation_matrix PRIVATE ${CMAKE_SOURCE_DIR}/include)
set_target_properties(test_quaternion_to_rotation_matrix PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "89"
)

# Parallel Gradient Accumulation テスト
add_executable(test_parallel_gradient_accumulation test_parallel_gradient_accumulation.cu)
target_link_libraries(test_parallel_gradient_accumulation xyz_autodiff gtest gtest_main)
target_include_directories(test_parallel_gradient_accumulation PRIVATE ${CMAKE_SOURCE_DIR}/include)
set_target_properties(test_parallel_gradient_accumulation PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "89"
)

# Shared Memory Atomic テスト
add_executable(test_shared_memory_atomic test_shared_memory_atomic.cu)
target_link_libraries(test_shared_memory_atomic xyz_autodiff gtest gtest_main)
target_include_directories(test_shared_memory_atomic PRIVATE ${CMAKE_SOURCE_DIR}/include)
set_target_properties(test_shared_memory_atomic PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "89"
)

# DAG Backward テスト
add_executable(test_dag_backward test_dag_backward.cu)
target_link_libraries(test_dag_backward xyz_autodiff gtest gtest_main)
target_include_directories(test_dag_backward PRIVATE ${CMAKE_SOURCE_DIR}/include)
set_target_properties(test_dag_backward PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "89"
)

# Standard Norm Operations テスト
add_executable(test_norm_operations operation/unary/test_norm_operations.cu)
target_link_libraries(test_norm_operations xyz_autodiff gtest gtest_main)
target_include_directories(test_norm_operations PRIVATE ${CMAKE_SOURCE_DIR}/include)
set_target_properties(test_norm_operations PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "89"
)

# Standard Symmetric Matrix 2x2 Inverse テスト
add_executable(test_sym_matrix2_inv operation/unary/test_sym_matrix2_inv.cu)
target_link_libraries(test_sym_matrix2_inv xyz_autodiff gtest gtest_main)
target_include_directories(test_sym_matrix2_inv PRIVATE ${CMAKE_SOURCE_DIR}/include)
set_target_properties(test_sym_matrix2_inv PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "89"
)

# Broadcast Operation テスト
add_executable(test_broadcast operation/unary/test_broadcast.cu)
target_link_libraries(test_broadcast xyz_autodiff gtest gtest_main)
target_include_directories(test_broadcast PRIVATE ${CMAKE_SOURCE_DIR}/include)
set_target_properties(test_broadcast PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "89"
)

# Mini Gaussian Splatting Individual Operation Tests
add_executable(test_covariance_generation ../examples/mini-gaussian-splatting/tests/test_covariance_generation.cu)
target_link_libraries(test_covariance_generation xyz_autodiff gtest gtest_main)
target_include_directories(test_covariance_generation PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/examples/mini-gaussian-splatting)
set_target_properties(test_covariance_generation PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "89"
)

# Removed: test_element_wise_exp and test_element_wise_multiply (now covered by standard operations)

add_executable(test_mahalanobis_distance ../examples/mini-gaussian-splatting/tests/test_mahalanobis_distance.cu)
target_link_libraries(test_mahalanobis_distance xyz_autodiff gtest gtest_main)
target_include_directories(test_mahalanobis_distance PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/examples/mini-gaussian-splatting)
set_target_properties(test_mahalanobis_distance PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "89"
)

add_executable(test_matrix_multiplication ../examples/mini-gaussian-splatting/tests/test_matrix_multiplication.cu)
target_link_libraries(test_matrix_multiplication xyz_autodiff gtest gtest_main)
target_include_directories(test_matrix_multiplication PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/examples/mini-gaussian-splatting)
set_target_properties(test_matrix_multiplication PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "89"
)

# Linear Regression Network テスト
add_executable(test_linear_regression_network operation/base/test_linear_regression_network.cu)
target_link_libraries(test_linear_regression_network xyz_autodiff gtest gtest_main)
target_include_directories(test_linear_regression_network PRIVATE ${CMAKE_SOURCE_DIR}/include)
set_target_properties(test_linear_regression_network PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "89"
)

# Variable Operators テスト (constant operators)
add_executable(test_variable_operators test_variable_operators.cu)
target_link_libraries(test_variable_operators xyz_autodiff gtest gtest_main)
target_include_directories(test_variable_operators PRIVATE ${CMAKE_SOURCE_DIR}/include)
set_target_properties(test_variable_operators PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "89"
)

# Binary Variable Operators テスト (Variable + Variable operators)
add_executable(test_binary_variable_operators operation/binary/test_variable_operators.cu)
target_link_libraries(test_binary_variable_operators xyz_autodiff gtest gtest_main)
target_include_directories(test_binary_variable_operators PRIVATE ${CMAKE_SOURCE_DIR}/include)
set_target_properties(test_binary_variable_operators PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "89"
)

# NEW GRADIENT VERIFICATION TESTS

# Binary Operations Gradient Tests
add_executable(test_binary_operations_gradient operation/binary/test_binary_operations_gradient.cu)
target_link_libraries(test_binary_operations_gradient xyz_autodiff gtest gtest_main)
target_include_directories(test_binary_operations_gradient PRIVATE ${CMAKE_SOURCE_DIR}/include)
set_target_properties(test_binary_operations_gradient PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "89"
)

# Basic Unary Operations Gradient Tests
add_executable(test_basic_unary_operations_gradient operation/unary/test_basic_unary_operations_gradient.cu)
target_link_libraries(test_basic_unary_operations_gradient xyz_autodiff gtest gtest_main)
target_include_directories(test_basic_unary_operations_gradient PRIVATE ${CMAKE_SOURCE_DIR}/include)
set_target_properties(test_basic_unary_operations_gradient PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "89"
)

# Trigonometric Operations Gradient Tests
add_executable(test_trigonometric_operations_gradient operation/unary/test_trigonometric_operations_gradient.cu)
target_link_libraries(test_trigonometric_operations_gradient xyz_autodiff gtest gtest_main)
target_include_directories(test_trigonometric_operations_gradient PRIVATE ${CMAKE_SOURCE_DIR}/include)
set_target_properties(test_trigonometric_operations_gradient PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "89"
)

# Specialized Operations Gradient Tests
add_executable(test_specialized_operations_gradient operation/unary/test_specialized_operations_gradient.cu)
target_link_libraries(test_specialized_operations_gradient xyz_autodiff gtest gtest_main)
target_include_directories(test_specialized_operations_gradient PRIVATE ${CMAKE_SOURCE_DIR}/include)
set_target_properties(test_specialized_operations_gradient PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "89"
)

# CTest にテストを登録
add_test(NAME VariableTest COMMAND test_variable)
add_test(NAME DenseMatrixTest COMMAND test_dense_matrix)
add_test(NAME DiagonalMatrixTest COMMAND test_diagonal_matrix)
add_test(NAME MatrixTransposeTest COMMAND test_matrix_transpose)
add_test(NAME SymmetricMatrixTest COMMAND test_symmetric_matrix)
add_test(NAME UnaryOperationsTest COMMAND test_unary_operations)
add_test(NAME MathDispatcherTest COMMAND test_math_dispatcher)
add_test(NAME GradientVerificationTest COMMAND test_gradient_verification)
add_test(NAME OperationChainingTest COMMAND test_operation_chaining)
add_test(NAME OperationConceptsTest COMMAND test_operation_concepts)
add_test(NAME QuaternionToRotationMatrixTest COMMAND test_quaternion_to_rotation_matrix)
add_test(NAME ParallelGradientAccumulationTest COMMAND test_parallel_gradient_accumulation)
add_test(NAME SharedMemoryAtomicTest COMMAND test_shared_memory_atomic)
add_test(NAME DAGBackwardTest COMMAND test_dag_backward)
add_test(NAME NormOperationsTest COMMAND test_norm_operations)
add_test(NAME SymMatrix2InvTest COMMAND test_sym_matrix2_inv)
add_test(NAME BroadcastTest COMMAND test_broadcast)
add_test(NAME CovarianceGenerationTest COMMAND test_covariance_generation)
# Removed: ElementWiseExpTest and ElementWiseMultiplyTest (now covered by standard operations)
add_test(NAME MahalanobisDistanceTest COMMAND test_mahalanobis_distance)
add_test(NAME MatrixMultiplicationTest COMMAND test_matrix_multiplication)
# Removed: NormAdditionTest (now covered by standard add operations in NormOperationsTest)
# Removed: SymmetricMatrixOpsTest (now covered by standard operations)
add_test(NAME LinearRegressionNetworkTest COMMAND test_linear_regression_network)
add_test(NAME VariableOperatorsTest COMMAND test_variable_operators)
add_test(NAME BinaryVariableOperatorsTest COMMAND test_binary_variable_operators)

# Norm Operations Gradient Tests
add_executable(test_norm_operations_gradient operation/unary/test_norm_operations_gradient.cu)
target_link_libraries(test_norm_operations_gradient xyz_autodiff gtest gtest_main)
target_include_directories(test_norm_operations_gradient PRIVATE ${CMAKE_SOURCE_DIR}/include)
set_target_properties(test_norm_operations_gradient PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "89"
)

# Exponential Operations Gradient Tests
add_executable(test_exponential_operations_gradient operation/unary/test_exponential_operations_gradient.cu)
target_link_libraries(test_exponential_operations_gradient xyz_autodiff gtest gtest_main)
target_include_directories(test_exponential_operations_gradient PRIVATE ${CMAKE_SOURCE_DIR}/include)
set_target_properties(test_exponential_operations_gradient PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "89"
)

# Matrix Operations Gradient Tests
add_executable(test_matrix_operations_gradient operation/unary/test_matrix_operations_gradient.cu)
target_link_libraries(test_matrix_operations_gradient xyz_autodiff gtest gtest_main)
target_include_directories(test_matrix_operations_gradient PRIVATE ${CMAKE_SOURCE_DIR}/include)
set_target_properties(test_matrix_operations_gradient PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "89"
)

# Mini Gaussian Splatting Gradient Tests
add_executable(test_mini_gaussian_splatting_gradient ../examples/mini-gaussian-splatting/tests/test_mini_gaussian_splatting_gradient.cu)
target_link_libraries(test_mini_gaussian_splatting_gradient xyz_autodiff gtest gtest_main)
target_include_directories(test_mini_gaussian_splatting_gradient PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/examples/mini-gaussian-splatting)
set_target_properties(test_mini_gaussian_splatting_gradient PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "89"
)

# NEW GRADIENT VERIFICATION TESTS
add_test(NAME BinaryOperationsGradientTest COMMAND test_binary_operations_gradient)
add_test(NAME BasicUnaryOperationsGradientTest COMMAND test_basic_unary_operations_gradient)
add_test(NAME TrigonometricOperationsGradientTest COMMAND test_trigonometric_operations_gradient)
add_test(NAME SpecializedOperationsGradientTest COMMAND test_specialized_operations_gradient)
add_test(NAME NormOperationsGradientTest COMMAND test_norm_operations_gradient)
add_test(NAME ExponentialOperationsGradientTest COMMAND test_exponential_operations_gradient)
add_test(NAME MatrixOperationsGradientTest COMMAND test_matrix_operations_gradient)
add_test(NAME MiniGaussianSplattingGradientTest COMMAND test_mini_gaussian_splatting_gradient)
