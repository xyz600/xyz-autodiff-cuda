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

# 既存のテストは削除

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

# Operation テスト
add_executable(test_operation test_operation.cu)
target_link_libraries(test_operation xyz_autodiff gtest gtest_main)
target_include_directories(test_operation PRIVATE ${CMAKE_SOURCE_DIR}/include)
set_target_properties(test_operation PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "89"
)

# Operation Chaining テスト
add_executable(test_operation_chaining test_operation_chaining.cu)
target_link_libraries(test_operation_chaining xyz_autodiff gtest gtest_main)
target_include_directories(test_operation_chaining PRIVATE ${CMAKE_SOURCE_DIR}/include)
set_target_properties(test_operation_chaining PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "89"
)

# Operation Chaining Simple テスト
add_executable(test_operation_chaining_simple test_operation_chaining_simple.cu)
target_link_libraries(test_operation_chaining_simple xyz_autodiff gtest gtest_main)
target_include_directories(test_operation_chaining_simple PRIVATE ${CMAKE_SOURCE_DIR}/include)
set_target_properties(test_operation_chaining_simple PROPERTIES
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

# Unary Functions テスト
add_executable(test_unary_functions test_unary_functions.cu)
target_link_libraries(test_unary_functions xyz_autodiff gtest gtest_main)
target_include_directories(test_unary_functions PRIVATE ${CMAKE_SOURCE_DIR}/include)
set_target_properties(test_unary_functions PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "89"
)

# Complex Operations テスト
add_executable(test_complex_operations test_complex_operations.cu)
target_link_libraries(test_complex_operations xyz_autodiff gtest gtest_main)
target_include_directories(test_complex_operations PRIVATE ${CMAKE_SOURCE_DIR}/include)
set_target_properties(test_complex_operations PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "89"
)

# Numerical Verification テスト
add_executable(test_numerical_verification test_numerical_verification.cu)
target_link_libraries(test_numerical_verification xyz_autodiff gtest gtest_main)
target_include_directories(test_numerical_verification PRIVATE ${CMAKE_SOURCE_DIR}/include)
set_target_properties(test_numerical_verification PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "89"
)

# CTest にテストを登録
add_test(NAME VariableTest COMMAND test_variable)
add_test(NAME DenseMatrixTest COMMAND test_dense_matrix)
add_test(NAME DiagonalMatrixTest COMMAND test_diagonal_matrix)
add_test(NAME OperationTest COMMAND test_operation)
add_test(NAME OperationChainingTest COMMAND test_operation_chaining)
add_test(NAME OperationChainingSimpleTest COMMAND test_operation_chaining_simple)
add_test(NAME MatrixTransposeTest COMMAND test_matrix_transpose)
add_test(NAME SymmetricMatrixTest COMMAND test_symmetric_matrix)
add_test(NAME UnaryFunctionsTest COMMAND test_unary_functions)
add_test(NAME ComplexOperationsTest COMMAND test_complex_operations)
add_test(NAME NumericalVerificationTest COMMAND test_numerical_verification)
