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
add_executable(test_unary_operations test_unary_operations.cu)
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
add_executable(test_gradient_verification test_gradient_verification.cu)
target_link_libraries(test_gradient_verification xyz_autodiff gtest gtest_main)
target_include_directories(test_gradient_verification PRIVATE ${CMAKE_SOURCE_DIR}/include)
set_target_properties(test_gradient_verification PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "89"
)

# OperationChaining テスト
add_executable(test_operation_chaining test_operation_chaining.cu)
target_link_libraries(test_operation_chaining xyz_autodiff gtest gtest_main)
target_include_directories(test_operation_chaining PRIVATE ${CMAKE_SOURCE_DIR}/include)
set_target_properties(test_operation_chaining PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "89"
)

# OperationConcepts テスト
add_executable(test_operation_concepts test_operation_concepts.cu)
target_link_libraries(test_operation_concepts xyz_autodiff gtest gtest_main)
target_include_directories(test_operation_concepts PRIVATE ${CMAKE_SOURCE_DIR}/include)
set_target_properties(test_operation_concepts PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "89"
)

# QuaternionToRotationMatrix テスト
add_executable(test_quaternion_to_rotation_matrix test_quaternion_to_rotation_matrix.cu)
target_link_libraries(test_quaternion_to_rotation_matrix xyz_autodiff gtest gtest_main)
target_include_directories(test_quaternion_to_rotation_matrix PRIVATE ${CMAKE_SOURCE_DIR}/include)
set_target_properties(test_quaternion_to_rotation_matrix PROPERTIES
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
