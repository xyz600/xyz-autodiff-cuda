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
    CUDA_ARCHITECTURES "50;60;70;75;80;86"
)

# DenseMatrix テスト
add_executable(test_dense_matrix test_dense_matrix.cu)
target_link_libraries(test_dense_matrix xyz_autodiff gtest gtest_main)
target_include_directories(test_dense_matrix PRIVATE ${CMAKE_SOURCE_DIR}/include)
set_target_properties(test_dense_matrix PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "50;60;70;75;80;86"
)

# DiagonalMatrix テスト
add_executable(test_diagonal_matrix test_diagonal_matrix.cu)
target_link_libraries(test_diagonal_matrix xyz_autodiff gtest gtest_main)
target_include_directories(test_diagonal_matrix PRIVATE ${CMAKE_SOURCE_DIR}/include)
set_target_properties(test_diagonal_matrix PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "50;60;70;75;80;86"
)

# CTest の有効化
enable_testing()
add_test(NAME VariableTest COMMAND test_variable)
add_test(NAME DenseMatrixTest COMMAND test_dense_matrix)
add_test(NAME DiagonalMatrixTest COMMAND test_diagonal_matrix)