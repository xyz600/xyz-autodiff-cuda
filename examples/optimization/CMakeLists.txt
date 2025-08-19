# Optimization Examples CMakeLists.txt

# Linear Regression SGD Example
add_executable(linear_regression_sgd linear_regression_sgd.cu)
target_link_libraries(linear_regression_sgd xyz_autodiff)
target_include_directories(linear_regression_sgd PRIVATE 
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)
set_target_properties(linear_regression_sgd PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "89"
)

# Tests
add_executable(test_linear_regression_network_gradient tests/test_linear_regression_network_gradient.cu)
target_link_libraries(test_linear_regression_network_gradient xyz_autodiff gtest gtest_main)
target_include_directories(test_linear_regression_network_gradient PRIVATE 
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)
set_target_properties(test_linear_regression_network_gradient PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "89"
)

# Linear Regression Gradient Test (using actual SGD kernel structure)
add_executable(test_linear_regression_gradient tests/test_linear_regression_gradient.cu)
target_link_libraries(test_linear_regression_gradient xyz_autodiff gtest gtest_main)
target_include_directories(test_linear_regression_gradient PRIVATE 
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)
set_target_properties(test_linear_regression_gradient PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "89"
)