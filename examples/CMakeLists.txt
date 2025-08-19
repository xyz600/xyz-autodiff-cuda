# Linear Regression SGD Optimization Example
add_executable(linear_regression_sgd optimization/linear_regression_sgd.cu)
target_link_libraries(linear_regression_sgd xyz_autodiff)
target_include_directories(linear_regression_sgd PRIVATE ${CMAKE_SOURCE_DIR}/include)
set_target_properties(linear_regression_sgd PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "89"
)

# Add Mini Gaussian Splatting subdirectory
add_subdirectory(mini-gaussian-splatting)