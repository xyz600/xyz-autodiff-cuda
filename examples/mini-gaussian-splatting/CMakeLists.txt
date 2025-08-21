# Mini Gaussian Splatting Example CMakeLists.txt

# Gaussian Splatting Training Application
add_executable(gaussian_splatting_training 
    gaussian_splatting_training.cu
    gaussian_parameters.cpp
    image_utils.cpp
    gaussian_splatting_kernel.cu
)
target_link_libraries(gaussian_splatting_training xyz_autodiff)
target_include_directories(gaussian_splatting_training PRIVATE 
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/external
)
set_target_properties(gaussian_splatting_training PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "89"
)