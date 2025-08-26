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
# Add optimization flags
target_compile_options(gaussian_splatting_training PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math>
    $<$<COMPILE_LANGUAGE:CUDA>:-maxrregcount=32>
    $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    $<$<COMPILE_LANGUAGE:CUDA>:-ftz=true>
    $<$<COMPILE_LANGUAGE:CUDA>:-prec-div=false>
    $<$<COMPILE_LANGUAGE:CUDA>:-prec-sqrt=false>
)

# Simple Gaussian Test
add_executable(test_simple_gaussian
    test_simple_gaussian.cu
    gaussian_parameters.cpp
    image_utils.cpp
    simple_gaussian_kernel.cu
)
target_link_libraries(test_simple_gaussian xyz_autodiff)
target_include_directories(test_simple_gaussian PRIVATE 
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/external
)
set_target_properties(test_simple_gaussian PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "89"
)
