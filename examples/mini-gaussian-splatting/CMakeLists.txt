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

# Single Pixel Test
add_executable(test_single_pixel 
    test_single_pixel.cu
    gaussian_parameters.cpp
)
target_link_libraries(test_single_pixel xyz_autodiff)
target_include_directories(test_single_pixel PRIVATE 
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/external
)
set_target_properties(test_single_pixel PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "89"
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

# Fixed Kernel Test (commented out - file not ready)
# add_executable(test_fixed_kernel
#     test_fixed_kernel.cu
#     gaussian_parameters.cpp
#     image_utils.cpp
#     gaussian_splatting_kernel.cu
#     gaussian_splatting_kernel_fixed.cu
# )
# target_link_libraries(test_fixed_kernel xyz_autodiff)
# target_include_directories(test_fixed_kernel PRIVATE 
#     ${CMAKE_SOURCE_DIR}/include
#     ${CMAKE_CURRENT_SOURCE_DIR}
#     ${CMAKE_CURRENT_SOURCE_DIR}/external
# )
# set_target_properties(test_fixed_kernel PROPERTIES
#     CUDA_SEPARABLE_COMPILATION ON
#     CUDA_ARCHITECTURES "89"
# )

# Hybrid Training (CPU gradients + GPU forward)
# add_executable(hybrid_gaussian_training
#     hybrid_gaussian_training.cu
#     gaussian_parameters.cpp
#     image_utils.cpp
# )
# target_link_libraries(hybrid_gaussian_training xyz_autodiff)
# target_include_directories(hybrid_gaussian_training PRIVATE 
#     ${CMAKE_SOURCE_DIR}/include
#     ${CMAKE_CURRENT_SOURCE_DIR}
#     ${CMAKE_CURRENT_SOURCE_DIR}/external
# )
# set_target_properties(hybrid_gaussian_training PROPERTIES
#     CUDA_SEPARABLE_COMPILATION ON
#     CUDA_ARCHITECTURES "89"
# )