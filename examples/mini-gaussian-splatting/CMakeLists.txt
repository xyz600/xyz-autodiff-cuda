# Mini Gaussian Splatting Example CMakeLists.txt

# Mini Gaussian Splatting Example
add_executable(mini_gaussian_splatting_example mini_gaussian_splatting_example.cu)
target_link_libraries(mini_gaussian_splatting_example xyz_autodiff)
target_include_directories(mini_gaussian_splatting_example PRIVATE 
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)
set_target_properties(mini_gaussian_splatting_example PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "89"
)

