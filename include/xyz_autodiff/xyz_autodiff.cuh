#pragma once

// Core types and concepts
#include <xyz_autodiff/concept/variable.cuh>
#include <xyz_autodiff/variable.cuh>
#include <xyz_autodiff/const_array.cuh>
#include <xyz_autodiff/dense_matrix.cuh>
#include <xyz_autodiff/diagonal_matrix_view.cuh>
#include <xyz_autodiff/symmetric_matrix_view.cuh>

// Utilities
#include <xyz_autodiff/util/cuda_unique_ptr.cuh>
#include <xyz_autodiff/util/cuda_managed_ptr.cuh>
#include <xyz_autodiff/util/error_checker.cuh>

// Math operations
#include <xyz_autodiff/operations/math.cuh>

// Binary operations
#include <xyz_autodiff/operations/binary/add_logic.cuh>
#include <xyz_autodiff/operations/binary/sub_logic.cuh>
#include <xyz_autodiff/operations/binary/mul_logic.cuh>
#include <xyz_autodiff/operations/binary/div_logic.cuh>
#include <xyz_autodiff/operations/binary/matmul_logic.cuh>

// Unary operations
#include <xyz_autodiff/operations/unary/add_constant_logic.cuh>
#include <xyz_autodiff/operations/unary/sub_constant_logic.cuh>
#include <xyz_autodiff/operations/unary/mul_constant_logic.cuh>
#include <xyz_autodiff/operations/unary/div_constant_logic.cuh>
#include <xyz_autodiff/operations/unary/exp_logic.cuh>
#include <xyz_autodiff/operations/unary/sin_logic.cuh>
#include <xyz_autodiff/operations/unary/cos_logic.cuh>
#include <xyz_autodiff/operations/unary/sigmoid_logic.cuh>
#include <xyz_autodiff/operations/unary/squared_logic.cuh>
#include <xyz_autodiff/operations/unary/neg_logic.cuh>
#include <xyz_autodiff/operations/unary/l1_norm_logic.cuh>
#include <xyz_autodiff/operations/unary/l2_norm_logic.cuh>
#include <xyz_autodiff/operations/unary/sum_logic.cuh>
#include <xyz_autodiff/operations/unary/broadcast.cuh>
#include <xyz_autodiff/operations/unary/to_rotation_matrix_logic.cuh>
#include <xyz_autodiff/operations/unary/sym_matrix2_inv_logic.cuh>

// Const array operations
#include <xyz_autodiff/operations/unary/const_array_add_logic.cuh>
#include <xyz_autodiff/operations/unary/const_array_sub_logic.cuh>
#include <xyz_autodiff/operations/unary/const_array_concepts.cuh>

// Operator overloads (must be last to see all operation definitions)
#include <xyz_autodiff/variable_operators.cuh>