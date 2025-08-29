#pragma once

#include <xyz_autodiff/variable.cuh>
#include <xyz_autodiff/operations/binary/add_logic.cuh>
#include <xyz_autodiff/operations/binary/sub_logic.cuh>
#include <xyz_autodiff/operations/binary/mul_logic.cuh>
#include <xyz_autodiff/operations/binary/div_logic.cuh>
#include <xyz_autodiff/operations/unary/add_constant_logic.cuh>
#include <xyz_autodiff/operations/unary/sub_constant_logic.cuh>
#include <xyz_autodiff/operations/unary/mul_constant_logic.cuh>
#include <xyz_autodiff/operations/unary/div_constant_logic.cuh>
#include <xyz_autodiff/operations/unary/const_array_add_logic.cuh>
#include <xyz_autodiff/operations/unary/const_array_sub_logic.cuh>
#include <xyz_autodiff/operations/unary/const_array_concepts.cuh>

namespace xyz_autodiff {

// ================================================================
// Universal operator overloads using concepts - much cleaner!
// ================================================================

// Constant operators: Variable-like + constant
template <DifferentiableVariableConcept Var>
__device__ auto operator+(Var& var, const typename Var::value_type& constant) {
    return op::add_constant(var, constant);
}

template <DifferentiableVariableConcept Var>
__device__ auto operator-(Var& var, const typename Var::value_type& constant) {
    return op::sub_constant(var, constant);
}

template <DifferentiableVariableConcept Var>
__device__ auto operator*(Var& var, const typename Var::value_type& constant) {
    return op::mul_constant(var, constant);
}

template <DifferentiableVariableConcept Var>
__device__ auto operator/(Var& var, const typename Var::value_type& constant) {
    return op::div_constant(var, constant);
}

// Variable-to-Variable operators: Variable-like + Variable-like (same dimensions and type)
template <DifferentiableVariableConcept Var1, DifferentiableVariableConcept Var2>
requires (Var1::size == Var2::size) && std::same_as<typename Var1::value_type, typename Var2::value_type>
__device__ auto operator+(Var1& var1, Var2& var2) {
    return op::add(var1, var2);
}

template <DifferentiableVariableConcept Var1, DifferentiableVariableConcept Var2>
requires (Var1::size == Var2::size) && std::same_as<typename Var1::value_type, typename Var2::value_type>
__device__ auto operator-(Var1& var1, Var2& var2) {
    return op::sub(var1, var2);
}

template <DifferentiableVariableConcept Var1, DifferentiableVariableConcept Var2>
requires (Var1::size == Var2::size) && std::same_as<typename Var1::value_type, typename Var2::value_type>
__device__ auto operator*(Var1& var1, Var2& var2) {
    return op::mul(var1, var2);
}

template <DifferentiableVariableConcept Var1, DifferentiableVariableConcept Var2>
requires (Var1::size == Var2::size) && std::same_as<typename Var1::value_type, typename Var2::value_type>
__device__ auto operator/(Var1& var1, Var2& var2) {
    return op::div(var1, var2);
}

// Const array operators: Variable + const array
template <DifferentiableVariableConcept Var, typename ConstArray>
requires op::ArrayLikeConcept<ConstArray>
__device__ auto operator+(Var& var, const ConstArray& const_array) {
    return op::const_add(var, const_array);
}

template <DifferentiableVariableConcept Var, typename ConstArray>
requires op::ArrayLikeConcept<ConstArray>
__device__ auto operator-(Var& var, const ConstArray& const_array) {
    return op::const_sub(var, const_array);
}

} // namespace xyz_autodiff