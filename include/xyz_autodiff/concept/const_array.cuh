#pragma once

#include <cstddef>
#include <type_traits>

namespace xyz_autodiff {

// Concept to check if a type behaves like a constant array
template<typename T>
concept ConstArrayLike = requires(T t, std::size_t i) {
    // Must have operator[] that returns a convertible type
    { t[i] } -> std::convertible_to<typename T::value_type>;
    // Must have a static size member (matching Variable/VariableRef interface)
    { T::size } -> std::convertible_to<std::size_t>;
    // Must have a value_type typedef
    typename T::value_type;
};

// Concept to check if two ConstArrayLike types are compatible for operations
template<typename T1, typename T2>
concept ConstArrayCompatible = ConstArrayLike<T1> && ConstArrayLike<T2> &&
    std::same_as<typename T1::value_type, typename T2::value_type>;

// Concept to check if two ConstArrayLike types have the same size (at compile time)
template<typename T1, typename T2>
concept ConstArraySameSize = ConstArrayLike<T1> && ConstArrayLike<T2> &&
    (T1::size == T2::size);

} // namespace xyz_autodiff