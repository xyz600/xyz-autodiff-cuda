#pragma once

#include <cstddef>

namespace xyz_autodiff {
namespace op {

// Concept to check if type supports operator[]
template<typename T>
concept ArrayLikeConcept = requires(T t, std::size_t i) {
    { t[i] } -> std::convertible_to<typename std::remove_reference_t<T>::value_type>;
};

} // namespace op
} // namespace xyz_autodiff