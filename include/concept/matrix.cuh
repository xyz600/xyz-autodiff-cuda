#pragma once

#include <concepts>
#include <type_traits>
#include <span>

namespace xyz_autodiff {

// Forward propagation に必要な MatrixView の要件
template <typename T>
concept MatrixViewConcept = requires(T view) {
    // コンパイル時に決定される行列サイズ
    { T::rows } -> std::convertible_to<std::size_t>;
    { T::cols } -> std::convertible_to<std::size_t>;
    
    { view(std::size_t{}, std::size_t{}) } -> std::convertible_to<typename T::value_type>;
    { std::as_const(view)(std::size_t{}, std::size_t{}) } -> std::convertible_to<typename T::value_type>;

    // データへの直接アクセス
    { view.data() } -> std::convertible_to<typename T::value_type*>;
    { view.data() } -> std::convertible_to<const typename T::value_type*>;
    
    // 疎行列サポート: 指定のセルの有効業を取得
    { view.is_active_cell(std::size_t{}, std::size_t{}) } -> std::convertible_to<bool>;
    
    // 型情報
    typename T::value_type;
} && std::is_copy_constructible_v<T>;

} // namespace xyz_autodiff
