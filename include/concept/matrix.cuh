#pragma once

#include <concepts>
#include <type_traits>
#include <span>

namespace xyz_autodiff {
namespace concept {

// Forward propagation に必要な MatrixView の要件
template <typename T>
concept MatrixView = requires(T view) {
    // コンパイル時に決定される行列サイズ
    { T::rows } -> std::convertible_to<std::size_t>;
    { T::cols } -> std::convertible_to<std::size_t>;
    
    // 要素へのアクセス (値)
    { view(std::size_t{}, std::size_t{}) } -> std::convertible_to<typename T::value_type&>;
    { view(std::size_t{}, std::size_t{}) } -> std::convertible_to<const typename T::value_type&>;
    
    // データへの直接アクセス
    { view.data() } -> std::convertible_to<typename T::value_type*>;
    { view.data() } -> std::convertible_to<const typename T::value_type*>;
    
    // 疎行列サポート: 指定列の有効行を取得 (bool配列)
    { view.active_rows_in_col(std::size_t{}) } -> std::convertible_to<std::span<const bool>>;
    
    // 疎行列サポート: 指定行の有効列を取得 (bool配列)
    { view.active_cols_in_row(std::size_t{}) } -> std::convertible_to<std::span<const bool>>;
    
    // 型情報
    typename T::value_type;
} && std::is_copy_constructible_v<T>;

// Backward propagation に必要な MatrixView の要件
template <typename T>
concept DifferentiableMatrixView = MatrixView<T> && requires(T view) {
    // 勾配へのアクセス
    { view.grad() } -> std::convertible_to<typename T::value_type*>;
    { view.grad() } -> std::convertible_to<const typename T::value_type*>;
    
    // 2次元アクセス (勾配)
    { view.grad(std::size_t{}, std::size_t{}) } -> std::convertible_to<typename T::value_type&>;
    { view.grad(std::size_t{}, std::size_t{}) } -> std::convertible_to<const typename T::value_type&>;
};


} // namespace concept
} // namespace xyz_autodiff
