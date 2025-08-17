#!/bin/bash

echo "=== 詳細なカーネル最適化分析 ==="

echo ""
echo "1. 定数最適化の確認（即値の埋め込み）"
echo "--- 3.0f (0x40400000) の使用 ---"
grep -n "1077936128\|0f40400000" ptx_output/kernel_O3.ptx

echo ""
echo "--- 4.0f (0x40800000) の使用 ---"  
grep -n "1082130432\|0f40800000" ptx_output/kernel_O3.ptx

echo ""
echo "--- 1.0f (0x3F800000) の使用 ---"
grep -n "0f3F800000" ptx_output/kernel_O3.ptx

echo ""
echo "2. メモリアクセスパターンの分析"
echo "--- グローバルメモリストア ---"
grep -n "st\.global" ptx_output/kernel_O3.ptx

echo ""
echo "--- グローバルメモリロード ---"
grep -n "ld\.global" ptx_output/kernel_O3.ptx

echo ""
echo "--- アトミック操作 ---"
grep -n "atom\." ptx_output/kernel_O3.ptx

echo ""
echo "3. 浮動小数点演算の分析"
echo "--- 加算命令 ---"
grep -n "add\.f32" ptx_output/kernel_O3.ptx

echo ""
echo "--- その他の浮動小数点演算 ---"
grep -n "\.f32" ptx_output/kernel_O3.ptx | grep -v "ld\|st\|add"

echo ""
echo "4. 制御フロー分析"
echo "--- 分岐命令 ---"
grep -n "bra\|branch" ptx_output/kernel_O3.ptx || echo "分岐命令なし（完全な線形実行）"

echo ""
echo "--- 関数呼び出し ---"
grep -n "call" ptx_output/kernel_O3.ptx || echo "関数呼び出しなし（完全インライン化）"

echo ""
echo "5. レジスタ使用量詳細"
echo "--- 各カーネルのレジスタ宣言 ---"
grep -A 5 "\.reg\." ptx_output/kernel_O3.ptx

echo ""
echo "6. Operation Chaining カーネルの分析"
echo "--- test_operation_chaining の命令数 ---"
sed -n '/_Z24test_operation_chaining/,/^}/p' ptx_output/kernel_O3.ptx | wc -l

echo ""
echo "=== 最適化サマリー ==="
echo "総行数: $(wc -l < ptx_output/kernel_O3.ptx)"
echo "カーネル数: $(grep -c "\.visible \.entry" ptx_output/kernel_O3.ptx)"
echo "浮動小数点演算数: $(grep -c "\.f32" ptx_output/kernel_O3.ptx)"
echo "メモリアクセス数: $(grep -c "ld\.\|st\." ptx_output/kernel_O3.ptx)"