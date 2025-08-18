#!/bin/bash

# PTX生成スクリプト
# 異なる最適化レベルでPTXを生成して比較

echo "=== Generating PTX for optimization analysis ==="

# 出力ディレクトリ作成
mkdir -p ptx_output

# 共通のnvccフラグ
NVCC_FLAGS="-std=c++20 -I./include --ptx --gpu-architecture=sm_89"

echo "1. O0 (最適化なし) でPTX生成..."
nvcc $NVCC_FLAGS -O0 tests/kernel_optimization_test.cu -o ptx_output/kernel_O0.ptx

echo "2. O1 (基本最適化) でPTX生成..."
nvcc $NVCC_FLAGS -O1 tests/kernel_optimization_test.cu -o ptx_output/kernel_O1.ptx

echo "3. O2 (標準最適化) でPTX生成..."
nvcc $NVCC_FLAGS -O2 tests/kernel_optimization_test.cu -o ptx_output/kernel_O2.ptx

echo "4. O3 (最大最適化) でPTX生成..."
nvcc $NVCC_FLAGS -O3 tests/kernel_optimization_test.cu -o ptx_output/kernel_O3.ptx

echo "5. O3 + 追加最適化フラグでPTX生成..."
nvcc $NVCC_FLAGS -O3 --use_fast_math --ftz=true tests/kernel_optimization_test.cu -o ptx_output/kernel_O3_fast.ptx

echo ""
echo "=== PTXファイル生成完了 ==="
ls -la ptx_output/

echo ""
echo "=== サイズ比較 ==="
wc -l ptx_output/*.ptx

echo ""
echo "=== 特定カーネルの関数シグネチャ確認 ==="
echo "--- test_variable_basic ---"
grep -A 5 "_Z18test_variable_basic" ptx_output/kernel_O3.ptx || echo "関数が見つかりません"

echo ""
echo "--- test_operation_internal ---"
grep -A 5 "_Z23test_operation_internal" ptx_output/kernel_O3.ptx || echo "関数が見つかりません"

echo ""
echo "=== レジスタ使用量確認 ==="
echo "O0:"
grep ".reg " ptx_output/kernel_O0.ptx | head -5
echo "O3:"
grep ".reg " ptx_output/kernel_O3.ptx | head -5

echo ""
echo "=== インライン化確認（関数呼び出し数） ==="
echo "O0 call命令数:"
grep -c "call " ptx_output/kernel_O0.ptx || echo "0"
echo "O3 call命令数:"
grep -c "call " ptx_output/kernel_O3.ptx || echo "0"

echo ""
echo "PTX解析完了。詳細は ptx_output/ ディレクトリ内のファイルを確認してください。"