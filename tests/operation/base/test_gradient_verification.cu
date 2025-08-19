#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "../../utility/gradient_test_utility.cuh"
#include "../../../include/operations/unary/sigmoid_logic.cuh"
#include "../../../include/operations/unary/exp_logic.cuh"
#include "../../../include/operations/binary/add_logic.cuh"
#include "../../../include/operations/binary/mul_logic.cuh"

using namespace xyz_autodiff;
using namespace xyz_autodiff::test;

// SigmoidLogicのテスト (UnaryOperation)
TEST_UNARY_GRADIENT(SigmoidLogic<3>, 3, 3, SigmoidLogic3D)
TEST_UNARY_GRADIENT(SigmoidLogic<1>, 1, 1, SigmoidLogic1D)
TEST_UNARY_GRADIENT(SigmoidLogic<5>, 5, 5, SigmoidLogic5D)

// ExpLogicのテスト (UnaryOperation)
TEST_UNARY_GRADIENT(ExpLogic<3>, 3, 3, ExpLogic3D)
TEST_UNARY_GRADIENT(ExpLogic<1>, 1, 1, ExpLogic1D)
TEST_UNARY_GRADIENT(ExpLogic<5>, 5, 5, ExpLogic5D)

// AddLogicのテスト (BinaryOperation) - 一旦コメントアウト（型推論の問題を解決するため）
// TEST_BINARY_GRADIENT(xyz_autodiff::op::AddLogic<VariableRef<double, 1>, VariableRef<double, 1>>, 1, 1, 1, AddLogic1D)

// より大きな次元でのテスト
TEST_UNARY_GRADIENT(SigmoidLogic<10>, 10, 10, SigmoidLogic10D)
TEST_UNARY_GRADIENT(ExpLogic<10>, 10, 10, ExpLogic10D)

// 特殊ケース: 異なる入力・出力次元
TEST_UNARY_GRADIENT(SigmoidLogic<7>, 7, 7, SigmoidLogic7D)
TEST_UNARY_GRADIENT(ExpLogic<8>, 8, 8, ExpLogic8D)

// より複雑なBinaryOperationテスト（将来的にAddLogic以外を追加する際に使用）
// TEST_BINARY_GRADIENT(xyz_autodiff::op::AddLogic<VariableRef<double, 3>, VariableRef<double, 3>>, 3, 3, 1, AddLogic3D)

// スケーラビリティテスト
TEST_UNARY_GRADIENT(SigmoidLogic<20>, 20, 20, SigmoidLogicLarge)
TEST_UNARY_GRADIENT(ExpLogic<20>, 20, 20, ExpLogicLarge)

class GradientVerificationTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaError_t err = cudaSetDevice(0);
        if (err != cudaSuccess) {
            GTEST_SKIP() << "CUDA device not available";
        }
    }
};

// 手動テスト例: より詳細な診断が必要な場合
TEST_F(GradientVerificationTest, ManualSigmoidTest) {
    // より詳細なテストロジックがここに書ける
    // 例: 特定の入力値での詳細なデバッグ
    xyz_autodiff::test::UnaryGradientTester<SigmoidLogic<2>, 2, 2>::test("ManualSigmoidTest");
}

TEST_F(GradientVerificationTest, ManualExpTest) {
    xyz_autodiff::test::UnaryGradientTester<ExpLogic<2>, 2, 2>::test("ManualExpTest");
}

// エッジケーステスト
TEST_F(GradientVerificationTest, EdgeCaseSingleDimension) {
    xyz_autodiff::test::UnaryGradientTester<SigmoidLogic<1>, 1, 1>::test("EdgeCaseSingleDimension");
}

// BinaryOperationのテスト（element-wise operations）
// Type aliases to handle commas in template arguments
using MulLogic1D_t = xyz_autodiff::op::MulLogic<VariableRef<double, 1>, VariableRef<double, 1>>;
using AddLogic1D_t = xyz_autodiff::op::AddLogic<VariableRef<double, 1>, VariableRef<double, 1>>;
using MulLogic3D_t = xyz_autodiff::op::MulLogic<VariableRef<double, 3>, VariableRef<double, 3>>;
using AddLogic3D_t = xyz_autodiff::op::AddLogic<VariableRef<double, 3>, VariableRef<double, 3>>;
using MulLogic5D_t = xyz_autodiff::op::MulLogic<VariableRef<double, 5>, VariableRef<double, 5>>;
using AddLogic5D_t = xyz_autodiff::op::AddLogic<VariableRef<double, 5>, VariableRef<double, 5>>;

// 1次元
TEST_BINARY_GRADIENT(MulLogic1D_t, 1, 1, 1, MulLogic1D)
TEST_BINARY_GRADIENT(AddLogic1D_t, 1, 1, 1, AddLogic1D)

// 3次元
TEST_BINARY_GRADIENT(MulLogic3D_t, 3, 3, 3, MulLogic3D)
TEST_BINARY_GRADIENT(AddLogic3D_t, 3, 3, 3, AddLogic3D)

// 5次元
TEST_BINARY_GRADIENT(MulLogic5D_t, 5, 5, 5, MulLogic5D)
TEST_BINARY_GRADIENT(AddLogic5D_t, 5, 5, 5, AddLogic5D)

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}