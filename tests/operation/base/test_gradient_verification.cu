#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <xyz_autodiff/testing.cuh>
#include <xyz_autodiff/operations/unary/sigmoid_logic.cuh>
#include <xyz_autodiff/operations/unary/exp_logic.cuh>
#include <xyz_autodiff/operations/binary/add_logic.cuh>
#include <xyz_autodiff/operations/binary/mul_logic.cuh>
#include <xyz_autodiff/variable_operators.cuh>

using namespace xyz_autodiff;
using namespace xyz_autodiff::testing;

// SigmoidLogicのテスト (UnaryOperation)
TEST_UNARY_GRADIENT(op::SigmoidLogic<3>, 3, 3, SigmoidLogic3D)
TEST_UNARY_GRADIENT(op::SigmoidLogic<1>, 1, 1, SigmoidLogic1D)
TEST_UNARY_GRADIENT(op::SigmoidLogic<5>, 5, 5, SigmoidLogic5D)

// ExpLogicのテスト (UnaryOperation)
TEST_UNARY_GRADIENT(op::ExpLogic<3>, 3, 3, ExpLogic3D)
TEST_UNARY_GRADIENT(op::ExpLogic<1>, 1, 1, ExpLogic1D)
TEST_UNARY_GRADIENT(op::ExpLogic<5>, 5, 5, ExpLogic5D)

// AddLogicのテスト (BinaryOperation) - 一旦コメントアウト（型推論の問題を解決するため）
// TEST_BINARY_GRADIENT(xyz_autodiff::op::AddLogic<VariableRef<1, double>, VariableRef<1, double>>, 1, 1, 1, AddLogic1D)

// より大きな次元でのテスト
TEST_UNARY_GRADIENT(op::SigmoidLogic<10>, 10, 10, SigmoidLogic10D)
TEST_UNARY_GRADIENT(op::ExpLogic<10>, 10, 10, ExpLogic10D)

// 特殊ケース: 異なる入力・出力次元
TEST_UNARY_GRADIENT(op::SigmoidLogic<7>, 7, 7, SigmoidLogic7D)
TEST_UNARY_GRADIENT(op::ExpLogic<8>, 8, 8, ExpLogic8D)

// より複雑なBinaryOperationテスト（将来的にAddLogic以外を追加する際に使用）
// TEST_BINARY_GRADIENT(xyz_autodiff::op::AddLogic<VariableRef<3, double>, VariableRef<3, double>>, 3, 3, 1, AddLogic3D)

// スケーラビリティテスト
TEST_UNARY_GRADIENT(op::SigmoidLogic<20>, 20, 20, SigmoidLogicLarge)
TEST_UNARY_GRADIENT(op::ExpLogic<20>, 20, 20, ExpLogicLarge)

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
    xyz_autodiff::testing::UnaryGradientTester<op::SigmoidLogic<2>, 2, 2>::test("ManualSigmoidTest");
}

TEST_F(GradientVerificationTest, ManualExpTest) {
    xyz_autodiff::testing::UnaryGradientTester<op::ExpLogic<2>, 2, 2>::test("ManualExpTest");
}

// エッジケーステスト
TEST_F(GradientVerificationTest, EdgeCaseSingleDimension) {
    xyz_autodiff::testing::UnaryGradientTester<op::SigmoidLogic<1>, 1, 1>::test("EdgeCaseSingleDimension");
}

// BinaryOperationのテスト（element-wise operations）
// Type aliases to handle commas in template arguments
using MulLogic1D_t = xyz_autodiff::op::MulLogic<VariableRef<1, double>, VariableRef<1, double>>;
using AddLogic1D_t = xyz_autodiff::op::AddLogic<VariableRef<1, double>, VariableRef<1, double>>;
using MulLogic3D_t = xyz_autodiff::op::MulLogic<VariableRef<3, double>, VariableRef<3, double>>;
using AddLogic3D_t = xyz_autodiff::op::AddLogic<VariableRef<3, double>, VariableRef<3, double>>;
using MulLogic5D_t = xyz_autodiff::op::MulLogic<VariableRef<5, double>, VariableRef<5, double>>;
using AddLogic5D_t = xyz_autodiff::op::AddLogic<VariableRef<5, double>, VariableRef<5, double>>;

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