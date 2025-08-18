# 静的な 計算グラフの構築＋自動微分ライブラリの設計

## 目的

最近、アルゴリズムで何かを微分して最適化するのが流行っているので、自分でも実装したい。
高速なので CUDA で実装したいけど、メンテナンス性が非常にしんどい。
自分の趣味で作るものなので、自分の肌に合ったライブラリを開発したい。

## 設計

### Variable

- パラメータを数値的に保存するためのクラス
- 何か計算をする時に便利にしたい場合、View をかぶせて演算する
    - 例：対角行列 view, sparse matrix view, ...
- forward / backward する時の chain は、 variable に対して行う
    - jacobbian や vjp の次元の大きさは、 variable の大きさを見て計算される

### Matrix View

- operation を視覚的に分かりやすく計算させたい(matrix の乗算など)が、レジスタ使用量の面で出来るだけパラメータ表現を圧縮させたい
- Repository を参照して行列演算を行い、repository に結果を書き込む役割を担う

### Operation

- N 個の variable を受け取って、 1個の variable に結果を書き込むもの
- 外部から受け取りたいパラメータがある場合は、constructor で受け取る
- operator() で順方向の計算を実行
- vjp を使って逆伝播の微分を実行
- numerical_vjp のサポート
    - operator() と入力の微小変化を使って計算
    - ユニットテスト用を想定

### 計算グラフの構築

operator() の返り値として Param に加えて計算結果の構造も返す。

例えば、
```c++
const auto v1 = Variable<3>(&param.v1, &param.v1_diff);
const auto v2 = Variable<4>(&param.v2, &param.v2_diff);
const auto v3 = Variable<4>(&param.v3, &param.v3_diff);

const auto op1 = Multiple();
const auto op2 = Multiple();

const auto v4 = TempVarialbe<3>();
// v4 に結果が入る
const auto intvar1 = op1(v1, v2, v4);
const auto v5 = TempVarialbe<1>();
// v5 に結果が入る
const auto result = op2(intvar1, v3, v5);

// param.v1_diff, param.v2_diff, param.v3_diff に微分した毛結果が全て代入される
result.backward();
```

のような構造を実現したい（雰囲気なので、部分的には変なコードになっている）。
この時、

- intvar1 は peration1<v1, v2>
- result は Operation2<Operation1<v1, v2>, v3>

のような構造を持っているべき。

### backward の実現

backward の計算には、ノードを順番に辿る仕組みと、jacobbian の伝播の仕組みが必要。

今回は実装の簡略化のため **計算グラフが木構造であることを強く仮定することにする**
こうすることで、トポロジカルソートのような一般の DAG を扱う仕組みを作ることなく、簡易的に実装できる。
