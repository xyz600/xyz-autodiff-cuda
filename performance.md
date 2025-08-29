# 概要

nsight-compute で計算を行った時のパフォーマンス上の問題点をメモする。

# プロファイル結果

## 1. 不用意に global memory にアクセスしている

例えば 

```c++
PixelOutput& pixel_out = output_image[pixel_idx]
pixel_out.color[0] = 0.0f;
```

等は、普通に global store が発生しちゃうらしい。ここはローカル変数に保存しておくとよい。

load する側も store する側も一旦レジスタに置いといて、flush するのがよさそう。

## 2. global memory からのロード

ループの中で global memory からロードするコードがそれなりにある。
K 個分の gaussian を shared memory に保存して、そこからロードする方針はよさそう

## 3. global memory -> shared memory の latency hiding

多分コア余ってるので、256 スレッドは保持するとして、もう 64 スレッド位立ちあげて、 320 スレッド立ち上げる。
64スレッド分は、 global memory -> shared memory へロードする処理をやる。
gaussian の 「32個分の 実体 + grad」 x 2を持っておいて、処理していない方の gaussian を裏で更新するようにする。

