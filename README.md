# Diffusion Geometry

拡散モデル(DDPM)における拡散過程を、幾何学的な情報をもとに解析することを目的としたプロジェクトである。  
データ集合としては、高次元ユークリッド空間に埋め込まれた、円や球、楕円、トーラスを考える.  
この時、各時刻ごとの管状近傍の割合を求めることができる. [第１ステップ]  
late initializationを行い、各時刻ごとの管状近傍の割合と、wasserstein距離による再構成誤差の2つを比較する. [第2ステップ]

```
# ddpmを訓練し、管状近傍の割合を求める
python s2_in_rn.py --config /path/to/config
```
```
# 訓練済みのddpmを用いて、late initializationによる評価を行う
python lateinit_sphere.py --config /path/to/config
```