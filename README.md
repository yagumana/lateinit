# Diffusion Geometry

拡散モデル(DDPM)における拡散過程を、幾何学的な情報をもとに解析することを目的としたプロジェクトである。  
データ集合としては、高次元ユークリッド空間に埋め込まれた、円や球、楕円、トーラスを考える.  
この時、各時刻ごとの管状近傍の割合を求めることができる. [第１ステップ]  
late initializationを行い、各時刻ごとの管状近傍の割合と、wasserstein距離による再構成誤差の2つを比較する. [第2ステップ]

## Requirements

### Dependencies


Our project uses Docker for environment setup. To install the necessary dependencies and start the environment, please follow the instructions below.

1. **Build the Docker Image:**
   ```bash
   docker build -t project_name:latest .
   ```

2. **Run the Docker container:**
    ```bash
    docker container run -v $(pwd):/workspace --gpus "device=0" -it --name <your_container_name> <your_image_name>
    ```

If you prefer using pip for local development, you can still install the requirements manually:
    
    pip install -r requirements.txt

### Training
To train the diffusion model, run this command:
```
python sn_in_rn_unet.py --config ./configs/s1.yaml
```

### Analyzing 
To analyze relationship between the proportion of the tubular neighbourhood and wasserstein distance, run this command:
```
python lateinit_sphere.py --config ./configs/s1_lateinit.yaml
```


<!-- 

```
# ddpmを訓練し、管状近傍の割合を求める
python sn_in_rn.py --config /path/to/config
```
```
# 訓練済みのddpmを用いて、late initializationによる評価を行う
python lateinit_sphere.py --config /path/to/config
```

- denoising-diffusion-pytorchフォルダに入って、pip でインストール
$ pip install denoising_diffusion_pytorch -->
