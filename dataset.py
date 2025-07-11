import numpy as np
import torch
import sys
import os

def s_0_dataset(n=8000, r=4, noise_level=0.01):
    """
    r次元のユークリッド空間に埋め込まれたS^0
    軽いノイズを追加して生成
    S^0 = {-1, 1}
    """
    embedding = np.zeros((n, r))
    embedding[:, 0] = np.random.choice([-1, 1], n)

    # 軽いノイズを追加
    noise = np.random.normal(0, noise_level, (n, r))
    embedding += noise

    return torch.from_numpy(embedding.astype(np.float32))
    

def circle_dataset(n=8000, r=4, noise_level=0.01):
    """
    r次元のユークリッド空間に埋め込まれた円周
    軽いノイズを追加して生成
    """
    rng = np.random.default_rng(42)
    x = rng.uniform(-0.5, 0.5, n)
    y = rng.uniform(-0.5, 0.5, n)

    norm = np.sqrt(x**2 + y**2) + 1e-10
    x /= norm
    y /= norm

    # 任意の次元数 r に埋め込むために、残りの次元にゼロを追加
    embedding = np.zeros((n, r))
    embedding[:, 0] = x
    embedding[:, 1] = y

    # 軽いノイズを追加
    noise = rng.normal(0, noise_level, (n, r))
    embedding += noise

    return torch.from_numpy(embedding.astype(np.float32))

def circle_half_dataset(n=8000, r=4, noise_level=0.01):
    """
    r次元のユークリッド空間に埋め込まれた円周
    軽いノイズを追加して生成
    """
    embedding = np.zeros((n, r))

    rng = np.random.default_rng(42)
    p = rng.uniform(0, np.pi/2, n//2)
    x = np.cos(p)
    y = np.sin(p)
    embedding[:n//2, 0] = x
    embedding[:n//2, 1] = y

    p = rng.uniform(np.pi, 3*np.pi/2, n//2)
    x = np.cos(p)
    y = np.sin(p)
    embedding[n//2:, 0] = x
    embedding[n//2:, 1] = y

    # 軽いノイズを追加
    # noise = rng.normal(0, noise_level, (n, r))
    # embedding += noise

    return torch.from_numpy(embedding.astype(np.float32))

def circle_half_dataset2(n=8000, dim=4, R=2, r=1, noise_level=0.01):
    """
    単射半径の異なる2つの円周の一部を組み合わせた多様体
    Rは大きい方の半径、rは小さい方の半径
    """
    embedding = np.zeros((n, dim))
    rng = np.random.default_rng(42)
    p = rng.uniform(np.pi/6, np.pi/3, n//2)
    x = r*np.cos(p)
    y = r*np.sin(p)
    embedding[:n//2, 0] = x
    embedding[:n//2, 1] = y

    p = rng.uniform(7*np.pi/6, 4*np.pi/3, n//2)
    x = R*np.cos(p)
    y = R*np.sin(p)
    embedding[n//2:, 0] = x
    embedding[n//2:, 1] = y

    return torch.from_numpy(embedding.astype(np.float32))

def circle_mixed3_dataset(n=9000, dim=4, RR=1.5, R=1, r=0.5, noise_level=0.01):
    """
    単射半径の異なる3つの円周の一部を組み合わせた多様体
    r1 > r2 > r3
    """
    embedding = np.zeros((n, dim))
    rng = np.random.default_rng(42)
    p = rng.uniform(0, np.pi/6, n//3)
    x = r*np.cos(p)
    y = r*np.sin(p)
    embedding[:n//3, 0] = x
    embedding[:n//3, 1] = y

    p = rng.uniform(np.pi*2/3, 5*np.pi/6, 2*n//3-n//3)
    x = R*np.cos(p)
    y = R*np.sin(p)
    embedding[n//3:2*n//3, 0] = x
    embedding[n//3:2*n//3, 1] = y

    p = rng.uniform(4*np.pi/3, 3*np.pi/2, n-2*n//3)
    x = RR*np.cos(p)
    y = RR*np.sin(p)
    embedding[2*n//3:, 0] = x
    embedding[2*n//3:, 1] = y

    return torch.from_numpy(embedding.astype(np.float32))



def sphere_dataset(n=8000, r=4, noise_level=0.01):
    """
    r次元のユークリッド空間に埋め込まれたS^2
    軽いノイズを追加して生成
    """
    rng = np.random.default_rng(42)
    x = rng.uniform(-0.5, 0.5, n)
    y = rng.uniform(-0.5, 0.5, n)
    z = rng.uniform(-0.5, 0.5, n)

    norm = np.sqrt(x**2 + y**2 + z**2) + 1e-10
    x /= norm
    y /= norm
    z /= norm

    embedding = np.zeros((n, r))
    embedding[:, 0] = x
    embedding[:, 1] = y
    embedding[:, 2] = z

    # 軽いノイズを追加
    noise = rng.normal(0, noise_level, (n, r))
    embedding += noise

    return torch.from_numpy(embedding.astype(np.float32))

def sphere_dataset_notuniform(n=8000, r=4, noise_level=0.01):
    """
    r次元のユークリッド空間に埋め込まれたS^2
    軽いノイズを追加して生成
    """
    rng = np.random.default_rng(42)
    x = rng.uniform(-0.5, 0.5, n)
    y = rng.uniform(-0.5, 0.5, n)
    z = rng.uniform(-0.5, 0, n)

    norm = np.sqrt(x**2 + y**2 + z**2) + 1e-10
    x /= norm
    y /= norm
    z /= norm

    embedding = np.zeros((n, r))
    embedding[:, 0] = x
    embedding[:, 1] = y
    embedding[:, 2] = z

    # 軽いノイズを追加
    noise = rng.normal(0, noise_level, (n, r))
    embedding += noise

    return torch.from_numpy(embedding.astype(np.float32))

def hyper_sphere_dataset(n=8000, r=48, s=20, noise_level=0.01):
    """
    r: ambiet space, s: sphere dimension
    21次元のランダムなベクトルをn個生成して、半径1の超球面にマッピング
    """
    # n個のランダムなベクトルを生成
    random_vec = np.random.randn(n, s+1)

    # 各ベクトルを正規化
    norm = np.linalg.norm(random_vec, axis=1, keepdims=True)
    unit_vec = random_vec / norm

    # r次元のユークリッド空間に埋め込む
    embedding = np.zeros((n, r))
    embedding[:, :s+1] = unit_vec

    return torch.from_numpy(embedding.astype(np.float32))



# トーラスのdatasetを作成して表示
def torus_dataset(n=8000, R=2, r=1, dim=7, noise_level=0.01):
    rng = np.random.default_rng(42)
    # Generate random points for angles p and t
    p = rng.uniform(0, 2*np.pi, n)
    t = rng.uniform(0, 2*np.pi, n)
    # Torus parametric equations
    x = (R + r * np.cos(p)) * np.cos(t)
    y = (R + r * np.cos(p)) * np.sin(t)
    z = r * np.sin(p)

    embedding = np.zeros((n, dim))
    embedding[:, 0] = x
    embedding[:, 1] = y
    embedding[:, 2] = z

    # 軽いノイズを追加
    noise = rng.normal(0, noise_level, (n, dim))
    embedding += noise

    return torch.from_numpy(embedding.astype(np.float32))

def ellipse_dataset(n=8000, a=1, b=1, dim=7, noise_level=0.01):
    rng = np.random.default_rng(42)
    t = rng.uniform(0, 2*np.pi, n)
    
    x = a * np.cos(t)
    y = b * np.sin(t)

    embedding = np.zeros((n, dim))
    embedding[:, 0] = x
    embedding[:, 1] = y

    noise = rng.normal(0, noise_level, (n, dim))
    embedding += noise

    return torch.from_numpy(embedding.astype(np.float32))

def embed_sphere_dataset(n=8000, dim_d=51, dataset_path=None):
    """
    hyperspherical vaeを用いて、潜在空間に埋め込んだdatasetをもとに、datasetを生成
    """
    print("loading data from hyperspherical vae...")
    # 保存されたテンソルをロード
    if not os.path.exists(dataset_path):
        sys.exit(f"dataset not found: {dataset_path}")
    z_mean_all = torch.load(dataset_path)

    # 指定された数のデータをランダムにサンプリング
    idx = np.random.choice(z_mean_all.shape[0], n, replace=False)
    z_sampled = z_mean_all[idx]
    print(z_sampled.shape)
    
    # 次元の拡張
    if dim_d > z_sampled.shape[1]:
        z_extended = torch.zeros((n, dim_d))
        z_extended[:, :z_sampled.shape[1]] = z_sampled
    else:
        z_extended = z_sampled[:, :dim_d]

    print(z_extended.shape)

    return z_extended
