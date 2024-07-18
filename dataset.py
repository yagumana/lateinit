import numpy as np
import torch
import sys

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
