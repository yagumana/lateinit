import argparse
import os
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import seaborn as sns
import sys
import random
import model
import utils
import dataset

dataset_name = "circle"
Data_size = 50000
is_train = True
dim_d = 2
R=2 # 大半径
r=1 # 小半径
path_name = dataset_name

if dataset_name == "torus" or dataset_name == "ellipse":
    base_path = os.getcwd()
    path_name = dataset_name + str(R) + str(r)
    img_dir = os.path.join(base_path, "images", path_name)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Making the plot outputs portable and reproducible
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['text.antialiased'] = True
plt.rcParams['lines.antialiased'] = True
sns.set_theme()

# ランダム性に関する評価
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# datasetの定義
if dataset_name == "circle":
    data = dataset.circle_dataset(n=Data_size, r=dim_d).numpy()

elif dataset_name == "ellipse":
    data = dataset.ellipse_dataset(n=Data_size, a=R, b=r, dim=dim_d).numpy()

tensor_data = TensorDataset(torch.tensor(data))
dataloader = DataLoader(tensor_data, batch_size=32, shuffle=True, drop_last=True)

if is_train == True:
    for i in range(5):
        nn_model = model.Unet1D(dim=dim_d, channels=1).to(device)
        noise_scheduler = model.NoiseScheduler(num_timesteps=1000, beta_schedule="linear")
        optimizer = torch.optim.AdamW(nn_model.parameters(), lr=1e-3)
        print(f"training model_{i}...")
        model.train(nn_model, dataloader, noise_scheduler, optimizer, device=device, N_epoch=100, dim_d=dim_d, i=i)
        torch.save(nn_model.state_dict(), f'/workspace/weights/{path_name}_in_r{dim_d}_{i}.pth')
    print("Successfuly trained ddpm model!")
nn_model = model.MLP(hidden_size=128, hidden_layers=3, emb_size=128, time_emb="sinusoidal", input_emb="sinusoidal", out_dim=dim_d)
noise_scheduler = model.NoiseScheduler(num_timesteps=1000, beta_schedule="linear")
optimizer = torch.optim.AdamW(nn_model.parameters(), lr=1e-3)
checkpoint = torch.load(f'/workspace/weights/{path_name}_in_r{dim_d}_0.pth')
nn_model.load_state_dict(checkpoint)
dim=2
grid_size = 64
x = np.linspace(-3, 3, grid_size)
y = np.linspace(-3, 3, grid_size)
xx, yy = np.meshgrid(x, y)
samples = np.stack([xx, yy], axis=-1).reshape(-1, dim).astype(np.float32)  # (256, 2) になる
samples = torch.from_numpy(samples)
print(samples.shape)
nn_model.eval()
nn_model.to(device)
timesteps = np.arange(0, 1001, 1).tolist()[::-1]
timestep_cnt = len(timesteps)
print(f"timesteps_cnt: {timestep_cnt}")
data_list = np.zeros((len(timesteps)+1, grid_size*grid_size, dim))
residual_stack = []
for i, t in enumerate(tqdm(timesteps)):
    t = torch.from_numpy(np.repeat(t, grid_size*grid_size)).long()
    with torch.no_grad():
        samples.to(device)
        t.to(device)
        residual = nn_model(samples, t, device=device)
        residual_stack.append(residual.to('cpu'))
residual_stack = np.stack(residual_stack)
# residual_stackをファイルに保存
file_path = f'./score_data/residual_stack_{path_name}_{grid_size}.npy'
np.save(file_path, residual_stack)
# データを読み込む
residual_stack = np.load(file_path)
print(residual_stack.shape)
# データフォルダの作成
output_dir = f'/workspace/data/{path_name}_vfield/png'
os.makedirs(output_dir, exist_ok=True)
# diffusionではスコアベクトルを現在の値から引く, つまり、マイナス1を掛けた方向に移動する
data_set = residual_stack * -1
# グリッドを設定します
x = np.linspace(-3, 3, grid_size)
y = np.linspace(-3, 3, grid_size)
xx, yy = np.meshgrid(x, y)
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_aspect('equal')

# 最初のタイムステップでカラーバーの範囲を計算
initial_residuals = data_set[0]
initial_residual_x = initial_residuals[:, 0].reshape(grid_size, grid_size)
initial_residual_y = initial_residuals[:, 1].reshape(grid_size, grid_size)
initial_magnitude = np.sqrt(initial_residual_x**2 + initial_residual_y**2)
vmin = initial_magnitude.min()
vmax = initial_magnitude.max()
for i in range(len(data_set)):
    residuals = data_set[i]
    residual_x = residuals[:, 0].reshape(grid_size, grid_size)
    residual_y = residuals[:, 1].reshape(grid_size, grid_size)
    magnitude = np.sqrt(residual_x**2 + residual_y**2)
    # ベクトル場をプロット
    quiver = ax.quiver(xx, yy, residual_x, residual_y, magnitude, cmap='viridis')
    quiver.set_clim(vmin, vmax)
    colorbar = fig.colorbar(quiver, ax=ax)
    colorbar.set_ticks(np.linspace(vmin, vmax, num=5))
    # ax.set_title(f'Timestep {i}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True)
    if dataset_name == "circle":
        # 単位円を追加
        unit_circle = plt.Circle((0, 0), 1, color='r', fill=False)
        ax.add_artist(unit_circle)
    elif dataset_name == "ellipse":
        # 楕円を追加
        ellipse = patches.Ellipse((0, 0), width=R*2, height=r*2, color='r', fill=False)
        ax.add_artist(ellipse)
    # 画像を保存
    plt.savefig(os.path.join(output_dir, f'vector_field_2d_{i:04d}.png'), dpi=500)
    # カラーバーとクイバーを削除
    colorbar.remove()
    quiver.remove()
    # プロットをクリア
    ax.cla()
    ax.set_aspect('equal')
plt.close(fig)