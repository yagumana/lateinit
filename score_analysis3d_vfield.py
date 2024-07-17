import argparse
import os

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Making the plot outputs portable and reproducible
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['text.antialiased'] = True
plt.rcParams['lines.antialiased'] = True
sns.set_theme()

# import datasets
# from positional_embeddings import PositionalEmbedding


class Block(nn.Module):
    def __init__(self, size: int):
        super().__init__()

        self.ff = nn.Linear(size, size)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor):
        return x + self.act(self.ff(x))


class MLP(nn.Module):
    def __init__(self, hidden_size: int = 128, hidden_layers: int = 3, emb_size: int = 128, out_dim: int = 2,
                 time_emb: str = "sinusoidal", input_emb: str = "sinusoidal"):
        super().__init__()

        self.hidden_size = hidden_size
        self.time_mlp = PositionalEmbedding(emb_size, time_emb)
        self.input_mlp1 = PositionalEmbedding(emb_size, input_emb, scale=25.0)
        self.input_mlp2 = PositionalEmbedding(emb_size, input_emb, scale=25.0)
        self.input_mlp3 = PositionalEmbedding(emb_size, input_emb, scale=25.0)
        self.out_dim = out_dim

        if self.out_dim==1:
            concat_size = len(self.time_mlp.layer) + len(self.input_mlp1.layer)
        elif self.out_dim==2:
            concat_size = len(self.time_mlp.layer) + \
            len(self.input_mlp1.layer) + len(self.input_mlp2.layer)
        elif self.out_dim==3:
            concat_size = len(self.time_mlp.layer) + \
            len(self.input_mlp1.layer) + len(self.input_mlp2.layer) + len(self.input_mlp3.layer)

        else:
            raise NotImplementedError("The value of 'out_dim' is not implemented.")

        layers = [nn.Linear(concat_size, hidden_size), nn.GELU()]
        for _ in range(hidden_layers):
            layers.append(Block(hidden_size))
        layers.append(nn.Linear(hidden_size, out_dim))
        self.joint_mlp = nn.Sequential(*layers)

    def forward(self, x, t):
        # print(f"x:shape: {x.shape}")
        batch_size, dim = x.shape
        if dim==1:
            x1_emb = self.input_mlp1(x[:, 0])
            t_emb = self.time_mlp(t)
            x = torch.cat((x1_emb, t_emb), dim=-1)

        elif dim==2:
            x1_emb = self.input_mlp1(x[:, 0])
            x2_emb = self.input_mlp2(x[:, 1])
            t_emb = self.time_mlp(t)
            x = torch.cat((x1_emb, x2_emb, t_emb), dim=-1)

        elif dim==3:
            x1_emb = self.input_mlp1(x[:, 0])
            x2_emb = self.input_mlp2(x[:, 1])
            x3_emb = self.input_mlp3(x[:, 2])
            t_emb = self.time_mlp(t)
            x = torch.cat((x1_emb, x2_emb, x3_emb, t_emb), dim=-1)

        x = self.joint_mlp(x)
        return x


class NoiseScheduler():
    def __init__(self,
                 num_timesteps=1000,
                 beta_start=0.0001,
                 beta_end=0.02,
                 beta_schedule="linear"):

        self.num_timesteps = num_timesteps
        if beta_schedule == "linear":
            self.betas = torch.linspace(
                beta_start, beta_end, num_timesteps, dtype=torch.float32)
        elif beta_schedule == "quadratic":
            self.betas = torch.linspace(
                beta_start ** 0.5, beta_end ** 0.5, num_timesteps, dtype=torch.float32) ** 2

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(
            self.alphas_cumprod[:-1], (1, 0), value=1.)

        # required for self.add_noise
        self.sqrt_alphas_cumprod = self.alphas_cumprod ** 0.5
        self.sqrt_one_minus_alphas_cumprod = (1 - self.alphas_cumprod) ** 0.5

        # required for reconstruct_x0
        self.sqrt_inv_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod)
        self.sqrt_inv_alphas_cumprod_minus_one = torch.sqrt(
            1 / self.alphas_cumprod - 1)

        # required for q_posterior
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)

    def reconstruct_x0(self, x_t, t, noise):
        s1 = self.sqrt_inv_alphas_cumprod[t]
        s2 = self.sqrt_inv_alphas_cumprod_minus_one[t]
        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)
        return s1 * x_t - s2 * noise

    def q_posterior(self, x_0, x_t, t):
        s1 = self.posterior_mean_coef1[t]
        s2 = self.posterior_mean_coef2[t]
        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)
        mu = s1 * x_0 + s2 * x_t
        return mu

    def get_variance(self, t):
        if t == 0:
            return 0

        variance = self.betas[t] * (1. - self.alphas_cumprod_prev[t]) / (1. - self.alphas_cumprod[t])
        variance = variance.clip(1e-20)
        return variance

    def step(self, model_output, timestep, sample):
        t = timestep
        pred_original_sample = self.reconstruct_x0(sample, t, model_output)
        pred_prev_sample = self.q_posterior(pred_original_sample, sample, t)

        variance = 0
        if t > 0:
            noise = torch.randn_like(model_output)
            variance = (self.get_variance(t) ** 0.5) * noise

        pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample

    def add_noise(self, x_start, x_noise, timesteps):
        s1 = self.sqrt_alphas_cumprod[timesteps]
        s2 = self.sqrt_one_minus_alphas_cumprod[timesteps]

        s1 = s1.reshape(-1, 1) # (32,1)
        s2 = s2.reshape(-1, 1) # (32, 1)
        # print(x_start.shape) # (32, 3)
        # print(x_noise.shape) # (32, 3)
        """
        s1はtが大きくなるにつれ小さくなる⇒のイズが大きくなる
        s1, s2のshape: (batch_size,1),
        x_start, x_noiseのshape: (batch_size, data_dim)
        """
        return s1 * x_start + s2 * x_noise

    def __len__(self):
        return self.num_timesteps


'''Different methods for positional embeddings. These are not essential for understanding DDPMs, but are relevant for the ablation study.'''

import torch
from torch import nn
from torch.nn import functional as F


class SinusoidalEmbedding(nn.Module):
    def __init__(self, size: int, scale: float = 1.0):
        super().__init__()
        self.size = size
        self.scale = scale

    def forward(self, x: torch.Tensor):
        x = x * self.scale
        half_size = self.size // 2
        emb = torch.log(torch.Tensor([10000.0])) / (half_size - 1)
        emb = torch.exp(-emb * torch.arange(half_size))
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)
        return emb

    def __len__(self):
        return self.size


class LinearEmbedding(nn.Module):
    def __init__(self, size: int, scale: float = 1.0):
        super().__init__()
        self.size = size
        self.scale = scale

    def forward(self, x: torch.Tensor):
        x = x / self.size * self.scale
        return x.unsqueeze(-1)

    def __len__(self):
        return 1


class LearnableEmbedding(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.size = size
        self.linear = nn.Linear(1, size)

    def forward(self, x: torch.Tensor):
        return self.linear(x.unsqueeze(-1).float() / self.size)

    def __len__(self):
        return self.size


class IdentityEmbedding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return x.unsqueeze(-1)

    def __len__(self):
        return 1


class ZeroEmbedding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return x.unsqueeze(-1) * 0

    def __len__(self):
        return 1


class PositionalEmbedding(nn.Module):
    def __init__(self, size: int, type: str, **kwargs):
        super().__init__()

        if type == "sinusoidal":
            self.layer = SinusoidalEmbedding(size, **kwargs)
        elif type == "linear":
            self.layer = LinearEmbedding(size, **kwargs)
        elif type == "learnable":
            self.layer = LearnableEmbedding(size)
        elif type == "zero":
            self.layer = ZeroEmbedding()
        elif type == "identity":
            self.layer = IdentityEmbedding()
        else:
            raise ValueError(f"Unknown positional embedding type: {type}")

    def forward(self, x: torch.Tensor):
        return self.layer(x)



# ランダム性に関する評価

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def sphere_dataset(n=8000):
    rng = np.random.default_rng(42)

    # x, y, z coordinates
    x = np.round(rng.uniform(-0.5, 0.5, n)/2, 1)*2
    y = np.round(rng.uniform(-0.5, 0.5, n)/2, 1)*2
    z = np.round(rng.uniform(-0.5, 0.5, n)/2, 1)*2

    # Normalize
    norm = np.sqrt(x**2 + y**2 + z**2) + 1e-10
    x /= norm
    y /= norm
    z /= norm

    # Add noise
    theta = 2 * np.pi * rng.uniform(0, 1, n)
    phi = np.pi * rng.uniform(0, 1, n)
    r = rng.uniform(0, 0.03, n)
    x += r * np.sin(phi) * np.cos(theta)
    y += r * np.sin(phi) * np.sin(theta)
    z += r * np.cos(phi)

    # Stack coordinates
    X = np.stack((x, y, z), axis=1)
    # X *= 3  # Scale

    return TensorDataset(torch.from_numpy(X.astype(np.float32)))


dataset = sphere_dataset(8000)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)
model = MLP(hidden_size=128, hidden_layers=3, emb_size=128, time_emb="sinusoidal", input_emb="sinusoidal", out_dim=3)
noise_scheduler = NoiseScheduler(num_timesteps=1000, beta_schedule="linear")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)


checkpoint = torch.load('./weights/3d_points_weights0.pth')
model.load_state_dict(checkpoint)

dim=3
grid_size = 16

x = np.linspace(-3, 3, grid_size)
y = np.linspace(-3, 3, grid_size)
z = np.linspace(-3, 3, grid_size)
xx, yy, zz = np.meshgrid(x, y, z)
samples = np.stack([xx, yy, zz], axis=-1).reshape(-1, dim).astype(np.float32)  # (grid_size**3, 3) になる
samples = torch.from_numpy(samples)
print(samples.shape)


model.eval()
# model.to(device)
# samples.to(device)


# timestepsの生成
timesteps = np.arange(0, 1001, 1).tolist()[::-1]
timesteps_cnt = len(timesteps)
print(f"timesteps_cnt: {timesteps_cnt}")
data_list = np.zeros((len(timesteps)+1, grid_size**3, dim))

residual_stack = []
for i, t in enumerate(tqdm(timesteps)):
    t = torch.from_numpy(np.repeat(t, grid_size**3)).long()
    # t.to(device)
    # samples.to(device)
    with torch.no_grad():

        residual = model(samples, t)
        residual_stack.append(residual)
residual_stack = np.stack(residual_stack)
print(residual_stack.shape)

# residual_stackをファイルに保存
file_path = f'./score_data/residual_stack_3d_{dim}.npy'
np.save(file_path, residual_stack)

# データを読み込む
residual_stack = np.load(file_path)


norms = torch.norm(samples, dim=1)
epsilon = 0.05
mask = (norms >= 1 - epsilon) & (norms <= 1+epsilon)
sphere_samples = samples[mask]

print(sphere_samples.shape)
residual_stack_inv = residual_stack[:, ~mask, :] * -1
print(residual_stack_inv.shape)



import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import to_rgba_array, Normalize

# データフォルダの作成
output_dir = f'./data/3d_vfield_{grid_size}/pdf2'
os.makedirs(output_dir, exist_ok=True)

# グリッドの設定
x = np.linspace(-3, 3, grid_size)
y = np.linspace(-3, 3, grid_size)
z = np.linspace(-3, 3, grid_size)
x, y, z = np.meshgrid(x, y, z)

# サンプルデータを仮に生成
residual_stack_inverse = residual_stack * -1  # サンプルデータを仮に生成
residual_stack_inverse[:, ~mask, :] = 0

# カラーバーの範囲を最初のタイムステップで決定
magnitude_all = np.sqrt(np.sum(residual_stack_inverse**2, axis=2)).flatten()
vmin = magnitude_all.min()
vmax = magnitude_all.max()
norm = Normalize(vmin=vmin, vmax=vmax)

# 初期設定
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# カラーバーの初期追加
sm = plt.cm.ScalarMappable(cmap=cm.viridis, norm=norm)
sm.set_array([])
# fig.colorbar(sm, ax=ax, shrink=0.5, aspect=5, label='Magnitude')
fig.colorbar(sm, ax=ax, shrink=0.5, aspect=5, label='')

for num in tqdm(range(timesteps_cnt)):
    residuals = residual_stack_inverse[num]
    u = residuals[:, 0].reshape((grid_size, grid_size, grid_size))
    v = residuals[:, 1].reshape((grid_size, grid_size, grid_size))
    w = residuals[:, 2].reshape((grid_size, grid_size, grid_size))
    magnitude = np.sqrt(u**2 + v**2 + w**2)

    # Maskを適用して不要なデータを除外
    u = u.flatten()[mask]
    v = v.flatten()[mask]
    w = w.flatten()[mask]
    x_masked = x.flatten()[mask]
    y_masked = y.flatten()[mask]
    z_masked = z.flatten()[mask]
    magnitude_masked = magnitude.flatten()[mask]

    # カラーマップの設定
    colors = cm.viridis(norm(magnitude_masked))

    # 新しいベクトル場を再描画
    quiver = ax.quiver(x_masked, y_masked, z_masked, u, v, w, color=colors, length=0.3, normalize=True)

    # ax.set_title(f"Timestep {num}")

    # 画像を保存
    plt.savefig(os.path.join(output_dir, f'{num:04d}.pdf'))

    # クイバーを削除
    quiver.remove()

    # プロットをクリア
    ax.cla()

plt.close(fig)