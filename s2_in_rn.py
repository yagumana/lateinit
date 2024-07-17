import os
import torch
from tqdm import tqdm
import numpy as np
import scipy as sp
import matplotlib
import seaborn as sns
from cycler import cycler
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.ticker import FormatStrFormatter

import dataset
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch import nn
import model
import utils
import sys
import argparse

def perser_args():
    parser = argparse.ArgumentParser(description="Script for training DDPM Network")

    # add each argument
    parser.add_argument('--time_step', type=int, default=1000, help='Number of time steps')
    parser.add_argument('--data_size', type=int, default=50000)
    parser.add_argument('--dim_d', type=int, default=7)
    parser.add_argument('--is_train', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--dataset_name', type=str, default="sphere")
    parser.add_argument('--R', type=float, default=3)
    parser.add_argument('--r', type=float, default=1)

    args = parser.parse_args()
    return args

# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



Time_step = 1000
Data_size = 50000
dim_d = 7 # Euclidean space R^d
is_train = False # if True, then train ddpm network
batch_size = 1000 # testデータのbatch_size
dataset_name = "sphere_notuniform" # Todo: ellipse
R = 3
r = 1
path_name = dataset_name
if dataset_name == "torus" or dataset_name == "ellipse":
    base_path = os.getcwd()
    path_name = dataset_name + str(R) + str(r)
    img_dir = os.path.join(base_path, "images", path_name)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
else:
    base_path = os.getcwd()
    img_dir = os.path.join(base_path, "images", path_name)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)


# 訓練dataの生成　shape: (n, r)
# dim_d次元のユークリッド空間に埋め込まれた2次元の単位円を生成
if dataset_name == "circle":
    data = dataset.circle_dataset(n=Data_size, r=dim_d).numpy()
elif dataset_name == "sphere":
    data = dataset.sphere_dataset(n=Data_size, r=dim_d).numpy()
elif dataset_name == "sphere_notuniform":
    data = dataset.sphere_dataset_notuniform(n=Data_size, r=dim_d).numpy()
elif dataset_name == "torus":
    data = dataset.torus_dataset(n=Data_size, R=R, r=r, dim=dim_d).numpy()
elif dataset_name == "ellipse":
    data = dataset.ellipse_dataset(n=Data_size, a=R, b=r, dim=dim_d).numpy()

sde = model.VP_SDE_dim(beta_min=0.1, beta_max=20, N=Time_step, T=1)

print("loading Us_forward data...")
Us = model.euler_maruyama_dim(data, sde)
print("Successfuly loaded Us_forward data!")
# cnt_prob, cnt_prob2 = neighbourhood_cnt(Us)
cnt_prob = utils.neighbourhood_cnt(Us, dataset_name, R=R, r=r) # ここは、図形依存だから、dataset_nameで良い（path_nameじゃない）

plt.figure(figsize=(9, 6))
plt.plot(cnt_prob, label='Currently Outside', color='b')
# plt.plot(cnt_prob2, label='At Any Time Outside', color='r')
plt.legend()
plt.grid(True)
plt.savefig(f'images/{path_name}/r{dim_d}_forward.png')

tensor_data = TensorDataset(torch.tensor(data))
dataloader = DataLoader(tensor_data, batch_size=32, shuffle=True, drop_last=True)

global_step = 0
frames = []
losses = []


if is_train == True:
    for i in range(5):
        nn_model = model.MLP(hidden_size=128, hidden_layers=3, emb_size=128, time_emb="sinusoidal", input_emb="sinusoidal", out_dim=dim_d).to(device)
        noise_scheduler = model.NoiseScheduler(num_timesteps=1000, beta_schedule="linear")
        optimizer = torch.optim.AdamW(nn_model.parameters(), lr=1e-3)
        print(f"training model_{i}...")
        model.train(nn_model, dataloader, noise_scheduler, optimizer, device=device, N_epoch=100)
        print(f"dim_d: {dim_d}")
        torch.save(nn_model.state_dict(), f'/workspace/weights/{path_name}_in_r{dim_d}_{i}.pth')
    print("Successfuly trained ddpm model!")

nn_model = model.MLP(hidden_size=128, hidden_layers=3, emb_size=128, time_emb="sinusoidal", input_emb="sinusoidal", out_dim=dim_d).to(device)
print("loading Us_backward data...")
Us_backward = model.load_experiments(roop=5, batch_size=1000, Time_step=1000, dim_d=dim_d, device=device, path_name=path_name)
Us_backward = np.stack(Us_backward) # [Gauss noise, ..., S_1]
print(f"Us_backward.shape: {Us_backward.shape}") 

prob_ls_stacked = []

for i in range(5):
    prob_ls = utils.neighbourhood_cnt(Us_backward[i], dataset_name, R=R, r=r) # ここは、図形依存だから、dataset_nameで良い（path_nameじゃない）

    prob_ls_stacked.append(prob_ls)

means = np.mean(prob_ls_stacked, axis=0)
stds = np.std(prob_ls_stacked, axis=0)
print(means)

plt.figure(figsize=(9, 6))
plt.plot(means)
plt.savefig(f'images/{path_name}/r{dim_d}_back.png')

plt.figure(figsize=(9, 6))
plt.scatter(Us_backward[0, 0, :, 0], Us_backward[0, 0, :, 1])
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.savefig(f'images/{path_name}/r{dim_d}_back_dim01_1.png')

plt.figure(figsize=(9, 6))
plt.scatter(Us_backward[1, 0, :, 0], Us_backward[1, 0, :, 1])
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.savefig(f'images/{path_name}/r{dim_d}_back_dim01_2.png')

plt.figure(figsize=(9, 6))
plt.scatter(Us_backward[2, 0, :, 2], Us_backward[2, 0, :, 3])
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.savefig(f'images/{path_name}/r{dim_d}_back_dim23_2.png')