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

# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



Time_step = 1000
Data_size = 50000
dim_d = 7 # Euclidean space R^d
is_train = True # if True, then train ddpm network
batch_size = 1000 # testデータのbatch_size
dataset_name = "circle" # Todo: まだcircleの場合しか、backwardは対応していない

# 訓練dataの生成　shape: (n, r)
# dim_d次元のユークリッド空間に埋め込まれた2次元の単位円を生成
if dataset_name == "circle":
    data = dataset.circle_dataset(n=Data_size, r=dim_d).numpy()
elif dataset_name == "sphere":
    data = dataset.sphere_dataset(n=Data_size, r=dim_d).numpy()

sde = model.VP_SDE_dim(beta_min=0.1, beta_max=20, N=Time_step, T=1)

print("loading Us_forward data...")
Us = model.euler_maruyama_dim(data, sde)
print("Successfuly loaded Us_forward data!")
# cnt_prob, cnt_prob2 = neighbourhood_cnt(Us)
cnt_prob = utils.neighbourhood_cnt(Us, dataset_name)

plt.figure(figsize=(9, 6))
plt.plot(cnt_prob, label='Currently Outside', color='b')
# plt.plot(cnt_prob2, label='At Any Time Outside', color='r')
plt.legend()
plt.grid(True)
plt.savefig(f'images/{dataset_name}/r{dim_d}.png')

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
        model.train(nn_model, dataloader, noise_scheduler, optimizer, device=device, N_epoch=200)
        torch.save(nn_model.state_dict(), f'/workspace/weights/{dataset_name}_in_r{dim_d}_{i}.pth')
    print("Successfuly trained ddpm model!")

nn_model = model.MLP(hidden_size=128, hidden_layers=3, emb_size=128, time_emb="sinusoidal", input_emb="sinusoidal", out_dim=dim_d).to(device)
print("loading Us_backward data...")
Us_backward = model.load_experiments(roop=5, batch_size=1000, Time_step=1000, dim_d=7, device=device)
Us_backward = np.stack(Us_backward) # [Gauss noise, ..., S_1]
print(f"Us_backward.shape: {Us_backward.shape}") 

prob_ls_stacked = []

for i in range(5):
    prob_ls = []
    for t in range(Time_step):
        cnt = 0
        for j in range(batch_size):
            if not utils.is_neighbour_2d(Us_backward[i][t][j]):
                cnt += 1
        prob_ls.append(cnt/batch_size)

    prob_ls_stacked.append(prob_ls)

means = np.mean(prob_ls_stacked, axis=0)
stds = np.std(prob_ls_stacked, axis=0)
print(means)

plt.figure(figsize=(9, 6))
plt.plot(means)
plt.savefig(f'images/{dataset_name}/r{dim_d}_back.png')

plt.figure(figsize=(9, 6))
plt.scatter(Us_backward[0, 0, :, 2], Us_backward[0, 0, :, 3])
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.savefig(f'images/{dataset_name}/r{dim_d}_back_last.png')
