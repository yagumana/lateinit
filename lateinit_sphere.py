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
import ot
import model
import utils
import sys

"""
すでに訓練済みのddpmの重みが存在することを前提にする. (s1_in_r3.pyファイルを参照)
"""


# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Time_step = 1000
Late_time = [0, 100, 200, 300, 400, 500, 600, 700, 725, 750, 775, 800, 825, 850, 875, 900, 925, 950, 960, 970, 980, 990, 999]
# Late_time の値を変換
Late_time_transformed = [Time_step - t for t in Late_time]

Data_size = 1000
dim_d = 49
dataset_name = "circle"
batch_size = 1000
Roop = 5
Load_exp = True
R=3
r=1

path_name = dataset_name
if dataset_name == "torus" or dataset_name == "ellipse":
    base_path = os.getcwd()
    path_name = dataset_name + str(R) + str(r)


def get_forward_Us(dim_d, roop=5, dataset_name = "circle", num_timesteps=1000, batch_size=1000):
    if dataset_name == "circle":
        train_data = dataset.circle_dataset(8000, dim_d)
    elif dataset_name == "sphere":
        train_data = dataset.sphere_dataset(8000, dim_d)
    elif dataset_name == "sphere_notuniform":
        train_data = dataset.sphere_dataset_notuniform(n=8000, r=dim_d).numpy()
    elif dataset_name == "torus":
        train_data = dataset.torus_dataset(8000, R=R, r=r, dim=dim_d)
    elif dataset_name == "ellipse":
        train_data = dataset.ellipse_dataset(n=8000, a=R, b=r, dim=dim_d).numpy()

    dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    nn_model = model.MLP(hidden_size=128, hidden_layers=3, emb_size=128, time_emb="sinusoidal", input_emb="sinusoidal", out_dim=dim_d)
    noise_scheduler = model.NoiseScheduler(num_timesteps=num_timesteps, beta_schedule="linear")
    # optimizer = torch.optim.AdamW(nn_model.parameters(), lr=1e-3)

    nn_model.train()
    Us_forward = []

    for i in range(Roop):
        utils.set_seed(i)
        batch = next(iter(dataloader)) # (1000, 7) = (batch_size, dim_d)
        # batch = batch[0]
        # batch_size = batch.shape[0]

        data_list = []
        for t in tqdm(range(num_timesteps)):
            noise = torch.randn(batch.shape)
            t = torch.from_numpy(np.repeat(t, batch_size)).long()
            noisy = noise_scheduler.add_noise(batch, noise, t)

            data_list.append(noisy)
        data_list = np.stack(data_list)
        Us_forward.append(data_list)

    Us_forward = np.stack(Us_forward)

    return Us_forward

# forwardのデータを取得
if Load_exp == True:
    print("loading Us forward data...")
    Us_forward = get_forward_Us(dim_d=dim_d, roop=5, dataset_name=dataset_name, num_timesteps=Time_step, batch_size=batch_size)
    Us_forward = np.array(Us_forward)
    np.save(f"Us_data/forward_{path_name}_in_{dim_d}.npy", Us_forward)
Us_forward = np.load(f"Us_data/forward_{path_name}_in_{dim_d}.npy")
# Us_forward = Us_forward.tolist()
print("Successufuly Loaded Us_forward data!")

# backwardのデータを取得
if Load_exp == True:
    print("loading Us backward data...")
    Us_backward = model.load_experiments(roop=5, batch_size=batch_size, Time_step=Time_step, dim_d=dim_d, device=device, late=0, path_name=path_name)
    Us_backward = np.array(Us_backward)
    Us_backward = Us_backward[:, ::-1, :, :] # Us_backwardを逆順にして、Us_forwradに合わせる

    np.save(f"Us_data/backward_{path_name}_in_{dim_d}.npy", Us_backward)
Us_backward = np.load(f"Us_data/backward_{path_name}_in_{dim_d}.npy")
# Us_backward = Us_backward.tolist()
print("Successufuly Loaded Us_backward data!")

for i in range(Roop):
    cnt_prob = utils.neighbourhood_cnt(Us_forward[i], dataset_name, R=R, r=r)

    plt.figure(figsize=(9, 6))
    plt.plot(cnt_prob, label='Currently Outside', color='b')
    # plt.plot(cnt_prob2, label='At Any Time Outside', color='r')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'images/{path_name}/r{dim_d}_forward_{i}.png')

for i in range(Roop):
    cnt_prob = utils.neighbourhood_cnt(Us_backward[i], dataset_name, R=R, r=r)

    plt.figure(figsize=(9, 6))
    plt.plot(cnt_prob, label='Currently Outside', color='b')
    # plt.plot(cnt_prob2, label='At Any Time Outside', color='r')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'images/{path_name}/r{dim_d}_backward_{i}.png')

print(f"Us_forward.shape: {Us_forward.shape}")
print(f"Us_backward.shape: {Us_backward.shape}")




distance_ls = []
for i in range(len(Late_time)):
    Us_backward = model.load_experiments(roop=5, batch_size=batch_size, Time_step=Time_step, dim_d=dim_d, device=device, late=Late_time[i], path_name=path_name)
    Us_backward = np.array(Us_backward)
    Us_backward = Us_backward[:, ::-1, :, :]
    ls = []
    for j in range(5):
        cost_matrix = ot.dist(Us_forward[j, 0], Us_backward[j, 0])
        a = np.ones(Us_forward[j][0].shape[0]) / Us_forward[j][0].shape[0]
        b = np.ones(Us_backward[j][0].shape[0]) / Us_backward[j][0].shape[0]

        distance = ot.emd2(a, b, cost_matrix, numItermax=200000)
        ls.append(distance)
    ls = np.array(ls)
    distance_ls.append(np.mean(ls))
print(distance_ls)


# backward processにおける管状近傍の外にある粒子の割合を評価
prob_ls_stacked = []
Us_backward = model.load_experiments(roop=5, batch_size=batch_size, Time_step=Time_step, dim_d=dim_d, device=device, late=0, path_name=path_name)
Us_backward = np.array(Us_backward)
Us_backward = Us_backward[:, ::-1, :, :]

# for i in range(5):
#     prob_ls = []
#     for t in range(Time_step):
#         cnt = 0
#         for j in range(batch_size):
#             if not utils.is_neighbour_2d(Us_backward[i][t][j]):
#                 cnt += 1
#         prob_ls.append(cnt/batch_size)

#     prob_ls_stacked.append(prob_ls)
for i in range(5):
    prob_ls = utils.neighbourhood_cnt(Us_backward[i], dataset_name, R=R, r=r)
    prob_ls_stacked.append(prob_ls)


prob_means = np.mean(prob_ls_stacked, axis=0)
prob_stds = np.std(prob_ls_stacked, axis=0)

# グラフの描画
fig, ax1 = plt.subplots(figsize=(10, 6))

# 左側の軸に対して distance_ls をプロット
color = 'tab:blue'
ax1.set_xlabel('Index')
ax1.set_ylabel('Wasserstein Distance', color=color)
ax1.plot(Late_time_transformed, distance_ls, color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_yscale('log')  # y軸を対数スケールに設定

if dataset_name == "circle":
    ax1.axvline(x=215, color='green', linestyle='--', linewidth=2)
elif dataset_name == "sphere":
    # 横軸177に縦線を引く
    ax1.axvline(x=177, color='green', linestyle='--', linewidth=2)

# 右側の軸に対して prob_ls をプロット
ax2 = ax1.twinx()  # 2つ目の軸を生成
color = 'tab:red'
ax2.set_ylabel('Probability of the particles outside tubular neighbourhood', color=color)
ax2.plot(prob_means, color=color)
ax2.tick_params(axis='y', labelcolor=color)

plt.savefig(f'images/{path_name}/r{dim_d}_back_lateinit.png')







plt.figure(figsize=(6, 6))
plt.scatter(Us_backward[0, 0, :, 5], Us_backward[0, 0, :, 6])
# plt.xlim(-3, 3)
# plt.ylim(-3, 3)
plt.savefig(f"images/{path_name}/r{dim_d}_back_last_test0.png")

plt.figure(figsize=(8, 6))
plt.scatter(Us_backward[1, 0, :, 0], Us_backward[1, 0, :, 1])
# plt.xlim(-3, 3)
# plt.ylim(-3, 3)
plt.savefig(f"images/{path_name}/r{dim_d}_back_last_test1.png")

plt.figure(figsize=(8, 6))
plt.scatter(Us_backward[2, 0, :, 0], Us_backward[2, 0, :, 1])
# plt.xlim(-3, 3)
# plt.ylim(-3, 3)
plt.savefig(f"images/{path_name}/r{dim_d}_back_last_test2.png")

plt.figure(figsize=(8, 6))
plt.scatter(Us_backward[3, 0, :, 5], Us_backward[3, 0, :, 6])
# plt.xlim(-3, 3)
# plt.ylim(-3, 3)
plt.savefig(f"images/{path_name}/r{dim_d}_back_last_test3.png")