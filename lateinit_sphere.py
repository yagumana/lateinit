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
import yaml
import argparse
import wandb
import sys

"""
すでに訓練済みのddpmの重みが存在することを前提にする. (s1_in_r3.pyファイルを参照)
"""

def load_config(config_path):
    with open(config_path) as file:
        config = yaml.safe_load(file)
    return config


def parser_args():
    parser = argparse.ArgumentParser(description='Script for late initialization')

    # add each argument
    parser.add_argument('--config', type=str, default='./configs/s1_lateinit.yaml', help='Path to the config file')
    parser.add_argument('--time_step', type=int, help='Number of time steps')
    parser.add_argument('--Late_time', type=list, help='Late time')
    parser.add_argument('--data_size', type=int)
    parser.add_argument('--dim_d', type=int)
    parser.add_argument('--dim_z', type=int)
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--Roop', type=int)
    parser.add_argument('--Load_exp', action='store_true', help='If provided, Load experiment data')
    parser.add_argument('--R', type=float)
    parser.add_argument('--r', type=float)
    parser.add_argument('--denoise_model', type=str, help='Denoise model type')


    args = parser.parse_args()
    return args

def get_forward_Us(dim_d, dim_z=20, roop=5, dataset_name = "circle", num_timesteps=1000, batch_size=1000):
    if dataset_name == "circle":
        train_data = dataset.circle_dataset(8000, dim_d)
    elif dataset_name == "circle_half":
        train_data = dataset.circle_half_dataset(8000, dim_d)
    elif dataset_name == "circle_two_injectivity":
        train_data = dataset.circle_half_dataset2(8000, dim_d)
    elif dataset_name == "sphere":
        train_data = dataset.sphere_dataset(8000, dim_d)
    elif dataset_name == "sphere_notuniform":
        train_data = dataset.sphere_dataset_notuniform(n=8000, r=dim_d).numpy()
    elif dataset_name == "torus":
        train_data = dataset.torus_dataset(8000, R=R, r=r, dim=dim_d)
    elif dataset_name == "ellipse":
        train_data = dataset.ellipse_dataset(n=8000, a=R, b=r, dim=dim_d).numpy()
    elif dataset_name == "embed_sphere":
        train_data = dataset.embed_sphere_dataset(n=50000, dim_d=dim_d, dataset_path=dataset_path).detach().numpy()
    elif dataset_name == "hyper_sphere":
        train_data = dataset.hyper_sphere_dataset(n=8000, r=dim_d, s=dim_z).detach().numpy()

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

if __name__ == "__main__":
    args = parser_args()

    # yamlファイルから設定を読み込む
    config = load_config(args.config)

    if args.time_step is not None:
        config['time_step'] = args.time_step
    if args.Late_time is not None:
        config['Late_time'] = args.Late_time
    if args.data_size is not None:
        config['data_size'] = args.data_size
    if args.dim_d is not None:
        config['dim_d'] = args.dim_d
    if args.dim_z is not None:
        config['dim_z'] = args.dim_z
    if args.dataset_name is not None:
        config['dataset_name'] = args.dataset_name
    if args.dataset_path is not None:
        config['dataset_path'] = args.dataset_path
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.Roop is not None:
        config['Roop'] = args.Roop
    if args.Load_exp:
        config['Load_exp'] = True
    if args.R is not None:
        config['R'] = args.R
    if args.r is not None:
        config['r'] = args.r
    if args.denoise_model is not None:
        config['denoise_model'] = args.denoise_model
    
    wandb.init(config=config, project="late initialization")

    # デバイスの設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # パラメータの設定
    Time_step = config['time_step']
    Late_time = config['Late_time']
    Late_time_transformed = [Time_step - t for t in Late_time]
    Data_size = config['data_size']
    dim_d = config['dim_d']
    dim_z = config['dim_z']
    dataset_name = config['dataset_name']
    if dataset_name == "embed_sphere":
        embed_data = config['embed_data'] # hyperspherical vaeを用いて埋め込んだデータセット名(ex. mnist, fashion_mnist)
    dataset_path = config['dataset_path']
    batch_size = config['batch_size']
    Roop = config['Roop']
    Load_exp = config['Load_exp']
    R = config['R']
    r = config['r']
    denoise_model = config['denoise_model']


    path_name = dataset_name
    if dataset_name == "torus" or dataset_name == "ellipse":
        base_path = os.getcwd()
        path_name = dataset_name + str(R) + str(r)
    if dataset_name == "embed_sphere":
        path_name = embed_data

    # forwardのデータを取得
    if Load_exp == True:
        print(f"loading Us forward data...{dataset_name}")
        Us_forward = get_forward_Us(dim_d=dim_d, dim_z=dim_z, roop=Roop, dataset_name=dataset_name, num_timesteps=Time_step, batch_size=batch_size)
        Us_forward = np.array(Us_forward)
        np.save(f"Us_data/forward_{path_name}_in_{dim_d}.npy", Us_forward)
    Us_forward = np.load(f"Us_data/forward_{path_name}_in_{dim_d}.npy")
    # Us_forward = Us_forward.tolist()
    print("Successufuly Loaded Us_forward data!")

    # backwardのデータを取得
    if Load_exp == True:
        print("loading Us backward data...")
        Us_backward = model.load_experiments(roop=Roop, batch_size=batch_size, Time_step=Time_step, dim_d=dim_d, dim_z=dim_z, device=device, late=0, path_name=path_name, denoise_model=denoise_model)
        Us_backward = np.array(Us_backward)
        Us_backward = Us_backward[:, ::-1, :, :] # Us_backwardを逆順にして、Us_forwradに合わせる

        np.save(f"Us_data/backward_{path_name}_in_{dim_d}.npy", Us_backward)
    Us_backward = np.load(f"Us_data/backward_{path_name}_in_{dim_d}.npy")
    # Us_backward = Us_backward.tolist()
    print("Successufuly Loaded Us_backward data!")

    for i in range(Roop):
        cnt_prob = utils.neighbourhood_cnt(Us_forward[i], dataset_name, R=R, r=r, dim_z=dim_z)

        plt.figure(figsize=(9, 6))
        plt.plot(cnt_prob, label='Currently Outside', color='b')
        # plt.plot(cnt_prob2, label='At Any Time Outside', color='r')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'images/{path_name}/r{dim_d}_forward_{i}.png')

    for i in range(Roop):
        cnt_prob = utils.neighbourhood_cnt(Us_backward[i], dataset_name, R=R, r=r, dim_z=dim_z)

        plt.figure(figsize=(9, 6))
        plt.plot(cnt_prob, label='Currently Outside', color='b')
        # plt.plot(cnt_prob2, label='At Any Time Outside', color='r')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'images/{path_name}/r{dim_d}_backward_{i}.png')

    print(f"Us_forward.shape: {Us_forward.shape}")
    print(f"Us_backward.shape: {Us_backward.shape}")

    # backward processにおける管状近傍の外にある粒子の割合を評価
    prob_ls_stacked = []
    Us_backward = model.load_experiments(roop=Roop, batch_size=batch_size, Time_step=Time_step, dim_d=dim_d, dim_z=dim_z, device=device, late=0, path_name=path_name, denoise_model=denoise_model)
    Us_backward = np.array(Us_backward)
    Us_backward = Us_backward[:, ::-1, :, :]

    for i in range(Roop):
        prob_ls = utils.neighbourhood_cnt(Us_backward[i], dataset_name, R=R, r=r, dim_z=dim_z)
        prob_ls_stacked.append(prob_ls)

    prob_means = np.mean(prob_ls_stacked, axis=0)
    prob_stds = np.std(prob_ls_stacked, axis=0)

    # prob_means が 0.95, 0.99, 0.999 以下になる最初のインデックスを探す
    index_10 = np.argmax(prob_means > 0.1)
    index_50 = np.argmax(prob_means > 0.5)
    index_90 = np.argmax(prob_means > 0.9)
    index_95 = np.argmax(prob_means > 0.95)
    index_99 = np.argmax(prob_means > 0.99)
    index_999 = np.argmax(prob_means > 0.999)
    print(index_10)
    print(index_50)
    print(index_90)
    print(index_95)
    print(index_99)
    print(index_999)

    # defaultのLate_timeに、index_95, index_99, index_999 に対応する時間を追加
    Late_time.extend([1000-index_10, 1000-index_50, 1000-index_90, 1000-index_95, 1000-index_99, 1000-index_999])


    distance_ls = []
    for i in range(len(Late_time)):
        Us_backward = model.load_experiments(roop=Roop, batch_size=batch_size, Time_step=Time_step, dim_d=dim_d, device=device, late=Late_time[i], path_name=path_name, dim_z=dim_z, denoise_model=denoise_model)
        Us_backward = np.array(Us_backward)
        Us_backward = Us_backward[:, ::-1, :, :]
        ls = []
        for j in range(Roop):
            cost_matrix = ot.dist(Us_forward[j, 0], Us_backward[j, 0])
            a = np.ones(Us_forward[j][0].shape[0]) / Us_forward[j][0].shape[0]
            b = np.ones(Us_backward[j][0].shape[0]) / Us_backward[j][0].shape[0]

            distance = ot.emd2(a, b, cost_matrix, numItermax=200000)
            ls.append(distance)
        ls = np.array(ls)
        distance_ls.append(np.mean(ls))
    print(distance_ls)

    # distance_lsのうち、index_95, 99, 999に対応する最後の3つの値を消す
    distance_ls = distance_ls[:-6]

    # dis_ls の最後の値の 1.2 倍を計算
    target_value = distance_ls[0] * 1.2
    # target_value 以上の最後のインデックスを見つける
    index = np.argmax(distance_ls >= target_value) - 1
    # index に対応する Late_time_transformed の値を取得
    target_time = Late_time_transformed[index]
    print(target_time)



    

    # グラフの描画
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 左側の軸に対して distance_ls をプロット
    color = 'tab:blue' 
    ax1.set_xlabel('Diffusion Time')
    ax1.set_ylabel('Wasserstein Distance', color=color)
    ax1.plot(Late_time_transformed, distance_ls, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_yscale('log')  # y軸を対数スケールに設定
    ax1.grid(True, which='both', axis='both', zorder=1)

    # target_time に対応する位置に青色の垂直線を引く
    # ax1.axvline(x=target_time, color=color, linestyle='--', label=f'Target Time: {target_time}')

    

    # if dataset_name == "circle":
    #     ax1.axvline(x=215, color='green', linestyle='--', linewidth=2)
    # elif dataset_name == "sphere":
    #     # 横軸177に縦線を引く
    #     ax1.axvline(x=177, color='green', linestyle='--', linewidth=2)

    # 右側の軸に対して prob_ls をプロット
    ax2 = ax1.twinx()  # 2つ目の軸を生成
    color = 'tab:red'
    ax2.set_ylabel('Probability of the particles outside tubular neighbourhood', color=color)
    ax2.fill_between(range(len(prob_means)), prob_means - prob_stds, prob_means + prob_stds, color='gray', alpha=0.2)
    ax2.plot(prob_means, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.grid(True, which='both', axis='both', zorder=1)

    # 追加: prob_means が 0.95, 0.99, 0.999 以下になるインデックスに垂直線を描画
    ax2.axvline(x=index_95, color='orange', linestyle='--', linewidth=2, label='0.95 Threshold')
    ax2.axvline(x=index_99, color='purple', linestyle='--', linewidth=2, label='0.99 Threshold')
    ax2.axvline(x=index_999, color='brown', linestyle='--', linewidth=2, label='0.999 Threshold')

    # レジェンドの表示
    ax2.legend(loc='upper right')    


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