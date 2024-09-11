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
import sys
import logging

sys.path.append('./denoising-diffusion-pytorch')

from denoising_diffusion_pytorch.denoising_diffusion_pytorch_1d import Unet1D, GaussianDiffusion1D, Trainer1D, Dataset1D


import dataset
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch import nn
import model
import utils
import argparse
import yaml
import wandb

def load_config(config_path):
    with open(config_path) as file:
        config = yaml.safe_load(file)
    return config

def parser_args():
    parser = argparse.ArgumentParser(description="Script for training DDPM Network")

    # add each argument
    parser.add_argument('--config', type=str, default='./configs/s1.yaml', help='Path to the config file')
    parser.add_argument('--time_step', type=int, help='Number of time steps')
    parser.add_argument('--data_size', type=int)
    parser.add_argument('--dim_d', type=int)
    parser.add_argument('--dim_z', type=int)
    parser.add_argument('--is_train', action='store_true', help='If provided, training mode is enabled')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--R', type=float)
    parser.add_argument('--r', type=float)
    parser.add_argument('--denoise_model', type=str, help='Denoise model type')


    args = parser.parse_args()
    return args

def update_config_with_args(config, args):
    if args.time_step is not None:
        config['time_step'] = args.time_step
    if args.data_size is not None:
        config['data_size'] = args.data_size
    if args.dim_d is not None:
        config['dim_d'] = args.dim_d
    if args.dim_z is not None:
        config['dim_z'] = args.dim_z
    if args.is_train:
        config['is_train'] = True
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.dataset_name is not None:
        config['dataset_name'] = args.dataset_name
    if args.dataset_path is not None:
        config['dataset_path'] = args.dataset_path
    if args.R is not None:
        config['R'] = args.R
    if args.r is not None:
        config['r'] = args.r
    if args.denoise_model is not None:
        config['denoise_model'] = args.denoise_model

    return config

def setup_paths(config):
    dataset_name = config['dataset_name']
    path_name = dataset_name
    base_path = os.getcwd()

    if dataset_name == "embed_sphere":
        path_name = config['embed_data']
    elif dataset_name in ["torus", "ellipse"]:
        path_name = dataset_name + str(config['R']) + str(config['r'])

    img_dir = os.path.join(base_path, "images", path_name)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    return dataset_name, path_name, img_dir


if __name__ == "__main__":

    # 引数を取得
    args = parser_args()

    # ログのセットアップ　（ログファイル名はここで指定）
    log_filename = "sn_in_rn_unet.log"
    utils.setup_logging(log_filename)

    # yamlファイルから設定を読み込む
    config = load_config(args.config)

    # 設定の上書き処理
    config = update_config_with_args(config, args)

    # ログでコンフィグの確認
    logging.info(f"Config loaded: {config}")


    wandb.init(config=config, project="ddpm")

    # デバイスの設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Time_step = config['time_step']
    Data_size = config['data_size']
    dim_d = config['dim_d'] # Euclidean space R^d
    dim_z = config['dim_z'] # Sphere S^z
    is_train = config['is_train'] # if True, then train ddpm network
    batch_size = config['batch_size'] # testデータのbatch_size
    R = config['R']
    r = config['r']

    # パスの設定と dataset_nameの取得
    dataset_name, path_name, denoise_model = setup_paths(config)

    logging.info(f"Training mode is {'enabled' if is_train else 'disabled'}")
    logging.info(f"Dataset name: {dataset_name}")

    # 訓練dataの生成　shape: (n, r)
    # dim_d次元のユークリッド空間に埋め込まれた2次元の単位円を生成
    if dataset_name == "circle":
        data = dataset.circle_dataset(n=Data_size, r=dim_d).numpy()
    elif dataset_name == "circle_half":
        data = dataset.circle_half_dataset(n=Data_size, r=dim_d).numpy()
    elif dataset_name == "circle_two_injectivity":
        data = dataset.circle_half_dataset2(n=Data_size, r=dim_d).numpy()
    elif dataset_name == "sphere":
        data = dataset.sphere_dataset(n=Data_size, r=dim_d).numpy()
    elif dataset_name == "sphere_notuniform":
        data = dataset.sphere_dataset_notuniform(n=Data_size, r=dim_d).numpy()
    elif dataset_name == "torus":
        data = dataset.torus_dataset(n=Data_size, R=R, r=r, dim=dim_d).numpy()
    elif dataset_name == "ellipse":
        data = dataset.ellipse_dataset(n=Data_size, a=R, b=r, dim=dim_d).numpy()
    elif dataset_name == "embed_sphere":
        data = dataset.embed_sphere_dataset(n=Data_size, dim_d=dim_d, dataset_path=dataset_path).detach().numpy()
    elif dataset_name == "hyper_sphere":
        data = dataset.hyper_sphere_dataset(n=Data_size, r=dim_d, s=dim_z).numpy()

    sde = model.VP_SDE_dim(beta_min=0.1, beta_max=20, N=Time_step, T=1)

    logging.info("loading Us_forward data...")
    Us = model.euler_maruyama_dim(data, sde)
    logging.info("Successfuly loaded Us_forward data!")
    logging.info(f"Us.shape: {Us.shape}")
    logging.info(f"dataset_name: {dataset_name}")
    cnt_prob = utils.neighbourhood_cnt(Us, dataset_name, R=R, r=r, dim_z=dim_z) # ここは、図形依存だから、dataset_nameで良い（path_nameじゃない）

    plt.figure(figsize=(9, 6))
    plt.plot(cnt_prob, label='Currently Outside', color='b')
    # plt.plot(cnt_prob2, label='At Any Time Outside', color='r')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'images/{path_name}/r{dim_d}_forward.png')

    plt.figure(figsize=(9, 6))
    plt.scatter(Us[0, :, 2], Us[0, :, 3])
    plt.savefig(f'images/{path_name}/r{dim_d}_forward_dim23_2.png')



    tensor_data = TensorDataset(torch.tensor(data))
    dataloader = DataLoader(tensor_data, batch_size=32, shuffle=True, drop_last=True)

    global_step = 0
    frames = []
    losses = []

    if is_train == True:
        for i in range(5):
            logging.info(f"denoise_model: {denoise_model}")
            if denoise_model == "MLP":
                nn_model = model.MLP(hidden_size=128, hidden_layers=3, emb_size=128, time_emb="sinusoidal", input_emb="sinusoidal", out_dim=dim_d).to(device)
            elif denoise_model == "Unet":
                nn_model = Unet1D(dim=dim_d, channels=1).to(device)

            noise_scheduler = model.NoiseScheduler(num_timesteps=1000, beta_schedule="linear")
            optimizer = torch.optim.AdamW(nn_model.parameters(), lr=1e-3)
            logging.info(f"training model_{i}...")
            if not os.path.exists(f'/workspace/weights'):
                os.makedirs(f'/workspace/weights')
            losses = model.train(nn_model, dataloader, noise_scheduler, optimizer, device=device, N_epoch=25, wandb=wandb, path_name=path_name, dim_d=dim_d, dim_z=dim_z, i=i)
            logging.info(f"losses: {losses}")
            # torch.save(nn_model.state_dict(), f'/workspace/weights/{path_name}_in_r{dim_d}_{i}.pth')
        logging.info("Successfuly trained ddpm model!")

    if denoise_model == "MLP":
        nn_model = model.MLP(hidden_size=128, hidden_layers=3, emb_size=128, time_emb="sinusoidal", input_emb="sinusoidal", out_dim=dim_d).to(device)
    elif denoise_model == "Unet":
        nn_model = Unet1D(dim=dim_d).to(device)

    logging.info("loading Us_backward data...")
    Us_backward = model.load_experiments(roop=5, batch_size=1000, Time_step=1000, dim_d=dim_d, dim_z=dim_z, device=device, path_name=path_name, denoise_model=denoise_model)
    Us_backward = np.stack(Us_backward) # [Gauss noise, ..., S_1]
    logging.info(f"Us_backward.shape: {Us_backward.shape}") 

    prob_ls_stacked = []

    for i in range(5):
        prob_ls = utils.neighbourhood_cnt(Us_backward[i], dataset_name, R=R, r=r, dim_z=dim_z) # ここは、図形依存だから、dataset_nameで良い（path_nameじゃない）

        prob_ls_stacked.append(prob_ls)

    means = np.mean(prob_ls_stacked, axis=0)
    stds = np.std(prob_ls_stacked, axis=0)
    logging.info(means)

    plt.figure(figsize=(9, 6))
    plt.plot(means)
    plt.savefig(f'images/{path_name}/r{dim_d}_back.png')

    plt.figure(figsize=(9, 6))
    plt.scatter(Us_backward[0, -1, :, 0], Us_backward[0, -1, :, 1])
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.savefig(f'images/{path_name}/r{dim_d}_back_dim01_1.png')

    plt.figure(figsize=(9, 6))
    plt.scatter(Us_backward[1, -1, :, 0], Us_backward[1, -1, :, 1])
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.savefig(f'images/{path_name}/r{dim_d}_back_dim01_2.png')
    