import os
import numpy as np

"""
mnist dataをbinaryにしないで学習したらどうなるのかをcheck
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import datasets, transforms
import torchvision
from collections import defaultdict
import sys 
import os
import wandb
from tqdm import tqdm
import torchvision.utils as vutils


# hyperspherical_vae`ディレクトリへのパスを追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hyperspherical_vae.distributions.von_mises_fisher import VonMisesFisher
from hyperspherical_vae.distributions.hyperspherical_uniform import HypersphericalUniform

# モデルをGPUに移動
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# train_loader = torch.utils.data.DataLoader(datasets.MNIST('./data', train=True, download=True,
#     transform=transforms.ToTensor()), batch_size=64, shuffle=True)
# test_loader = torch.utils.data.DataLoader(datasets.MNIST('./data', train=False, download=True,
#     transform=transforms.ToTensor()), batch_size=64)


class ModelVAE(torch.nn.Module):
    
    def __init__(self, h_dim, z_dim, activation=F.relu, distribution='normal', model_type='mlp'):
        """
        ModelVAE initializer
        :param h_dim: dimension of the hidden layers
        :param z_dim: dimension of the latent representation
        :param activation: callable activation function
        :param distribution: string either `normal` or `vmf`, indicates which distribution to use
        """
        super(ModelVAE, self).__init__()
        
        self.z_dim, self.activation, self.distribution = z_dim, activation, distribution
        
        # 2 hidden layers encoder
        # self.fc_e0 = nn.Linear(784, h_dim * 2)
        # self.fc_e1 = nn.Linear(h_dim * 2, h_dim)

        self.conv_e0 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv_e1 = nn.Conv2d(in_channels=16, out_channels=48, kernel_size=3, stride=1, padding=1)
        self.conv_e2 = nn.Conv2d(in_channels=48, out_channels=144, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.batch_e0 = nn.BatchNorm2d(16)
        self.batch_e1 = nn.BatchNorm2d(48)
        self.batch_e2 = nn.BatchNorm2d(144)

        self.fc_e0 = nn.Linear(144 * 3 * 3, h_dim*2)
        self.fc_e1 = nn.Linear(h_dim*2, h_dim)

        self.fc_d0 = nn.Linear(z_dim, h_dim)
        self.fc_d1 = nn.Linear(h_dim, h_dim * 2)
        self.fc_d2 = nn.Linear(h_dim * 2, 64*7*7)
        self.conv_d0 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.conv_d1 = nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=4, stride=2, padding=1)


        if self.distribution == 'normal':
            # compute mean and std of the normal distribution
            self.fc_mean = nn.Linear(h_dim, z_dim)
            self.fc_var =  nn.Linear(h_dim, z_dim)
        elif self.distribution == 'vmf':
            # compute mean and concentration of the von Mises-Fisher
            self.fc_mean = nn.Linear(h_dim, z_dim)
            self.fc_var = nn.Linear(h_dim, 1)
        else:
            raise NotImplemented
            
        # # 2 hidden layers decoder
        # self.fc_d0 = nn.Linear(z_dim, h_dim)
        # self.fc_d1 = nn.Linear(h_dim, h_dim * 2)
        # self.fc_logits = nn.Linear(h_dim * 2, 784)

    def encode(self, x):
        # 2 hidden layers encoder
        # x = self.activation(self.fc_e0(x))
        # x = self.activation(self.fc_e1(x))
        x = self.activation(self.pool(self.batch_e0(self.conv_e0(x))))
        x = self.activation(self.pool(self.batch_e1(self.conv_e1(x))))
        x = self.activation(self.pool(self.batch_e2(self.conv_e2(x))))
        x = x.view(x.size(0), -1)
        # print(f"encode: {x.shape}")
        x = self.activation(self.fc_e0(x))
        x = self.activation(self.fc_e1(x))
        
        if self.distribution == 'normal':
            # compute mean and std of the normal distribution
            z_mean = self.fc_mean(x)
            z_var = F.softplus(self.fc_var(x))
        elif self.distribution == 'vmf':
            # compute mean and concentration of the von Mises-Fisher
            z_mean = self.fc_mean(x)
            z_mean = z_mean / z_mean.norm(dim=-1, keepdim=True)
            # the `+ 1` prevent collapsing behaviors
            z_var = F.softplus(self.fc_var(x)) + 1
        else:
            raise NotImplemented
        
        return z_mean, z_var
        
    def decode(self, z):
        
        x = self.activation(self.fc_d0(z))
        x = self.activation(self.fc_d1(x))
        x = self.activation(self.fc_d2(x))
        # print(x.shape)
        x = x.view(-1, 64, 7, 7)
        # print(x.shape)

        x = self.activation(self.conv_d0(x))
        x = torch.sigmoid(self.conv_d1(x))

        # x = self.activation(self.fc_d0(z))
        # x = self.activation(self.fc_d1(x))
        # x = self.fc_logits(x)
        
        return x
        
    def reparameterize(self, z_mean, z_var):
        if self.distribution == 'normal':
            q_z = torch.distributions.normal.Normal(z_mean, z_var)
            p_z = torch.distributions.normal.Normal(torch.zeros_like(z_mean), torch.ones_like(z_var))
        elif self.distribution == 'vmf':
            q_z = VonMisesFisher(z_mean, z_var)
            p_z = HypersphericalUniform(self.z_dim - 1)
        else:
            raise NotImplemented

        return q_z, p_z
        
    def forward(self, x): 
        z_mean, z_var = self.encode(x)
        q_z, p_z = self.reparameterize(z_mean, z_var)
        z = q_z.rsample()
        x_ = self.decode(z)
        
        return (z_mean, z_var), (q_z, p_z), z, x_
    
    
def log_likelihood(model, x, n=10):
    """
    :param model: model object
    :param optimizer: optimizer object
    :param n: number of MC samples
    :return: MC estimate of log-likelihood
    """

    # z_mean, z_var = model.encode(x.reshape(-1, 784))
    z_mean, z_var = model.encode(x)
    q_z, p_z = model.reparameterize(z_mean, z_var)
    z = q_z.rsample(torch.Size([n]))
    x_mb_ = model.decode(z)

    log_p_z = p_z.log_prob(z).to(device)

    if model.distribution == 'normal':
        log_p_z = log_p_z.sum(-1)
    
    # print(f"x.shape : {x.reshape(-1, 784).repeat(n, 1, 1).shape}")

    # log_p_x_z = -nn.BCEWithLogitsLoss(reduction='none')(x_mb_, x.reshape(-1, 784).repeat((n, 1, 1))).sum(-1)
    log_p_x_z = -nn.MSELoss(reduction='none')(x_mb_.reshape(n, -1, 784), x.reshape(-1, 784).repeat((n, 1, 1))).sum(-1)


    log_q_z_x = q_z.log_prob(z)

    if model.distribution == 'normal':
        log_q_z_x = log_q_z_x.sum(-1)

    return ((log_p_x_z + log_p_z - log_q_z_x).t().logsumexp(-1) - np.log(n)).mean()


def train(model, optimizer, epoch, run):
    model.train()
    train_loss = 0
    for i, (x_mb, y_mb) in enumerate(tqdm(train_loader)):
        x_mb = x_mb.to(device)
        y_mb = y_mb.to(device)
            
        optimizer.zero_grad()
        
        # dynamic binarization
        # x_mb = (x_mb > torch.distributions.Uniform(0, 1).sample(x_mb.shape)).float()
        # _, (q_z, p_z), _, x_mb_ = model(x_mb.reshape(-1, 784))
        _, (q_z, p_z), _, x_mb_ = model(x_mb)

        # loss_recon = nn.BCEWithLogitsLoss(reduction='none')(x_mb_, x_mb.reshape(-1, 784)).sum(-1).mean()
        loss_recon = nn.MSELoss(reduction='none')(x_mb_.reshape(-1, 784), x_mb.reshape(-1, 784)).sum(-1).mean()
        if model.distribution == 'normal':
            loss_KL = torch.distributions.kl.kl_divergence(q_z, p_z).sum(-1).mean()
        elif model.distribution == 'vmf':
            loss_KL = torch.distributions.kl.kl_divergence(q_z, p_z).mean()
        else:
            raise NotImplemented
        loss = loss_recon + loss_KL.to(device)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader.dataset)
    run.log({"Train Loss": train_loss, "Epoch": epoch})
            
            
def test(model, optimizer, epoch, run):
    model.eval()
    test_loss = 0
    test_ll = 0
    step = 0
    print_ = defaultdict(list)
    for x_mb, y_mb in test_loader:
        x_mb = x_mb.to(device)
        y_mb = y_mb.to(device)
        
        # dynamic binarization
        # x_mb = (x_mb > torch.distributions.Uniform(0, 1).sample(x_mb.shape)).float()
        
        # _, (q_z, p_z), _, x_mb_ = model(x_mb.reshape(-1, 784))
        _, (q_z, p_z), _, x_mb_ = model(x_mb)

        
        # print_['recon loss'].append(float(nn.BCEWithLogitsLoss(reduction='none')(x_mb_,
        #     x_mb.reshape(-1, 784)).sum(-1).mean().data))
    
        print_['recon loss'].append(float(nn.MSELoss(reduction='none')(x_mb_.reshape(-1, 784), x_mb.reshape(-1, 784)).sum(-1).mean().data))
        
        if model.distribution == 'normal':
            print_['KL'].append(float(torch.distributions.kl.kl_divergence(q_z, p_z).sum(-1).mean().data))
        elif model.distribution == 'vmf':
            print_['KL'].append(float(torch.distributions.kl.kl_divergence(q_z, p_z).mean().data))
        else:
            raise NotImplemented
        
        print_['ELBO'].append(- print_['recon loss'][-1] - print_['KL'][-1])
        print_['LL'].append(float(log_likelihood(model, x_mb).data))
        loss = - print_['recon loss'][-1] - print_['KL'][-1]
        test_loss += loss * -1
        test_ll += print_['LL'][-1]
        step += 1
    
    print({k: np.mean(v) for k, v in print_.items()})
    test_loss /= step
    test_ll /= step
    run.log({"Test Loss": test_loss, "Test LL": test_ll, "Epoch": epoch})


def visualize(model, output_dir, output_path):
    x_mb, y_mb = next(iter(test_loader))
    # 画像をグリッド形式に変換
    img_grid = torchvision.utils.make_grid(x_mb)
    # テンソルをNumpy配列に変換
    np_img = img_grid.numpy()
    # チャンネルの順序を変換
    np_img = np.transpose(np_img, (1, 2, 0))
    
    # 画像を描画
    plt.figure(figsize=(10, 10))
    plt.imshow(np_img, interpolation='nearest')
    plt.title('MNIST Images')
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'original.png'))

    model.eval()
    # dynamic binarization
    # x_mb = (x_mb > torch.distributions.Uniform(0, 1).sample(x_mb.shape)).float()

    _, (q_z, p_z), _, x_mb_ = model(x_mb.to(device))

    x_mb_ = x_mb_.reshape(-1, 1, 28, 28).cpu().detach()

    # 画像をグリッド形式に変換
    img_grid_recon = torchvision.utils.make_grid(x_mb_)
    # テンソルをNumpy配列に変換
    np_img_recon = img_grid_recon.numpy()
    # チャンネルの順序を変換
    np_img_recon = np.transpose(np_img_recon, (1, 2, 0))
    
    # 画像を描画
    plt.figure(figsize=(10, 10))
    plt.imshow(np_img_recon, interpolation='nearest')
    plt.title('MNIST Images')
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, output_path))


if __name__ == '__main__':



    data_dir = "/workspace/Us_data"
    path_name = "mnist"

    mnist_path = os.path.join(data_dir, "backward_mnist_in_24.npy")
    fmnist_path = os.path.join(data_dir, "backward_fashion_mnist_in_24.npy")

    Us_backward_mnist = np.load(mnist_path)
    Us_backward_fmnist = np.load(fmnist_path)
    print("Successufully Loaded Us_backward data!")

    print(f"Us_backward_mnist.shape: {Us_backward_mnist.shape}")
    print(f"Us_backward_fmnist.shape: {Us_backward_fmnist.shape}")

    # hidden dimension and dimension of latent space
    H_DIM = 128
    Z_DIM = 20
    modelS = ModelVAE(H_DIM, Z_DIM+1, distribution='vmf').to(device)
    checkpoint = torch.load(f'/workspace/hyperspherical_vae/.checkpoints/{path_name}/svae20.pt', map_location=device)
    for key, value in checkpoint.items():
        print(f"{key}: {value.shape}")
    modelS.load_state_dict(torch.load(f"/workspace/hyperspherical_vae/.checkpoints/{path_name}/svae20.pt", map_location=device))

    dim_d = 24
    Late_time = [0, 100, 200, 300, 400, 500, 600, 650, 700, 725, 750, 775, 800, 810, 820, 830, 840, 850, 860, 870, 880, 890, 900, 910, 920, 930, 940, 950, 960, 970, 980, 990, 999]
    num_images = 64
    grid_size = int(np.sqrt(num_images))  # グリッドサイズ (8x8など)

    for i in range(len(Late_time)):
        Us_data = np.load(f"Us_data/backward_{path_name}_in_{dim_d}_{Late_time[i]}.npy")
        Us_data = Us_data[:, :21]
        Us_data = torch.tensor(Us_data).float()
        Us_data = Us_data.to(device)
        decode_data = modelS.decode(Us_data)
        print(decode_data.shape)

        # decode_dataからランダムに64枚の画像を選択
        selected_images = decode_data[:num_images]  # 最初の64枚を選択 (ランダム選択したい場合は np.random.choice を使用)

        # 画像をグリッド形式に配置
        grid_img = vutils.make_grid(selected_images, nrow=grid_size, normalize=True, scale_each=True)

        # 画像を表示
        plt.figure(figsize=(8, 8))
        plt.imshow(grid_img.permute(1, 2, 0).cpu().numpy())  # (C, H, W) -> (H, W, C) に変換
        plt.axis('off')
        plt.title(f'Images at Late_time {Late_time[i]}')

        # 画像を保存
        plt.savefig(f"/workspace/images/{path_name}/decoded_images_{Late_time[i]}.png", bbox_inches='tight', pad_inches=0)
        plt.show()

    print("Finished!")
        

    
    # Us_backward_mnistから、いくつかを取り出し、decodeerに入れる



    