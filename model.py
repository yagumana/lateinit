import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from positional_embeddings import PositionalEmbedding
import utils
import sys

sys.path.append('./denoising-diffusion-pytorch')

from denoising_diffusion_pytorch.denoising_diffusion_pytorch_1d import Unet1D, GaussianDiffusion1D, Trainer1D, Dataset1D

class VP_SDE_dim:
    """ A class used to define a Stochastic Differential Equation (SDE)."""
    def __init__(self,  beta_min=0.1, beta_max=20, N=1000, T=1, sigma=1):
        """
        Initializes the parameters of the SDE.
        Args:
            beta_min, beta_max (float): The minimum and maximum values of beta.
            N (int): The total number of steps in the Euler Maruyama method.
            T (float): The total time.
            sigma (float): The standard deviation of the distribution.
        """
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.N = N
        self.T = T
        self.sigma = sigma

    def f(self, x, t):
        """ Defines the drift term of the SDE. """
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        return -0.5 * beta_t * x

    def g(self, t):
        """ Defines the diffusion term of the SDE.m """
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        return self.sigma * np.sqrt(beta_t)



def euler_maruyama_dim(u0, SDE):
    """
    Implements the Euler Maruyama numerical approximation to the SDE.
    Args:
        u0 (np.array): The initial state.
        SDE (object): An instance of the VP_SDE class.
    """

    # Creates time partition based on the SDE configuration
    T=SDE.T; N=SDE.N
    dt = T/N

    # Initial condition
    batch_num, dim = u0.shape
    U = np.zeros((N+1, batch_num, dim))
    U[0] = u0

    # Iterate to over the whole time partition, N steps.
    for n in range(N):
        tn = n*dt
        z = np.random.randn(batch_num, dim)
        dW = np.sqrt(dt)*z

        U[n+1] = U[n] + dt*SDE.f(U[n], tn) + SDE.g(tn) * dW
    return U


class Block(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()

        self.ff = nn.Linear(input_size, input_size)
        self.ff2 = nn.Linear(input_size, output_size)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor):
        x = x + self.act(self.ff(x))
        x = self.act(self.ff2(x))
        return x


    
class MLP(nn.Module):
    def __init__(self, hidden_size: int = 128, hidden_layers: int = 3, emb_size: int = 128, out_dim: int = 2,
                 time_emb: str = "sinusoidal", input_emb: str = "sinusoidal"):
        super().__init__()

        self.hidden_size = hidden_size
        self.time_mlp = PositionalEmbedding(emb_size, time_emb)
        self.input_mlps = nn.ModuleList([PositionalEmbedding(emb_size, input_emb, scale=25.0) for _ in range(out_dim)])

        concat_size = len(self.time_mlp.layer) + len(self.input_mlps) * len(self.input_mlps[0].layer)
        
        layers = [nn.Linear(concat_size, concat_size//2), nn.GELU()]
        # for _ in range(hidden_layers):
        layers.append(Block(concat_size//2, concat_size//4))
        layers.append(Block(concat_size//4, concat_size//8))
        layers.append(Block(concat_size//8, concat_size//16))
        layers.append(nn.Linear(concat_size//16, out_dim))
        self.joint_mlp = nn.Sequential(*layers)

    def forward(self, x, t):
        # print(f"x:shape: {x.shape}")
        device = x.device
        batch_size, dim = x.shape
        x_embs = [self.input_mlps[i](x[:, i].to(device)) for i in range(dim)]
        t_emb = self.time_mlp(t.to(device))
        x = torch.cat(x_embs + [t_emb], dim=-1)
        x = self.joint_mlp(x)
        return x
    

class NoiseScheduler():
    def __init__(self,
                 num_timesteps=1000,
                 beta_start=0.0001,
                 beta_end=0.02,
                 beta_schedule="linear"):

        self.num_timesteps = num_timesteps
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if beta_schedule == "linear":
            self.betas = torch.linspace(
                beta_start, beta_end, num_timesteps, dtype=torch.float32).to(device)
        elif beta_schedule == "quadratic":
            self.betas = (torch.linspace(
                beta_start ** 0.5, beta_end ** 0.5, num_timesteps, dtype=torch.float32) ** 2).to(device)
        

        self.alphas = (1.0 - self.betas).to(device)
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0).to(device)
        self.alphas_cumprod_prev = F.pad(
            self.alphas_cumprod[:-1], (1, 0), value=1.).to(device)

        # required for self.add_noise
        self.sqrt_alphas_cumprod = (self.alphas_cumprod ** 0.5).to(device)
        self.sqrt_one_minus_alphas_cumprod = ((1 - self.alphas_cumprod) ** 0.5).to(device)

        # required for reconstruct_x0
        self.sqrt_inv_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod).to(device)
        self.sqrt_inv_alphas_cumprod_minus_one = torch.sqrt(
            1 / self.alphas_cumprod - 1).to(device)

        # required for q_posterior
        # print(self.betas.device)
        self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)).to(device)
        self.posterior_mean_coef2 = ((1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)).to(device)

    def reconstruct_x0(self, x_t, t, noise):
        device = x_t.device
        s1 = self.sqrt_inv_alphas_cumprod[t].to(device)
        s2 = self.sqrt_inv_alphas_cumprod_minus_one[t].to(device)
        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)
        return s1 * x_t - s2 * noise

    def q_posterior(self, x_0, x_t, t):
        device = x_0.device
        s1 = self.posterior_mean_coef1[t].to(device)
        s2 = self.posterior_mean_coef2[t].to(device)
        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)
        mu = s1 * x_0 + s2 * x_t
        return mu

    def get_variance(self, t):
        if t == 0:
            return torch.tensor(0.0).to(self.betas.device)

        variance = self.betas[t] * (1. - self.alphas_cumprod_prev[t]) / (1. - self.alphas_cumprod[t]).to(self.betas.device)
        variance = variance.clip(1e-20)
        return variance

    def step(self, model_output, timestep, sample):
        device = model_output.device
        t = timestep.to(device)
        pred_original_sample = self.reconstruct_x0(sample, t, model_output)
        pred_prev_sample = self.q_posterior(pred_original_sample, sample, t)

        variance = torch.tensor(0.0).to(device)
        if t > 0:
            noise = torch.randn_like(model_output).to(device)
            variance = (self.get_variance(t) ** 0.5).to(device) * noise

        pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample

    def add_noise(self, x_start, x_noise, timesteps):
        device = x_start.device
        s1 = self.sqrt_alphas_cumprod[timesteps].to(device)
        s2 = self.sqrt_one_minus_alphas_cumprod[timesteps].to(device)

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
    

def train(nn_model, dataloader, noise_scheduler, optimizer, device, N_epoch=100, wandb=None):
    global_step = 0
    frames = []
    losses = []

    for epoch in tqdm(range(N_epoch)):
        nn_model.train()
        total_loss = 0
        for step, batch in enumerate(dataloader):
            batch = batch[0].to(device)
            dim_d = batch.shape[-1]

            noise = torch.randn(batch.shape).to(device)
            timesteps = torch.randint(0, noise_scheduler.num_timesteps, (batch.shape[0],)).long().to(device)
            noisy = noise_scheduler.add_noise(batch, noise, timesteps)

            # print(f"debug: {noisy.shape}, {timesteps.shape}")
            noisy = noisy.unsqueeze(1).to(device)
            # print(f"debug: {noisy.shape}, {timesteps.shape}")

            noise_pred = nn_model(noisy, timesteps)
            # print(noise_pred.shape)
            # print(noise.shape)
            noise = noise.unsqueeze(1)
            # sys.exit()
            loss = F.mse_loss(noise_pred, noise)
            loss.backward()

            nn.utils.clip_grad_norm_(nn_model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.detach().item()
        avg_loss = total_loss / len(dataloader)
        losses.append(avg_loss)
        if wandb:
            wandb.log({'epoch': epoch, 'loss': avg_loss})
        
        # エポックごとにlossを出力
        print(f"Epoch [{epoch+1}/{N_epoch}], Loss: {avg_loss:.4f}")

    return losses
    
def get_denoise_data(nn_model, noise_scheduler, device, N=1000, batch_size=1000, dim = 7, late=0):
    """
    訓練済みのdiffusionモデル(nn_model)を用いて、dataがgauss noiseからN stepかけてdataが生成される過程を、data_listに保存して返す.
    """
    nn_model.eval()
    sample = torch.randn(batch_size, 1, dim).to(device)
    # timesteps = list(range(len(noise_scheduler)))[::-1]
    timesteps = list(range(len(noise_scheduler)-late))[::-1]
    data_list = np.zeros((len(timesteps), batch_size, dim))
    for i, t in enumerate(tqdm(timesteps)):
        t = torch.from_numpy(np.repeat(t, batch_size)).long().to(device)
        with torch.no_grad():
            residual = nn_model(sample, t)
        sample.to(device)
        sample = noise_scheduler.step(residual, t[0], sample)
        data_list[i] = sample.squeeze(1).cpu().numpy()
    return data_list

def load_experiments(device=None, roop=5, batch_size=1000, Time_step = 1000, dim_d=7, late=0, path_name="sphere", denoise_model="MLP"):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_list_stacked = []
    for i in range(roop):
        utils.set_seed(i)
        if denoise_model == "MLP":
            nn_model = MLP(hidden_size=128, hidden_layers=3, emb_size=128, time_emb="sinusoidal", input_emb="sinusoidal", out_dim=dim_d).to(device)
        elif denoise_model == "Unet":
            nn_model = Unet1D(dim=dim_d, channels=1).to(device)
        # nn_model = MLP(hidden_size=128, hidden_layers=3, emb_size=128, time_emb="sinusoidal", input_emb="sinusoidal", out_dim=dim_d).to(device)
        noise_scheduler = NoiseScheduler(num_timesteps=Time_step, beta_schedule="linear")

        # load pretrained weights
        print(f"loading pretrained weight_{i}...")
        nn_model.load_state_dict(torch.load(f'/workspace/weights/{path_name}_in_r{dim_d}_{i}.pth', map_location=device))
        
        if late==0:
            data_list = get_denoise_data(nn_model, noise_scheduler, batch_size=batch_size, dim=dim_d, device=device, late=late)
        else:
            data_list = get_denoise_data(nn_model, noise_scheduler, batch_size=batch_size, dim=dim_d, device=device, late=late)
        data_list_stacked.append(data_list)

    return data_list_stacked
