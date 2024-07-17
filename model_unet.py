import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from positional_embeddings import PositionalEmbedding
import dataset
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.nn import functional as F
import model
import utils

class UNet1D(nn.Module):
    def __init__(self, in_channels, out_channels, init_features=32, emb_size=128, time_emb="sinusoidal"):
        super(UNet1D, self).__init__()
        features = init_features
        self.encoder1 = UNet1D._block(in_channels + emb_size, features, name="enc1")
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.encoder2 = UNet1D._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.encoder3 = UNet1D._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.encoder4 = UNet1D._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.bottleneck = UNet1D._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose1d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet1D._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose1d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet1D._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose1d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet1D._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose1d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet1D._block(features * 2, features, name="dec1")

        self.conv = nn.Conv1d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

        # Time embedding layer
        self.time_mlp = PositionalEmbedding(emb_size, time_emb)
        
    def forward(self, x, t):
        # Time embedding
        t_emb = self.time_mlp(t)
        print(t_emb.shape)  # (batch_size, 128)
        
        # Concatenate time embedding to the input
        x = torch.cat((x, t_emb), dim=1)  # (batch_size, in_channels + emb_size)
        print(x.shape)  # (batch_size, in_channels + emb_size)
        
        x = x.unsqueeze(2)  # Adding sequence dimension for Conv1d compatibility: (batch_size, in_channels + emb_size, 1)
        
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        
        output = self.conv(dec1).squeeze(2)  # Remove sequence dimension after Conv1d: (batch_size, out_channels)
        return output

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=features,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(num_features=features),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                in_channels=features,
                out_channels=features,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(num_features=features),
            nn.ReLU(inplace=True),
        )


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
        self.input_mlps = nn.ModuleList([PositionalEmbedding(emb_size, input_emb, scale=25.0) for _ in range(out_dim)])

        concat_size = len(self.time_mlp.layer) + len(self.input_mlps) * len(self.input_mlps[0].layer)
        layers = [nn.Linear(concat_size, hidden_size), nn.GELU()]
        for _ in range(hidden_layers):
            layers.append(Block(hidden_size))
        layers.append(nn.Linear(hidden_size, out_dim))
        self.joint_mlp = nn.Sequential(*layers)

    def forward(self, x, t, device):
        # print(f"x:shape: {x.shape}")
        batch_size, dim = x.shape
        x_embs = [self.input_mlps[i](x[:, i].to(device)) for i in range(dim)]
        # print(x_embs[0].shape)
        t_emb = self.time_mlp(t.to(device))
        # print(t_emb.shape)
        x = torch.cat(x_embs + [t_emb], dim=-1)
        # print(x.shape)
        # print("ok")
        x = self.joint_mlp(x)
        return x

if __name__ == '__main__':
    data = dataset.circle_dataset(n=1000, r=7).numpy()
    tensor_data = TensorDataset(torch.tensor(data))
    dataloader = DataLoader(tensor_data, batch_size=32, shuffle=True, drop_last=True)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    noise_scheduler = model.NoiseScheduler(num_timesteps=1000, beta_schedule="linear")

    model2 = MLP(hidden_size=128, hidden_layers=3, emb_size=128, time_emb="sinusoidal", input_emb="sinusoidal", out_dim=7)


    # バッチサイズ、入力チャネル数、シーケンス長さ、出力チャネル数の設定
    batch_size = 8
    in_channels = 7
    # sequence_length = 64
    out_channels = 1

    for step, batch in enumerate(dataloader):
        batch = batch[0].to(device)
        # print(batch.shape)
        dim_d = batch.shape[-1]
        noise = torch.randn(batch.shape).to(device)
        timesteps = torch.randint(0, noise_scheduler.num_timesteps, (batch.shape[0],)).long()
        # print(timesteps.shape)
        noisy = noise_scheduler.add_noise(batch, noise, timesteps)
        # print(noisy.shape)
        # print(timesteps.shape)
        noise_pred = model2(noisy, timesteps, device=device)
        # print(noise_pred)
        break


    # UNet1Dモデルのインスタンスを作成
    model1 = UNet1D(in_channels=in_channels, out_channels=out_channels)

    # ランダムな入力データと時間ステップを生成
    # x = torch.randn(batch_size, sequence_length)
    x = next(iter(dataloader))[0]
    print(x.shape) #(32, 7)
    t = torch.randint(0, 1000, (x.shape[0],)).long()
    print(t.shape) # (32)

    
    

    # # モデルの出力を計算
    # y2 = model2(x, t, device)
    y = model1(x, t)
    # 出力の形状を表示
    print(f"Output shape: {y.shape}")
