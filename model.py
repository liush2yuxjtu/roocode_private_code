import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from einops import rearrange
import numpy as np
from tqdm import tqdm

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(),
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU()
        )
        self.downsample = nn.MaxPool3d(2)
        
    def forward(self, x):
        skip = self.conv(x)
        return self.downsample(skip), skip

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.ConvTranspose3d(in_channels, in_channels//2, 2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(),
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU()
        )
        
    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class UNet3D(nn.Module):
    def __init__(self, in_channels=1, base_channels=32):
        super().__init__()
        self.inc = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, 3, padding=1),
            nn.BatchNorm3d(base_channels),
            nn.LeakyReLU()
        )
        
        self.down1 = DownBlock(base_channels, base_channels*2)
        self.down2 = DownBlock(base_channels*2, base_channels*4)
        self.down3 = DownBlock(base_channels*4, base_channels*8)
        
        self.up1 = UpBlock(base_channels*8, base_channels*4)
        self.up2 = UpBlock(base_channels*4, base_channels*2)
        self.up3 = UpBlock(base_channels*2, base_channels)
        
        self.outc = nn.Conv3d(base_channels, in_channels, 1)
        
    def forward(self, x):
        x = self.inc(x)
        x1, skip1 = self.down1(x)
        x2, skip2 = self.down2(x1)
        x3, skip3 = self.down3(x2)
        
        x = self.up1(x3, skip3)
        x = self.up2(x, skip2)
        x = self.up3(x, skip1)
        return self.outc(x)

class DiffusionModel:
    def __init__(self, model, betas=(1e-4, 0.02), n_steps=1000, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.n_steps = n_steps
        
        # 创建噪声调度
        self.betas = torch.linspace(betas[0], betas[1], n_steps).to(device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
    
    def add_noise(self, x, t):
        """向输入添加t步噪声"""
        alpha_bar = self.alpha_bars[t]
        
        # 重复alpha_bar以匹配x的形状
        alpha_bar = alpha_bar.reshape(-1, 1, 1, 1, 1)
        
        eps = torch.randn_like(x)
        return torch.sqrt(alpha_bar) * x + torch.sqrt(1 - alpha_bar) * eps, eps
    
    def train_step(self, x, optimizer):
        """单步训练"""
        self.model.train()
        
        # 随机选择时间步
        t = torch.randint(0, self.n_steps, (x.shape[0],)).to(self.device)
        
        # 添加噪声
        noisy_x, target_noise = self.add_noise(x, t)
        
        # 预测噪声
        pred_noise = self.model(noisy_x)
        
        # 计算损失
        loss = F.mse_loss(pred_noise, target_noise)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    @torch.no_grad()
    def sample(self, batch_size=1, size=64):
        """从噪声生成样本"""
        self.model.eval()
        
        # 从标准正态分布采样
        x = torch.randn(batch_size, 1, size, size, size).to(self.device)
        
        # 逐步去噪
        for t in tqdm(range(self.n_steps-1, -1, -1), desc="Sampling"):
            # 预测噪声
            eps = self.model(x)
            
            alpha = self.alphas[t]
            alpha_bar = self.alpha_bars[t]
            beta = self.betas[t]
            
            # 无噪声估计
            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = 0
                
            x = 1/torch.sqrt(alpha) * (x - (beta/torch.sqrt(1-alpha_bar)) * eps) + torch.sqrt(beta) * noise
            
        return x

def train(diffusion, train_loader, epochs, lr=1e-4, device='cuda'):
    """训练函数"""
    optimizer = Adam(diffusion.model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            batch = batch.unsqueeze(1).float().to(device)  # 添加通道维度
            loss = diffusion.train_step(batch, optimizer)
            total_loss += loss
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")