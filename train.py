import torch
from torch.utils.data import Dataset, DataLoader
from model import get_ldm_model
import os
import numpy as np
from torch.optim import Adam
import matplotlib.pyplot as plt
from pathlib import Path

class GaussianSphereDataset(Dataset):
    def __init__(self, data_dir):
        self.data_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npy')])
        self.data_dir = data_dir

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        data_path = os.path.join(self.data_dir, self.data_files[idx])
        data = np.load(data_path)
        data = torch.from_numpy(data).float()
        data = data.unsqueeze(0)  # 添加通道维度
        return data

def train_diffusion(
    diffusion_model,
    train_loader,
    num_epochs,
    device,
    learning_rate=1e-4,
    save_interval=10
):
    optimizer = Adam(diffusion_model.parameters(), lr=learning_rate)
    
    # 创建保存目录
    save_dir = Path("checkpoints")
    save_dir.mkdir(exist_ok=True)
    
    diffusion_model.train()
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0
        
        for batch in train_loader:
            x = batch.to(device)
            
            optimizer.zero_grad()
            loss = diffusion_model(x)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        # 计算平均损失
        avg_loss = epoch_loss / num_batches
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")
        
        # 定期保存模型
        if (epoch + 1) % save_interval == 0:
            torch.save(
                diffusion_model.state_dict(),
                save_dir / f"diffusion_model_epoch_{epoch+1}.pth"
            )

def visualize_samples(diffusion_model, device, num_samples=4):
    diffusion_model.eval()
    with torch.no_grad():
        samples = diffusion_model.sample(
            num_samples=num_samples,
            device=device
        )
    
    # 创建图表
    fig, axes = plt.subplots(2, num_samples, figsize=(4*num_samples, 8))
    
    for i, sample in enumerate(samples):
        # 显示中心 xy 切片
        axes[0, i].imshow(sample[0, sample.shape[2]//2].cpu())
        axes[0, i].set_title(f'Sample {i+1} (XY)')
        axes[0, i].axis('off')
        
        # 显示中心 yz 切片
        axes[1, i].imshow(sample[0, :, :, sample.shape[3]//2].cpu())
        axes[1, i].set_title(f'Sample {i+1} (YZ)')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('generated_samples.png')
    plt.close()

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 初始化数据集
    print("Loading dataset...")
    dataset = GaussianSphereDataset('data')
    train_loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4
    )
    
    # 初始化模型
    print("Initializing model...")
    diffusion_model = get_ldm_model()
    diffusion_model = diffusion_model.to(device)
    
    # 训练模型
    print("Starting training...")
    train_diffusion(
        diffusion_model,
        train_loader,
        num_epochs=100,
        device=device,
        learning_rate=1e-4
    )
    
    # 生成和可视化样本
    print("Generating samples...")
    visualize_samples(diffusion_model, device)
    
    print("Training complete!")

if __name__ == "__main__":
    main()