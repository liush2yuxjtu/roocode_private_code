import torch
from torch.utils.data import TensorDataset, DataLoader
from model import UNet3D, DiffusionModel
from data_generation import generate_dataset, visualize_sample
import matplotlib.pyplot as plt

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 生成数据集
    print("生成训练数据...")
    num_samples = 100
    data = generate_dataset(num_samples=num_samples)
    data_tensor = torch.from_numpy(data).float()
    
    # 创建数据加载器
    dataset = TensorDataset(data_tensor)
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)

    # 初始化模型
    print("初始化模型...")
    model = UNet3D(in_channels=1, base_channels=32)
    diffusion = DiffusionModel(
        model=model,
        betas=(1e-4, 0.02),
        n_steps=1000,
        device=device
    )

    # 训练模型
    print("开始训练...")
    epochs = 100
    train(diffusion, train_loader, epochs, lr=1e-4, device=device)

    # 生成样本
    print("生成示例...")
    samples = diffusion.sample(batch_size=1)
    sample = samples[0, 0].cpu().numpy()  # 移除批次和通道维度

    # 可视化结果
    print("可视化结果...")
    plt.figure(figsize=(15, 5))
    
    # 显示原始数据
    plt.subplot(121)
    center_slice = data[0, data.shape[1]//2]
    plt.imshow(center_slice)
    plt.title("原始数据 (中心切片)")
    
    # 显示生成的数据
    plt.subplot(122)
    gen_center_slice = sample[sample.shape[0]//2]
    plt.imshow(gen_center_slice)
    plt.title("生成的数据 (中心切片)")
    
    plt.savefig('results.png')
    plt.close()
    
    # 保存模型
    print("保存模型...")
    torch.save(diffusion.model.state_dict(), 'diffusion_model.pth')
    print("完成！")

if __name__ == "__main__":
    main()