# 3D Latent Diffusion Model with Gaussian Sphere Data

这个项目实现了一个3D潜在扩散模型(LDM)，使用3D高斯球形作为虚拟训练数据。

## 数据生成

生成的3D高斯球具有以下特性：
- 数据形状: (64, 64, 64)
- 球心范围: (r, 64-r)，其中r为球半径
- 半径范围: 16 < r < 32

## 项目结构

```
.
├── README.md
├── requirements.txt
├── data_generation.py  # 生成3D高斯球数据
├── model.py           # 3D UNet和扩散模型实现
└── train.py          # 训练和评估脚本
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

1. 生成数据并查看示例：
```bash
python data_generation.py
```

2. 训练模型：
```bash
python train.py
```

训练完成后会生成：
- `results.png`: 对比原始数据和生成数据的可视化
- `diffusion_model.pth`: 保存的模型权重

## 模型架构

- 使用3D UNet作为骨干网络
- 实现了完整的扩散过程：
  - 前向扩散过程（添加噪声）
  - 反向扩散过程（去噪）
- 使用线性β调度

## 环境要求

- Python 3.8+
- PyTorch 1.8+
- CUDA支持（推荐）

## 参考

基于MONAI教程修改：https://github.com/Project-MONAI/tutorials/tree/main/generation/3d_ldm