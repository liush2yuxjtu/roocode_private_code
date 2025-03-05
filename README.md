# 3D Latent Diffusion Model with Gaussian Sphere Data

本项目使用 MONAI 框架实现了一个 3D Latent Diffusion Model，并使用 3D 高斯球体作为虚拟训练数据。

## 项目结构

```
.
├── data_generation.py  # 生成 3D 高斯球体数据
├── model.py           # 模型定义（DiffusionModelUNet 和 DDPM）
├── train.py          # 训练脚本
└── README.md         # 项目文档
```

## 数据生成

生成的虚拟数据是 3D 高斯球体，具有以下特征：
- 形状：64x64x64
- 半径范围：16-32
- 条件：r < center < 64-r

生成数据集：
```bash
python data_generation.py
```

## 训练模型

训练脚本将：
1. 加载生成的数据集
2. 初始化 3D Latent Diffusion Model
3. 训练模型并定期保存检查点
4. 生成示例并可视化结果

运行训练：
```bash
python train.py
```

## 依赖

- MONAI
- PyTorch
- NumPy
- Matplotlib

## 模型架构

该模型使用 MONAI 的 DiffusionModelUNet，具有以下特征：
- 空间维度：3D
- 输入/输出通道：1
- 特征通道：[32, 64, 128, 256]
- 注意力层级：[False, True, True, True]

扩散过程使用 DDPM (Denoising Diffusion Probabilistic Models) 实现，具有以下参数：
- 时间步数：1000
- Beta 调度：线性
- Beta 范围：1e-4 到 0.02