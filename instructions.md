# 使用说明

## 环境要求
确保你的 Python 环境中已安装以下依赖：
```bash
pip install torch monai numpy matplotlib
```

## 上传到GitHub
```bash
git init
git add -A
git commit -m "Initial commit"
git remote add origin https://github.com/liush2yuxjtu/roocode_private_code.git
git push -u origin main  # 注意：使用 main 分支，不是 master
```

## 在Jupyter服务器上运行

1. 克隆仓库并进入目录
```bash
git clone https://github.com/liush2yuxjtu/roocode_private_code.git
cd roocode_private_code
```

2. 生成训练数据
```python
# 在 Python 脚本中运行
python data_generation.py

# 或在 Jupyter notebook 中运行
%run data_generation.py
```
这将在当前目录下创建 'data' 文件夹，并生成 100 个 3D 高斯球体数据样本（形状：64x64x64，半径范围：16-32）。

3. 训练模型
```python
# 在 Python 脚本中运行
python train.py

# 或在 Jupyter notebook 中运行
%run train.py
```

## 输出说明
- 数据生成：
  - 在 `data/` 目录下生成 .npy 格式的训练数据文件
  - 每个数据样本大小为 64x64x64
  - 总共生成 100 个样本

- 训练过程：
  - 模型检查点保存在 `checkpoints/` 目录
  - 每 10 个 epoch 保存一次模型
  - 训练完成后生成 `generated_samples.png` 显示生成结果

## 代码说明
- `data_generation.py`: 生成 3D 高斯球体数据
- `model.py`: 定义 3D Latent Diffusion Model
- `train.py`: 模型训练和采样脚本