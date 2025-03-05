import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def create_3d_gaussian_sphere(size=64, r_min=16, r_max=32):
    """
    创建3D高斯球形数据
    size: 体素大小
    r_min: 最小半径
    r_max: 最大半径
    """
    # 随机生成球心和半径
    r = np.random.uniform(r_min, r_max)
    center_range = (r, size-r)
    center = np.random.uniform(center_range[0], center_range[1], 3)
    
    # 创建网格点
    x, y, z = np.mgrid[0:size, 0:size, 0:size]
    
    # 计算到球心的距离
    dist = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
    
    # 创建高斯球
    sigma = r/3  # 使高斯分布在球半径处衰减到约0.01
    sphere = np.exp(-(dist**2)/(2*sigma**2))
    
    return sphere

def generate_dataset(num_samples=100, size=64, r_min=16, r_max=32):
    """
    生成数据集
    num_samples: 样本数量
    """
    dataset = []
    for _ in tqdm(range(num_samples)):
        sample = create_3d_gaussian_sphere(size, r_min, r_max)
        dataset.append(sample)
    return np.array(dataset)

def visualize_sample(sample):
    """
    可视化3D数据的中心切片
    """
    center_slice = sample.shape[0]//2
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(sample[center_slice, :, :])
    axes[0].set_title('XY平面')
    axes[1].imshow(sample[:, center_slice, :])
    axes[1].set_title('XZ平面')
    axes[2].imshow(sample[:, :, center_slice])
    axes[2].set_title('YZ平面')
    plt.show()

if __name__ == "__main__":
    # 生成示例数据
    sample = create_3d_gaussian_sphere()
    print("数据形状:", sample.shape)
    visualize_sample(sample)
    
    # 生成数据集
    dataset = generate_dataset(num_samples=5)
    print("数据集形状:", dataset.shape)