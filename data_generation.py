import numpy as np

def generate_gaussian_sphere_data(shape=(64, 64, 64), r_min=16, r_max=32):
    """
    生成 3D 高斯球虚拟数据。

    Args:
        shape (tuple): 数据形状 (height, width, depth).
        r_min (int): 最小半径.
        r_max (int): 最大半径.

    Returns:
        numpy.ndarray: 生成的 3D 数据.
    """
    data = np.zeros(shape, dtype=np.float32)
    center = np.array([s // 2 for s in shape])  # 中心点坐标
    r = np.random.randint(r_min, r_max + 1)  # 随机半径

    for x in range(shape[0]):
        for y in range(shape[1]):
            for z in range(shape[2]):
                dist = np.sqrt(np.sum((np.array([x, y, z]) - center) ** 2))
                if dist < r and dist > 0:  # 修改：dist > 0 避免中心点为0
                    data[x, y, z] = np.exp(-dist**2 / (2 * (r/3)**2))  # 高斯分布，半径1/3处衰减到峰值的约60%

    return data

def generate_dataset(num_samples=100, save_dir='data'):
    """
    生成数据集并保存。

    Args:
        num_samples (int): 要生成的样本数量
        save_dir (str): 保存数据的目录
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    for i in range(num_samples):
        data = generate_gaussian_sphere_data()
        save_path = os.path.join(save_dir, f'sample_{i:04d}.npy')
        np.save(save_path, data)
        if i % 10 == 0:
            print(f'Generated {i+1}/{num_samples} samples')

if __name__ == '__main__':
    # 生成示例数据集
    generate_dataset(num_samples=100)
    print("Data generation complete.")