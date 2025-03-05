### 可重用提示词规范 (v2.0)

#### 1. 代码架构规范
**开发流程**  
```markdown
- 分阶段开发：  
  `DataPipeline → ModelProto → TrainCore → EvalKit → Deployment`
- 阶段间接口：  
  `Dataset → DataLoader → Model → Trainer → Validator → Exporter`
- 模块化设计：  
  ```bash
  src/
  ├── data/          # 数据管道
  ├── models/        # 模型架构
  ├── engine/        # 训练引擎
  ├── utils/         # 可视化/日志
  └── configs/       # 超参数配置
class BaseDataset(torch.utils.data.Dataset):
    @abstractmethod
    def generate_sample(self) -> Tuple[Tensor, dict]:
        """必须返回 (数据张量, 元数据)"""

#### 2. 测试规范
**测试金字塔**  
```markdown
- 单元测试：数据生成逻辑/模型基础算子
  ```python
  def test_sphere_generation():
      sample = dataset[0]
      assert sample[0].shape == (64,64,64)
集成测试：DataLoader吞吐量/GPU内存占用

E2E测试：完整训练流水线验证

**数据完整性校验**  
```markdown
- 分布检查：均值和方差是否符合预设范围
- 异常检测：空样本/NaN值/数据类型校验
- 可视化协议：  
  ```python
  def render_3d_slices(data: Tensor) -> Figure:
      """返回包含XY/XZ/YZ切面的matplotlib图像"""
3. 训练工程化
配置管理
# configs/train.yaml
defaults:
  - base_cfg
  - _self_

train:
  batch_size: 8
  epochs: 100
  checkpoint:
    save_every: 1  # epoch
    max_keep: 5     # 最多保留的ckpt数
日志规范

markdown
复制
- 必须记录：
  ```python
  {
      "timestamp": "ISO8601",
      "epoch": int,
      "loss": {"train": float, "val": float},
      "metrics": {"SSIM": float, "PSNR": float},
      "hardware": {"GPU_mem": "8/24GB"} 
  }
可视化要求：

TensorBoard/PyTorch Lightning格式

复制

#### 4. 环境适配
**跨平台支持**  
```markdown
- 环境检测逻辑：  
  ```python
  def detect_env() -> Dict:
      return {
          "OS": platform.system(),
          "Python": sys.version,
          "CUDA": torch.version.cuda,
          "Shell": os.environ.get('SHELL', 'CMD/Powershell')
      }
依赖管理：

bash
复制
# 层级化安装 (基础→训练→推理)
pip install -r requirements/core.txt
pip install -r requirements/train.txt  # 包含xformers等优化库
5. 自动化协议
持续集成检查

yaml
复制
# .github/workflows/ci.yml
jobs:
  test:
    runs-on: [ubuntu-latest, windows-latest]
    steps:
      - name: 代码静态检查
        run: flake8 src --max-line-length=120
      - name: 训练冒烟测试
        run: python -m pytest tests/train_smoke_test.py -v
文档自动化

markdown
复制
- API文档生成：  
  ```bash
  pdoc --html src -o docs/ --force
训练记录自动更新README：

自动生成的训练曲线示例

复制

### 部署增强
```markdown
1. **Colab快速启动**  
   ```python
   # 环境准备单元
   !git clone {repo_url} && cd {repo_dir}
   !pip install -r requirements.txt --quiet
   from src.utils.env import setup_colab
   setup_colab(enable_gpu=True)  # 自动检测GPU并配置环境

2. **生产级部署**  
   ```bash
   # 支持Docker部署
   docker build -t 3dldm .
   docker run --gpus all -v ./data:/app/data 3dldm train --config configs/prod.yaml
监控集成

python
复制
# Prometheus指标端点
@app.route('/metrics')
def training_metrics():
    return generate_latest(REGISTRY)
复制

---

### 优化要点说明
1. **分层架构**：将原始平面结构改为金字塔式规范，适配不同规模项目
2. **接口契约**：明确定义模块间交互协议，提升代码可维护性
3. **自动化增强**：新增CI/CD和文档自动化要求，降低协作成本
4. **生产就绪**：增加Docker部署和监控指标输出能力
5. **环境感知**：通过动态环境检测实现跨平台兼容

