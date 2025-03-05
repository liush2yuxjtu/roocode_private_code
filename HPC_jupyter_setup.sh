 #!/usr/bin/env bash
# 最小可行脚本：在 HPC 上创建 Python 环境并启动 Jupyter Notebook 端口 8890

# 1. 创建并激活 Python 虚拟环境（可换成 conda 或其他工具）
python -m venv hpc_venv
source hpc_venv/bin/activate

# 2. 安装 Jupyter
pip install --upgrade pip
pip install jupyter

# 3. 启动 Jupyter Notebook（无需浏览器）
jupyter notebook --no-browser --port=8890 --ip=0.0.0.0

# 注：
#   在本地执行 SSH 隧道映射，如：
#   ssh -L 8890:localhost:8890 youruser@your_hpc_address
#   然后本地浏览器访问 http://localhost:8890