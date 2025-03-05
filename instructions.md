# 使用说明

## 上传到GitHub
```bash
git init
git add -A
git commit -m "Initial commit"
git remote add origin https://github.com/liush2yuxjtu/roocode_private_code.git
git push -u origin master
```

## 在Jupyter服务器上克隆并运行
```bash
git clone https://github.com/liush2yuxjtu/roocode_private_code.git
cd roocode_private_code
python gpu_demo.py