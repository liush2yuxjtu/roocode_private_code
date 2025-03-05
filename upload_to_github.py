import os
import subprocess

def run_command(command):
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = process.communicate()
    print("输出:", output.decode())
    print("错误:", error.decode())
    return output.decode(), error.decode()

def main():
    try:
        # 配置Git用户信息
        print("配置Git用户信息...")
        run_command('git config --global user.email "liush2yux@163.com"')
        run_command('git config --global user.name "liush2yuxjtu"')
        
        # 初始化git仓库
        print("\n初始化Git仓库...")
        run_command("git init")
        
        # 添加文件
        print("添加文件...")
        run_command("git add -A")
        
        # 提交更改
        print("提交更改...")
        run_command('git commit -m "添加GPU示例代码和使用说明"')
        
        # 设置远程仓库
        repo_name = "gpu_demo"
        repo_url = f"https://github.com/liush2yuxjtu/{repo_name}.git"
        print("添加远程仓库:", repo_url)
        run_command("git remote remove origin")  # 移除已存在的origin
        run_command(f"git remote add origin {repo_url}")
        
        # 创建main分支并推送
        print("推送到GitHub...")
        run_command("git branch -M main")
        run_command("git push -u origin main")
        
        print("完成！")
        print("仓库URL:", repo_url)
            
    except Exception as e:
        print("发生错误:", str(e))

if __name__ == "__main__":
    main()