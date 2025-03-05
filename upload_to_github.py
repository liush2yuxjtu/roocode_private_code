import os
import subprocess

def run_command(command):
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = process.communicate()
    return output.decode(), error.decode()

def main():
    # 初始化git仓库
    print("初始化Git仓库...")
    run_command("git init")
    
    # 添加文件
    print("添加文件...")
    run_command("git add gpu_demo.py instructions.md")
    
    # 提交更改
    print("提交更改...")
    run_command('git commit -m "添加GPU示例代码和使用说明"')
    
    # 添加远程仓库
    repo_url = input("请输入GitHub仓库URL (格式: https://github.com/username/repo.git): ")
    print("添加远程仓库...")
    run_command(f"git remote add origin {repo_url}")
    
    # 推送到GitHub
    print("推送到GitHub...")
    output, error = run_command("git push -u origin master")
    
    if error:
        print("错误:", error)
    else:
        print("成功上传到GitHub!")
        print("仓库URL:", repo_url)

if __name__ == "__main__":
    main()