import paramiko
from scp import SCPClient
import os
import time

# ==============================================================
# 通用云服务器自动化热部署脚本 (适配 Ubuntu 22.04 / 24.04 等脱壳环境)
# ==============================================================

# 请修改下方的服务器连接信息
HOST = '101.245.110.98'
USER = 'root'
PASSWORD = 'LYH@2005'
REMOTE_PATH = '/opt/cloud_storage_gateway'
LOCAL_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 核心的远端生产环境变量注入
PRODUCTION_ENV_CONFIG = """
INTERNAL_API_KEY=your_production_secret_key
OBS_BUCKET=medical-model-data
OBS_ENDPOINT=obs.cn-southwest-2.myhuaweicloud.com
IAM_REGION=cn-southwest-2
USE_ECS_AGENCY=True
"""

def deploy():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    print(f"[*] 正在通过 SSH 登录目标服务器: {HOST}")
    client.connect(HOST, 22, USER, PASSWORD)
    
    # 1. 基础系统包检查与安装保护
    print("[*] 正在准备底层环境包 (python3-venv)")
    client.exec_command("apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y python3-venv python3-pip")
    time.sleep(5) # 给 apt 缓存释放留一点时间
    
    # 2. 上传工程文件
    print("[*] 正在向服务器 SCP 全量注入最新网关代码...")
    client.exec_command(f"rm -rf {REMOTE_PATH}/*")
    client.exec_command(f"mkdir -p {REMOTE_PATH}/services {REMOTE_PATH}/routers")
    
    with SCPClient(client.get_transport()) as scp:
        root_files = ['main.py', 'config.py', 'schemas.py', 'dependencies.py', 'requirements.txt']
        for f in root_files:
            abs_path = os.path.join(LOCAL_PATH, f)
            if os.path.exists(abs_path):
                scp.put(abs_path, remote_path=REMOTE_PATH)
        
        for folder in ['services', 'routers']:
            folder_path = os.path.join(LOCAL_PATH, folder)
            if os.path.exists(folder_path):
                scp.put(folder_path, remote_path=REMOTE_PATH, recursive=True)
                
    # 3. 环境变量下发
    cmd_write_env = f"cat << 'EOF' > {REMOTE_PATH}/.env\n{PRODUCTION_ENV_CONFIG}EOF\n"
    client.exec_command(cmd_write_env)

    # 4. Pip 高速安装
    print("[*] 在云端拉起 VENV 并安装镜像依赖包...")
    # 针对国内服务器挂载加速镜像环境提升构建速度防 timeout
    _, stdout, stderr = client.exec_command(
        f"cd {REMOTE_PATH} && "
        "python3 -m venv venv && "
        "./venv/bin/pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple"
    )
    print(" >>> " + stdout.read().decode().strip().replace("\\n", "\n >>> "))
    if err := stderr.read().decode().strip():
        print(" [!] Pip 警告日志:", err)

    # 5. Uvicorn 进程守护拉起
    print("[*] 清洗历史残留进程, 重建 Systemd Uvicorn...")
    client.exec_command("fuser -k 8000/tcp")
    client.exec_command("systemctl stop cloudgateway.service")
    time.sleep(2)
    
    # 使用瞬态 systemd 使得 FastAPI 以最高稳定性脱离 SSH Session 的挂断生命周期保持长活
    bootstrap_cmd = f"systemd-run --unit=cloudgateway.service --remain-after-exit /bin/bash -c 'cd {REMOTE_PATH} && ./venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000'"
    client.exec_command(bootstrap_cmd)
    
    print("[*] 自动部署流程全绿完成！网关系统已上云并上线服务。")
    client.close()

if __name__ == '__main__':
    deploy()
