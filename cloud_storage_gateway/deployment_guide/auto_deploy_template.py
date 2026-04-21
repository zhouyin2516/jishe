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
INTERNAL_API_KEY=test_secret_key
OBS_BUCKET=medical-model-data
OBS_ENDPOINT=obs.cn-southwest-2.myhuaweicloud.com
IAM_REGION=cn-southwest-2
USE_ECS_AGENCY=True
"""


# Systemd 守护进程模板 (支持服务器重启自启动)
SYSTEMD_SERVICE_TEMPLATE = """
[Unit]
Description=Cloud Storage Gateway for Federated Learning
After=network.target

[Service]
User=root
WorkingDirectory={remote_path}
ExecStart={remote_path}/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
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

    # 5. Uvicorn 进程守护拉起 (改为持久化 Systemd 服务以支持开机自启)
    print("[*] 清洗历史残留进程, 配置持久化 Systemd 守护进程...")
    client.exec_command("fuser -k 8000/tcp")
    client.exec_command("systemctl stop cloudgateway.service")
    # 强制清理掉可能存在的同名瞬态 Unit (防止 'transient or generated' 冲突)
    client.exec_command("rm -f /run/systemd/system/cloudgateway.service /run/systemd/transient/cloudgateway.service*")
    time.sleep(2)
    
    # 写入 Service 配置文件 (改用本地生成 + SCP 物理上传以确保数据持久化完整性)
    service_content = SYSTEMD_SERVICE_TEMPLATE.format(remote_path=REMOTE_PATH)
    local_tmp_path = os.path.join(LOCAL_PATH, "cloudgateway.service.tmp")
    with open(local_tmp_path, "w", encoding="utf-8") as f:
        f.write(service_content)
    
    with SCPClient(client.get_transport()) as scp:
        print("[*] 正在 SCP 服务配置文件到远程暂存区...")
        scp.put(local_tmp_path, remote_path="/tmp/cloudgateway.service")
    
    if os.path.exists(local_tmp_path):
        os.remove(local_tmp_path) 

    # 负载并启动
    print("[*] 激活并启动开机自启动服务 (systemctl enable & restart)...")
    full_setup_cmd = (
        "mv /tmp/cloudgateway.service /etc/systemd/system/cloudgateway.service && "
        "chmod 644 /etc/systemd/system/cloudgateway.service && "
        "systemctl daemon-reload && "
        "systemctl enable cloudgateway.service && "
        "systemctl restart cloudgateway.service && "
        "sync"
    )
    _, stdout, stderr = client.exec_command(full_setup_cmd)
    if err := stderr.read().decode().strip():
        # 忽略正常状态输出
        if "Created symlink" not in err:
            print(" [!] Systemd 配置提示:", err)
    
    print("[*] 自动部署流程全绿完成！网关系统已上云并开启开机自启。")


    client.close()

if __name__ == '__main__':
    deploy()
