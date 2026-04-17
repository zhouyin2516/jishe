#!/usr/bin/env python3
"""
简化版训练启动脚本
直接在当前环境中启动服务器和客户端
"""
import os
import sys
import subprocess
import time
import threading

# 获取当前目录
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 配置
SERVER_SCRIPT = "server.py"
CLIENT_SCRIPT = "client.py"
NODES = 4  # 客户端数量

# 日志目录
LOG_DIR = os.path.join(CURRENT_DIR, "output")

print("=== 简化版训练启动脚本 ===")
print(f"当前目录: {CURRENT_DIR}")
print(f"服务器脚本: {SERVER_SCRIPT}")
print(f"客户端脚本: {CLIENT_SCRIPT}")
print(f"客户端数量: {NODES}")
print(f"日志目录: {LOG_DIR}")

# 检查脚本是否存在
if not os.path.exists(SERVER_SCRIPT):
    print(f"错误: {SERVER_SCRIPT} 不存在")
    sys.exit(1)

if not os.path.exists(CLIENT_SCRIPT):
    print(f"错误: {CLIENT_SCRIPT} 不存在")
    sys.exit(1)

# 创建日志目录
os.makedirs(LOG_DIR, exist_ok=True)

# 启动服务器
def start_server():
    print("\n启动服务器...")
    server_log = os.path.join(LOG_DIR, "server_log.txt")
    with open(server_log, "w") as f:
        server_process = subprocess.Popen(
            [sys.executable, SERVER_SCRIPT],
            stdout=f,
            stderr=subprocess.STDOUT
        )
    return server_process

# 启动客户端
def start_client(client_id, gpu_id):
    print(f"\n启动客户端 {client_id} (GPU {gpu_id})...")
    client_log = os.path.join(LOG_DIR, f"client{client_id}_log.txt")
    
    # 设置环境变量，指定使用的GPU
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    with open(client_log, "w") as f:
        client_process = subprocess.Popen(
            [sys.executable, CLIENT_SCRIPT],
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env
        )
    return client_process

# 主函数
def main():
    processes = []
    
    try:
        # 启动服务器
        server_proc = start_server()
        processes.append(("server", server_proc))
        
        # 等待服务器启动
        time.sleep(10)
        
        # 启动客户端
        for i in range(NODES):
            client_id = i + 1
            gpu_id = i % 4  # 为每个客户端分配不同的GPU (0-3)
            client_proc = start_client(client_id, gpu_id)
            processes.append((f"client_{client_id}", client_proc))
            time.sleep(1)
        
        print("\n=== 训练已开始 ===")
        print("所有进程已启动")
        print(f"服务器日志：{os.path.join(LOG_DIR, 'server_log.txt')}")
        for i in range(NODES):
            print(f"客户端 {i+1} 日志：{os.path.join(LOG_DIR, f'client{i+1}_log.txt')}")
        
        # 监控进程
        print("\n按 Ctrl+C 停止训练")
        while True:
            time.sleep(1)
            
            # 检查所有进程是否还在运行
            all_finished = True
            for name, proc in processes:
                if proc.poll() is None:  # 进程还在运行
                    all_finished = False
                    break
            
            # 如果所有进程都结束了，自动退出
            if all_finished:
                print("\n所有进程已结束，训练完成！")
                break
            
    except KeyboardInterrupt:
        print("\n正在停止训练...")
    finally:
        # 停止所有进程
        for name, proc in processes:
            print(f"停止 {name}...")
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except:
                proc.kill()
        print("\n训练已停止")

if __name__ == "__main__":
    main()