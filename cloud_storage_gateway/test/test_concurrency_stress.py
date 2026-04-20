import concurrent.futures
import requests
import time

HOST = "http://101.245.110.98:8000"
TOKEN = "test_secret_key"
HEADERS = {
    "X-Internal-Token": TOKEN,
    "Content-Type": "application/json"
}

def simulate_upload_request(worker_id):
    """模拟不同组多个人同时申请上传链路"""
    group_id = f"group_{worker_id}"
    file_key = f"model_v{worker_id}.zip"
    payload = {"group_id": group_id, "file_key": file_key, "action": "upload"}
    
    start_time = time.time()
    try:
        response = requests.post(
            f"{HOST}/api/internal/storage/generate_presigned_url", 
            headers=HEADERS, 
            json=payload,
            timeout=5
        )
        latency = (time.time() - start_time) * 1000
        if response.status_code == 200:
            return f"✅ [上传申请] Worker {worker_id} 成功拿到 {group_id} 的上传链接! (耗时: {latency:.2f}ms)"
        else:
            return f"❌ [上传申请] Worker {worker_id} 报错: {response.text}"
    except Exception as e:
        return f"❌ [上传申请] Worker {worker_id} 崩溃: {str(e)}"

def simulate_download_request(worker_id):
    """模拟同一个组多人同时疯狂抢夺下载链接"""
    group_id = "group_shared"
    file_key = "popular_model.zip"
    payload = {"group_id": group_id, "file_key": file_key, "action": "download"}
    
    start_time = time.time()
    try:
        response = requests.post(
            f"{HOST}/api/internal/storage/generate_presigned_url", 
            headers=HEADERS, 
            json=payload,
            timeout=5
        )
        latency = (time.time() - start_time) * 1000
        if response.status_code == 200:
            return f"✅ [下载申请] Worker {worker_id} 成功拿到高热度下载直连! (耗时: {latency:.2f}ms)"
        else:
            return f"❌ [下载申请] Worker {worker_id} 报错: {response.text}"
    except Exception as e:
        return f"❌ [下载申请] Worker {worker_id} 崩溃: {str(e)}"

def run_stress_test(concurrency=50):
    print(f"🚀 开始发起 {concurrency} 并发海啸冲击测试...\n")
    
    results = []
    # 使用线程池瞬间打满并发
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        # 一半人疯狂申请上传（不同组），一半人疯狂排队下载（同组）
        upload_tasks = [executor.submit(simulate_upload_request, i) for i in range(concurrency // 2)]
        download_tasks = [executor.submit(simulate_download_request, i) for i in range(concurrency // 2)]
        
        for future in concurrent.futures.as_completed(upload_tasks + download_tasks):
            results.append(future.result())

    # 打印最终并发战绩
    success_count = sum(1 for r in results if "✅" in r)
    for res in results[:10]: # 只预览前十条日志以防刷屏
        print(res)
    print("...\n")
    print(f"📊 压测总结: 发起 {concurrency} 笔并发，成功 {success_count} 笔，失败 {concurrency - success_count} 笔。")

if __name__ == "__main__":
    run_stress_test(20) # 你可以在这里修改甚至调整到 200, 500 来测试！
