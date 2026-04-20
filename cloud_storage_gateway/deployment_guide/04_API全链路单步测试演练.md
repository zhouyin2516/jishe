# 网关 API 全链路单步测试演练 (Postman / cURL 适用)

这份操作文档将带你走通一个模型文件的**完整生命周期**：获取上传地址 -> 真正上传空包 -> 列表查询验证 -> 读取云端指纹属性 -> 获取下载直链获取内容 -> 物理销毁服务器残余数据。

> **测试环境与鉴权变量准备：** 
> 
> 在接下来的所有涉及到 `http://` 接口调用的演示中，你都需要替换或了解以下两个变量是怎么来的：
> 1. **服务器 IP 地址 (`101.245.110.98`)**：这代表了你已经部署完成微服务的公网（或局域网）IP。
> 2. **内部安全握手密钥 (Header 限定: `X-Internal-Token`)**：
>     - **这是什么？**：这是为了防止公网恶意访问，我们设立的一道关卡。任何请求如果不在 Header 头里挂上这行字和相应的密码，直接无视并报 `403 Forbidden`。
>     - **怎么填？该填什么？**：它对应你服务器代码目录下的 `.env` 文件（或者是你在 `auto_deploy_template.py` 中下发的那个变量）里明文规定的 **`INTERNAL_API_KEY`** 这个参数的值。
>     - **本次测试的默认值**：由于我们用了一键脚本部署，该脚本默认把 `INTERNAL_API_KEY` 写成了 **`test_secret_key`**，所以在下面的命令中，你会看到通篇全部在使用 `-H "X-Internal-Token: test_secret_key"` 进行通关。在未来的真实生产中，当后端 Java 找你要的时候，如果 `.env` 里的 `INTERNAL_API_KEY=ABC12345`，你就要把下面测试里所有的 `test_secret_key` 换成 `ABC12345`。

---

## 步骤 1：在本地创建一个空的模型文件
在你本地的命令行环境（CMD 或 PowerShell），创造一个极小的测试文件模型用来跑通全流程：
```bat
echo "AI Model V1.0 Test Data" > test_model.txt
tar -a -c -f test_model.zip test_model.txt
```
> *(注：如果 tar 报错，你可以直接上传 `test_model.txt`，后面的 `file_key` 换成 .txt 即可)*

---

## 步骤 2：获取『上传』专属预签名 URL
你必须要向咱们的后台独立系统“打报告申请”，才能获得跟华为云直接交流的能力。

**使用终端 (cURL 单行版，兼容 CMD/PowerShell 复制执行)**
```bat
curl -X POST "http://101.245.110.98:8000/api/internal/storage/generate_presigned_url" -H "X-Internal-Token: test_secret_key" -H "Content-Type: application/json" -d "{\"group_id\": \"federated_group_1\", \"file_key\": \"test_model.zip\", \"action\": \"upload\"}"
```

**Postman 操作对照法**
- Method: `POST` / URL: `http://101.245.110.98:8000/api/internal/storage/generate_presigned_url`
- Headers 添加: `X-Internal-Token` => `test_secret_key`
- Body 选择 Raw -> JSON: 贴入上述 `-d` 后面的 JSON 字符串格式。

**预期的成功响应**:
```json
{
  "url": "https://medical-model-data.obs.cn-north-4.myhuaweicloud.com/federated_group_1/models/test_model.zip?AWSAccessKeyId=......"
}
```

---

## 步骤 3：跳过内网服务器，直接推入华为云！
复制你在**步骤 2** 收到的那一长段 `"url"`。这是专属授权，任何拦截或安全组都已经通过了。

**使用终端 (cURL)** (将下面 `<YOUR_OBS_SIGNED_URL>` 替换为刚拿到的极长字符串)
```powershell
curl -X PUT -T "test_model.zip" "<YOUR_OBS_SIGNED_URL>"
```
*如果你使用 Postman*：新建一个页签，选择 `PUT`，框内填入 URL，Body 选择 `binary` 并选中你刚才电脑桌面的 `test_model.zip` 上传，点击 Send。

**预期反应**: 如果控制台不报错（华为通常返回空字符或 `HTTP 200 OK`），代表模型已经神不知鬼不觉上云了。

---

## 步骤 4：网关查询验证 (List Models)
刚才通过步骤3我们直接送了资源给 OBS，那么你的内部网关知道它存在了吗？测一下：

**使用终端 (cURL 单行版)**
```bat
curl "http://101.245.110.98:8000/api/internal/storage/list_models?group_id=federated_group_1" -H "X-Internal-Token: test_secret_key"
```

**预期反应**: 你们群组的列表模型已经列出了刚才的文件和上传时间：
```json
{
  "group_id": "federated_group_1",
  "models": [
    {
      "file_key": "test_model.zip",
      "size_bytes": 165,
      "last_modified": "2026-04-20T14:30:22.000Z"
    }
  ]
}
```

---

## 步骤 5：申请『下载』此模型的提货预签直链
由于是纯对象存储架构，你要读出模型去参与训练，还得再申请个提取证：

```bat
curl -X POST "http://101.245.110.98:8000/api/internal/storage/generate_presigned_url" -H "X-Internal-Token: test_secret_key" -H "Content-Type: application/json" -d "{\"group_id\": \"federated_group_1\", \"file_key\": \"test_model.zip\", \"action\": \"download\"}"
```

**预期反应**: 再次吐给客户端一个全新的签名 URL。

---

## 步骤 6：通过 GET 直链下载
就像在任意下载网站下资源一样简单了。利用你在**步骤 5** 获取到的 `url`：

```powershell
curl -O "<YOUR_DOWNLOAD_SIGNED_URL>"
```
或者，**最震撼的做法**：因为这是 `GET` 方法提取文件，你拿着这段具有一小时寿命的网链直接丢进 Chrome 或 Safari 的浏览器网址栏敲回车，浏览器将立刻开始原生的满速下载行为！

---

## 步骤 7：物理毁尸灭迹 (Delete Object)
测试闭环最重要的一点，释放我们云主机的无用大文件，向网关发号施令做强制驱逐：

```bat
curl -X POST "http://101.245.110.98:8000/api/internal/storage/delete_object" -H "X-Internal-Token: test_secret_key" -H "Content-Type: application/json" -d "{\"group_id\": \"federated_group_1\", \"file_key\": \"test_model.zip\"}"
```

**预期反应**:
```json
{"success": true}
```
此时你再回头执行**步骤 4** 去查询，`models` 数组已经空空如也。完整周期测试结束！
