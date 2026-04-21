# Cloud Storage Gateway (云存储网关微服务)

## 项目简介
本项目是专为联邦学习系统设计的云端存储内部网关，作为一个独立的高性能微服务运行。其核心职责是将客户端的高负载、大文件（如深度学习大模型，动辄 GB 级）的上传与下载动作从 Java 业务线中解耦出来。

**核心架构：Pre-signed URL (预签名直通)**
客户端不再通过中间业务服务器中转巨型模型数据，而是由后端微服务向本网关发送请求，网关利用其具备的高权限签发一个带有时效性和操作范围限制的 **华为云 OBS 预签名 URL (Pre-signed URL)**。客户端拿到这个 URL 后，直接与华为云 OBS 建立直连并进行极速的上传或断点续传下载。

基于华为云 ECS 的委托（Agency）鉴权模型，本网关**完全无状态且实现了 “零长期安全密钥泄露风险”**，所有的鉴权操作基于无感的 ECS Metadata 和细粒度的 URL 签署规则。

## 核心功能
*   **预签名 URL 动态颁发**: 为每个指定的联邦学习组别和文件下发极其精准的预签名直连，大文件授权的时间硬编码限制为一小时以保安全不断传。
*   **免密/委托机制适配**:
    *   在华为云 ECS 部署时，启用 `USE_ECS_AGENCY=True` 即自动调用云内网元数据授权，免去繁琐的 AK/SK 保存风险。
    *   在本地调试时，自动降级读取 `.env` 中的开发凭据完成签名。
*   **私有 API 认证隔离**: 内置 Header 中间件，所有请求必须持有正确的 `X-Internal-Token`，杜绝公网直接恶意调用网关刷流量。
*   **OBS 目录结构规范化**: 将所有的 Bucket 对象严格进行前缀隔离：`{group_id}/models/{file_key}`。

- **语言框架**: Python 3.10+ / FastAPI / Uvicorn
- **核心包**: pydantic, pydantic-settings, requests
- **华为云核心 SDK**: `esdk-obs-python` (强制 V4 签名, 区域识别补丁)


---

## 如果你是一名前端/后端开发人员：网关调用指南

网关目前已经通过自动化部署运行在远端（或本地网络内），提供 5 个符合标准 RESTful 的原子接口。

> **⚠️核心安全约定与防火墙警告**：
> 1. 所有的内网调用，都必须在 HTTP Request Header 中带上身份验证票据 `X-Internal-Token: 你在微服务配置中设置好的密钥`。
> 2. **该网关微服务千万不要对 `0.0.0.0` 全网裸奔开放端口！** 请后端研发及运维务必在华为云控制台设置安全组规则，令网关所在虚拟机的 8000 端口**仅仅对后端 Java 业务系统所在的公网 IP（或同 VPC 内网 IP）开放白名单路由**。网关从物理上对公网客户端隐身，客户端仅拿生成的预签名 URL 直通云厂大带宽管道。

### 1. 申请直传/下载授权签名 (最核心 API)
当联邦客户端准备开始发送模型时，业务服务器必须代替客户端调用此接口获取 URL，再下发给客户端自己去传输。

*   **Endpoint**: `POST /api/internal/storage/generate_presigned_url`
*   **Payload (JSON)**:
    ```json
    {
        "group_id": "federated_group_1",
        "file_key": "unet_epoch_10.zip",
        "action": "upload",
        "content_type": "application/zip", // 可选
        "expires": 3600 // 可选，单位秒。默认为 3600 (1小时)
    }

    ```
*   **Response (JSON)**:
    ```json
    {
        "url": "https://...",
        "curl_command": "curl -X PUT -T \"YOUR_LOCAL_FILE_PATH\" -H \"Content-Type: ...\" \"https://...\"" 
        // 提示：这是自动生成的防错上传命令，直接复制执行即可，无需手动拼凑。
    }
    ```
*   **客户端操作指引**: 
    客户端将直接对该 `"url"` 发起 HTTP `PUT` (upload) 或 `GET` (download)。
    **注意**：如果申请时指定了 `content_type`，上传时的 HTTP Header 必须完全一致。


### 2. 远端物理文件删除
如果在系统里废弃了某个模型，可跨越后台直删 OBS 源文件。
*   **Endpoint**: `POST /api/internal/storage/delete_object`
*   **Payload (JSON)**: `{"group_id": "grp1", "file_key": "unet.zip"}`

### 3. 获取文件元数据校验 (ContentLength & ETag)
*   **Endpoint**: `GET /api/internal/storage/object_meta?group_id={dir}&file_key={file}`

### 4. 获取顶层租户/组别列表 (List Groups)
*   **Endpoint**: `GET /api/internal/storage/list_groups`

### 5. 拉取全量分组下的模型文件
*   **Endpoint**: `GET /api/internal/storage/list_models?group_id={dir}`
*   **说明**: 获取包含文件大小和上传时间的数据，用于呈现模型版本仓库。

---

具体请参见本项目附带的全新教程目录 `deployment_guide/`。

**开发测试辅助工具：**
在 `deployment_guide/upload_test.py` 目录下有一个自动化上传演示脚本。你可以使用该脚本一键测试全链路（申请URL -> 上传数据），该工具能有效规避终端复制粘贴时产生换行符导致的鉴权失败问题。

