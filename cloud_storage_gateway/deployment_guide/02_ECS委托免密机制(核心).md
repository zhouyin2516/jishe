# 核心安全机制：ECS 委托 (Agency) 绑定指南

本网关的一个巨大的架构优势是**代码“零安全风险”**（No Hardcoded AK/SK）。这就要求服务端通过“委托机制”无感获取临时凭证。

> ⚠️ 注意：在使用非华为云服务器（如本地电脑、物理机或阿里云）时，该方案无效，你只能退而求其次配置环境文件 `.env` 里面的 `USE_ECS_AGENCY=False` 强行填入明文密码。但在华为云上，我们强制推荐以下设定：

## 1. 创立 ECS 读写 OBS 的委托 (IAM Agency)

1. 进入华为云控制台首页，搜索并进入 **统一身份认证服务 (IAM)**。
2. 左边栏找到 **“委托”**，点击 **“创建委托”**。
3. **关键参数填写**：
   - **委托名称**：填写 `OBS-Medical-Gateway-Agency` (必须与代码配置一致)。
   - **委托类型**：必须选择 **“云服务”**，并勾选 **“弹性云服务器 ECS”**。
   - **持续时间**：选择 **“永久”**。

4. **授权**：在下面搜索并添加 `OBS Administrator` （OBS 所有权限），或者你可以遵循最小化原则，在 IAM 创建一个自定义 Policy（如我们之前的例子），严格只写 `obs:object:PutObject`，`GetObject`，`DeleteObject` 赋予特定的组，再授权给该委托。

## 2. 将委托挂载给云服务器

1. 回到**弹性云服务器 (ECS)** 的控制台。
2. 找到你需要部署和发送微服务的机器（例如 101.245...）。
3. 选择该机器，点击【更多】->【管理】->【委托】。
4. 在弹出的选框中挂载刚刚建立好的 `OBS-Medical-Gateway-Agency`。
5. 等待 1-2 分钟让元数据映射生效（通常无需重启）。


## 3. 服务器端自检
登录服务器，输入下述内网请求命令进行诊断：
```bash
# 推荐路径 (Managed Service 路径)
curl http://169.254.169.254/managed_service/security_token/OBS-Medical-Gateway-Agency

# 备选路径 (OpenStack 兼容路径)
curl http://169.254.169.254/openstack/latest/securitykey
```

