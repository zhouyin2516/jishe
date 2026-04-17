import torch.nn as nn
import torch
import torch.nn.functional as F
import yaml

# 加载配置文件
def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

# 加载配置
config = load_config()

# 时间步范围配置
# 支持多个时间步范围，每个范围可以有不同的适配器配置
timestep_ranges = config.get("server", {}).get("timestep_ranges", [])
if not timestep_ranges:
    # 兼容旧配置
    default_range = config.get("server", {}).get("timestep_range", [500, 600])
    timestep_ranges = [{
        "range": default_range,
        "adapter": config.get("adapter", {})
    }]

class ImageAdapter(nn.Module):
    """图像适配器：处理 UNet 上采样层特征，注入医学图像结构"""

    def __init__(self, img_dim):
        super().__init__()
        self.img_dim = img_dim
        
        # 存储每个时间步范围的适配器配置和网络
        # 使用普通列表，因为其中包含的是字典而不是模块
        self.range_adapters = []
        
        # 为每个时间步范围创建适配器
        for range_config in timestep_ranges:
            time_range = range_config.get("range", [500, 600])
            adapter_config = range_config.get("adapter", {})
            
            # 解析适配器配置
            hidden_layers = adapter_config.get("hidden_layers", 2)
            hidden_size = adapter_config.get("hidden_size", 128)
            activation = adapter_config.get("activation", "gelu")
            
            # 支持不同大小的隐藏层
            if isinstance(hidden_size, list):
                if len(hidden_size) != hidden_layers:
                    # 如果列表长度与隐藏层层数不匹配，使用列表中的第一个值
                    hidden_sizes = [hidden_size[0]] * hidden_layers
                else:
                    hidden_sizes = hidden_size
            else:
                # 如果是单个值，所有隐藏层使用相同大小
                hidden_sizes = [hidden_size] * hidden_layers
            
            # 获取激活函数
            def get_activation(name):
                if name == "relu":
                    return nn.ReLU()
                elif name == "gelu":
                    return nn.GELU()
                elif name == "sigmoid":
                    return nn.Sigmoid()
                elif name == "tanh":
                    return nn.Tanh()
                else:
                    return nn.GELU()  # 默认使用 GELU
            
            # 构建适配器网络
            layers = []
            # 第一层：输入到第一个隐藏层
            first_hidden_size = hidden_sizes[0]
            if img_dim != first_hidden_size:
                layers.append(nn.Conv2d(img_dim, first_hidden_size, kernel_size=1))
                layers.append(get_activation(activation))
            # 然后使用 3x3 卷积进行特征提取
            layers.append(nn.Conv2d(first_hidden_size, first_hidden_size, kernel_size=3, padding=1))
            layers.append(get_activation(activation))
            
            # 中间隐藏层
            for i in range(1, hidden_layers):
                current_size = hidden_sizes[i-1]
                next_size = hidden_sizes[i]
                if current_size != next_size:
                    # 如果当前隐藏层大小与下一层不同，使用 1x1 卷积调整通道数
                    layers.append(nn.Conv2d(current_size, next_size, kernel_size=1))
                    layers.append(get_activation(activation))
                layers.append(nn.Conv2d(next_size, next_size, kernel_size=3, padding=1))
                layers.append(get_activation(activation))
            
            # 特征投影
            last_hidden_size = hidden_sizes[-1]
            feature_proj = nn.Conv2d(last_hidden_size, img_dim, kernel_size=1)
            
            # 使用普通字典存储，将模块注册为子模块
            adapter_dict = {
                "range": time_range,
                "structural_encoder": nn.Sequential(*layers),
                "feature_proj": feature_proj
            }
            # 将 structural_encoder 和 feature_proj 注册为子模块
            self.register_module(f"structural_encoder_{len(self.range_adapters)}", adapter_dict["structural_encoder"])
            self.register_module(f"feature_proj_{len(self.range_adapters)}", adapter_dict["feature_proj"])
            self.range_adapters.append(adapter_dict)
        
        # 参数
        self.kappa_scale = 1

    def forward(self, img_feat, timestep=None, kappa=None):
        batch_size = img_feat.shape[0]
        timestep_in_range = False
        selected_adapter = None
        
        if timestep is not None:
            # 检查时间步是否在任何配置的范围内
            for adapter_info in self.range_adapters:
                time_range = adapter_info["range"]
                if ((timestep >= time_range[0]) & (timestep <= time_range[1])).any():
                    timestep_in_range = True
                    selected_adapter = adapter_info
                    break
        
        if not timestep_in_range or selected_adapter is None:
            return img_feat

        if kappa is None:
            # 直接生成 [B,1,1,1] 的默认值，避免维度混乱
            kappa_scaled = torch.full((batch_size, 1, 1, 1), 0.5, device=img_feat.device)
        else:
            # 确保kappa的设备与img_feat一致
            kappa_scaled = kappa.to(img_feat.device) * self.kappa_scale

            # 处理kappa批量大小与img_feat批量大小不匹配的情况
            if kappa_scaled.shape[0] != batch_size:
                # 如果kappa的批量大小为1但img_feat的批量大小大于1，则广播kappa
                if kappa_scaled.shape[0] == 1:
                    kappa_scaled = kappa_scaled.expand(batch_size, *kappa_scaled.shape[1:])
                else:
                    # 其他情况，取第一个元素并广播
                    kappa_scaled = kappa_scaled[0:1].expand(batch_size, *kappa_scaled.shape[1:])

            # 确保kappa_scaled至少是2维的 [B, ...]
            while kappa_scaled.dim() < 2:
                kappa_scaled = kappa_scaled.unsqueeze(-1)

            # 将kappa统一转为4维 [B,1,1,1]，适配图像特征广播
            if kappa_scaled.dim() == 2:
                kappa_scaled = kappa_scaled.unsqueeze(-1).unsqueeze(-1)
            elif kappa_scaled.dim() == 3:
                kappa_scaled = kappa_scaled.unsqueeze(-1)
            # 如果已经是4维，则保持不变

            # 数值限制
            kappa_scaled = torch.clamp(kappa_scaled, min=0.1, max=0.8)

        # 使用选中的适配器处理特征
        # 确保适配器的设备与输入特征一致
        structural_encoder = selected_adapter["structural_encoder"].to(img_feat.device)
        feature_proj = selected_adapter["feature_proj"].to(img_feat.device)
        
        structured_feat = structural_encoder(img_feat)
        adapted_feat = feature_proj(structured_feat)

        output = (1 - kappa_scaled ) * img_feat + kappa_scaled  * adapted_feat

        return output
