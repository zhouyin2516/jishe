import socket
from socket import SO_REUSEADDR
import time
import math
from copy import deepcopy
import torch.nn as nn
import torch
import numpy as np
import random
import os
import yaml

# 加载配置文件
def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

# 加载配置
config = load_config()


# 设置固定随机种子以确保实验可重复性
def set_random_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# 在程序开始时调用
set_random_seed()

from control_algorithm.adaptive_tau import ControlAlgAdaptiveTauServer
from statistic.collect_stat import CollectStatistics
from util.utils import send_msg, recv_msg

# Configurations are in a separate config.py file
from config import parse_args, generate_random_params, calculate_time_energy, get_adapt_layers, get_img_dims_per_level
from Adapter import ImageAdapter
import accelerate
import torch.utils.checkpoint
from accelerate.state import AcceleratorState
from transformers import CLIPTextModel

from diffusers.utils import check_min_version
import warnings

# 忽略 PyTorch 的 FutureWarning（仅针对 torch.load 的 weights_only 提示）
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.storage")
warnings.filterwarnings("ignore", message="NOTE: Redirects are currently not supported in Windows or MacOs.")


def init_adapter(adapter_module):
    """初始化适配器参数（对齐客户端单输出ImageKappaHyperNetwork）"""
    for level in adapter_module.keys():
        # 图像适配器初始化（up0-up3）
        if level in ["up0", "up1", "up2", "up3"]:
            # 图像适配器初始化
            img_adapter = adapter_module[level]["img_adapter"]
            # 遍历所有时间步范围的适配器
            for range_adapter in img_adapter.range_adapters:
                for m in range_adapter["structural_encoder"].modules():
                    if isinstance(m, nn.Conv2d):
                        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0.05)
                if hasattr(range_adapter, "feature_proj"):
                    nn.init.kaiming_normal_(range_adapter["feature_proj"].weight, mode='fan_out', nonlinearity='relu')
                    if range_adapter["feature_proj"].bias is not None:
                        nn.init.constant_(range_adapter["feature_proj"].bias, 0.05)

        # 图像kappa超网络初始化（适配单输出版本）
        elif level == "img_kappa_hypernet":
            hypernet = adapter_module[level]
            # 全局编码器初始化
            for m in hypernet.global_encoder.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=0.5)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.05)
            # 层级预测器初始化（单输出，无需区分text/img）
            for _, predictor in hypernet.level_predictors.items():
                for i, module in enumerate(predictor):
                    if isinstance(module, nn.Linear):
                        if i == len(predictor) - 2:  # 倒数第二个线性层（最后一个是Sigmoid）
                            nn.init.xavier_uniform_(module.weight, gain=0.1)
                            if module.bias is not None:
                                nn.init.constant_(module.bias, 0.5)
                        else:
                            nn.init.xavier_uniform_(module.weight)
                            if module.bias is not None:
                                nn.init.constant_(module.bias, 0.05)
            # 层级嵌入初始化
            for emb in hypernet.level_embeddings.values():
                nn.init.normal_(emb, mean=0.0, std=0.02)


def get_timestep_embedding(timesteps, embedding_dim):
    """
    将时间步转换为正弦位置编码（与客户端完全一致，对齐dtype）
    """
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=timesteps.dtype) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb.to(timesteps.dtype)


# 对齐客户端的ImageKappaHyperNetwork
# 替换服务器端原有的ImageKappaHyperNetwork类
class ImageKappaHyperNetwork(nn.Module):
    """图像专用kappa超网络：仅预测图像kappa（与客户端完全一致）"""

    def __init__(self, text_dim=768, time_dim=512, level_embed_dim=32, levels=None):
        super().__init__()
        self.text_dim = text_dim
        self.time_dim = time_dim
        self.level_embed_dim = level_embed_dim
        self.levels = levels or ["up0", "up1", "up2", "up3"]

        self.global_encoder = nn.Sequential(
            nn.Linear(text_dim + time_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        self.level_predictors = nn.ModuleDict({
            level: nn.Sequential(
                nn.Linear(512 + level_embed_dim, 128),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(128, 1),  # 仅输出图像kappa（单值）
                nn.Sigmoid()
            ) for level in self.levels
        })

        self.level_embeddings = nn.ParameterDict({
            level: nn.Parameter(torch.randn(level_embed_dim))
            for level in self.levels
        })
        self._init_level_embeddings()

    def _init_level_embeddings(self):
        for level in self.level_embeddings:
            nn.init.normal_(self.level_embeddings[level], mean=0.0, std=0.02)

    def forward(self, text_emb, timestep, current_level=None):
        dtype = text_emb.dtype  # 对齐客户端的dtype逻辑
        batch_size = text_emb.shape[0]
        text_feat = text_emb.mean(dim=1).to(dtype)
        timestep_emb = get_timestep_embedding(timestep, self.time_dim).to(dtype)
        global_cond = torch.cat([text_feat, timestep_emb], dim=1).to(dtype)
        global_feat = self.global_encoder(global_cond).to(dtype)

        if current_level is not None:
            level_emb = self.level_embeddings[current_level].unsqueeze(0).expand(batch_size, -1).to(dtype)
            level_aware_feat = torch.cat([global_feat, level_emb], dim=1).to(dtype)
            img_kappa = self.level_predictors[current_level](level_aware_feat).to(dtype)
            return img_kappa[:, 0]
        else:
            all_img_kappas = {}
            for level in self.levels:
                level_emb = self.level_embeddings[level].unsqueeze(0).expand(batch_size, -1).to(dtype)
                level_aware_feat = torch.cat([global_feat, level_emb], dim=1).to(dtype)
                img_kappa = self.level_predictors[level](level_aware_feat).to(dtype)
                all_img_kappas[level] = img_kappa[:, 0].unsqueeze(-1).unsqueeze(-1).to(dtype)
            return all_img_kappas


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.22.0.dev0")

# 解析参数
args = parse_args()
print(args.output_dir)


def deepspeed_zero_init_disabled_context_manager():
    """
    returns either a context list that includes one that will disable zero.Init or an empty context list
    """
    deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
    if deepspeed_plugin is None:
        return []
    return [deepspeed_plugin.zero3_init_context_manager(enable=False)]


# 加载CLIP文本编码器
text_encoder = CLIPTextModel.from_pretrained(
    config["model"].get("base_model_path", "model"), subfolder=config["model"].get("text_encoder_subfolder", "text_encoder"), revision=config["model"].get("revision", None)
)
SERVER_ADDR = config["server"].get("server_addr", "localhost")
SERVER_PORT = config["server"].get("server_port", 5100)
n_nodes = config["server"].get("n_nodes", 4)
max_training_rounds = config["server"].get("max_training_rounds")  # 允许为 None，由后端控制
use_momentum = True  # 始终使用动量加速
momentum_value = config["server"].get("momentum_value", 0.9)
use_min_loss = True  # 始终使用最小损失策略
dataset_root = config["server"].get("dataset_root", "data/medical_images")
# 时间步范围配置
timestep_ranges = config["server"].get("timestep_ranges", [])
if not timestep_ranges:
    # 兼容旧配置
    default_range = config["server"].get("timestep_range", [500, 601])
    timestep_ranges = [{
        "range": default_range,
        "adapter": config.get("adapter", {})
    }]

# 训练时使用的时间步范围（使用第一个范围作为默认）
timestep_range = tuple(timestep_ranges[0]["range"]) if timestep_ranges else (500, 601)
# 控制参数
control_param_phi = 0.00005

# 是否在所有运行中估计beta和delta
iestimate_beta_delta_in_all_runs = False

# 初始化服务器socket
listening_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
listening_sock.setsockopt(socket.SOL_SOCKET, SO_REUSEADDR, 1)
listening_sock.bind((SERVER_ADDR, SERVER_PORT))
client_sock_all = []

# 建立与所有客户端的连接
while len(client_sock_all) < n_nodes:
    listening_sock.listen(5)
    print("Waiting for incoming connections...")
    (client_sock, (ip, port)) = listening_sock.accept()
    print('Got connection from ', (ip, port))
    client_sock_all.append(client_sock)

# 初始化统计模块
# 创建每个训练情况的结果文件
case_name = args.output_dir.split('/')[1]  # 获取case名称
results_file_dir = os.path.join(args.output_dir, 'results')
os.makedirs(results_file_dir, exist_ok=True)

# 初始化统计
single_run_file_path = os.path.join(results_file_dir, 'SingleRun.csv')
stat = CollectStatistics(results_file_name=single_run_file_path, is_single_run=True)

# 重新设置随机种子，确保适配器初始化独立
set_random_seed()

# 初始化全局适配器（只保留图像适配器）
from config import get_adapt_layers, get_img_dims_per_level
# 自动检测UNet上采样层数并选择最接近输出的两层
adapt_layers = get_adapt_layers()
# 自动检测UNet上采样层的输出维度
img_dims_per_level = get_img_dims_per_level()
print(f"自动检测到的上采样层: {adapt_layers}")
print(f"自动检测到的上采样层维度: {img_dims_per_level}")

adapter_global = nn.ModuleDict()
for level in adapt_layers:
    adapter_global[level] = nn.ModuleDict({
        "img_adapter": ImageAdapter(
            img_dim=img_dims_per_level[level],
        )
    })
# 只保留图像kappa超网络
adapter_global["img_kappa_hypernet"] = ImageKappaHyperNetwork(
    text_dim=768,
    time_dim=512,
    level_embed_dim=32,
    levels=adapt_layers
)

# 使用可用设备（优先GPU）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
adapter_global = adapter_global.to(device)
init_adapter(adapter_global)

# 初始化统计
stat.init_stat_new_global_round()

adapter_global_min_loss = None
loss_min = np.inf
prev_loss_is_min = False

# 设置tau配置（始终使用自适应）
is_adapt_local = True
tau_config = [5] * n_nodes
use_adapt_local = True
moving_average_holding_param = 0.9

# 初始化控制算法
control_alg = ControlAlgAdaptiveTauServer(is_adapt_local, client_sock_all, n_nodes,
                                          control_param_phi, moving_average_holding_param)

# 发送初始化消息给所有客户端
for n in range(0, n_nodes):
    msg = ['MSG_INIT_SERVER_TO_CLIENT', control_alg,
           use_min_loss, adapter_global.state_dict(), n]
    send_msg(client_sock_all[n], msg)

# -------------------------- 初始化服务器端损失记录文件 --------------------------
# 创建结果目录（如果不存在）
os.makedirs(os.path.join(os.path.dirname(__file__), 'results'), exist_ok=True)

# 创建损失记录文件路径
loss_record_dir = os.path.join(args.output_dir, 'results')
os.makedirs(loss_record_dir, exist_ok=True)
loss_record_path = os.path.join(loss_record_dir,
                                f'server_aggregated_loss_record_tau_adaptive.csv')

# 检查文件是否存在，不存在则创建并写入表头
if not os.path.isfile(loss_record_path):
    with open(loss_record_path, 'w') as f:
        f.write('step,loss,iter_time,total_energy\n')

# 初始化全局step计数器
global_step = 0

print('All clients connected')

# 等待所有客户端完成数据准备
try:
    for n in range(0, n_nodes):
        recv_msg(client_sock_all[n], 'MSG_DATA_PREP_FINISHED_CLIENT_TO_SERVER', timeout=120)
    print('Start learning')
except Exception as e:
    print(f"错误：等待客户端数据准备时出错: {str(e)}")
    # 清理并退出
    for sock in client_sock_all:
        try:
            sock.close()
        except:
            pass
    try:
        listening_sock.close()
    except:
        pass
    raise

# 时间/能量统计变量初始化
current_round = 0
is_last_round_tmp = False
time_global_aggregation_all = None
total_time = 0
total_time_recomputed = 0
cumulative_time_with_transfer = 0  # 累积的包含传输时间的总时间
is_last_round = False
is_eval_only = False
tau_new_resume = None
delt_f = 0
gam_f = 0
E_total = 0
adapter_global_state = None
start_time = time.time()  # 记录训练开始时间

# 停止条件相关变量
stop_conditions = config["training"].get("stop_conditions", {})
time_limit = stop_conditions.get("time_limit")  # 不再使用默认值
energy_limit = stop_conditions.get("energy_limit")  # 不再使用默认值
convergence_rounds = stop_conditions.get("convergence_rounds", 20)  # 默认20轮
loss_improvement_percentage = stop_conditions.get("loss_improvement_percentage", 1.0)  # 默认1%

# 收敛检查相关变量
consecutive_no_improvement = 0
last_min_loss = np.inf

# 重构梯度聚合结构（只保留图像适配器和图像超网络）
from config import adapt_layers
grad_global = {
    "img": {level: [] for level in adapt_layers},
    "img_hypernet": {
        "global_encoder": [],
        "level_predictors": {level: [] for level in adapt_layers},
        "level_embeddings": {level: [] for level in adapt_layers}
    }
}

# 联邦训练主循环
while True:
    current_round += 1
    print('---------------------------------------------------------------------------')
    if max_training_rounds is not None:
        print(f'Current training round: {current_round}/{max_training_rounds}')
    else:
        print(f'Current training round: {current_round}')
    print('current tau config:', tau_config)

    if max_training_rounds is not None and current_round >= max_training_rounds - 1:
        is_last_round_tmp = True
        print(f'Reach max training rounds ({max_training_rounds}), preparing to stop.')

    time_total_all_start = time.time()

    # 发送权重和tau配置给所有客户端
    for n in range(0, n_nodes):
        msg = ['MSG_WEIGHT_TAU_SERVER_TO_CLIENT',
               adapter_global.state_dict(),
               tau_config[n],
               is_last_round,
               prev_loss_is_min,
               adapter_global_min_loss.state_dict() if adapter_global_min_loss is not None else None,
               grad_global]
        send_msg(client_sock_all[n], msg)

    w_global_prev = deepcopy(adapter_global)
    print('Waiting for local iteration at client')

    # 初始化聚合变量
    adapter_global = None
    loss_last_global = 0.0
    loss_w_prev_min_loss = 0.0
    received_loss_local_w_prev_min_loss = False
    data_size_total = 0
    time_all_local_all = 0
    data_size_local_all = []
    tau_actual = 0
    tau_each = []
    tau_new = []
    time_l = []
    which_is_slowest = 0
    t = 0.0
    client_adapters = [None] * n_nodes

    loss_local_list = []
    iter_time_local_list = []
    E_local_list = []

    # 重置梯度聚合结构
    grad_global = {
        "img": {"up0": [], "up1": [], "up2": [], "up3": []},
        "img_hypernet": {
            "global_encoder": [],
            "level_predictors": {"up0": [], "up1": [], "up2": [], "up3": []},
            "level_embeddings": {"up0": [], "up1": [], "up2": [], "up3": []}
        }
    }

    # 接收所有客户端的本地更新
    try:
        for n in range(0, n_nodes):
            msg = recv_msg(client_sock_all[n], 'MSG_WEIGHT_TIME_SIZE_CLIENT_TO_SERVER', timeout=300)
            adapter_local = msg[1]
            time_all_local = msg[2]
            time_l.append(time_all_local)

            # 记录最慢客户端
            if t < time_all_local:
                which_is_slowest = n
            t = max(t, time_all_local)
            time_all_local_all = t

            tau_each.append(msg[3])
            tau_actual = max(tau_actual, msg[3])
            data_size_local = msg[4]
            data_size_local_all.append(data_size_local)
            data_size_total += data_size_local

            loss_local_last_global = msg[5]
            loss_local_w_prev_min_loss = msg[6]
            grad_local = msg[7]

            # 保存当前客户端的适配器用于后续传输能量计算
            client_adapters[n] = adapter_local

            # 接收客户端的损失和迭代时间
            loss_local = msg[8]
            iter_time_local = msg[9]

            # -------------------------- 梯度聚合（按数据量加权） --------------------------
            # 1. 图像适配器梯度
            from config import adapt_layers
            for level in adapt_layers:
                if not grad_global["img"][level]:
                    grad_global["img"][level] = [g * data_size_local for g in grad_local["img"][level]]
                else:
                    for i in range(len(grad_global["img"][level])):
                        grad_global["img"][level][i] += grad_local["img"][level][i] * data_size_local



            # 5. 图像kappa超网络梯度
            # 全局编码器
            if not grad_global["img_hypernet"]["global_encoder"]:
                grad_global["img_hypernet"]["global_encoder"] = [
                    g * data_size_local for g in grad_local["img_hypernet"]["global_encoder"]
                ]
            else:
                for i in range(len(grad_global["img_hypernet"]["global_encoder"])):
                    grad_global["img_hypernet"]["global_encoder"][i] += \
                        grad_local["img_hypernet"]["global_encoder"][i] * data_size_local
            # 层级预测器
            for level in adapt_layers:
                if not grad_global["img_hypernet"]["level_predictors"][level]:
                    grad_global["img_hypernet"]["level_predictors"][level] = [
                        g * data_size_local for g in grad_local["img_hypernet"]["level_predictors"][level]
                    ]
                else:
                    for i in range(len(grad_global["img_hypernet"]["level_predictors"][level])):
                        grad_global["img_hypernet"]["level_predictors"][level][i] += \
                            grad_local["img_hypernet"]["level_predictors"][level][i] * data_size_local
            # 层级嵌入
            for level in adapt_layers:
                if not grad_global["img_hypernet"]["level_embeddings"][level]:
                    grad_global["img_hypernet"]["level_embeddings"][level] = [
                        g * data_size_local for g in grad_local["img_hypernet"]["level_embeddings"][level]
                    ]
                else:
                    for i in range(len(grad_global["img_hypernet"]["level_embeddings"][level])):
                        grad_global["img_hypernet"]["level_embeddings"][level][i] += \
                            grad_local["img_hypernet"]["level_embeddings"][level][i] * data_size_local

            # -------------------------- 参数聚合（按数据量加权） --------------------------
            if adapter_global is None:
                adapter_global = deepcopy(adapter_local)
                # 初始化参数为 local * data_size_local
                for level in adapt_layers:
                    # 图像适配器
                    for param_g, param_l in zip(
                            adapter_global[level]["img_adapter"].parameters(),
                            adapter_local[level]["img_adapter"].parameters()
                    ):
                        param_g.data = param_l.data * data_size_local
                # 图像kappa超网络
                for param_g, param_l in zip(
                        adapter_global["img_kappa_hypernet"].parameters(),
                        adapter_local["img_kappa_hypernet"].parameters()
                ):
                    param_g.data = param_l.data * data_size_local
            else:
                # 累加参数
                for level in adapt_layers:
                    # 图像适配器
                    for param_g, param_l in zip(
                            adapter_global[level]["img_adapter"].parameters(),
                            adapter_local[level]["img_adapter"].parameters()
                    ):
                        param_g.data += param_l.data * data_size_local
                # 图像kappa超网络
                for param_g, param_l in zip(
                        adapter_global["img_kappa_hypernet"].parameters(),
                        adapter_local["img_kappa_hypernet"].parameters()
                ):
                    param_g.data += param_l.data * data_size_local

            # -------------------------- 损失聚合 --------------------------
            if use_min_loss:
                loss_last_global += loss_local_last_global * data_size_local
                if loss_local_w_prev_min_loss is not None:
                    loss_w_prev_min_loss += loss_local_w_prev_min_loss * data_size_local
                    received_loss_local_w_prev_min_loss = True

            # 存储到列表中用于后续聚合
            loss_local_list.append(loss_local)
            iter_time_local_list.append(iter_time_local)

        # 接收MSG_INFO_CLIENT_TO_SERVER消息
        E_local_list = []
        for n in range(0, n_nodes):
            msg = recv_msg(client_sock_all[n], 'MSG_INFO_CLIENT_TO_SERVER')
            E_local = msg[1]  # 本地计算能量
            E_local_list.append(E_local)
    except Exception as e:
        print(f"错误：接收客户端权重更新时出错: {str(e)}")
        # 清理并退出
        for sock in client_sock_all:
            try:
                sock.close()
            except:
                pass
        try:
            listening_sock.close()
        except:
            pass
        raise

    # -------------------------- 梯度/参数归一化 --------------------------
    # 1. 损失归一化
    if use_min_loss:
        loss_last_global /= data_size_total
        if received_loss_local_w_prev_min_loss:
            loss_w_prev_min_loss /= data_size_total

    # 2. 图像适配器梯度
    for level in adapt_layers:
        if grad_global["img"][level]:
            grad_global["img"][level] = [g / data_size_total for g in grad_global["img"][level]]

    # 4. 图像kappa超网络梯度
    if grad_global["img_hypernet"]["global_encoder"]:
        grad_global["img_hypernet"]["global_encoder"] = [g / data_size_total for g in
                                                         grad_global["img_hypernet"]["global_encoder"]]
    for level in adapt_layers:
        if grad_global["img_hypernet"]["level_predictors"][level]:
            grad_global["img_hypernet"]["level_predictors"][level] = [g / data_size_total for g in
                                                                      grad_global["img_hypernet"]
                                                                          ["level_predictors"][level]]
            if grad_global["img_hypernet"]["level_embeddings"][level]:
                grad_global["img_hypernet"]["level_embeddings"][level] = [g / data_size_total for g in
                                                                          grad_global["img_hypernet"]
                                                                              ["level_embeddings"][level]]

    # 5. 参数归一化
    for level in adapt_layers:
        # 图像适配器
        for param in adapter_global[level]["img_adapter"].parameters():
            param.data /= data_size_total
    # 图像kappa超网络
    for param in adapter_global["img_kappa_hypernet"].parameters():
        param.data /= data_size_total

    # -------------------------- NaN检查（只检查图像适配器和图像超网络） --------------------------
    has_nan = False
    # 检查图像适配器
    for level in adapt_layers:
        for param in adapter_global[level]["img_adapter"].parameters():
            if torch.isnan(param).any():
                has_nan = True
                break
        if has_nan:
            break
    # 检查图像kappa超网络
    if not has_nan:
        for param in adapter_global["img_kappa_hypernet"].parameters():
            if torch.isnan(param).any():
                has_nan = True
                break

    # NaN回滚逻辑
    if has_nan:
        print("*** 检测到NaN，回滚到上一轮权重 ***")
        adapter_global = w_global_prev
        grad_global = deepcopy(prev_grad_global)  # 需在回滚前保存上一轮grad_global
        use_w_global_prev_due_to_nan = True
    else:
        prev_grad_global = deepcopy(grad_global)  # 正常轮次保存梯度
        use_w_global_prev_due_to_nan = False

    # -------------------------- 最小损失逻辑 --------------------------
    if use_min_loss:

        # 更新最小损失适配器
        if loss_last_global < loss_min:
            loss_min = loss_last_global
            adapter_global_min_loss = deepcopy(w_global_prev)
            prev_loss_is_min = True
        else:
            prev_loss_is_min = False

        if loss_last_global > loss_min:
            # 展平所有梯度
            flat_grad_list = []
            # 图像适配器梯度
            for level in adapt_layers:
                for g in grad_global["img"][level]:
                    flat_grad_list.append(g.contiguous().view(-1))
            # 图像kappa超网络梯度
            for g in grad_global["img_hypernet"]["global_encoder"]:
                flat_grad_list.append(g.contiguous().view(-1))
            for level in adapt_layers:
                for g in grad_global["img_hypernet"]["level_predictors"][level]:
                    flat_grad_list.append(g.contiguous().view(-1))
                for g in grad_global["img_hypernet"]["level_embeddings"][level]:
                    flat_grad_list.append(g.contiguous().view(-1))

            flat_grad = torch.cat(flat_grad_list)

        print(f"客户端初始损失聚合值: {loss_last_global:.6f}")
        print(f"当前最小损失: {loss_min:.6f}")

    # -------------------------- 生成与客户端相同的随机参数 --------------------------
    # 使用当前轮次生成相同的参数，确保与客户端一致
    round_params = generate_random_params(n_nodes, current_round)
    p = round_params['p']  # 发射功率 (W)
    r = round_params['r']  # 传输速率 (bps)

    # 计算传输能量
    E_round = 0  # 本轮训练能量消耗
    for n in range(n_nodes):
        # 加上本地计算能量
        if n < len(E_local_list):
            E_round += E_local_list[n]
        # 计算传输能量
        if client_adapters[n] is not None:
            data_size_trans = sum(p.numel() * p.element_size() for p in client_adapters[n].parameters())  # 字节
            time_trans = data_size_trans * 8 / r[n]  # 传输时间（秒）
            E_trans = time_trans * p[n]  # 传输能量（焦耳）
            E_round += E_trans

    # 将本轮能量消耗累加到全局总能量中
    E_total += E_round
    print(f"累计能量消耗: {E_total:.6f} J")

    # -------------------------- 新增指标聚合计算 --------------------------
    # 计算新增指标的平均值
    if loss_local_list:
        average_loss = sum(loss_local_list) / len(loss_local_list)
    else:
        average_loss = 0.0

    if iter_time_local_list:
        average_iter_time = sum(iter_time_local_list) / len(iter_time_local_list)
    else:
        average_iter_time = 0.0

    print(f"客户端训练损失平均值: {average_loss:.6f}")
    print(f"客户端迭代时间平均值: {average_iter_time:.6f}秒")
    print(f"客户端初始损失聚合值: {loss_last_global:.6f}")

    # -------------------------- 写入聚合后的指标到CSV文件 --------------------------
    with open(loss_record_path, 'a') as f:
        # 写入客户端训练后的损失平均值，这更能反映训练进展
        # 使用累积的包含传输时间的总时间并添加总能耗
        f.write(f'{global_step},{average_loss},{cumulative_time_with_transfer},{E_total}\n')

    # 递增全局步骤计数器
    global_step += 1

    # -------------------------- 计算下一轮tau --------------------------
    tau_new = []
    if not use_w_global_prev_due_to_nan:
        if control_alg is not None:
            print("调用 compute_new_tau 前")
            tau_new, delt_f, gam_f = control_alg.compute_new_tau(
                data_size_local_all=data_size_local_all,
                data_size_total=data_size_total,
                tau=min(tau_config),
                delt_f=delt_f,
                gam_f=gam_f
            )
            print("compute_new_tau 返回")
        else:
            tau_new = tau_config if tau_new_resume is None else tau_new_resume
    else:
        tau_new_resume = tau_config
        tau_new = [1] * n_nodes

    # -------------------------- 时间统计 --------------------------
    time_total_all_end = time.time()
    time_total_all = time_total_all_end - time_total_all_start

    # 使用真实时间计算
    total_time_recomputed = time.time() - start_time  # 从开始到现在的真实时间
    total_time = total_time_recomputed  # 实际测量总时间（用于打印）

    stat.collect_stat_end_local_round(tau_actual, control_alg, total_time_recomputed,
                                      loss_last_global)
    print("已经过的时间:", total_time_recomputed)

    if max_training_rounds is None or current_round < max_training_rounds:
        tau_config = tau_new

    # 检查收敛条件
    if last_min_loss == np.inf:
        # 第一次迭代，直接更新
        consecutive_no_improvement = 0
        last_min_loss = loss_min
    else:
        # 使用百分比计算损失改善
        improvement_percentage = ((last_min_loss - loss_min) / last_min_loss) * 100
        if improvement_percentage >= loss_improvement_percentage:
            consecutive_no_improvement = 0
            last_min_loss = loss_min
        else:
            consecutive_no_improvement += 1
    
    # 检查停止条件
    stop_reason = None
    if max_training_rounds is not None and current_round >= max_training_rounds:
        stop_reason = f"达到最大训练轮数: {max_training_rounds}"
    elif time_limit is not None and total_time_recomputed >= time_limit:
        stop_reason = f"达到时间上限: {total_time_recomputed:.2f}秒"
    elif energy_limit is not None and E_total >= energy_limit:
        stop_reason = f"达到能量上限: {E_total:.2f}焦耳"
    elif convergence_rounds is not None and consecutive_no_improvement >= convergence_rounds:
        stop_reason = f"达到收敛条件: 连续{consecutive_no_improvement}轮损失未改善"
    
    if stop_reason:
        print(f"\n训练停止原因: {stop_reason}")
        is_last_round = True
    
    # 判断是否终止训练
    if is_last_round:
        break
    if is_eval_only:
        tau_config = [1] * n_nodes
        is_last_round = True
    if is_last_round_tmp:
        if use_min_loss:
            is_eval_only = True
        else:
            is_last_round = True

# 训练循环结束后保存最佳模型
# 保存最佳模型
w_eval = None
if use_min_loss and adapter_global_min_loss is not None:
    w_eval = adapter_global_min_loss
elif adapter_global is not None:
    w_eval = adapter_global

if w_eval is not None:
    save_path = os.path.join(args.output_dir, f"adapter_best_tau_adaptive.pt")
    torch.save(w_eval.state_dict(), save_path)
    print(f"最佳适配器已保存至: {save_path}")
else:
    print("警告：没有可用的模型可以保存")

# 收集全局统计
stat.collect_stat_end_global_round(
    tau_setup="adaptive",
    total_time=total_time,
    total_time_recomputed=total_time_recomputed,
    E_total=E_total
)

# 关闭所有连接
for sock in client_sock_all:
    sock.close()
listening_sock.close()
print("所有客户端连接已关闭，训练完成")