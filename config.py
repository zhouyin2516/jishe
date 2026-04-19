
import numpy as np
import os
import argparse
import yaml
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration

# 加载配置文件
def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

# 加载配置
config = load_config()

SERVER_ADDR= config["server"].get("server_addr", "localhost")   # When running in a real distributed setting, change to the server's IP address
SERVER_PORT = config["server"].get("server_port", 5100)

dataset_file_path = os.path.join(os.path.dirname(__file__), 'datasets')
dataset_root = config["server"].get("dataset_root", "data/medical_images")
results_file_path = os.path.join(os.path.dirname(__file__), 'results')
single_run_results_file_path = results_file_path + '/SingleRun.csv'
multi_run_results_file_path = results_file_path + '/MultipleRuns.csv'

control_param_phi = 0.00005

n_nodes = config["server"].get("n_nodes", 4)  # Specifies the total number of clients

moving_average_holding_param = config["training"].get("moving_average_holding_param", 0.9)  # Moving average coefficient to smooth the estimation of beta, delta, and rho

# Choose whether to run a single instance and plot the instantaneous results or
# run multiple instances and plot average results
single_run = config["training"].get("single_run", True)



# If true, the weight corresponding to minimum loss (the loss is estimated if using stochastic gradient descent)
# returned. If false, the weight at the end is returned. Setting use_min_loss = True corresponds to the latest
# theoretical bound for the **DISTRIBUTED** case.
# For the **CENTRALIZED** case, set use_min_loss = False,
# because convergence of the final value can be guaranteed in the centralized case.
use_min_loss = True  # 始终使用最小损失策略

# Specifies whether all the data should be read when using stochastic gradient descent.
# Reading all the data requires much more memory but should avoid slowing down due to file reading.
read_all_data_for_stochastic = True

MAX_CASE = 4  # Specifies the maximum number of cases, this should be a constant equal to 4
tau_max = 100  # Specifies the maximum value of tau

# 只使用自适应策略，tau_setup = -1
single_run = config["training"].get("single_run", True)
case_range = config["training"].get("case_range", [0])   # 只使用一个case
tau_setup_all = config["training"].get("tau_setup_all", [-1])   # 只使用自适应策略
sim_runs = config["training"].get("sim_runs", [0])   # 只使用一个随机种子

max_training_rounds = config["training"]["stop_conditions"].get("num_train_epochs", 200)  # 从停止条件中读取训练轮数

# 动量设置（始终启用）
momentum_value = config["server"].get("momentum_value", 0.9)  # 动量值



# 固定参数配置
N0_dBm = -174  # 噪声功率谱密度 (dBm/Hz)
N0 = 10**(N0_dBm / 10) * 1e-3  # 转换为 W/Hz

# 生成每轮随机参数的函数
def generate_random_params(n_nodes, round_num):
    """
    生成每轮的随机参数，使用高斯分布
    相同轮次的不同迭代策略将生成相同的参数，以便比较
    
    参数：
    n_nodes: 客户端数量
    round_num: 当前轮次编号
    
    返回：字典，包含每轮的随机参数
    """
    params = {}
    
    # 为当前轮次设置固定的随机种子，确保不同迭代策略在相同轮次生成相同参数
    # 使用轮次编号作为种子，确保相同轮次生成相同参数
    np.random.seed(round_num)
    
    # 子信道带宽 b[i]d​：0.5 MHz ~ 1 MHz
    # 总和限制在3 MHz以内
    total_bandwidth_limit = 3.0  # MHz
    
    # 生成初始带宽分布
    b_mhz = np.random.normal(0.75, 0.125, n_nodes)
    # 确保在[0.5, 1]范围内
    b_mhz = np.clip(b_mhz, 0.5, 1)
    
    # 检查并调整总和
    current_total = np.sum(b_mhz)
    if current_total > total_bandwidth_limit:
        # 按比例缩放以满足总和限制
        scaling_factor = total_bandwidth_limit / current_total
        b_mhz *= scaling_factor
        # 确保缩放后仍在[0.5, 1]范围内
        b_mhz = np.clip(b_mhz, 0.5, 1)
        # 再次检查总和，若仍超过则将最大的带宽调整为剩余值
        current_total = np.sum(b_mhz)
        if current_total > total_bandwidth_limit:
            # 找到最大的带宽值并调整
            max_index = np.argmax(b_mhz)
            b_mhz[max_index] = max(0.5, total_bandwidth_limit - (current_total - b_mhz[max_index]))
    
    params['b'] = b_mhz * 1e6  # 转换为 Hz
    
    # 发射功率 p[i]d​：5 dBm ~ 20 dBm
    # 均值为12.5 dBm，标准差为3.75 dBm
    p_dBm = np.random.normal(12.5, 3.75, n_nodes)
    # 确保在[5, 20]范围内
    p_dBm = np.clip(p_dBm, 5, 20)
    params['p_dBm'] = p_dBm
    params['p'] = 10**(p_dBm / 10) * 1e-3  # 转换为 W
    
    # CPU 频率 f[i]d​：0.5 GHz ~ 1 GHz
    # 均值为0.75 GHz，标准差为0.125 GHz
    f_ghz = np.random.normal(0.75, 0.125, n_nodes)
    # 确保在[0.5, 1]范围内
    f_ghz = np.clip(f_ghz, 0.5, 1)
    params['f'] = f_ghz * 1e9  # 转换为 Hz
    
    # 计算能量消耗系数 κ[i]d​：10−28 焦耳·秒²/周期² ~ 5×10−28 焦耳·秒²/周期²
    # 均值为3e-28 焦耳·秒²/周期²，标准差为1e-28 焦耳·秒²/周期²
    kappa = np.random.normal(3e-28, 1e-28, n_nodes)
    # 确保在[1e-28, 5e-28]范围内
    kappa = np.clip(kappa, 1e-28, 5e-28)
    params['kappa'] = kappa
    
    # 单样本处理 CPU 周期 o[i]d
    # 均值为4e7 cycles，标准差为5e6 cycles
    o = np.random.normal(4e7, 5e6, n_nodes)
    # 确保在[3e7, 5e7]范围内
    o = np.clip(o, 3e7, 5e7)
    params['o'] = o

    # 随机生成信道增益
    # 随机分布用户的位置，基站在中心位置（一次性随机初始化）
    # 500m x 500m 区域，基站位于中心
    user_positions = np.random.uniform(0, 500, (n_nodes, 2))
    bs_position = np.array([250, 250])
    # 计算每个客户端到服务器的距离（km）
    distances = np.array([np.linalg.norm(pos - bs_position) / 1000 for pos in user_positions])
    PL = 128.1 + 37.6 * np.log10(distances)
    params['PL'] = PL
    # 计算信道增益（线性值）
    g = 10**(-PL / 10)
    params['g'] = g
    
    # 计算传输速率 r[i]d​（使用香农公式）
    # r = b * log2(1 + (p * g) / (b * N0))
    r = []
    for i in range(n_nodes):
        signal_power = params['p'][i] * g[i]  # 接收信号功率
        noise_power = params['b'][i] * N0  # 噪声功率
        sinr = signal_power / noise_power  # 信噪比
        rate = params['b'][i] * np.log2(1 + sinr)  # 香农容量（bps）
        r.append(rate)
    params['r'] = np.array(r)
    
    return params

# 计算客户端每一轮的迭代时间和能耗
def calculate_time_energy(tau, C, sample_size, f, kappa):
    """
    计算客户端每一轮的迭代时间和能耗
    参数：
    tau: 迭代次数
    C: 单样本处理 CPU 周期
    sample_size: 样本处理量
    f: CPU 频率 (Hz)
    kappa: 计算能量消耗系数 (J/cycle)
    
    返回：
    t: 迭代时间 (s)
    energy: 能耗 (J)
    """
    # 迭代时间t[i]d = 迭代次数tau[i]d * 单样本处理 CPU 周期 C[i]d * 样本处理量 / CPU 频率 f[i]d
    t = tau * C * sample_size / f
    
    # 能耗 = 计算能量消耗系数 κ[i]d * 迭代次数tau[i]d * 样本处理量 * 单样本处理 CPU 周期 C[i]d * CPU 频率 f[i]d^2
    energy = kappa * tau * sample_size * C * (f ** 2)
    
    return t, energy

# 简化的参数解析函数
def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--base_model_path",
        type=str,
        default=config["model"].get("base_model_path", "model"),
        required=False,
        help="Path to base pretrained model directory.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=config["model"].get("revision", None),
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--unet_subfolder",
        type=str,
        default=config["model"].get("unet_subfolder", "unets/4/unet"),
        required=False,
        help="UNet subfolder path.",
    )
    parser.add_argument(
        "--text_encoder_subfolder",
        type=str,
        default=config["model"].get("text_encoder_subfolder", "text_encoder"),
        required=False,
        help="Text encoder subfolder path.",
    )
    parser.add_argument(
        "--vae_subfolder",
        type=str,
        default=config["model"].get("vae_subfolder", "vae"),
        required=False,
        help="VAE subfolder path.",
    )
    parser.add_argument(
        "--scheduler_subfolder",
        type=str,
        default=config["model"].get("scheduler_subfolder", "scheduler"),
        required=False,
        help="Scheduler subfolder path.",
    )
    parser.add_argument(
        "--tokenizer_subfolder",
        type=str,
        default=config["model"].get("tokenizer_subfolder", "tokenizer"),
        required=False,
        help="Tokenizer subfolder path.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=config["model"].get("output_dir", "output/"),
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=config["model"].get("resolution", 512),
        help=("The resolution for input images, all the images in the train/validation dataset will be resized to this resolution"),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=config["model"].get("train_batch_size", 16), help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=config["training"]["stop_conditions"].get("num_train_epochs", 100))
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=config["model"].get("gradient_accumulation_steps", 1),
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        default=config["model"].get("gradient_checkpointing", False),
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=config["model"].get("learning_rate", 0.0001),
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=config["model"].get("mixed_precision", "fp16"),
        choices=["no", "fp16", "bf16"],
        help="Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10.and an Nvidia Ampere GPU. Default to the value of accelerate config of the current system or the flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=config["model"].get("checkpointing_steps", 100),
        help="Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming training using `--resume_from_checkpoint`.",
    )
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=config["model"].get("non_ema_revision", None),
        required=False,
        help="Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or remote repository specified with --pretrained_model_name_or_path.",
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args

# 自动检测UNet上采样层数并选择最接近输出的两层
def get_adapt_layers(unet=None):
    """自动检测UNet上采样层数并返回最接近输出的两层"""
    if unet is None:
        # 默认值，当UNet未提供时使用
        return ["up2", "up3"]
    
    # 检测up_blocks的长度
    num_up_blocks = len(unet.up_blocks)
    if num_up_blocks < 2:
        # 如果上采样层数少于2，则使用所有层
        return [f"up{i}" for i in range(num_up_blocks)]
    else:
        # 选择最接近输出的两层（最后两层）
        return [f"up{i}" for i in range(num_up_blocks-2, num_up_blocks)]

# 自动检测UNet上采样层的输出维度
def get_img_dims_per_level(unet=None):
    """自动检测UNet上采样层的输出维度"""
    if unet is None:
        # 默认值，当UNet未提供时使用
        return {
            "up0": 1280,
            "up1": 1280,
            "up2": 640,
            "up3": 320
        }
    
    # 检测每个上采样层的输出维度
    img_dims = {}
    for i, up_block in enumerate(unet.up_blocks):
        # 假设up_block的最后一个模块是输出卷积层
        # 尝试获取输出通道数
        if hasattr(up_block, 'resnets') and up_block.resnets:
            # 对于有resnets的情况，获取最后一个resnet的输出通道
            last_resnet = up_block.resnets[-1]
            if hasattr(last_resnet, 'conv2') and hasattr(last_resnet.conv2, 'out_channels'):
                img_dims[f"up{i}"] = last_resnet.conv2.out_channels
        elif hasattr(up_block, 'conv') and hasattr(up_block.conv, 'out_channels'):
            # 对于直接有conv的情况
            img_dims[f"up{i}"] = up_block.conv.out_channels
        else:
            # 如果无法自动检测，使用默认值
            img_dims[f"up{i}"] = 1280 if i < 2 else 640 if i < 3 else 320
    
    return img_dims

# 共享模型架构参数（默认值，实际运行时会自动检测）
img_dims_per_level = get_img_dims_per_level()

# 要适配的上采样层（默认值，实际运行时会自动检测）
adapt_layers = get_adapt_layers()


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

args = parse_args()
args.num_processes=1
args.mixed_precision=config["model"]["mixed_precision"]

args.use_ema = True
args.resolution=config["model"]["resolution"]
args.center_crop = True
args.random_flip = True
args.train_batch_size=config["model"]["train_batch_size"]
args.gradient_accumulation_steps=config["model"]["gradient_accumulation_steps"]
args.gradient_checkpointing = config["model"]["gradient_checkpointing"]
args.learning_rate=config["model"]["learning_rate"]
args.max_grad_norm=1
args.validation_prompts = config["text"]["training_text"]
args.output_dir=config["model"]["output_dir"]
args.logging_dir = "logs"
args.base_model_path=config["model"]["base_model_path"]
args.unet_subfolder=config["model"]["unet_subfolder"]
args.text_encoder_subfolder=config["model"]["text_encoder_subfolder"]
args.vae_subfolder=config["model"]["vae_subfolder"]
args.scheduler_subfolder=config["model"]["scheduler_subfolder"]
args.tokenizer_subfolder=config["model"]["tokenizer_subfolder"]
args.revision = config["model"].get("revision")
args.non_ema_revision = config["model"].get("non_ema_revision")
args.unet_name = config["model"].get("unet_name")
logging_dir = os.path.join(args.output_dir, args.logging_dir)
accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

# 初始化 Accelerator（不使用日志集成）
accelerator = Accelerator(
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    mixed_precision=args.mixed_precision,
    project_config=accelerator_project_config,
)
