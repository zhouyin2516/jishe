import gc
from itertools import chain
import socket
import time
import struct
import warnings
import yaml

import cv2
import numpy as np
import pydicom
import torch
import torch.nn.functional as F
from pathlib import Path
from torch import nn
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import DDPMScheduler, AutoencoderKL, UNet2DConditionModel
from torchvision import transforms
from Adapter import ImageAdapter
from control_algorithm.adaptive_tau import ControlAlgAdaptiveTauClient, ControlAlgAdaptiveTauServer
from util.utils import send_msg, recv_msg
from diffusers.utils import is_wandb_available, make_image_grid
import os
import torch.utils.checkpoint

from PIL import Image
import copy
import math

# 加载配置文件
def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

# 加载配置
config = load_config()


torch.backends.cudnn.benchmark = True
torch.utils.checkpoint.set_checkpoint_early_stop(True)

if is_wandb_available():
    import wandb
# Configurations are in a separate config.py file
from config import SERVER_ADDR, SERVER_PORT, parse_args, generate_random_params, accelerator, img_dims_per_level, \
    timestep_range, n_nodes, timestep_ranges


# 解析参数
args = parse_args()

# 恢复完整的能量消耗计算函数
def calculate_energy_consumption(tau_actual, accelerator, round_params=None, client_id=None):
    """
    计算CPU/GPU的能耗，使用CPU仿真参数，对齐参考论文公式
    Args:
        tau_actual: 本地实际迭代次数
        accelerator: 加速设备对象（获取设备类型、索引）
        round_params: 当前轮次的仿真参数（包含f, kappa, o等）
        client_id: 客户端ID，用于从round_params数组中选择对应的值
    Returns:
        Ek: 总能耗（J）
    """
    # 记录实际运行的设备类型和信息
    Ek = None

    # 使用CPU仿真参数进行计算
    if round_params is not None:
        # 如果是numpy数组且提供了client_id，则选择对应客户端的值
        f = round_params['f'][client_id] if hasattr(round_params['f'], '__getitem__') and client_id is not None else \
            round_params['f']
        kappa = round_params['kappa'][client_id] if hasattr(round_params['kappa'],
                                                            '__getitem__') and client_id is not None else round_params[
            'kappa']
        o = round_params['o'][client_id] if hasattr(round_params['o'], '__getitem__') and client_id is not None else \
            round_params['o']

        # 能耗 Ek = kappa * tau * 样本处理量 * 单样本处理 CPU 周期 * CPU 频率^2
        Ek = kappa * tau_actual * args.train_batch_size * o * (f ** 2)

    return Ek


class MedicalDataset(torch.utils.data.Dataset):
    def __init__(self, data_paths, transforms):
        self.data_paths = data_paths
        self.transforms = transforms

    def __getitem__(self, idx):
        return str(self.data_paths[idx])  # 返回图像路径

    def __len__(self):
        return len(self.data_paths)


class HeavyBallMGD(torch.optim.Optimizer):
    def __init__(self, params, lr=5e-4, momentum=0.9):
        # 论文中动量系数默认0.9（Section 5.1实验设置）
        defaults = dict(lr=lr, momentum=momentum)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data  # 当前梯度（论文中的?L(θ?)）

                # 累积历史梯度（动量项）
                state = self.state[p]
                # 初始化动量缓存（存储历史梯度的加权累积）
                if 'momentum_buffer' not in state:
                    # 初始缓存为0（无历史梯度）
                    state['momentum_buffer'] = torch.zeros_like(grad)

                # 核心修正2：动量项 = 动量系数×历史缓存 + 当前梯度（论文逻辑）
                momentum_buffer = state['momentum_buffer']
                momentum_buffer.mul_(momentum).add_(grad)  # 等价于：buffer = β·buffer + ?L(θ?)

                # 核心修正3：兼容混合精度（论文支持异构设备）
                if hasattr(p.grad, 'scale'):
                    # 处理GradScaler缩放的梯度
                    update = -lr * p.grad.scale * momentum_buffer
                else:
                    update = -lr * momentum_buffer

                # 参数更新（论文公式：θ??? = θ? - α·(β·buffer + ?L(θ?))）
                p.data.add_(update)

                # 更新动量缓存（保存当前累积梯度）
                state['momentum_buffer'] = momentum_buffer
        return loss


def get_timestep_embedding(timesteps, embedding_dim):
    """
    将时间步转换为正弦位置编码
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


class ImageKappaHyperNetwork(nn.Module):
    """图像专用kappa超网络：仅预测图像kappa"""

    def __init__(self, text_dim=768, time_dim=512, level_embed_dim=32, levels=None):
        super().__init__()
        self.text_dim = text_dim
        self.time_dim = time_dim
        self.level_embed_dim = level_embed_dim

        # 默认层级，如果没有提供
        if levels is None:
            levels = ["up0", "up1", "up2", "up3"]

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
                nn.Linear(128, 1),  # 仅输出图像kappa
                nn.Sigmoid()
            ) for level in levels
        })

        self.level_embeddings = nn.ParameterDict({
            level: nn.Parameter(torch.randn(level_embed_dim))
            for level in levels
        })
        self._init_level_embeddings()

    def _init_level_embeddings(self):
        for level in self.level_embeddings:
            nn.init.normal_(self.level_embeddings[level], mean=0.0, std=0.02)

    def forward(self, text_emb, timestep, current_level=None):
        dtype = text_emb.dtype
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
            # 使用实际的层级列表，而不是硬编码的列表
            for level in self.level_predictors.keys():
                level_emb = self.level_embeddings[level].unsqueeze(0).expand(batch_size, -1).to(dtype)
                level_aware_feat = torch.cat([global_feat, level_emb], dim=1).to(dtype)
                img_kappa = self.level_predictors[level](level_aware_feat).to(dtype)
                all_img_kappas[level] = img_kappa[:, 0].unsqueeze(-1).unsqueeze(-1).to(dtype)
            return all_img_kappas


def save_model_card(
        args,
        repo_id: str,
        images=None,
        repo_folder=None,
):
    img_str = ""
    if len(images) > 0:
        image_grid = make_image_grid(images, 1, len(args.validation_prompts))
        image_grid.save(os.path.join(repo_folder, "val_imgs_grid.png"))
        img_str += "![val_imgs_grid](./val_imgs_grid.png)\n"

    yaml = f"""
---
license: creativeml-openrail-m
base_model: {args.base_model_path}
datasets:
- {args.dataset_name}
tags:
- stable-diffusion
- stable-diffusion-diffusers
- text-to-image
- diffusers
inference: true
---
    """
    model_card = f"""
# Text-to-image finetuning - {repo_id}

This pipeline was finetuned from **{args.base_model_path}** on the **{args.dataset_name}** dataset. Below are some example images generated with the finetuned pipeline using the following prompts: {args.validation_prompts}: \n
{img_str}

## Pipeline usage

You can use the pipeline like so:

```python
from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained("{repo_id}", torch_dtype=torch.float16)
prompt = "{args.validation_prompts[0]}"
image = pipeline(prompt).images[0]
image.save("my_image.png")
```

## Training info

These are the key hyperparameters used during training:

* Epochs: {args.num_train_epochs}
* Learning rate: {args.learning_rate}
* Batch size: {args.train_batch_size}
* Gradient accumulation steps: {args.gradient_accumulation_steps}
* Image resolution: {args.resolution}
* Mixed-precision: {args.mixed_precision}

"""
    wandb_info = ""
    if is_wandb_available():
        wandb_run_url = None
        if wandb.run is not None:
            wandb_run_url = wandb.run.url

    if wandb_run_url is not None:
        wandb_info = f"""
More information on all the CLI arguments and the environment are available on your [`wandb` run page]({wandb_run_url}).
"""

    model_card += wandb_info

    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)


num_inference_steps = 100
args = parse_args()
sock = socket.socket()
sock.connect((SERVER_ADDR, SERVER_PORT))
from config import dataset_root
import pathlib

# 转换为Path对象
dataset_root = pathlib.Path(dataset_root)
unet = UNet2DConditionModel.from_pretrained(
    args.base_model_path, subfolder=args.unet_subfolder, revision=args.non_ema_revision
)
if args.gradient_checkpointing:
    unet.enable_gradient_checkpointing()
text_encoder = CLIPTextModel.from_pretrained(
    args.base_model_path, subfolder=args.text_encoder_subfolder, revision=args.revision
)
vae = AutoencoderKL.from_pretrained(
    args.base_model_path, subfolder=args.vae_subfolder, revision=args.revision
)

print('---------------------------------------------------------------------------')

prompt_file_path = Path(__file__).parent / "prompt.txt"
with open(prompt_file_path, "r", encoding="utf-8") as f:
    text = f.read()
batch_size_prev = None
total_data_prev = None
sim_prev = None
omega = 0
device = None
try:
    while True:
        msg = recv_msg(sock, 'MSG_INIT_SERVER_TO_CLIENT')
        # ['MSG_INIT_SERVER_TO_CLIENT',
        # use_control_alg, use_min_loss,
        # adapter, client_id]

        control_alg_server_instance = msg[1]
        use_min_loss = msg[2]
        adapter_global_state_dict = msg[3]
        client_id = msg[4]

        from config import get_adapt_layers, get_img_dims_per_level
        # 自动检测UNet上采样层数并选择最接近输出的两层
        adapt_layers = get_adapt_layers()
        # 自动检测UNet上采样层的输出维度
        img_dims_per_level = get_img_dims_per_level()
        print(f"自动检测到的上采样层: {adapt_layers}")
        print(f"自动检测到的上采样层维度: {img_dims_per_level}")
        
        adapter = nn.ModuleDict()
        for level in adapt_layers:
            adapter[level] = nn.ModuleDict({
                "img_adapter": ImageAdapter(img_dim=img_dims_per_level[level])  # 通道数与服务器一致
            })
        # 图像超网络保持不变
        adapter["img_kappa_hypernet"] = ImageKappaHyperNetwork(
            text_dim=768,
            time_dim=512,
            level_embed_dim=32,
            levels=adapt_layers
        )
        # 加载服务器发送的参数（覆盖客户端初始化的参数）
        adapter.load_state_dict(adapter_global_state_dict)
        print("客户端适配器已成功加载服务器参数")

        adapter_min = copy.deepcopy(adapter)
        # 使用配置文件中的数据集路径
        client_data_dir = Path(config["client"].get("dataset_path", "dataset_case4")) / f'{client_id + 1}'

        # 搜索多种常见的图像格式
        dataset = []
        dataset.extend(list(client_data_dir.glob('*.png')))
        dataset.extend(list(client_data_dir.glob('*.jpg')))
        dataset.extend(list(client_data_dir.glob('*.jpeg')))
        dataset.extend(list(client_data_dir.glob('*.bmp')))
        dataset.extend(list(client_data_dir.glob('*.tif')))
        dataset.extend(list(client_data_dir.glob('*.tiff')))

        # 检查数据集是否为空
        if not dataset:
            print(f"警告：在目录 {client_data_dir} 中没有找到任何图像文件！")
            print(f"目录内容：{list(client_data_dir.iterdir())}")

        # 数据载入
        # Load scheduler, tokenizer and models.
        noise_scheduler = DDPMScheduler.from_pretrained(args.base_model_path, subfolder=args.scheduler_subfolder)

        tokenizer = CLIPTokenizer.from_pretrained(
            args.base_model_path, subfolder=args.tokenizer_subfolder, revision=args.revision
        )


        # 改进的数据预处理：支持大小差异大的医学图像
        class AdaptiveResize(object):
            def __init__(self, size, interpolation=transforms.InterpolationMode.BILINEAR, pad_value=0):
                self.size = size  # 目标尺寸（正方形）
                self.interpolation = interpolation
                self.pad_value = pad_value  # 医学图像常用黑色填充

            def __call__(self, img):
                w, h = img.size
                aspect_ratio = w / h

                # 步骤1：按纵横比缩放到目标尺寸的最大边，保持比例
                if w > h:
                    new_w = self.size
                    new_h = int(new_w / aspect_ratio)
                else:
                    new_h = self.size
                    new_w = int(new_h * aspect_ratio)
                img = transforms.Resize((new_h, new_w), interpolation=self.interpolation)(img)

                # 步骤2：计算padding值（居中填充）
                pad_w = max(self.size - new_w, 0)
                pad_h = max(self.size - new_h, 0)
                pad_left = pad_w // 2
                pad_right = pad_w - pad_left
                pad_top = pad_h // 2
                pad_bottom = pad_h - pad_top

                # 步骤3：填充到正方形（避免拉伸）
                img = transforms.Pad((pad_left, pad_top, pad_right, pad_bottom), fill=self.pad_value)(img)
                return img


        # 增强的窗宽窗位调整类
        class AdaptiveWindowing(object):
            def __call__(self, img):
                # 转换为numpy数组
                img_np = np.array(img, dtype=np.float32)

                # 分析图像统计信息以动态调整窗宽窗位
                img_mean = np.mean(img_np)
                img_std = np.std(img_np)

                # 根据图像亮度动态选择窗位
                if img_mean < 100:  # 较暗图像（可能是肺窗）
                    window_center = -600  # 肺窗窗位
                    window_width = 1500  # 肺窗窗宽
                else:  # 较亮图像（可能是纵隔窗）
                    window_center = 40  # 纵隔窗窗位
                    window_width = 400  # 纵隔窗窗宽

                # 窗宽窗位调整
                min_val = window_center - window_width / 2
                max_val = window_center + window_width / 2
                img_np = np.clip(img_np, min_val, max_val)
                img_np = (img_np - min_val) / (max_val - min_val)

                # 根据图像质量决定是否应用增强
                if img_std < 30:  # 低对比度图像
                    # 应用对比度增强
                    img_np = np.clip(img_np * 1.5 - 0.25, 0, 1)

                return Image.fromarray((img_np * 255).astype(np.uint8))


        train_transforms = transforms.Compose(
            [
                AdaptiveResize(args.resolution),
                transforms.Grayscale(num_output_channels=3),
                AdaptiveWindowing(),
                transforms.RandomRotation(degrees=5),  # 添加轻微旋转增强
                transforms.ColorJitter(brightness=0.1, contrast=0.1),  # 添加亮度对比度增强
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )


        # 2. 统一的医学图像加载（适配CT/MRI，移除重复窗宽窗位）
        def load_medical_image(image_path, modal_type="MRI"):
            """修复后的医学图像加载函数：
            补充 DICOM 的 HU 值转换
            仅执行一次窗宽窗位调整
            修复伪影去除逻辑
            增加异常处理"""

            try:
                if image_path.endswith(".dcm"):
                    ds = pydicom.dcmread(image_path)
                    img_np = ds.pixel_array.astype(np.float32)

                    # 修复缩进：HU值转换应在DICOM判断内
                    if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                        img_np = img_np * ds.RescaleSlope + ds.RescaleIntercept
                        if modal_type == "CT":
                            q1, q99 = np.percentile(img_np, [1, 99])
                            window_center = (q1 + q99) / 2
                            window_width = q99 - q1
                        elif modal_type == "MRI":
                            window_center = np.mean(img_np)
                            window_width = 2 * np.std(img_np)
                        else:
                            window_center = 127.5
                            window_width = 255.0
                        min_val = window_center - window_width / 2
                        max_val = window_center + window_width / 2
                        img_np = np.clip(img_np, min_val, max_val)
                        img_np = (img_np - min_val) / (max_val - min_val + 1e-8)

                        img_np_uint8 = (img_np * 255).astype(np.uint8)
                        img_np_uint8 = cv2.medianBlur(img_np_uint8, ksize=3)
                        img = Image.fromarray(img_np_uint8).convert("L")
                    else:
                        img = Image.fromarray((img_np).astype(np.uint8)).convert("L")
                else:
                    img = Image.open(image_path).convert("L")
                    img_np = np.array(img, dtype=np.float32) / 255.0

                    if modal_type == "CT":
                        window_center = 40
                        window_width = 400
                    else:
                        window_center = np.mean(img_np)
                        window_width = 2 * np.std(img_np)
                    min_val = window_center - window_width / 2
                    max_val = window_center + window_width / 2
                    img_np = np.clip(img_np, min_val, max_val)
                    img_np = (img_np - min_val) / (max_val - min_val + 1e-8)
                    img = Image.fromarray((img_np * 255).round().astype(np.uint8)).convert("L")
                    img.info = {}
                return img
            except Exception as e:
                warnings.warn(f"Failed to load {image_path}: {e}, return blank image")
                return Image.new("L", (512, 512), color=0)


        # 4. 批量加载优化（collate_fn+合理batch_size）
        def medical_collate_fn(batch):
            """确保批量图像尺寸一致"""
            images = []
            for img_path in batch:
                img = load_medical_image(img_path, modal_type="MRI")  # 按需切换CT/MRI
                img = train_transforms(img)
                images.append(img)
            return torch.stack(images)


        def image_to_latent_base(image_paths, vae, transforms, timesteps):
            """生成指定timestep的带噪潜变量（内存优化版本）"""

            # 预分配内存
            device = vae.device
            dtype = vae.dtype

            # 单样本处理以减少内存使用
            latent_list = []
            noise_list = []

            with torch.no_grad():
                for i, img_path in enumerate(image_paths):
                    img = load_medical_image(img_path)
                    tensor = transforms(img).to(device, dtype)

                    # 单样本编码
                    encoder_output = vae.encode(tensor.unsqueeze(0), return_dict=True)
                    target_latent = encoder_output.latent_dist.sample() * 0.18215

                    noise = torch.randn_like(target_latent)
                    noisy_latent = noise_scheduler.add_noise(target_latent, noise, timesteps[i].unsqueeze(0))

                    latent_list.append(noisy_latent)
                    noise_list.append(noise)

                    # 清理内存
                    del tensor, encoder_output, target_latent
                    torch.cuda.empty_cache()

            # 最后再堆叠，减少内存峰值
            noisy_latent = torch.cat(latent_list, dim=0)
            noise = torch.cat(noise_list, dim=0)

            # 生成目标潜变量（只用于计算，不存储完整批次）
            return noisy_latent, timesteps, noise


        # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
        # as these weights are only used for inference, keeping weights in full precision is not required.
        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
            args.mixed_precision = accelerator.mixed_precision

        # Move text_encode and vae to gpu and cast to weight_dtype
        text_encoder.to(accelerator.device, dtype=weight_dtype)
        vae.to(accelerator.device, dtype=weight_dtype)

        data_size_local = len(dataset)

        if isinstance(control_alg_server_instance, ControlAlgAdaptiveTauServer):
            control_alg = ControlAlgAdaptiveTauClient()
        else:
            control_alg = None

        w_prev_min_loss = None
        w_last_global = None

        msg = ['MSG_DATA_PREP_FINISHED_CLIENT_TO_SERVER']
        send_msg(sock, msg)

        Dk = len(dataset)

        batch_size = args.train_batch_size  # 设置统一的批量大小
        dataset_loader = torch.utils.data.DataLoader(
            MedicalDataset(dataset, train_transforms),
            batch_size=batch_size,
            shuffle=True,
            drop_last=True
        )
        if device is None:
            device = accelerator.device if accelerator else torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
            unet.to(device)
            adapter.to(device)

        # 添加全局step计数器，确保损失记录的step是连续的
        global_loss_step = 0

        # 添加轮次计数器，用于生成固定的随机参数
        round_counter = 0

        while True:
            round_counter += 1  # 每轮递增
            print('---------------------------------------------------------------------------')

            unet.eval()
            text_encoder.eval()
            vae.eval()

            for param in chain(unet.parameters(), vae.parameters(), text_encoder.parameters()):
                param.requires_grad = False

            msg = recv_msg(sock, 'MSG_WEIGHT_TAU_SERVER_TO_CLIENT')
            # ['MSG_WEIGHT_TAU_SERVER_TO_CLIENT', w_global, lambda, is_last_round, prev_loss_is_min, w_global_min_loss]
            adapter_dict = msg[1]
            tau_config = msg[2]
            is_last_round = msg[3]
            prev_loss_is_min = msg[4]
            w_global_min_loss_dict = msg[5]
            grad_global = msg[6]
            from config import adapt_layers
            if grad_global is not None:
                for adapt_type in ["img"]:
                    for level in adapt_layers:
                        # 遍历该层级的每个梯度张量，移动设备
                        grad_global[adapt_type][level] = [
                            g.to(accelerator.device) for g in grad_global[adapt_type][level]
                        ]

                    # 图像超网络梯度加载
                    if "img_hypernet" in grad_global:
                        grad_global["img_hypernet"]["global_encoder"] = [g.to(accelerator.device) for g in
                                                                         grad_global["img_hypernet"]["global_encoder"]]
                        for level in adapt_layers:
                            grad_global["img_hypernet"]["level_predictors"][level] = [g.to(accelerator.device) for g in
                                                                                      grad_global["img_hypernet"]
                                                                                          ["level_predictors"][level]]
                            grad_global["img_hypernet"]["level_embeddings"][level] = [g.to(accelerator.device) for g in
                                                                                      grad_global["img_hypernet"]
                                                                                          ["level_embeddings"][level]]
            adapter.load_state_dict(adapter_dict)
            # 确保 adapter 在 GPU 上
            adapter = adapter.to(accelerator.device)
            # 使用 config.py 中定义的 img_dims_per_level（与服务器相同）

            # 初始化 w_global_min_loss with the same structure as server's adapter_global
            from config import adapt_layers
            w_global_min_loss = nn.ModuleDict()
            for level in adapt_layers:
                w_global_min_loss[level] = nn.ModuleDict({
                    "img_adapter": ImageAdapter(
                        img_dim=img_dims_per_level[level]
                    )
                })
            w_global_min_loss["img_kappa_hypernet"] = ImageKappaHyperNetwork(
                text_dim=768, time_dim=512, level_embed_dim=32, levels=adapt_layers
            )
            if w_global_min_loss_dict is not None:
                w_global_min_loss.load_state_dict(w_global_min_loss_dict)
                adapter_min = w_global_min_loss
                # 确保 adapter_min 也被移动到 GPU
                adapter_min = adapter_min.to(accelerator.device)

            if prev_loss_is_min or ((w_prev_min_loss is None) and (w_last_global is not None)):
                w_prev_min_loss = w_last_global

            if control_alg is not None:
                control_alg.init_new_round()

            # 每轮开始时生成随机参数，传入轮次编号确保相同轮次生成相同参数
            round_params = generate_random_params(n_nodes, round_counter)

            # Perform local iteration
            loss_last_global = None  # Only the loss at starting time is from global model parameter

            tau_actual = 0
            last_grad = None
            last_loss = None

            last_iter_time = 0.0
            initial_loss = None
            w_last_global = copy.deepcopy(adapter)
            loss_w_prev_min_loss = None
            accumulated_grads = None

            # We need to initialize the trackers we use, and also store our configuration.
            # The trackers initializes automatically on the main process.
            # Train!

            global_step = 0

            completed_steps = 0
            break_training = False

            # 初始化损失记录
            # 如果使用自适应迭代策略，使用-1作为标识符，否则使用实际的tau_config值
            tau_file_name = -1 if control_alg is not None else tau_config
            loss_record_path = os.path.join(os.path.dirname(__file__), 'results',
                                            f'loss_record_tau_{tau_file_name}.csv')

            adapter = adapter.to(accelerator.device)
            adapter_min = adapter_min.to(accelerator.device)

            # 定义优化器 - 分开文本和图像适配器的参数
            from config import adapt_layers
            img_adapter_params = []
            img_hypernet_params = []

            for level in adapt_layers:
                if hasattr(adapter[level], 'img_adapter'):
                    img_adapter_params.extend(adapter[level].img_adapter.parameters())
            img_hypernet_params.extend(adapter["img_kappa_hypernet"].parameters())

            # 创建优化器
            from config import momentum_value

            img_adapter_optimizer = HeavyBallMGD(
                img_adapter_params,
                lr=args.learning_rate,
                momentum=momentum_value
            ) if img_adapter_params else None

            img_hypernet_optimizer = HeavyBallMGD(
                img_hypernet_params,
                lr=args.learning_rate,
                momentum=momentum_value
            ) if img_hypernet_params else None

            from config import adapt_layers
            target_layers = {}
            # unet.up_blocks的顺序是从深层到浅层（从up_blocks[0]到up_blocks[-1]）
            # up_blocks[i]对应up_i，其中up0是最深层（远离输出），up3是最浅层（接近输出）
            for level in adapt_layers:
                # 从层名提取索引（例如"up3" -> 3）
                level_idx = int(level.replace("up", ""))
                # 直接使用level_idx作为up_blocks的索引
                target_layers[level] = unet.up_blocks[level_idx]

            for layer in target_layers.values():
                if not hasattr(layer, 'adapter_features'):
                    layer.adapter_features = {}

            device = accelerator.device if accelerator else torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")

            loader_iter = iter(dataset_loader)

            for tau in range(tau_config):
                # 每次本地迭代前重置FLOPs计数器

                if img_adapter_optimizer:
                    img_adapter_optimizer.zero_grad(set_to_none=True)
                if img_hypernet_optimizer:
                    img_hypernet_optimizer.zero_grad(set_to_none=True)

                if break_training:
                    break

                idx = completed_steps % len(dataset)
                try:
                    image_paths = next(loader_iter)  # 读取批量4个路径（无需[0]）
                except StopIteration:
                    loader_iter = iter(dataset_loader)
                    image_paths = next(loader_iter)

                # 生成时间步（从所有配置的时间步范围中随机选择）
                # 收集所有时间步范围
                all_timesteps = []
                for range_config in timestep_ranges:
                    time_range = range_config.get("range", [500, 600])
                    all_timesteps.extend(range(time_range[0], time_range[1]))
                
                if all_timesteps:
                    # 从所有时间步中随机选择
                    timestep_scalar = torch.tensor(np.random.choice(all_timesteps), device=device).long()
                    timestep = timestep_scalar.repeat(batch_size)
                else:
                    # 如果没有配置时间步范围，使用默认范围
                    timestep_list = torch.arange(timestep_range[0], timestep_range[1], device=device).long()
                    timestep_idx = torch.randint(0, len(timestep_list), size=(), device=device)
                    timestep_scalar = timestep_list[timestep_idx]
                    timestep = timestep_scalar.repeat(batch_size)

                # 生成初始文本嵌入
                with torch.amp.autocast('cuda', enabled=True, dtype=torch.float32):
                    text_inputs = tokenizer(
                        [text] * batch_size,
                        padding="max_length",
                        max_length=tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt"
                    ).to(device)
                    initial_text_emb = text_encoder(**text_inputs).last_hidden_state

                # 记录当前迭代的开始时间
                iter_start_time = time.time()

                # 使用超网络生成所有层级的图像kappa
                with torch.amp.autocast('cuda', enabled=True, dtype=torch.float32):
                    all_img_kappas = adapter["img_kappa_hypernet"](initial_text_emb, timestep)
                    adapted_text_emb = initial_text_emb  # 直接使用原始文本嵌入，不进行文本适配

                    # 1. 获取带噪潜空间和目标潜空间
                    noisy_latent, timesteps, noise = image_to_latent_base(
                        image_paths, vae, train_transforms, timestep
                    )

                    # if noisy_latent.shape[0] > 1:
                    #     noisy_latent = noisy_latent[:1]

                    noisy_latent = noisy_latent.to(device)


                    # 定义钩子创建函数避免闭包陷阱
                    def create_hook(lvl, current_timesteps, img_kappa_val):
                        def hook(module, input, output):
                            feat = output[0] if isinstance(output, tuple) else output
                            if hasattr(adapter[lvl], 'img_adapter') and adapter[lvl][
                                "img_adapter"] is not None:
                                adapted_feat = adapter[lvl]["img_adapter"](feat, current_timesteps,
                                                                           img_kappa_val)
                                return (adapted_feat,) if isinstance(output, tuple) else adapted_feat
                            return output

                        return hook


                    hooks = []
                    for level, layer in target_layers.items():
                        img_kappa = all_img_kappas[level]
                        hook = layer.register_forward_hook(
                            create_hook(level, timesteps, img_kappa)
                        )
                        hooks.append(hook)

                    # 3. UNet前向传播
                    optimized_noise_pred = unet(
                        noisy_latent,
                        timesteps,
                        encoder_hidden_states=adapted_text_emb
                    ).sample

                    for hook in hooks:
                        hook.remove()

                    # 计算噪声预测损失
                    mse_loss = F.mse_loss(optimized_noise_pred, noise)
                    with torch.no_grad():
                        # 调用scheduler.step得到当前时间步的去噪潜变量（x_theta ≈ 真实潜变量x0的近似）
                        denoised_output = noise_scheduler.step(
                            model_output=optimized_noise_pred,  # UNet预测的噪声
                            timestep=timestep_scalar,  # 当前训练的时间步
                            sample=noisy_latent  # 当前时间步的带噪潜变量
                        )
                        x_theta = denoised_output.pred_original_sample

                    loss = mse_loss

                    # 7. 反向传播
                    loss.backward()

                    # 更新所有优化器
                    if img_adapter_optimizer:
                        img_adapter_optimizer.step()
                    if img_hypernet_optimizer:
                        img_hypernet_optimizer.step()

                    # 计算迭代时间
                    iter_time = time.time() - iter_start_time

                    # 保存最后一次迭代的损失和FLOPs
                    last_loss = loss.item()

                    last_iter_time = iter_time

                    del loss, mse_loss, optimized_noise_pred, x_theta
                    torch.cuda.empty_cache()
                    gc.collect()

                # 梯度累积逻辑保持不变...
                # 只对adapt_layers中的层初始化梯度结构
                img_grads = {}
                level_predictors_grads = {}
                level_embeddings_grads = {}
                for level in adapt_layers:
                    img_grads[level] = []
                    level_predictors_grads[level] = []
                    level_embeddings_grads[level] = []
                
                current_grads = {
                    "img": img_grads,
                    "img_hypernet": {
                        "global_encoder": [],
                        "level_predictors": level_predictors_grads,
                        "level_embeddings": level_embeddings_grads
                    }
                }

                # 收集适配器梯度
                for level in adapt_layers:
                    if hasattr(adapter[level], 'img_adapter'):
                        for param in adapter[level].img_adapter.parameters():
                            if param.grad is not None:
                                current_grads["img"][level].append(param.grad.detach().clone().cpu())
                            else:
                                current_grads["img"][level].append(torch.zeros_like(param).cpu())

                # 图像超网络梯度收集
                for param in adapter["img_kappa_hypernet"].global_encoder.parameters():
                    if param.grad is not None:
                        current_grads["img_hypernet"]["global_encoder"].append(param.grad.detach().clone().cpu())
                    else:
                        current_grads["img_hypernet"]["global_encoder"].append(torch.zeros_like(param).cpu())
                for level in adapt_layers:
                    for param in adapter["img_kappa_hypernet"].level_predictors[level].parameters():
                        if param.grad is not None:
                            current_grads["img_hypernet"]["level_predictors"][level].append(
                                param.grad.detach().clone().cpu())
                        else:
                            current_grads["img_hypernet"]["level_predictors"][level].append(
                                torch.zeros_like(param).cpu())
                    param = adapter["img_kappa_hypernet"].level_embeddings[level]
                    if param.grad is not None:
                        current_grads["img_hypernet"]["level_embeddings"][level].append(
                            param.grad.detach().clone().cpu())
                    else:
                        current_grads["img_hypernet"]["level_embeddings"][level].append(torch.zeros_like(param).cpu())

                if accumulated_grads is None:
                    accumulated_grads = current_grads
                else:
                    # 累积图像适配器梯度
                    for level in adapt_layers:
                        accumulated_grads["img"][level] = [
                            acc + cur for acc, cur in
                            zip(accumulated_grads["img"][level], current_grads["img"][level])
                        ]

                    # 图像超网络梯度累积
                    accumulated_grads["img_hypernet"]["global_encoder"] = [
                        acc + cur for acc, cur in zip(accumulated_grads["img_hypernet"]["global_encoder"],
                                                      current_grads["img_hypernet"]["global_encoder"])
                    ]
                    for level in adapt_layers:
                        accumulated_grads["img_hypernet"]["level_predictors"][level] = [
                            acc + cur for acc, cur in zip(accumulated_grads["img_hypernet"]["level_predictors"][level],
                                                          current_grads["img_hypernet"]["level_predictors"][level])
                        ]
                        accumulated_grads["img_hypernet"]["level_embeddings"][level] = [
                            acc + cur for acc, cur in zip(accumulated_grads["img_hypernet"]["level_embeddings"][level],
                                                          current_grads["img_hypernet"]["level_embeddings"][level])
                        ]

                if initial_loss is None:
                    initial_loss = last_loss

                if control_alg is not None:
                    is_last_local = control_alg.update_after_each_local(tau, grad_global)

                    if is_last_local:
                        break

                # 训练逻辑
                completed_steps += 1
                if completed_steps >= tau_config:
                    break_training = True
                    break

                if tau == 0:
                    if use_min_loss and w_prev_min_loss is not None:
                        # 定义安全的钩子函数
                        hooks = []
                        # 使用与之前相同的逻辑来映射层
                        target_layers = {}
                        for level in adapt_layers:
                            level_idx = int(level.replace("up", ""))
                            target_layers[level] = unet.up_blocks[level_idx]
                        adapted_text_emb_min = initial_text_emb.clone()  # 直接使用原始文本嵌入，不进行文本适配


                        # 定义钩子创建函数避免闭包陷阱
                        def create_hook(lvl, current_timesteps, img_kappa_val):
                            def hook(module, input, output):
                                feat = output[0] if isinstance(output, tuple) else output
                                if hasattr(adapter_min[lvl], 'img_adapter') and adapter_min[lvl][
                                    "img_adapter"] is not None:
                                    adapted_feat = adapter_min[lvl]["img_adapter"](feat, current_timesteps,
                                                                                   img_kappa_val)
                                    return (adapted_feat,) if isinstance(output, tuple) else adapted_feat
                                return output

                            return hook


                        all_img_kappas_min = adapter_min["img_kappa_hypernet"](initial_text_emb, timestep)
                        # 为每个层级创建专属钩子
                        for level, layer in target_layers.items():
                            img_kappa_min = all_img_kappas_min[level]
                            hook = layer.register_forward_hook(create_hook(level, timesteps, img_kappa_min))
                            hooks.append(hook)

                        optimized_noise_pred = unet(
                            noisy_latent,
                            timesteps,
                            encoder_hidden_states=adapted_text_emb_min
                        ).sample

                        for hook in hooks:
                            hook.remove()
                        # 3.1 计算MSE损失
                        mse_loss_min = F.mse_loss(optimized_noise_pred, noise)
                        with torch.no_grad():
                            # 调用scheduler.step得到当前时间步的去噪潜变量（x_theta ≈ 真实潜变量x0的近似）
                            denoised_output = noise_scheduler.step(
                                model_output=optimized_noise_pred,  # UNet预测的噪声
                                timestep=timestep_scalar,  # 当前训练的时间步（600-750）
                                sample=noisy_latent  # 当前时间步的带噪潜变量
                            )
                            x_theta = denoised_output.pred_original_sample

                        loss_w_prev_min_loss = mse_loss_min.item()

                torch.cuda.empty_cache()
                del noisy_latent, timesteps, noise, initial_text_emb
                gc.collect()

            tau_actual = completed_steps

            if accumulated_grads is not None and tau_actual > 0:
                for level in adapt_layers:
                    for i in range(len(accumulated_grads["img"][level])):
                        accumulated_grads["img"][level][i] = accumulated_grads["img"][level][i] / tau_actual
                # 图像超网络梯度平均
                for i in range(len(accumulated_grads["img_hypernet"]["global_encoder"])):
                    accumulated_grads["img_hypernet"]["global_encoder"][i] = \
                        accumulated_grads["img_hypernet"]["global_encoder"][i] / tau_actual
                for level in adapt_layers:
                    for i in range(len(accumulated_grads["img_hypernet"]["level_predictors"][level])):
                        accumulated_grads["img_hypernet"]["level_predictors"][level][i] = \
                            accumulated_grads["img_hypernet"]["level_predictors"][level][i] / tau_actual
                    for i in range(len(accumulated_grads["img_hypernet"]["level_embeddings"][level])):
                        accumulated_grads["img_hypernet"]["level_embeddings"][level][i] = \
                            accumulated_grads["img_hypernet"]["level_embeddings"][level][i] / tau_actual

            # Local operation finished, global aggregation starts
            o = round_params['o'][client_id] if hasattr(round_params['o'], '__getitem__') and client_id is not None else \
                round_params['o']
            f = round_params['f'][client_id] if hasattr(round_params['f'], '__getitem__') and client_id is not None else \
                round_params['f']
            time_all_local = tau_actual * o * batch_size / f
            print('time_all_local =', time_all_local)

            loss_last_global = initial_loss

            if control_alg is not None:
                control_alg.update_after_all_local(accumulated_grads, last_loss,
                                                   adapter, w_last_global, loss_last_global, w_global_min_loss, omega)

            # 使用服务器端类似的代码计算上传时间
            # 计算要发送的适配器参数的总大小（字节）
            data_size_trans = sum(p.numel() * p.element_size() for p in adapter.parameters())  # 字节

            # 从round_params中获取发送速度r（与服务器端的r[n]对应）
            send_speed = round_params['r'][client_id] if hasattr(round_params['r'],
                                                                 '__getitem__') and client_id is not None else \
                round_params['r']

            # 计算传输时间（秒）
            time_trans = data_size_trans * 8 / send_speed

            # 将传输时间加到总时间中
            time_all_local_with_transfer = time_all_local + time_trans

            # 计算传输能耗（焦耳）
            p = round_params['p'][client_id] if hasattr(round_params['p'], '__getitem__') and client_id is not None else \
                round_params['p']
            E_trans = time_trans * p  # 传输能耗 = 传输时间 × 发射功率

            # 打印仿真计算时间和仿真传输时间
            print('仿真计算时间 (time_all_local) =', time_all_local, '秒')
            print('仿真传输时间 (time_trans) =', time_trans, '秒')
            print('总仿真时间 (time_all_local_with_transfer) =', time_all_local_with_transfer, '秒')

            # 发送包含上传时间的总时间
            msg = ['MSG_WEIGHT_TIME_SIZE_CLIENT_TO_SERVER', adapter, time_all_local_with_transfer, tau_actual,
                   data_size_local,
                   loss_last_global, loss_w_prev_min_loss, accumulated_grads, last_loss, last_iter_time]
            send_msg(sock, msg)

            if time_all_local is not None:
                try:
                    # 计算能耗，使用CPU仿真参数
                    Ek = calculate_energy_consumption(
                        tau_actual=tau_actual,
                        accelerator=accelerator,
                        round_params=round_params,
                        client_id=client_id
                    )

                    # 打印计算能耗和传输能耗
                    print('计算能耗 (computation energy) =', Ek, '焦耳')
                    print('传输能耗 (transmission energy) =', E_trans, '焦耳')
                    print('总能耗 (total energy) =', Ek + E_trans, '焦耳')

                    msg = ['MSG_INFO_CLIENT_TO_SERVER', Ek]
                    send_msg(sock, msg)
                except Exception as e:
                    print(f'警告：计算能耗时出错: {str(e)}，使用默认值继续')
                    # 使用默认值发送消息，避免训练中断
                    Ek = 0.0
                    E_trans = 0.0
                    print('计算能耗 (computation energy) =', Ek, '焦耳')
                    print('传输能耗 (transmission energy) =', E_trans, '焦耳')
                    print('总能耗 (total energy) =', Ek + E_trans, '焦耳')
                    msg = ['MSG_INFO_CLIENT_TO_SERVER', Ek]
                    send_msg(sock, msg)

            if control_alg is not None:
                control_alg.send_to_server(sock)

            if is_last_round:
                break

            torch.cuda.empty_cache()

except (struct.error, socket.error):
    print('Server has stopped')
    pass
