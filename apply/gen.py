# 最关键的修复：在导入任何库之前解析命令行参数并清空sys.argv
import os
import sys
import argparse
import yaml

# 加载配置文件
def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

# 加载配置
config = load_config()

# 保存原始的sys.argv以便于调试
original_argv = sys.argv.copy()

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument(
    "--base_model_path",
    type=str,
    default=config["model"].get("base_model_path", "model"),
    help="The base pretrained model path."
)
parser.add_argument(
    "--unet_subfolder",
    type=str,
    default=config["model"].get("unet_subfolder", "unets/4/unet"),
    required=False,
    help="UNet subfolder path."
)
parser.add_argument(
    "--prompt",
    type=str,
    default=config["text"].get("training_text", "COVID-19 chest CT: Compared with normal lung parenchyma, typical viral pneumonia features: bilateral multiple patchy opacities (ground-glass opacities with visible vessels, focal consolidation), subpleural distribution, ill-defined margins, no obvious lobulation, spiculation or large necrotic hypodense areas."),
    help="The prompt to guide the generation."
)
parser.add_argument(
    "--img_num",
    type=int,
    default=1,
    help="How many images to generate."
)
parser.add_argument(
    "--device",
    type=str,
    default='cuda:0',
    help="Device used."
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="image_result",
    help="Output directory for generated images."
)
parser.add_argument(
    "--num_inference_steps",
    type=int,
    default=100,
    help="How many steps taken when model generate each image."
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Random seed for reproducible generation."
)
parser.add_argument(
    "--iteration_case",
    type=str,
    default="case1",
    help="Iteration case identifier (e.g., case1, case2, case3)."
)
parser.add_argument(
    "--momentum",
    type=str,
    default="with",
    choices=["with", "without"],
    help="Whether to use momentum (with/without)."
)
parser.add_argument(
    "--config_case",
    type=int,
    default=1,
    choices=[1, 2, 3, 4],
    help="Configuration case number (1-4)."
)
parser.add_argument(
    "--adapter_checkpoint_dir",
    type=str,
    default="output",
    help="Directory where the adapter checkpoint is stored."
)

# 解析参数并清空sys.argv
args = parser.parse_args()
sys.argv = [sys.argv[0]]  # 清空sys.argv，防止diffusers库干扰

# 调试信息：确认脚本被正确执行
print("=== 正在执行 RunMINIM/apply/gen.py 脚本 ===")
print(f"脚本路径: {os.path.abspath(__file__)}")
print(f"当前工作目录: {os.getcwd()}")
print(f"Python版本: {sys.version}")
print(f"原始命令行参数: {original_argv}")

# 确定项目根目录路径
# 方法1：从当前工作目录向上查找，直到找到Adapter.py文件
root_dir = os.getcwd()
while root_dir and not os.path.exists(os.path.join(root_dir, 'Adapter.py')):
    new_dir = os.path.dirname(root_dir)
    if new_dir == root_dir:  # 已经到达文件系统根目录
        break
    root_dir = new_dir

# 方法2：如果方法1失败，默认使用当前工作目录
if not os.path.exists(os.path.join(root_dir, 'Adapter.py')):
    root_dir = os.getcwd()

sys.path.append(root_dir)

# 现在才导入其他库
import gc
from diffusers import (
    StableDiffusionPipeline,
    UNet2DConditionModel,
    DDPMScheduler
)
import math
import numpy as np
import random
from torch import nn
from Adapter import ImageAdapter
import torch
from config import img_dims_per_level, target_text_level
from diffusers.utils import logging
import torch.nn.functional as F
from transformers import CLIPTokenizer, CLIPTextModel


# 设置固定随机种子以确保实验可重复性
def set_random_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_timestep_embedding(timesteps, embedding_dim):
    """
    将时间步转换为正弦位置编码（与训练阶段（客户端/服务器）完全一致）
    """
    if timesteps.dim() == 0:
        timesteps = timesteps.unsqueeze(0)
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class ImageKappaHyperNetwork(nn.Module):
    """图像专用kappa超网络：仅预测图像kappa"""

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
                nn.Linear(128, 1),  # 仅输出图像kappa
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
        batch_size = text_emb.shape[0]
        text_feat = text_emb.mean(dim=1)
        timestep_emb = get_timestep_embedding(timestep, self.time_dim)
        global_cond = torch.cat([text_feat, timestep_emb], dim=1)
        global_feat = self.global_encoder(global_cond)

        if current_level is not None:
            level_emb = self.level_embeddings[current_level].unsqueeze(0).expand(batch_size, -1)
            level_aware_feat = torch.cat([global_feat, level_emb], dim=1)
            img_kappa = self.level_predictors[current_level](level_aware_feat)
            return img_kappa[:, 0]
        else:
            all_img_kappas = {}
            for level in self.levels:
                level_emb = self.level_embeddings[level].unsqueeze(0).expand(batch_size, -1)
                level_aware_feat = torch.cat([global_feat, level_emb], dim=1)
                img_kappa = self.level_predictors[level](level_aware_feat)
                all_img_kappas[level] = img_kappa[:, 0].unsqueeze(-1).unsqueeze(-1)
            return all_img_kappas





logger = logging.get_logger(__name__)


def main():
    # 设置固定随机种子
    set_random_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    latents_list = []

    class UNetWithTimestepTrack(UNet2DConditionModel):
        def forward(self, sample, timestep, encoder_hidden_states, **kwargs):
            self.current_timestep = timestep.to(sample.device)
            return super().forward(sample, timestep, encoder_hidden_states, **kwargs)

    # 加载UNet并转换为fp16
    unet = UNetWithTimestepTrack.from_pretrained(
        os.path.join(args.base_model_path, args.unet_subfolder)
    ).to(device)

    # 初始化Adapter并转换为fp16
    from config import get_adapt_layers, get_img_dims_per_level
    # 自动检测UNet上采样层数并选择最接近输出的两层
    adapt_layers = get_adapt_layers(unet)
    # 自动检测UNet上采样层的输出维度
    img_dims_per_level = get_img_dims_per_level(unet)
    print(f"自动检测到的上采样层: {adapt_layers}")
    print(f"自动检测到的上采样层维度: {img_dims_per_level}")
    
    adapter = nn.ModuleDict()
    # 使用自动检测到的上采样层维度
    for level in adapt_layers:
        adapter[level] = nn.ModuleDict({
            "img_adapter": ImageAdapter(img_dim=img_dims_per_level[level])
        })

    adapter["img_kappa_hypernet"] = ImageKappaHyperNetwork(
        text_dim=768,
        time_dim=512,
        level_embed_dim=32,
        levels=adapt_layers
    )
    adapter = adapter.to(device)

    # 全量加载权重
    # 默认检查点路径，使用--adapter-checkpoint-dir参数指定的目录
    checkpoint_path = os.path.join(args.adapter_checkpoint_dir, "adapter_best_tau-1.pt")
    pretrained_dict = torch.load(checkpoint_path, map_location=device)
    adapter.load_state_dict(pretrained_dict, strict=True)
    adapter.eval()

    # 初始化DDPMScheduler并设置有效Timestep
    scheduler = DDPMScheduler.from_pretrained(
        args.base_model_path,
        subfolder=config["model"]["scheduler_subfolder"]
    )

    tokenizer = CLIPTokenizer.from_pretrained(
        args.base_model_path, subfolder=config["model"]["tokenizer_subfolder"]
    )

    text_encoder = CLIPTextModel.from_pretrained(
        args.base_model_path, subfolder=config["model"]["text_encoder_subfolder"]
    )

    # 初始化管道
    pipe = StableDiffusionPipeline.from_pretrained(
        args.base_model_path,
        unet=unet,
        safety_checker=None,
        scheduler=scheduler,
        tokenizer=tokenizer,
        text_encoder=text_encoder
    ).to(device)

    # 创建基础输出目录
    base_output_dir = "image_result"
    os.makedirs(base_output_dir, exist_ok=True)

    # 创建无适配器图片目录
    without_adapter_dir = os.path.join(base_output_dir, "without_adapter")
    os.makedirs(without_adapter_dir, exist_ok=True)

    # 生成无Adapter对比图
    for i in range(args.img_num):
        latents = torch.randn(
            (1, pipe.unet.config.in_channels, pipe.unet.config.sample_size, pipe.unet.config.sample_size),
            device=device
        )
        latents_list.append(latents.clone())

        with torch.no_grad():
            result = pipe(
                prompt=args.prompt,
                num_inference_steps=args.num_inference_steps,
                latents=latents,
            )
            image = result.images[0]
            image.save(os.path.join(without_adapter_dir, f"without_adapter{i}.png"))
            del result, image
            torch.cuda.empty_cache()

    # 目标层与文本层级
    target_layers = {
        "up0": unet.up_blocks[0],
        "up1": unet.up_blocks[1],
        "up2": unet.up_blocks[2],
        "up3": unet.up_blocks[3]
    }
    # 使用config.py中定义的target_text_level

    # 文本预处理（确保输出为fp16）
    text_inputs = pipe.tokenizer(
        args.prompt,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    ).input_ids.to(device)
    with torch.no_grad():
        text_embeddings = pipe.text_encoder(text_inputs).last_hidden_state

    # 提前生成所有有效Timestep的图像kappa
    all_timestep_kappas = {}
    for ts in scheduler.timesteps:
        ts = ts.to(device)
        ts_item = ts.item()
        img_kappas = adapter["img_kappa_hypernet"](text_embeddings, ts)
        all_timestep_kappas[ts_item] = img_kappas
    # 直接使用原始文本嵌入，不进行文本适配
    adapted_text_emb = text_embeddings

    # 创建带适配器图片目录（按照新的路径格式）
    with_adapter_dir = os.path.join(base_output_dir, f"with_adapter_case{args.iteration_case}_{args.momentum}_momentum_-1")
    os.makedirs(with_adapter_dir, exist_ok=True)

    # 生成带Adapter图像
    for i in range(args.img_num):
        latents = latents_list[i]

        def create_adapter_hook_factory(level, unet_instance):
            def hook(module, input, output):
                feat = output[0] if isinstance(output, tuple) else output
                current_timestep = unet_instance.current_timestep
                if current_timestep is None:
                    # 容错处理：使用默认kappa值，而非直接返回原始输出
                    img_kappa = torch.tensor([0.5], device=output.device)
                else:
                    current_ts = current_timestep.item()
                    img_kappa = all_timestep_kappas[current_ts][level]

                adapted_feat = adapter[level]["img_adapter"](feat, current_timestep, img_kappa)
                if isinstance(output, tuple):
                    output_list = list(output)
                    output_list[0] = adapted_feat
                    return tuple(output_list)
                else:
                    return adapted_feat

            return hook

        # 注册钩子
        hooks = []
        for level, layer in target_layers.items():
            hook = layer.register_forward_hook(create_adapter_hook_factory(level, unet))
            hooks.append(hook)

        try:
            # 生成图像
            with torch.no_grad():
                result = pipe(
                    prompt_embeds=adapted_text_emb,
                    num_inference_steps=args.num_inference_steps,
                    latents=latents
                )
            image = result.images[0]

            # 生成简化的文件名
            image_name = f"{i}.png"
            save_path = os.path.join(with_adapter_dir, image_name)
            image.save(save_path)

        finally:
            # 移除钩子
            for hook in hooks:
                hook.remove()
            hooks.clear()

        # 清理内存
        del result, image
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    main()