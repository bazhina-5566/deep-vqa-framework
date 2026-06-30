#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
独立推理脚本：加载训练好的 checkpoint，对单张图片或单个视频文件进行质量评分预测。

设计原则：
- checkpoint 里已经保存了完整的 config（见 src/core/engine.py `_save_checkpoint` 中的
  `state["config"] = self.config`），所以推理时不需要重新走 load_system_config /
  dataset_config.yaml 那一套，直接从 checkpoint 里还原配置和模型结构，避免"训练用的配置"
  和"推理用的配置"再次出现不一致。
- 不依赖 DataEDA / config_loader / trainer / engine / path_manager，只依赖
  src.models.iqavqa_net.IQAVQANet 这一个项目内文件，方便单独交付给后端组部署。
- 后端组在部署时需在项目根目录执行 `uv sync` 同步依赖，然后启动 `api.py` 服务。

模型路由方式：
    forward(self, x: torch.Tensor) -> torch.Tensor
    - 4D Tensor [B, 3, H, W]    -> 图片，走单帧空间特征提取分支
    - 5D Tensor [B, F, 3, H, W] -> 视频，先逐帧提取特征再做时序融合
    模型不接受额外的 mode 参数，完全靠输入 tensor 的维度自动路由。

自动模型选择：
    - 单文件或批量推理时，按文件类型自动选择模型
    - 图片 → IQA 模型 (iqa-models/tid2013_best.pt)
    - 视频 → VQA 模型 (vqa-models/konvid_best.pt)
    - 混合目录 → 分别加载两组模型，各自推理
    - 可通过 -c 手动指定统一模型覆盖自动选择

完整性检查：
    - 图片：可解码、分辨率有效、非全黑/全白、颜色种类 ≥ 2
    - 视频：可解码、实际解码帧数与视频文件声称的帧数偏差不超过 10%、
            黑帧/白帧比例、坏帧比例、跳帧检测（基于时间戳间隔统计）

使用方式：
    # 自动选择模型（推荐）
    uv run python -m deploy.infer -i test.jpg
    uv run python -m deploy.infer -i test.mp4
    uv run python -m deploy.infer -i ./mixed_dir/ -o results.json  # 混合目录自动分组

    # 手动指定模型（所有文件用同一个）
    uv run python -m deploy.infer -c model.pt -i test.jpg

============================================================
快速开始（后端同学看这里）
============================================================

1. 部署目录结构（必须）：

    deploy/
    ├── infer.py
    ├── api.py                       # 你们要写的服务层
    ├── iqa-models/
    │   └── tid2013_best.pt          # IQA 模型权重，文件名必须完全匹配
    └── vqa-models/
        └── konvid_best.pt           # VQA 模型权重，文件名必须完全匹配

    路径常量见下方 DEFAULT_IQA_MODEL / DEFAULT_VQA_MODEL，
    如果实际文件名不一样，改这两个常量就行，不用改其他代码。

2. 如果要做成常驻服务（实时推理），不要每次请求都跑一遍这个脚本，
   而是复用里面的两个函数：

        from deploy.infer import load_checkpoint, predict_single

        # 服务启动时只执行一次（耗时操作：读盘 + 模型搬到 GPU）
        model, config = load_checkpoint("iqa-models/tid2013_best.pt", device="cuda")

        # 每次收到请求时只调用这个，很快
        result = predict_single(model, file_path, config, device="cuda")
        # result 形如:
        # {"file": "...", "raw_score": 0.62, "mos_score": 3.41,
        #  "task_type": "iqa", "model_name": "IQAVQANet"}

   图片和视频要用不同的 model/config（分别来自 load_checkpoint(iqa_path) 和
   load_checkpoint(vqa_path)），不能用同一个模型混着推，参考 main() 里
   "自动模式"那部分分组逻辑。

3. mos_score 为 None 是正常情况，发生在 checkpoint 里没有保存
   mos_min/mos_max 的情况下（早期训练出的模型），此时只能拿到 raw_score
   （[0,1] 区间），不代表推理失败，前端展示时要分别处理这两种情况。

4. 完整性检查只会在控制台打 logger.warning，不会中断流程、不会让
   predict_single 报错 —— 除非文件本身完全无法解码 / 分辨率非法 / 颜色种类
   过少，这几种情况才会抛 ValueError，需要在 api.py 里 try/except 接住，
   返回给前端一个明确的"文件无效"错误，而不是 500。

5. CLI 跑通自检（建议先在本地这样测一遍，确认环境装对了再接服务层）：

        uv run python -m deploy.infer -i test.jpg
        uv run python -m deploy.infer -i test.mp4
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from loguru import logger

try:
    from decord import VideoReader, cpu
    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False
    logger.warning("⚠️ Decord 未安装，视频读取将回退到 OpenCV。")

# ==================== 路径处理 ====================
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_DEPLOY_ROOT = Path(__file__).resolve().parent

from src.models.iqavqa_net import IQAVQANet

# ==================== 文件类型常量 ====================
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# ==================== 默认模型路径 ====================
# 这两个路径是相对 deploy/ 目录的，后端同学只需要把权重文件放到对应位置，
# 或者改这两个字符串指向实际路径，不需要改其他任何代码。
DEFAULT_IQA_MODEL = "iqa-models/tid2013_best.pt"
DEFAULT_VQA_MODEL = "vqa-models/konvid_best.pt"

# ==================== 完整性检查阈值 ====================
# 以下全部是经验性阈值，不是严格推导得出的，如果实际使用中发现误报/漏报
# 较多（比如正常视频频繁触发"黑帧比例过高"告警），可以直接调整这里的数值，
# 不需要改其他逻辑。这些检查全部只产生 warning 日志，不会让推理失败
# （唯二会真正抛错中断推理的情况是：图片完全无法解码、颜色种类过少）。
MIN_COLORS = 2                       # 图片颜色种类下限，低于此值判定为无效图片（纯色块等）
BLACK_FRAME_THRESHOLD = 8.0          # 灰度均值低于此值视为黑帧
WHITE_FRAME_THRESHOLD = 245.0        # 灰度均值高于此值视为白帧
BAD_FRAME_DIFF_THRESHOLD = 0.5       # 相邻帧灰度差低于此值视为坏帧（卡顿/花屏/重复帧）
MAX_BLACK_WHITE_RATIO = 0.3          # 黑/白帧比例超过 30% 触发告警
MAX_BAD_RATIO = 0.3                  # 坏帧比例超过 30% 触发告警
MAX_FRAME_DEVIATION = 0.10           # 实际解码帧数与视频声称帧数偏差超过 10% 触发告警
FRAME_DROP_INTERVAL_THRESHOLD = 1.5  # 帧间隔超过平均间隔的 1.5 倍判定为跳帧
VIDEO_MEAN_WARNING_THRESHOLD = 0.01  # 整段视频 tensor 均值低于此值视为几乎全黑


# ==================== 路由函数 ====================
def detect_media_type(file_path: Union[str, Path]) -> str:
    """根据文件后缀判断是图片还是视频，用于自动模式下选择对应模型。

    不支持的后缀会直接抛 ValueError。在 main() 的分组逻辑里，单个不支持
    的文件只会被跳过并打印 warning，不会影响其他文件继续推理；但如果是
    其他地方单独调用这个函数，调用方需要自己处理这个异常。
    """
    ext = Path(file_path).suffix.lower()
    if ext in IMAGE_EXTS:
        return "image"
    elif ext in VIDEO_EXTS:
        return "video"
    raise ValueError(f"不支持的文件类型: {ext}")


# ==================== Checkpoint 加载 ====================
def load_checkpoint(checkpoint_path: Union[str, Path], device: str = "cuda") -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """从 checkpoint 还原模型结构和训练时的完整配置。

    💎 这是个相对耗时的操作（读盘 + 反序列化 + 模型搬到 GPU），如果要做
    实时服务，必须只在服务启动时调用一次，不能放在每次请求的处理路径里，
    否则每次推理都会有几秒甚至更久的额外延迟。

    Args:
        checkpoint_path: .pt 文件路径
        device: 推理设备

    Returns:
        (model, config): 模型实例和训练时的 config

    Raises:
        FileNotFoundError: checkpoint 不存在
        KeyError: checkpoint 中没有 config 字段
    """
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"❌ Checkpoint 不存在: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)

    if "config" not in checkpoint:
        raise KeyError(
            "🚨 Checkpoint 中没有保存 config 字段，无法还原模型结构。"
            "请确认该 checkpoint 是由 TrainerEngine._save_checkpoint 保存的。"
        )

    config = checkpoint["config"]
    model = IQAVQANet(config=config)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    epoch = checkpoint.get("epoch", "?")
    metrics = checkpoint.get("metrics", {})
    logger.info(f"✅ [Infer] 已加载 checkpoint: {ckpt_path.name} (epoch={epoch})")
    if metrics:
        logger.info(f"   └─ 该 checkpoint 训练时的验证指标: {metrics}")

    return model, config


# ==================== 反归一化 ====================
def denormalize(score: float, mos_min: Optional[float], mos_max: Optional[float]) -> Optional[float]:
    """将 [0,1] 归一化分数还原到原始 MOS 尺度。

    mos_min/mos_max 任一缺失都返回 None（而不是报错），调用方需要自己
    判断 None 的情况，不代表出错，只是该 checkpoint 没保存这组参数。
    """
    if mos_min is None or mos_max is None:
        return None
    return float(score) * (mos_max - mos_min) + mos_min


# ==================== 数据预处理 ====================
class Preprocessor:
    """
    单样本预处理器，逻辑与训练时保持一致。
    同时增加了完整性检查：
        - 图片：可解码、分辨率有效、非全黑/全白、颜色种类 ≥ 2
        - 视频：可解码、帧数偏差、黑帧/白帧比例、坏帧比例、跳帧检测
    """

    def __init__(self, num_frames: int = 8, input_size: int = 224):
        self.num_frames = num_frames
        self.input_size = input_size

    def process(self, file_path: Union[str, Path]) -> torch.Tensor:
        """
        预处理图片或视频文件。

        Returns:
            图片: Tensor (3, H, W)       -> 后续 unsqueeze(0) 得到 4D [B,3,H,W]
            视频: Tensor (T, 3, H, W)    -> 后续 unsqueeze(0) 得到 5D [B,F,3,H,W]
        """
        ext = Path(file_path).suffix.lower()
        if ext in VIDEO_EXTS:
            return self._process_video(file_path)
        elif ext in IMAGE_EXTS:
            return self._process_image(file_path)
        else:
            raise ValueError(f"❌ 不支持的文件类型: {ext}")

    # ==================== 视频处理 ====================
    def _process_video(self, file_path: Union[str, Path]) -> torch.Tensor:
        """视频预处理 + 完整性检查（优先 Decord，失败回退 OpenCV）"""
        if DECORD_AVAILABLE:
            try:
                vr = VideoReader(str(file_path), ctx=cpu(0))
                total_frames = len(vr)

                # ========== 1. 帧数偏差检查 ==========
                claimed_frames = self._get_claimed_frame_count(str(file_path))
                if claimed_frames > 0:
                    deviation = abs(total_frames - claimed_frames) / max(claimed_frames, 1)
                    if deviation > MAX_FRAME_DEVIATION:
                        logger.warning(f"⚠️ 帧数偏差 {deviation*100:.1f}%（声称={claimed_frames}, 实际={total_frames}）")

                # ========== 2. 采样帧 ==========
                if total_frames >= self.num_frames:
                    indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int).tolist()
                else:
                    indices = list(range(total_frames))

                frames = vr.get_batch(indices).asnumpy()
                if frames.size == 0:
                    raise ValueError(f"Decord 返回了空帧序列: {file_path}")

                # ========== 3. 跳帧检测（基于时间戳间隔） ==========
                timestamps = self._get_frame_timestamps(vr, indices)
                if len(timestamps) > 2:
                    intervals = np.diff(timestamps)
                    mean_interval = np.mean(intervals)
                    max_interval = np.max(intervals)
                    if max_interval > mean_interval * FRAME_DROP_INTERVAL_THRESHOLD:
                        logger.warning(f"⚠️ 检测到跳帧: max间隔={max_interval:.1f}ms, mean间隔={mean_interval:.1f}ms")

                # ========== 4. 黑帧/白帧/坏帧检测 ==========
                frame_stats = self._analyze_frames(frames)
                total = len(frames)
                black_ratio = frame_stats["black"] / max(total, 1)
                white_ratio = frame_stats["white"] / max(total, 1)
                bad_ratio = frame_stats["bad"] / max(total, 1)

                if black_ratio > MAX_BLACK_WHITE_RATIO:
                    logger.warning(f"⚠️ 黑帧比例过高: {black_ratio*100:.1f}%")
                if white_ratio > MAX_BLACK_WHITE_RATIO:
                    logger.warning(f"⚠️ 白帧比例过高: {white_ratio*100:.1f}%")
                if bad_ratio > MAX_BAD_RATIO:
                    logger.warning(f"⚠️ 坏帧比例过高: {bad_ratio*100:.1f}%")

                # ========== 5. 解码帧数不足 ==========
                if len(frames) < self.num_frames:
                    logger.warning(f"⚠️ 帧数不足 {len(frames)}/{self.num_frames}，将用最后一帧填充")

                # ========== 6. 预处理 ==========
                resized = [cv2.resize(f, (self.input_size, self.input_size)) for f in frames]
                video_np = np.stack(resized)
                tensor = torch.from_numpy(video_np).permute(0, 3, 1, 2).float() / 255.0

                if tensor.size(0) < self.num_frames:
                    pad = tensor[-1].unsqueeze(0).repeat(self.num_frames - tensor.size(0), 1, 1, 1)
                    tensor = torch.cat([tensor, pad], dim=0)

                # ========== 7. 全黑视频检测 ==========
                if tensor.mean() < VIDEO_MEAN_WARNING_THRESHOLD:
                    logger.warning(f"⚠️ 视频几乎全黑: {file_path}")

                return tensor

            except Exception as e:
                logger.warning(f"⚠️ [Decord] 解析失败，回退到 OpenCV: {e}")

        return self._process_video_opencv(file_path)

    def _process_video_opencv(self, file_path: Union[str, Path]) -> torch.Tensor:
        """OpenCV 备用视频读取 + 完整性检查，逻辑与 Decord 分支保持一致"""
        cap = cv2.VideoCapture(str(file_path))
        if not cap.isOpened():
            raise ValueError(f"❌ 无法打开视频: {file_path}")

        claimed_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frames = []
        timestamps = []
        while len(frames) < self.num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
            timestamps.append(timestamp)
            frame = cv2.resize(frame, (self.input_size, self.input_size))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()

        if not frames:
            raise ValueError(f"❌ 视频没有读到任何有效帧: {file_path}")

        if claimed_frames > 0:
            actual = len(frames)
            deviation = abs(actual - claimed_frames) / max(claimed_frames, 1)
            if deviation > MAX_FRAME_DEVIATION:
                logger.warning(f"⚠️ 帧数偏差 {deviation*100:.1f}%（声称={claimed_frames}, 实际={actual}）")

        if len(timestamps) > 2:
            intervals = np.diff(timestamps)
            mean_interval = np.mean(intervals)
            max_interval = np.max(intervals)
            if max_interval > mean_interval * FRAME_DROP_INTERVAL_THRESHOLD:
                logger.warning(f"⚠️ 检测到跳帧: max间隔={max_interval:.1f}ms, mean间隔={mean_interval:.1f}ms")

        frame_stats = self._analyze_frames(frames)
        total = len(frames)
        black_ratio = frame_stats["black"] / max(total, 1)
        white_ratio = frame_stats["white"] / max(total, 1)
        bad_ratio = frame_stats["bad"] / max(total, 1)

        if black_ratio > MAX_BLACK_WHITE_RATIO:
            logger.warning(f"⚠️ 黑帧比例过高: {black_ratio*100:.1f}%")
        if white_ratio > MAX_BLACK_WHITE_RATIO:
            logger.warning(f"⚠️ 白帧比例过高: {white_ratio*100:.1f}%")
        if bad_ratio > MAX_BAD_RATIO:
            logger.warning(f"⚠️ 坏帧比例过高: {bad_ratio*100:.1f}%")

        if len(frames) < self.num_frames:
            logger.warning(f"⚠️ 帧数不足 {len(frames)}/{self.num_frames}，将用最后一帧填充")

        video_np = np.stack(frames)
        tensor = torch.from_numpy(video_np).permute(0, 3, 1, 2).float() / 255.0

        if tensor.size(0) < self.num_frames:
            pad = tensor[-1].unsqueeze(0).repeat(self.num_frames - tensor.size(0), 1, 1, 1)
            tensor = torch.cat([tensor, pad], dim=0)

        if tensor.mean() < VIDEO_MEAN_WARNING_THRESHOLD:
            logger.warning(f"⚠️ 视频几乎全黑: {file_path}")

        return tensor

    def _get_claimed_frame_count(self, file_path: str) -> int:
        """获取视频文件元数据里声称的帧数，用于和实际解码帧数对比"""
        try:
            cap = cv2.VideoCapture(file_path)
            if cap.isOpened():
                count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                return count
        except Exception:
            return 0

    def _get_frame_timestamps(self, vr, indices) -> List[float]:
        """获取帧时间戳（毫秒），仅用于跳帧检测这个辅助性告警逻辑。

        vr.get_frame_timestamp 在部分视频编码格式下可能不可用或报错，
        这种情况下 fallback 假设固定 30fps（33.33ms/帧）来估算时间戳，
        估算误差不影响推理结果本身，只可能让跳帧告警不够精确。
        """
        timestamps = []
        for idx in indices:
            try:
                ts = vr.get_frame_timestamp(idx)[0] * 1000
                timestamps.append(ts)
            except Exception:
                timestamps.append(idx * 33.33)
        return timestamps

    def _analyze_frames(self, frames) -> Dict[str, int]:
        """分析帧序列：黑帧、白帧、坏帧（相邻帧几乎无变化，可能是卡顿/重复帧）"""
        stats = {"black": 0, "white": 0, "bad": 0}
        prev_gray = None

        for frame in frames:
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            else:
                gray = frame
            mean_gray = np.mean(gray)

            if mean_gray < BLACK_FRAME_THRESHOLD:
                stats["black"] += 1
            if mean_gray > WHITE_FRAME_THRESHOLD:
                stats["white"] += 1
            if prev_gray is not None:
                diff = np.mean(np.abs(gray - prev_gray))
                if diff < BAD_FRAME_DIFF_THRESHOLD:
                    stats["bad"] += 1
            prev_gray = gray

        return stats

    # ==================== 图片处理 ====================
    def _process_image(self, file_path: Union[str, Path]) -> torch.Tensor:
        """
        图片预处理 + 完整性检查。

        💎 这里抛出的 ValueError（解码失败/无效分辨率/颜色种类过少）是
        整个脚本里真正会中断单次推理的异常，api.py 必须 try/except 接住。
        """
        img = cv2.imread(str(file_path))
        if img is None:
            raise ValueError(f"❌ 图像解码失败: {file_path}")

        h, w = img.shape[:2]
        if h <= 0 or w <= 0:
            raise ValueError(f"❌ 无效分辨率: {w}x{h}")

        unique_colors = len(np.unique(img))
        if unique_colors < MIN_COLORS:
            raise ValueError(f"❌ 颜色种类过少: {unique_colors}")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean_gray = np.mean(gray)
        if mean_gray < BLACK_FRAME_THRESHOLD:
            logger.warning(f"⚠️ 图像几乎全黑: {file_path} (mean={mean_gray:.1f})")
        if mean_gray > WHITE_FRAME_THRESHOLD:
            logger.warning(f"⚠️ 图像几乎全白: {file_path} (mean={mean_gray:.1f})")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (self.input_size, self.input_size))
        return torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0


# ==================== 批量推理核心 ====================
@torch.no_grad()
def predict_single(
    model: torch.nn.Module,
    file_path: Union[str, Path],
    config: Dict[str, Any],
    device: str = "cuda",
    mos_min: Optional[float] = None,
    mos_max: Optional[float] = None,
) -> Dict[str, Any]:
    """
    对单个文件（图片或视频）进行推理。

    💎 这是 api.py 接入时最主要会用到的函数，返回字段说明：
        file        : 输入文件路径（原样返回，便于前端对照）
        raw_score   : 模型输出的原始分数，范围 [0,1]，始终存在
        mos_score   : 反归一化回原始 MOS 量纲的分数；如果 checkpoint 没保存
                      mos_min/mos_max 且调用时也没手动传，这里会是 None，
                      不代表推理出错，只是没法换算成业务可读的分数
        task_type   : "iqa" 或 "vqa"，可用于前端展示口径
        model_name  : 模型结构名（来自 config["model"]["name"]）

    💎 注意：这个函数不会捕获异常，文件解码失败/格式不支持会直接抛
       ValueError 往上传，调用方（api.py）必须自己 try/except，
       建议接住后返回明确的"文件无效"错误，而不是裸的 500。

    💎 如果要做实时单文件接口，直接复用这个函数即可；model/config 只需要
       在服务启动时通过 load_checkpoint 加载一次，不要在这里重复加载。
    """
    num_frames = config.get("model", {}).get("num_frames", 8)
    input_size = config.get("model", {}).get("input_size", 224)

    preprocessor = Preprocessor(num_frames=num_frames, input_size=input_size)
    data_tensor = preprocessor.process(file_path)
    data_tensor = data_tensor.unsqueeze(0).to(device)

    output = model(data_tensor)
    output = output.float()
    if output.ndim > 1 and output.size(-1) == 1:
        output = output.squeeze(-1)
    raw_score = float(output.flatten()[0].cpu().item())

    if mos_min is None or mos_max is None:
        dataset_info = config.get("dataset_info", {}) or {}
        mos_min = mos_min if mos_min is not None else dataset_info.get("mos_min")
        mos_max = mos_max if mos_max is not None else dataset_info.get("mos_max")

    real_score = denormalize(raw_score, mos_min, mos_max)

    return {
        "file": str(file_path),
        "raw_score": round(raw_score, 6),
        "mos_score": round(real_score, 4) if real_score is not None else None,
        "task_type": config.get("task_type", "unknown"),
        "model_name": config.get("model", {}).get("name", "unknown"),
    }


def predict_batch(
    model: torch.nn.Module,
    input_paths: List[Union[str, Path]],
    config: Dict[str, Any],
    device: str = "cuda",
    mos_min: Optional[float] = None,
    mos_max: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    批量推理多个文件，用于离线打分场景。

    💎 跟 predict_single 不同：这个函数内部吃掉了所有异常，单个文件失败
       不会中断整批推理，失败的文件在返回列表里会是
       {"file": "...", "error": "错误信息"} 而不是抛异常。
       如果是做"实时单文件接口"，应该直接用 predict_single（会抛异常，
       便于 api.py 统一处理成 HTTP 错误码）；如果是"批量离线打分"
       场景，才用这个。
    """
    results = []
    for path in input_paths:
        try:
            results.append(predict_single(model, path, config, device, mos_min, mos_max))
        except Exception as e:
            logger.error(f"🚨 推理失败: {path} | {e}")
            results.append({"file": str(path), "error": str(e)})
    return results


# ==================== CLI 入口 ====================
def main():
    parser = argparse.ArgumentParser(
        description="Deep VQA/IQA 推理脚本（自动选择 IQA/VQA 模型，支持混合目录）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 自动选择模型（单文件）
  uv run python -m deploy.infer -i test.jpg
  uv run python -m deploy.infer -i test.mp4

  # 混合目录自动分组推理
  uv run python -m deploy.infer -i ./mixed_dir/ -o results.json

  # 手动指定统一模型
  uv run python -m deploy.infer -c model.pt -i test.jpg
"""
    )
    parser.add_argument("-c", "--checkpoint", type=str, default=None, help="模型路径 (可选，不指定则自动选择)")
    parser.add_argument("-i", "--input", type=str, required=True, help="输入文件或目录路径")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--mos_min", type=float, default=None, help="反归一化下限")
    parser.add_argument("--mos_max", type=float, default=None, help="反归一化上限")
    parser.add_argument("-o", "--output", type=str, default=None, help="结果 JSON 输出路径")
    parser.add_argument("--cpu", action="store_true", help="强制使用 CPU")
    args = parser.parse_args()

    if args.cpu:
        args.device = "cpu"

    input_path = Path(args.input)

    # 收集文件
    targets = []
    if input_path.is_dir():
        for ext in VIDEO_EXTS | IMAGE_EXTS:
            targets.extend(input_path.rglob(f"*{ext}"))
        logger.info(f"📁 在目录 {input_path} 中找到 {len(targets)} 个文件")
    else:
        targets = [input_path]

    if not targets:
        logger.error(f"❌ 在 {input_path} 没有找到任何支持的图片/视频文件")
        sys.exit(1)

    # 按类型分组：图片和视频用不同模型，混合目录场景下避免用错模型推理
    image_files, video_files = [], []
    for f in targets:
        ext = f.suffix.lower()
        if ext in IMAGE_EXTS:
            image_files.append(f)
        elif ext in VIDEO_EXTS:
            video_files.append(f)
        else:
            logger.warning(f"⚠️ 跳过不支持的格式: {f}")

    results = []

    if args.checkpoint:
        # 手动模式：用户明确指定了一个模型，不管文件是图片还是视频，
        # 全部丢给这一个模型推理。
        # 💎 注意：如果传入的模型跟文件类型不匹配（比如拿 VQA 模型推图片），
        # 不会有任何拦截，会直接喂给 model() 调用，可能因为 4D/5D 维度不
        # 匹配而在 forward 内部报错，或者更糟——如果某天模型对两种维度
        # 都不报错，会跑出语义上完全错误但形式上"正常"的分数。所以手动
        # 指定模型时，使用者要自己确保模型和文件类型对得上，不确定的话
        # 优先用下面的自动模式。
        model_path = Path(args.checkpoint)
        if not model_path.exists():
            logger.error(f"❌ 模型不存在: {model_path}")
            sys.exit(1)
        model, config = load_checkpoint(model_path, device=args.device)
        all_files = image_files + video_files
        if all_files:
            results = predict_batch(model, all_files, config, args.device, args.mos_min, args.mos_max)
    else:
        # 自动模式（推荐）：按文件后缀自动分组，图片用 IQA 模型，
        # 视频用 VQA 模型，分别加载分别推理，不会出现类型不匹配的问题。
        if image_files:
            iqa_path = _DEPLOY_ROOT / DEFAULT_IQA_MODEL
            if not iqa_path.exists():
                logger.error(f"❌ IQA 模型不存在: {iqa_path}")
            else:
                logger.info(f"🖼️ 加载 IQA 模型，推理 {len(image_files)} 张图片")
                model, config = load_checkpoint(iqa_path, device=args.device)
                results.extend(predict_batch(model, image_files, config, args.device, args.mos_min, args.mos_max))

        if video_files:
            vqa_path = _DEPLOY_ROOT / DEFAULT_VQA_MODEL
            if not vqa_path.exists():
                logger.error(f"❌ VQA 模型不存在: {vqa_path}")
            else:
                logger.info(f"🎬 加载 VQA 模型，推理 {len(video_files)} 个视频")
                model, config = load_checkpoint(vqa_path, device=args.device)
                results.extend(predict_batch(model, video_files, config, args.device, args.mos_min, args.mos_max))

    if not results:
        logger.warning("⚠️ 没有成功推理任何文件")
        sys.exit(0)

    logger.info("=" * 60)
    logger.info("推理结果:")
    for r in results:
        if "error" in r:
            logger.error(f"  ❌ {r['file']}: {r['error']}")
        else:
            mos_str = f"{r['mos_score']:.4f}" if r["mos_score"] is not None else "N/A"
            logger.info(f"  ✅ {r['file']}: raw={r['raw_score']:.4f} | mos={mos_str}")
    logger.info("=" * 60)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"💾 结果已保存到: {out_path}")


if __name__ == "__main__":
    main()