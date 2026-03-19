#!/usr/bin/env python3
"""
AlpaSim driver for Alpamayo-R1-10B VLA model.

This driver replaces the VAM (VideoActionModel) driver with NVIDIA's Alpamayo-R1-10B,
a Vision-Language-Action model for autonomous driving.

Model: nvidia/Alpamayo-R1-10B
Architecture: Qwen3-VL-2B + Expert CoT + Diffusion action head
Output: 20-step trajectory (steering + speed)
"""

from __future__ import annotations

import logging
import os
import platform
from collections import OrderedDict
from contextlib import nullcontext

import einops
import numpy as np
import torch
import torch.serialization
import omegaconf.dictconfig
import omegaconf.listconfig
import scipy.spatial.transform as spt

from ..schema import ModelConfig
from .base import BaseTrajectoryModel, DriveCommand, ModelPrediction, PredictionInput

logger = logging.getLogger(__name__)

torch.serialization.add_safe_globals([
    omegaconf.listconfig.ListConfig,
    omegaconf.dictconfig.DictConfig,
])

CAMERA_ORDER = [
    "CAMERA_CROSS_LEFT_120FOV",
    "CAMERA_FRONT_WIDE_120FOV", 
    "CAMERA_CROSS_RIGHT_120FOV",
    "CAMERA_FRONT_TELE_30FOV",
]

DTYPE = torch.bfloat16
NUM_CAMERAS = 4
NUM_FRAMES = 4
NUM_HISTORY_STEPS = 16
DT = 0.1


def load_alpamayo(checkpoint_dir: str, device: torch.device):
    """Load Alpamayo-R1-10B from local checkpoint directory."""
    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
    from alpamayo_r1 import helper
    from transformers import AutoTokenizer

    logger.info(f"Loading Alpamayo-R1-10B from {checkpoint_dir}")
    model = AlpamayoR1.from_pretrained(checkpoint_dir, dtype=DTYPE).to(device)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-VL-2B-Instruct",
        trust_remote_code=True,
    )
    processor = helper.get_processor(tokenizer)
    
    logger.info(f"Alpamayo loaded on {device}, dtype={DTYPE}")
    return model, processor, helper


class AlpamayoCameraBuffer:
    """Ring buffer for accumulating camera frames per camera."""
    
    def __init__(self, num_cameras: int, max_frames: int = 16):
        self.num_cameras = num_cameras
        self.max_frames = max_frames
        self.frames: dict[str, list[np.ndarray]] = {}
        self.timestamps: dict[str, list[int]] = {}
    
    def add(self, camera_id: str, timestamp_us: int, image: np.ndarray) -> None:
        if camera_id not in self.frames:
            self.frames[camera_id] = []
            self.timestamps[camera_id] = []
        self.frames[camera_id].append(image)
        self.timestamps[camera_id].append(timestamp_us)
    
    def get_image_tensor(self) -> torch.Tensor:
        """Get image tensor of shape (N_cameras, num_frames, 3, H, W)."""
        camera_ids = sorted(self.frames.keys())
        if len(camera_ids) < self.num_cameras:
            logger.warning(
                f"Only {len(camera_ids)} cameras available, expected {self.num_cameras}"
            )
        
        tensors = []
        for cam_id in CAMERA_ORDER:
            if cam_id in self.frames:
                frames = self.frames[cam_id]
            elif camera_ids:
                frames = self.frames[camera_ids[0]]
            else:
                h, w = 256, 256
                frames = [np.zeros((3, h, w), dtype=np.uint8)]
            
            frames_arr = frames[-NUM_FRAMES:]
            if len(frames_arr) < NUM_FRAMES:
                h, w = frames_arr[0].shape[1], frames_arr[0].shape[2] if len(frames_arr) > 0 else (256, 256)
                pad = [np.zeros((3, h, w), dtype=np.uint8) for _ in range(NUM_FRAMES - len(frames_arr))]
                frames_arr = pad + list(frames_arr)
            
            stacked = np.stack(frames_arr, axis=0)
            if stacked.ndim == 4:
                stacked = np.transpose(stacked, (0, 3, 1, 2))
            elif stacked.shape[-1] == 3:
                stacked = np.transpose(stacked, (0, 3, 1, 2))
            tensors.append(torch.from_numpy(stacked))
        
        return torch.stack(tensors, dim=0)
    
    def get_timestamps(self) -> torch.Tensor:
        """Get relative timestamps of shape (N_cameras, num_frames)."""
        camera_ids = sorted(self.timestamps.keys())
        timestamps = []
        for cam_id in CAMERA_ORDER:
            if cam_id in self.timestamps:
                ts = self.timestamps[cam_id][-NUM_FRAMES:]
            elif camera_ids and camera_ids[0] in self.timestamps:
                ts = self.timestamps[camera_ids[0]][1][:NUM_FRAMES]
            else:
                ts = [0] * NUM_FRAMES
            if len(ts) < NUM_FRAMES:
                ts = [ts[0] if ts else 0] * (NUM_FRAMES - len(ts)) + ts
            timestamps.append(torch.tensor(ts, dtype=torch.int64))
        return torch.stack(timestamps, dim=0)
    
    def clear(self) -> None:
        self.frames.clear()
        self.timestamps.clear()


def poses_to_xyz_rot(poses: list, num_steps: int = NUM_HISTORY_STEPS) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert pose history to xyz and rotation matrices.
    
    Returns:
        ego_xyz: (1, 1, num_steps, 3)
        ego_rot: (1, 1, num_steps, 3, 3)
    """
    n = len(poses)
    xyz = np.zeros((1, 1, num_steps, 3), dtype=np.float32)
    rot = np.zeros((1, 1, num_steps, 3, 3), dtype=np.float32)
    
    identity = np.eye(3, dtype=np.float32)
    
    for i in range(min(n, num_steps)):
        pose = poses[-(n - i)] if i < n else poses[-1]
        xyz[0, 0, i] = [pose.pose.vec.x, pose.pose.vec.y, pose.pose.vec.z]
        r = spt.Rotation.from_quat([
            pose.pose.quat.x, pose.pose.quat.y, 
            pose.pose.quat.z, pose.pose.quat.w
        ])
        rot[0, 0, i] = r.as_matrix()
    
    for i in range(n, num_steps):
        xyz[0, 0, i] = xyz[0, 0, n-1] if n > 0 else [0, 0, 0]
        rot[0, 0, i] = rot[0, 0, n-1] if n > 0 else identity
    
    return torch.from_numpy(xyz), torch.from_numpy(rot)


def downsample_trajectory(xyz: np.ndarray, num_output: int = 20) -> np.ndarray:
    """Downsample trajectory from 64 to num_output points.
    
    Args:
        xyz: (T, 3) trajectory in world frame
        num_output: number of output points
    
    Returns:
        (num_output, 2) x,y offsets from current position in rig frame
    """
    T = xyz.shape[0]
    if T <= num_output:
        indices = list(range(T))
        indices += [T-1] * (num_output - T)
    else:
        step = (T - 1) / (num_output - 1)
        indices = [int(i * step) for i in range(num_output - 1)] + [T-1]
    
    selected = xyz[indices]
    
    origin = selected[0]
    offsets = selected - origin
    
    x_offset = offsets[:, 0]
    y_offset = offsets[:, 2]
    
    cos0 = np.cos(np.arctan2(selected[0, 2], selected[0, 0]))
    sin0 = np.sin(np.arctan2(selected[0, 2], selected[0, 0]))
    rot = np.array([[cos0, -sin0], [sin0, cos0]], dtype=np.float32)
    local = (rot @ np.stack([x_offset, y_offset])).T
    
    return local.astype(np.float32)


class AlpamayoModel(BaseTrajectoryModel):
    """Alpamayo-R1-10B VLA driver for AlpaSim.
    
    This model takes multi-camera images and ego pose history,
    runs vision-language reasoning, and outputs trajectory predictions.
    
    Architecture:
    - Vision backbone: Qwen3-VL-2B
    - Reasoning: Chain-of-thought via autoregressive VLM
    - Action: Diffusion-based trajectory prediction
    
    Output: 20-step trajectory in rig frame (x,y offsets + headings)
    """
    
    NUM_CAMERAS = NUM_CAMERAS
    EXPECTED_HEIGHT = 256
    EXPECTED_WIDTH = 256
    
    @classmethod
    def from_config(
        cls,
        model_cfg: ModelConfig,
        device: torch.device,
        camera_ids: list[str],
        context_length: int | None,
        output_frequency_hz: int,
    ) -> "AlpamayoModel":
        """Create AlpamayoModel from driver configuration."""
        if model_cfg.checkpoint_path is None:
            raise ValueError("Alpamayo model requires checkpoint_path")
        return cls(
            checkpoint_path=model_cfg.checkpoint_path,
            device=device,
            camera_ids=camera_ids,
            context_length=context_length or NUM_FRAMES,
            output_frequency_hz=output_frequency_hz or 2,
        )
    
    def __init__(
        self,
        checkpoint_path: str,
        device: torch.device,
        camera_ids: list[str],
        context_length: int = NUM_FRAMES,
        output_frequency_hz: int = 2,
        num_traj_samples: int = 1,
        top_p: float = 0.98,
        temperature: float = 0.6,
    ):
        """Initialize Alpamayo model.
        
        Args:
            checkpoint_path: Path to Alpamayo checkpoint (e.g., /tmp/alpamayo_model/)
            device: Torch device for inference
            camera_ids: List of camera IDs (must be exactly 4)
            context_length: Number of temporal frames
            output_frequency_hz: Trajectory output frequency
            num_traj_samples: Number of trajectory samples to average
            top_p: Nucleus sampling parameter
            temperature: Sampling temperature
        """
        if len(camera_ids) != self.NUM_CAMERAS:
            raise ValueError(
                f"Alpamayo requires exactly {self.NUM_CAMERAS} cameras, got {len(camera_ids)}"
            )
        
        self._model, self._processor, self._helper = load_alpamayo(checkpoint_path, device)
        self._device = device
        self._camera_ids = camera_ids
        self._context_length = context_length
        self._output_frequency_hz = output_frequency_hz
        self._num_traj_samples = num_traj_samples
        self._top_p = top_p
        self._temperature = temperature
        
        self._camera_buffers: dict[str, AlpamayoCameraBuffer] = {}
        self._use_autocast = device.type == "cuda"
        
        logger.info(f"AlpamayoModel initialized: {checkpoint_path}")
        logger.info(f"  Cameras: {camera_ids}")
        logger.info(f"  Output: {output_frequency_hz}Hz, {num_traj_samples} samples")
    
    @property
    def camera_ids(self) -> list[str]:
        return self._camera_ids
    
    @property
    def context_length(self) -> int:
        return self._context_length
    
    @property
    def output_frequency_hz(self) -> int:
        return self._output_frequency_hz
    
    def _encode_command(self, command: DriveCommand) -> str:
        """Convert DriveCommand to natural language for Alpamayo.
        
        Alpamayo is a VLM - it accepts text prompts for reasoning.
        """
        COMMAND_TEXT = {
            DriveCommand.LEFT: "Turn left at the next intersection.",
            DriveCommand.STRAIGHT: "Continue straight.",
            DriveCommand.RIGHT: "Turn right at the next intersection.",
            DriveCommand.UNKNOWN: "Proceed with caution.",
        }
        return COMMAND_TEXT.get(command, "Continue straight.")
    
    def _build_model_input(
        self,
        camera_images: dict[str, list],
        ego_pose_history: list,
    ) -> dict:
        """Build Alpamayo input from AlpaSim camera images and pose history.
        
        Args:
            camera_images: dict of camera_id -> [(timestamp, image), ...]
            ego_pose_history: list of PoseAtTime objects
        
        Returns:
            Alpamayo model input dict
        """
        image_frames = []
        for cam_id in CAMERA_ORDER:
            if cam_id in camera_images:
                frames = camera_images[cam_id]
            elif self._camera_ids:
                frames = list(camera_images.values())[0]
            else:
                h, w = self.EXPECTED_HEIGHT, self.EXPECTED_WIDTH
                frames = [(0, np.zeros((h, w, 3), dtype=np.uint8))]
            
            for _, img in frames[-NUM_FRAMES:]:
                if img.ndim == 3 and img.shape[-1] == 3:
                    img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                elif img.ndim == 3 and img.shape[0] == 3:
                    img_t = torch.from_numpy(img).float() / 255.0
                else:
                    img_t = torch.zeros(3, self.EXPECTED_HEIGHT, self.EXPECTED_WIDTH)
                image_frames.append(img_t)
        
        image_tensor = torch.stack(image_frames, dim=0).reshape(
            NUM_CAMERAS, NUM_FRAMES, 3, self.EXPECTED_HEIGHT, self.EXPECTED_WIDTH
        )
        
        ego_xyz, ego_rot = poses_to_xyz_rot(ego_pose_history, NUM_HISTORY_STEPS)
        
        return {
            "image_frames": image_tensor,
            "ego_history_xyz": ego_xyz,
            "ego_history_rot": ego_rot,
        }
    
    def _run_inference(self, data: dict, command: DriveCommand) -> tuple[np.ndarray, str]:
        """Run Alpamayo inference on a single input.
        
        Returns:
            (T, 2) trajectory: x,y offsets in rig frame
            reasoning_text: Chain-of-thought text
        """
        model_inputs = self._build_model_input(
            data.camera_images,
            data.ego_pose_history,
        )
        
        frames = model_inputs["image_frames"]
        messages = self._helper.create_message(frames)
        
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=True,
            return_tensors="pt",
        )
        
        tokenized_data = {
            "tokenized_data": inputs,
            "ego_history_xyz": model_inputs["ego_history_xyz"],
            "ego_history_rot": model_inputs["ego_history_rot"],
        }
        tokenized_data = self._helper.to_device(tokenized_data, self._device, DTYPE)
        
        torch.cuda.manual_seed_all(42)
        
        autocast_ctx = (
            torch.amp.autocast(self._device.type, dtype=DTYPE)
            if self._use_autocast
            else nullcontext()
        )
        
        with torch.no_grad():
            with autocast_ctx:
                pred_xyz, _, extra = self._model.sample_trajectories_from_data_with_vlm_rollout(
                    data=tokenized_data,
                    top_p=self._top_p,
                    temperature=self._temperature,
                    num_traj_samples=self._num_traj_samples,
                    max_generation_length=64,
                    return_extra=True,
                )
        
        pred_xyz_np = pred_xyz.detach().float().cpu().numpy()
        trajectory_xy = downsample_trajectory(pred_xyz_np[0, 0, 0], num_output=20)
        
        reasoning = ""
        if extra is not None:
            try:
                cot_list = extra.get("cot", [])
                if isinstance(cot_list, list) and len(cot_list) > 0:
                    reasoning = str(cot_list[0])[:500]
            except Exception:
                pass
        
        command_text = self._encode_command(command)
        reasoning_text = f"{command_text} | {reasoning}" if reasoning else command_text
        
        return trajectory_xy, reasoning_text
    
    def predict(self, prediction_input: PredictionInput) -> ModelPrediction:
        """Generate trajectory prediction for a single input."""
        self._validate_cameras(prediction_input.camera_images)
        
        trajectory_xy, reasoning_text = self._run_inference(
            prediction_input, prediction_input.command
        )
        headings = self._compute_headings_from_trajectory(trajectory_xy)
        
        return ModelPrediction(
            trajectory_xy=trajectory_xy,
            headings=headings,
            reasoning_text=reasoning_text,
        )
    
    def predict_batch(
        self, prediction_inputs: list[PredictionInput]
    ) -> list[ModelPrediction]:
        """Generate trajectory predictions for a batch of inputs."""
        if not prediction_inputs:
            return []
        
        results = []
        for inp in prediction_inputs:
            results.append(self.predict(inp))
        return results
