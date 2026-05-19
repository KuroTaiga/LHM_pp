#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Function      : Export Gaussian Splatting as PLY (T-pose or SMPL-X frame; CLI + loaders)

"""
Export canonical (T-pose) Gaussian Splatting as PLY.

This script mirrors the model / image / SMPL-X shape loading path used by
``scripts/test/test_app_case.py`` and ``scripts/inference/app_inference.py``,
then runs ``infer_single_view`` → ``inference_gs`` → ``GaussianModel.save_ply``.

Supported models must appear in ``core.utils.model_card.GS_RENDER_SUPPORTED_MODEL_NAMES``
(models with Gaussian/3DGS output). Using any other ``--model_name`` is rejected before load.

Usage:
    # 仅图像：合成相机 + 规范 T-pose（``--pose_dir`` 默认为空）
    python scripts/inference/to_gs_ply.py \\
        --model_name LHMPP-700M-SMPLX-FREE \\
        --image_glob "./assets/example_multi_images/00000_yuliang_*.png"

    # 指定某一帧 SMPL-X JSON：用该文件内相机与姿态做 animation 导出（不经视频/mask 管线）
    python scripts/inference/to_gs_ply.py \\
        --pose_dir "./motion_video/BasketBall_I/smplx_params/00014.json" \\
        --image_glob "./assets/example_multi_images/00000_yuliang_*.png"
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import math
import os
import re
import sys
from pathlib import Path
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

_LHM_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _LHM_ROOT)

import numpy as np
import torch
from accelerate import Accelerator
from PIL import Image

torch._dynamo.config.disable = True


def _load_module(unique_name: str, rel_path: str) -> Any:
    """Load a project module by path.

    Avoids ``import scripts...`` because a PyPI distribution named ``scripts`` can
    shadow the project's ``scripts`` package.
    """
    path = os.path.join(_LHM_ROOT, rel_path)
    spec = importlib.util.spec_from_file_location(unique_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_ai = _load_module("_lhmpp_app_inference", "scripts/inference/app_inference.py")
build_app_model = _ai.build_app_model
parse_app_configs = _ai.parse_app_configs

from core.utils.model_card import (
    GS_RENDER_SUPPORTED_MODEL_NAMES,
    MODEL_CONFIG,
    model_supports_gs_render,
)
from core.utils.model_download_utils import AutoModelQuery
from core.utils.video import images_to_video


FRAME_VARYING_SMPLX_KEYS = (
    "root_pose",
    "body_pose",
    "jaw_pose",
    "leye_pose",
    "reye_pose",
    "lhand_pose",
    "rhand_pose",
    "trans",
    "focal",
    "princpt",
    "img_size_wh",
    "expr",
)


def _require_gs_output_model(model_name: str) -> None:
    """Fail fast if checkpoint is not wired for Gaussian export (infer_single_view + inference_gs)."""
    if model_supports_gs_render(model_name):
        return
    allowed = ", ".join(GS_RENDER_SUPPORTED_MODEL_NAMES) or "(none configured)"
    raise ValueError(
        f"T-pose Gaussian PLY export requires a GS-output model (--model_name in "
        f"GS_RENDER_SUPPORTED_MODEL_NAMES). Got {model_name!r}; supported: {allowed}. "
        f"Extend the allowlist in core/utils/model_card.py when a new GS build is validated."
    )


def _parse_smplx_raw(smplx_raw_data: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Match ``core.runners.infer.utils._parse_smplx_param`` without importing the runners package."""
    return {
        k: torch.FloatTensor(v)
        for k, v in smplx_raw_data.items()
        if "pad_ratio" not in k
    }


def _gs_model_name_choices() -> list[str]:
    if not GS_RENDER_SUPPORTED_MODEL_NAMES:
        raise RuntimeError(
            "GS_RENDER_SUPPORTED_MODEL_NAMES is empty in core/utils/model_card.py; "
            "add at least one model with 3DGS output."
        )
    return list(GS_RENDER_SUPPORTED_MODEL_NAMES)


def _effective_pose_dir(args: argparse.Namespace) -> str | None:
    """Unset, empty string, or whitespace ``--pose_dir`` => no pose file (T-pose / synthetic camera)."""
    v = getattr(args, "pose_dir", None)
    if v is None:
        return None
    s = str(v).strip()
    return s if s else None


def _resolved_pose_input(args: argparse.Namespace) -> str | None:
    pose_path = _effective_pose_dir(args)
    if pose_path is None:
        return None
    return str(Path(pose_path).expanduser().resolve())


def _pose_input_mode(args: argparse.Namespace) -> str:
    """Return ``tpose``, ``single_pose``, or ``pose_sequence``."""
    pose_path = _resolved_pose_input(args)
    if pose_path is None:
        return "tpose"
    if os.path.isdir(pose_path):
        return "pose_sequence"
    if os.path.isfile(pose_path):
        return "single_pose"
    raise FileNotFoundError(
        f"--pose_dir must be empty, a SMPL-X JSON file, or a directory of JSONs. Got: {pose_path}"
    )


def _derive_pose_seq_folder_name(pose_json_path: str) -> str:
    """Derive a stable motion sequence folder name from a JSON or JSON directory path."""
    p = Path(pose_json_path).resolve()
    if p.is_dir():
        if p.name == "smplx_params":
            return _sanitize_export_folder_name(p.parent.name)
        return _sanitize_export_folder_name(p.name)
    if p.parent.name == "smplx_params":
        return _sanitize_export_folder_name(p.parent.parent.name)
    return _sanitize_export_folder_name(p.parent.name)


def _sanitize_export_folder_name(name: str) -> str:
    """Single path segment for `{folder_name}.ply` (no slashes, trimmed)."""
    name = (name or "").strip()
    name = os.path.basename(name.replace("\\", "/"))
    for c in '/:*?"<>|':
        name = name.replace(c, "_")
    name = name.strip("._") or "tpose_export"
    return name[:200]


def _derive_input_image_folder_name(args: argparse.Namespace) -> str:
    """Subfolder / parent-dir name for reference images (no motion in path)."""
    if args.images_dir is not None:
        return _sanitize_export_folder_name(
            os.path.basename(os.path.normpath(args.images_dir))
        )
    paths = _resolve_image_paths(args)
    if paths:
        parent = os.path.dirname(os.path.abspath(paths[0]))
        base = os.path.basename(parent)
        if base and base not in (".", ""):
            return _sanitize_export_folder_name(base)
        return _sanitize_export_folder_name(
            os.path.splitext(os.path.basename(paths[0]))[0]
        )
    return "ref_images"


def _derive_folder_name_for_tpose_output(args: argparse.Namespace) -> str:
    """Filename stem for image-only export under ``outputs/tpose_output/``."""
    return _derive_input_image_folder_name(args)


def default_export_output_path(args: argparse.Namespace) -> str:
    """Default output target for the active mode.

    T-pose:
        ``outputs/tpose_output/{ref}.ply``
    Single posed frame:
        ``outputs/animation_output/{seq}/{ref}_{frame}.ply``
    Pose sequence folder:
        ``outputs/animation_output/{seq}/{ref}/``
    """
    mode = _pose_input_mode(args)
    if mode == "tpose":
        folder = _derive_folder_name_for_tpose_output(args)
        return os.path.join(_LHM_ROOT, "outputs", "tpose_output", f"{folder}.ply")
    pose_path = _resolved_pose_input(args)
    assert pose_path is not None
    motion_key = _derive_pose_seq_folder_name(pose_path)
    img_key = _derive_input_image_folder_name(args)
    if mode == "pose_sequence":
        return os.path.join(
            _LHM_ROOT,
            "outputs",
            "animation_output",
            motion_key,
            img_key,
        )
    frame_stem = _sanitize_export_folder_name(Path(pose_path).stem)
    fname = f"{img_key}_{frame_stem}.ply"
    return os.path.join(
        _LHM_ROOT,
        "outputs",
        "animation_output",
        motion_key,
        fname,
    )


def prior_model_check(save_dir: str = "./pretrained_models") -> None:
    """Same behavior as ``app.prior_model_check`` without importing Gradio (``app.py``)."""
    human_model_path = os.path.join(save_dir, "human_model_files")
    if os.path.exists(human_model_path):
        return
    if os.path.islink(human_model_path):
        try:
            os.unlink(human_model_path)
            print("Removed broken symlink: human_model_files")
        except OSError as e:
            print(f"Failed to remove broken symlink: {e}")
    print("Prior models not found or invalid. Downloading...")
    auto_query = AutoModelQuery(save_dir=save_dir)
    auto_query.download_all_prior_models()
    print("Prior models ready.")


@contextmanager
def _easy_memory_manager(model: torch.nn.Module, device: str = "cuda"):
    """Same behavior as ``scripts.inference.utils.easy_memory_manager`` without importing Gradio."""
    del device
    if torch.cuda.is_available():
        dev = f"cuda:{torch.cuda.current_device()}"
    else:
        dev = "cpu"
    model.to(dev)
    try:
        yield model
    finally:
        model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def build_arg_parser() -> argparse.ArgumentParser:
    gs_names = _gs_model_name_choices()
    parser = argparse.ArgumentParser(
        description=(
            "Load LHM++ and export Gaussian Splatting as either canonical PLY, a single "
            "posed-frame PLY, or a pose-sequence package (cano_gs.ply + frame_*.ply + MP4)."
        )
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=gs_names[0],
        choices=gs_names,
        help=(
            "Model card key (configs + HF/MS ids). Restricted to "
            "GS_RENDER_SUPPORTED_MODEL_NAMES in core/utils/model_card.py."
        ),
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Local checkpoint directory; skips AutoModelQuery when set.",
    )
    parser.add_argument(
        "--image_glob",
        type=str,
        default="./assets/example_multi_images/00000_yuliang_*.png",
        help="Glob for reference images (ignored if --images_dir is set).",
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default=None,
        help="Directory of images (*.png, *.jpg, ...); overrides --image_glob when set.",
    )
    parser.add_argument(
        "--pose_dir",
        type=str,
        default="",
        help=(
            "Path to either one SMPL-X parameter JSON (single posed-frame mode) or a directory "
            "containing per-frame SMPL-X JSONs (sequence mode). Optional matching FLAME sidecars "
            "are loaded from ../flame_params/<same_name>.json when present. Default empty => "
            "canonical T-pose export with a synthetic camera (no motion file)."
        ),
    )
    parser.add_argument(
        "--ref_view",
        type=int,
        default=8,
        help="Number of reference views (same as Gradio / test_app_case).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Target output path. In T-pose or single-JSON mode this is a PLY path. In JSON-folder "
            "sequence mode this is an output directory. Default when ``--pose_dir`` is empty: "
            "<repo>/outputs/tpose_output/{ref_images_parent}.ply; "
            "with one pose JSON: <repo>/outputs/animation_output/{seq}/{ref_images_parent}_{json_stem}.ply; "
            "with a pose directory: <repo>/outputs/animation_output/{seq}/{ref_images_parent}/"
        ),
    )
    parser.add_argument(
        "--work_dir",
        type=str,
        default=None,
        help="Working directory for intermediates (default: <output_parent>/debug/tpose_gs_work).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for model and tensors.",
    )
    parser.add_argument(
        "--video_fps",
        type=int,
        default=30,
        help="Sequence-mode preview video FPS.",
    )
    parser.add_argument(
        "--video_renderer",
        type=str,
        default="gs",
        choices=("gs", "neural"),
        help=(
            'Sequence-mode preview renderer. "gs" uses Gaussian splat RGB only; '
            '"neural" uses the refinement decoder when available.'
        ),
    )
    return parser


def _list_images_from_dir(images_dir: str, max_views: int) -> list[str]:
    patterns = ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG")
    paths: list[str] = []
    for pat in patterns:
        paths.extend(glob.glob(os.path.join(images_dir, pat)))
    paths = sorted(set(paths))[:max_views]
    if not paths:
        raise FileNotFoundError(f"No images found in directory: {images_dir}")
    return paths


def _resolve_image_paths(args: argparse.Namespace) -> list[str]:
    if args.images_dir is not None:
        return _list_images_from_dir(args.images_dir, max_views=8)
    paths = sorted(glob.glob(args.image_glob))[:8]
    if not paths:
        raise FileNotFoundError(f"No images matched glob: {args.image_glob}")
    return paths


def _synthetic_render_hw(cfg) -> tuple[int, int]:
    """Match ``prepare_motion_seqs_eval`` sizing: ``tgt_h = tgt_size * aspect_standard``."""
    render_res = int(cfg.get("render_size", 420))
    aspect_standard = 5.0 / 3.0
    tgt_h = int(round(render_res * aspect_standard))
    tgt_w = render_res
    return tgt_h, tgt_w


def _load_pose_minimal(pose: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Same as ``core.runners.infer.utils._load_pose`` (no package import)."""
    intrinsic = torch.eye(4)
    intrinsic[0, 0] = pose["focal"][0]
    intrinsic[1, 1] = pose["focal"][1]
    intrinsic[0, 2] = pose["princpt"][0]
    intrinsic[1, 2] = pose["princpt"][1]
    intrinsic = intrinsic.float()
    c2w = torch.eye(4).float()
    return c2w, intrinsic


def _stack_smplx_params_list_minimal(
    smplx_params_list: list[dict[str, torch.Tensor]], shape_param: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Same as ``_stack_smplx_params_list`` in infer utils."""
    from collections import defaultdict

    stacked: dict[str, list] = defaultdict(list)
    for sp in smplx_params_list:
        for k, v in sp.items():
            stacked[k].append(v)
    result = {k: torch.stack(v) for k, v in stacked.items()}
    result["betas"] = shape_param
    return result


def _to_motion_batch_dict_minimal(
    c2ws: torch.Tensor,
    intrs: torch.Tensor,
    bg_colors: torch.Tensor,
    smplx_params: dict[str, torch.Tensor],
    rgbs: list,
    vis_motion: bool,
) -> dict[str, Any]:
    """Same as ``_to_motion_batch_dict`` with ``vis_motion=False`` (no SMPL mesh render)."""
    if vis_motion:
        raise NotImplementedError("vis_motion True requires full infer utils")
    for k, v in smplx_params.items():
        smplx_params[k] = v.unsqueeze(0)
    rgbs_out = rgbs.unsqueeze(0) if len(rgbs) > 0 else rgbs
    return {
        "render_c2ws": c2ws.unsqueeze(0),
        "render_intrs": intrs.unsqueeze(0),
        "render_bg_colors": bg_colors.unsqueeze(0),
        "smplx_params": smplx_params,
        "rgbs": rgbs_out,
        "vis_motion_render": None,
    }


def _build_synthetic_motion_seq(cfg) -> dict[str, Any]:
    """One-frame motion dict for ``infer_single_view`` (synthetic camera + neutral pose)."""

    tgt_h, tgt_w = _synthetic_render_hw(cfg)
    Wf, Hf = float(tgt_w), float(tgt_h)
    fx = max(Wf, Hf)
    fy = max(Wf, Hf)
    cx = Wf / 2.0
    cy = Hf / 2.0

    zf = torch.float32
    betas = torch.zeros(10, dtype=zf)
    frame: dict[str, torch.Tensor] = {
        "betas": betas,
        "root_pose": torch.zeros(3, dtype=zf),
        "body_pose": torch.zeros(21, 3, dtype=zf),
        "jaw_pose": torch.zeros(3, dtype=zf),
        "leye_pose": torch.zeros(3, dtype=zf),
        "reye_pose": torch.zeros(3, dtype=zf),
        "lhand_pose": torch.zeros(15, 3, dtype=zf),
        "rhand_pose": torch.zeros(15, 3, dtype=zf),
        "trans": torch.zeros(3, dtype=zf),
        "expr": torch.zeros(100, dtype=zf),
        "focal": torch.tensor([fx, fy], dtype=zf),
        "princpt": torch.tensor([cx, cy], dtype=zf),
        "img_size_wh": torch.tensor([Wf, Hf], dtype=zf),
    }
    c2w, intr = _load_pose_minimal(frame)
    stacked = _stack_smplx_params_list_minimal([frame], betas)
    c2ws = c2w.unsqueeze(0)
    intrs = intr.unsqueeze(0)
    bg_colors = torch.tensor([1.0], dtype=torch.float32).unsqueeze(-1).repeat(1, 3)
    base = _to_motion_batch_dict_minimal(c2ws, intrs, bg_colors, stacked, [], vis_motion=False)
    base["offset_list"] = [[1.0, 1.0, 0.0, 0.0]]
    base["ori_size"] = (tgt_h, tgt_w)
    base["motion_seqs"] = []
    return base


def _build_motion_seq_from_pose_json(json_path: str, cfg: Any) -> dict[str, Any]:
    """Single-frame ``motion_seq`` dict from one SMPL-X JSON (no video / mask / ``get_motion_information``)."""
    path = str(Path(json_path).expanduser().resolve())
    if not os.path.isfile(path):
        raise FileNotFoundError(f"--pose_dir must be an existing SMPL-X JSON file: {path}")
    if not path.lower().endswith(".json"):
        raise ValueError(
            f"--pose_dir must point to a .json file (e.g. .../smplx_params/00014.json), got {path!r}"
        )

    raw = _load_pose_json_with_flame(path)

    smplx_param = _parse_smplx_raw(raw)
    c2w, intr = _load_pose_minimal(smplx_param)
    shape_param = smplx_param["betas"]
    stacked = _stack_smplx_params_list_minimal([smplx_param], shape_param)
    c2ws = c2w.unsqueeze(0)
    intrs = intr.unsqueeze(0)
    bg_colors = torch.tensor([1.0], dtype=torch.float32).unsqueeze(-1).repeat(1, 3)
    base = _to_motion_batch_dict_minimal(c2ws, intrs, bg_colors, stacked, [], vis_motion=False)
    if "img_size_wh" in smplx_param:
        wh = smplx_param["img_size_wh"]
        tgt_w, tgt_h = int(wh[0].item()), int(wh[1].item())
    else:
        tgt_h, tgt_w = _synthetic_render_hw(cfg)
    base["offset_list"] = [[1.0, 1.0, 0.0, 0.0]]
    base["ori_size"] = (tgt_h, tgt_w)
    base["motion_seqs"] = []
    return base


def _zero_hand_pose_template() -> list[list[float]]:
    return [[0.0, 0.0, 0.0] for _ in range(15)]


def _normalize_smplx_raw(raw: dict[str, Any]) -> dict[str, Any]:
    rename_map = {
        "global_orient": "root_pose",
        "transl": "trans",
        "left_hand_pose": "lhand_pose",
        "right_hand_pose": "rhand_pose",
    }
    for old_key, new_key in rename_map.items():
        if old_key in raw and new_key not in raw:
            raw[new_key] = raw.pop(old_key)

    raw.pop("meta", None)
    raw.setdefault("root_pose", [0.0, 0.0, 0.0])
    raw.setdefault("trans", [0.0, 0.0, 0.0])
    raw.setdefault("jaw_pose", [0.0, 0.0, 0.0])
    raw.setdefault("leye_pose", [0.0, 0.0, 0.0])
    raw.setdefault("reye_pose", [0.0, 0.0, 0.0])
    raw.setdefault("lhand_pose", _zero_hand_pose_template())
    raw.setdefault("rhand_pose", _zero_hand_pose_template())
    raw.setdefault("expr", [0.0] * 100)
    return raw


def _load_pose_json_with_flame(json_path: str) -> dict[str, Any]:
    path = Path(json_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Pose JSON not found: {path}")
    if path.suffix.lower() != ".json":
        raise ValueError(f"Pose input must be a .json file, got: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw: dict[str, Any] = json.load(f)
    raw = _normalize_smplx_raw(raw)

    if path.parent.name == "smplx_params":
        flame_path = path.parent.parent / "flame_params" / path.name
    else:
        flame_path = path.parent / "flame_params" / path.name

    if flame_path.is_file():
        with open(flame_path, "r", encoding="utf-8") as f:
            flame_params = json.load(f)
        raw["expr"] = flame_params.get("expcode", raw["expr"])
        posecode = flame_params.get("posecode", [])
        eyecode = flame_params.get("eyecode", [])
        raw["jaw_pose"] = posecode[3:6] if len(posecode) >= 6 else raw["jaw_pose"]
        raw["leye_pose"] = eyecode[:3] if len(eyecode) >= 3 else raw["leye_pose"]
        raw["reye_pose"] = eyecode[3:6] if len(eyecode) >= 6 else raw["reye_pose"]

    return raw


def _natural_sort_key(value: str) -> list[Any]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", value)
    ]


def _sorted_pose_json_paths(pose_dir: str) -> list[str]:
    root = Path(pose_dir).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Expected a pose JSON directory, got: {root}")
    json_paths = [str(path.resolve()) for path in root.glob("*.json") if path.is_file()]
    json_paths = sorted(json_paths, key=lambda path: _natural_sort_key(Path(path).stem))
    if not json_paths:
        raise FileNotFoundError(f"No pose JSONs found in directory: {root}")
    return json_paths


def _build_motion_seq_from_pose_dir(pose_dir: str, cfg: Any) -> dict[str, Any]:
    """Multi-frame ``motion_seq`` dict from a directory of per-frame SMPL-X JSONs."""
    json_paths = _sorted_pose_json_paths(pose_dir)
    c2ws: list[torch.Tensor] = []
    intrs: list[torch.Tensor] = []
    smplx_params_list: list[dict[str, torch.Tensor]] = []
    shape_param: torch.Tensor | None = None

    for json_path in json_paths:
        raw = _load_pose_json_with_flame(json_path)
        smplx_param = _parse_smplx_raw(raw)
        c2w, intr = _load_pose_minimal(smplx_param)
        c2ws.append(c2w)
        intrs.append(intr)
        smplx_params_list.append(smplx_param)
        if shape_param is None:
            shape_param = smplx_param["betas"]

    if shape_param is None:
        raise RuntimeError(f"Failed to load any SMPL-X parameters from {pose_dir}")

    stacked = _stack_smplx_params_list_minimal(smplx_params_list, shape_param)
    render_c2ws = torch.stack(c2ws, dim=0)
    render_intrs = torch.stack(intrs, dim=0)
    bg_colors = torch.ones((len(json_paths), 3), dtype=torch.float32)
    base = _to_motion_batch_dict_minimal(
        render_c2ws,
        render_intrs,
        bg_colors,
        stacked,
        [],
        vis_motion=False,
    )

    if "img_size_wh" in smplx_params_list[0]:
        wh = smplx_params_list[0]["img_size_wh"]
        tgt_w, tgt_h = int(wh[0].item()), int(wh[1].item())
    else:
        tgt_h, tgt_w = _synthetic_render_hw(cfg)

    base["ori_size"] = (tgt_h, tgt_w)
    base["motion_seqs"] = json_paths
    return base


def slice_motion_seq_to_single_frame(
    motion_seq: dict[str, Any], frame_idx: int = 0
) -> dict[str, Any]:
    """Keep the first (or given) frame on the time dimension for cameras and SMPL-X."""
    out: dict[str, Any] = dict(motion_seq)
    sl = slice(frame_idx, frame_idx + 1)
    out["render_c2ws"] = motion_seq["render_c2ws"][:, sl]
    out["render_intrs"] = motion_seq["render_intrs"][:, sl]
    out["render_bg_colors"] = motion_seq["render_bg_colors"][:, sl]
    rgbs = motion_seq.get("rgbs")
    if isinstance(rgbs, torch.Tensor) and rgbs.numel() > 0 and rgbs.dim() >= 2:
        out["rgbs"] = rgbs[:, sl]

    smplx_in = motion_seq["smplx_params"]
    smplx: dict[str, torch.Tensor] = {}
    time_varying = {
        "root_pose",
        "body_pose",
        "jaw_pose",
        "leye_pose",
        "reye_pose",
        "lhand_pose",
        "rhand_pose",
        "trans",
        "expr",
        "focal",
        "princpt",
        "img_size_wh",
    }
    for k, v in smplx_in.items():
        if k in time_varying and v.dim() >= 2 and v.shape[1] > 1:
            smplx[k] = v[:, sl].clone()
        else:
            smplx[k] = v.clone()

    out["smplx_params"] = smplx
    off = motion_seq.get("offset_list")
    if isinstance(off, list) and len(off) > frame_idx:
        out["offset_list"] = [off[frame_idx]]
    mseq = motion_seq.get("motion_seqs")
    if isinstance(mseq, list) and len(mseq) > frame_idx:
        out["motion_seqs"] = [mseq[frame_idx]]
    msks = motion_seq.get("masks")
    if isinstance(msks, list) and len(msks) > frame_idx:
        out["masks"] = [msks[frame_idx]]
    return out


def cano_body_pose_template(
    device: torch.device | str, dtype: torch.dtype
) -> torch.Tensor:
    """Same canonical body_pose tweaks as ``BaseGSRender._prepare_smplx_data`` (last frame)."""
    bp = torch.zeros(1, 1, 21, 3, device=device, dtype=dtype)
    # leg
    bp[0, 0, 0, -1] = math.pi / 12
    bp[0, 0, 1, -1] = -math.pi / 12
    # hands
    bp[0, 0, 15, -1] = -math.pi / 6
    bp[0, 0, 16, -1] = math.pi / 6
    return bp


def build_tpose_smplx_params(
    motion_seq_one: dict[str, Any],
    transform_mat_neutral_pose: torch.Tensor,
    merged_betas: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    """One-frame T-pose SMPL-X dict: canonical pose + merged betas + neutral transform."""
    sp: dict[str, torch.Tensor] = {}
    for k, v in motion_seq_one["smplx_params"].items():
        sp[k] = v.to(device=device, dtype=dtype)

    sp["betas"] = merged_betas.to(device=device, dtype=dtype)
    sp["transform_mat_neutral_pose"] = transform_mat_neutral_pose.to(
        device=device, dtype=dtype
    )

    z31 = torch.zeros(1, 1, 3, device=device, dtype=dtype)
    z_hand = torch.zeros(1, 1, 15, 3, device=device, dtype=dtype)
    n_expr = sp["expr"].shape[-1]
    sp["root_pose"] = z31
    sp["body_pose"] = cano_body_pose_template(device, dtype)
    sp["jaw_pose"] = z31
    sp["leye_pose"] = z31
    sp["reye_pose"] = z31
    sp["lhand_pose"] = z_hand
    sp["rhand_pose"] = z_hand
    sp["trans"] = z31
    sp["expr"] = torch.zeros(1, 1, n_expr, device=device, dtype=dtype)
    return sp


def build_animation_frame_smplx_params(
    motion_seq_one: dict[str, Any],
    transform_mat_neutral_pose: torch.Tensor,
    merged_betas: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    """First-frame motion SMPL-X for GS export (clip frame 0 pose, not canonical T-pose)."""
    sp: dict[str, torch.Tensor] = {}
    for k, v in motion_seq_one["smplx_params"].items():
        sp[k] = v.to(device=device, dtype=dtype)

    sp["betas"] = merged_betas.to(device=device, dtype=dtype)
    sp["transform_mat_neutral_pose"] = transform_mat_neutral_pose.to(
        device=device, dtype=dtype
    )
    return sp


def _run_infer_single_view(
    model: torch.nn.Module,
    ref_imgs_tensor: torch.Tensor,
    motion_seq: dict[str, Any],
    device: str,
) -> SimpleNamespace:
    """Run ``infer_single_view`` once and keep the reusable tensors for GS/video export."""
    dev = torch.device(device)
    smplx_dev = {k: v.to(dev) for k, v in motion_seq["smplx_params"].items()}
    ref_batch = ref_imgs_tensor.unsqueeze(0)
    ref_mask = torch.ones(
        ref_imgs_tensor.shape[0], dtype=torch.bool, device=dev
    ).unsqueeze(0)
    use_pred_render = getattr(model, "use_pred_shape_for_render", False)

    model_outputs = model.infer_single_view(
        ref_batch,
        None,
        None,
        render_c2ws=motion_seq["render_c2ws"].to(dev),
        render_intrs=motion_seq["render_intrs"].to(dev),
        render_bg_colors=motion_seq["render_bg_colors"].to(dev),
        smplx_params=smplx_dev,
        ref_imgs_bool=ref_mask,
        return_pred_shape=use_pred_render,
    )

    pred_shape = None
    if len(model_outputs) == 8:
        (
            gs_model_list,
            query_points,
            transform_mat_neutral_pose,
            gs_hidden_features,
            image_latents,
            motion_emb,
            pos_emb,
            pred_shape,
        ) = model_outputs
    elif len(model_outputs) == 7:
        (
            gs_model_list,
            query_points,
            transform_mat_neutral_pose,
            gs_hidden_features,
            image_latents,
            motion_emb,
            pos_emb,
        ) = model_outputs
    elif len(model_outputs) == 6:
        (
            gs_model_list,
            query_points,
            transform_mat_neutral_pose,
            gs_hidden_features,
            image_latents,
            motion_emb,
        ) = model_outputs
        pos_emb = None
    else:
        raise RuntimeError(f"Unexpected infer_single_view outputs: {len(model_outputs)}")

    merged = type(model).smplx_params_with_pred_shape_betas(smplx_dev, pred_shape)
    merged_betas = merged["betas"]
    return SimpleNamespace(
        device=dev,
        smplx_dev=smplx_dev,
        gs_model_list=gs_model_list,
        query_points=query_points,
        transform_mat_neutral_pose=transform_mat_neutral_pose,
        gs_hidden_features=gs_hidden_features,
        image_latents=image_latents,
        motion_emb=motion_emb,
        pos_emb=pos_emb,
        pred_shape=pred_shape,
        merged_betas=merged_betas,
    )


def _export_gaussian_model(
    model: torch.nn.Module,
    infer_ctx: SimpleNamespace,
    motion_seq_one: dict[str, Any],
    output_ply: str,
    *,
    export_animation_pose: bool,
) -> None:
    """Save one canonical or posed Gaussian PLY using cached ``infer_single_view`` outputs."""
    dtype = infer_ctx.merged_betas.dtype
    if export_animation_pose:
        gs_smplx = build_animation_frame_smplx_params(
            motion_seq_one,
            infer_ctx.transform_mat_neutral_pose,
            infer_ctx.merged_betas,
            infer_ctx.device,
            dtype,
        )
    else:
        gs_smplx = build_tpose_smplx_params(
            motion_seq_one,
            infer_ctx.transform_mat_neutral_pose,
            infer_ctx.merged_betas,
            infer_ctx.device,
            dtype,
        )

    render_c2ws = motion_seq_one["render_c2ws"].to(infer_ctx.device)
    render_intrs = motion_seq_one["render_intrs"].to(infer_ctx.device)
    render_bg_colors = motion_seq_one["render_bg_colors"].to(infer_ctx.device)
    view_idx = 0
    renderer = model.renderer

    if export_animation_pose:
        smplx_view = renderer.get_single_view_smpl_data(gs_smplx, view_idx)
        smplx_one = renderer._get_single_batch_data(smplx_view, 0)
        anim_models, _ = renderer.animate_gs_model(
            infer_ctx.gs_model_list[0],
            infer_ctx.query_points["neutral_coords"][0],
            smplx_one,
            debug=False,
            mesh_meta=infer_ctx.query_points["mesh_meta"],
        )
        if anim_models:
            gs_model = anim_models[0]
        else:
            gs_model = model.inference_gs(
                infer_ctx.gs_model_list,
                infer_ctx.query_points,
                gs_smplx,
                render_c2ws,
                render_intrs,
                render_bg_colors,
                infer_ctx.gs_hidden_features,
                pad_forward=False,
            )
    else:
        gs_model = model.inference_gs(
            infer_ctx.gs_model_list,
            infer_ctx.query_points,
            gs_smplx,
            render_c2ws,
            render_intrs,
            render_bg_colors,
            infer_ctx.gs_hidden_features,
            pad_forward=False,
        )

    out_abs = os.path.abspath(output_ply)
    out_dir = os.path.dirname(out_abs)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    gs_model.save_ply(out_abs)
    if export_animation_pose:
        print(f"Saved SMPL-X pose Gaussian PLY to {out_abs}")
    else:
        print(f"Saved T-pose canonical Gaussian PLY to {out_abs}")


def _render_sequence_frames(
    model: torch.nn.Module,
    infer_ctx: SimpleNamespace,
    motion_seq: dict[str, Any],
    *,
    infer_output_renderer: str,
    batch_size: int = 40,
) -> np.ndarray:
    """Render RGB preview frames for a motion sequence using cached ``infer_single_view`` outputs."""
    video_size = int(motion_seq["render_c2ws"].shape[1])
    offset_list = motion_seq.get("offset_list")
    ori_h, ori_w = motion_seq.get("ori_size", (512, 512))
    output_rgb = torch.ones((ori_h, ori_w, 3))

    batch_smplx_params: dict[str, torch.Tensor] = {
        "betas": infer_ctx.merged_betas,
        "transform_mat_neutral_pose": infer_ctx.transform_mat_neutral_pose,
    }
    batch_rgb_list: list[np.ndarray] = []
    num_batches = (video_size + batch_size - 1) // batch_size

    for batch_idx in range(0, video_size, batch_size):
        current_batch = batch_idx // batch_size + 1
        print(f"Rendering preview batch {current_batch}/{num_batches}")
        batch_smplx_params.update(
            {
                key: motion_seq["smplx_params"][key][
                    :, batch_idx : batch_idx + batch_size
                ].to(infer_ctx.device)
                for key in FRAME_VARYING_SMPLX_KEYS
                if key in motion_seq["smplx_params"]
            }
        )

        anim_kwargs: dict[str, Any] = {
            "gs_model_list": infer_ctx.gs_model_list,
            "query_points": infer_ctx.query_points,
            "smplx_params": batch_smplx_params,
            "render_c2ws": motion_seq["render_c2ws"][
                :, batch_idx : batch_idx + batch_size
            ].to(infer_ctx.device),
            "render_intrs": motion_seq["render_intrs"][
                :, batch_idx : batch_idx + batch_size
            ].to(infer_ctx.device),
            "render_bg_colors": motion_seq["render_bg_colors"][
                :, batch_idx : batch_idx + batch_size
            ].to(infer_ctx.device),
            "gs_hidden_features": infer_ctx.gs_hidden_features,
            "image_latents": infer_ctx.image_latents,
            "motion_emb": infer_ctx.motion_emb,
            "infer_output_renderer": infer_output_renderer,
        }
        if infer_ctx.pos_emb is not None:
            anim_kwargs["pos_emb"] = infer_ctx.pos_emb
        if offset_list is not None:
            anim_kwargs["offset_list"] = offset_list[batch_idx : batch_idx + batch_size]
            anim_kwargs["output_rgb"] = output_rgb

        batch_rgb, _batch_mask = model.animation_infer(**anim_kwargs)
        batch_rgb_list.append(
            (batch_rgb.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
        )

    return np.concatenate(batch_rgb_list, axis=0)


@torch.no_grad()
def run_tpose_export(
    model: torch.nn.Module,
    ref_imgs_tensor: torch.Tensor,
    motion_seq: dict[str, Any],
    device: str,
    output_ply: str,
    *,
    export_animation_pose: bool = False,
) -> None:
    """``infer_single_view`` → ``inference_gs`` → ``GaussianModel.save_ply``.

    SMPL-X for the forward pass is taken from the first frame of ``motion_seq`` (see
    ``slice_motion_seq_to_single_frame``), so it matches cameras and body pose length.

    If ``export_animation_pose`` is True (non-empty ``--pose_dir``), the final GS matches that
    JSON pose. Otherwise it uses canonical T-pose in SMPL-X angles.
    """
    motion_one = slice_motion_seq_to_single_frame(motion_seq, frame_idx=0)
    infer_ctx = _run_infer_single_view(model, ref_imgs_tensor, motion_one, device)
    _export_gaussian_model(
        model,
        infer_ctx,
        motion_one,
        output_ply,
        export_animation_pose=export_animation_pose,
    )


@torch.no_grad()
def run_pose_sequence_export(
    model: torch.nn.Module,
    ref_imgs_tensor: torch.Tensor,
    motion_seq: dict[str, Any],
    device: str,
    output_dir: str,
    *,
    video_fps: int,
    video_renderer: str,
) -> None:
    """Export a canonical PLY, frame-wise posed PLYs, and a preview MP4 for a pose JSON folder."""
    out_dir = os.path.abspath(output_dir)
    os.makedirs(out_dir, exist_ok=True)
    frame_count = int(motion_seq["render_c2ws"].shape[1])
    infer_ctx = _run_infer_single_view(model, ref_imgs_tensor, motion_seq, device)

    cano_path = os.path.join(out_dir, "cano_gs.ply")
    canonical_motion = slice_motion_seq_to_single_frame(motion_seq, frame_idx=0)
    _export_gaussian_model(
        model,
        infer_ctx,
        canonical_motion,
        cano_path,
        export_animation_pose=False,
    )

    for frame_idx in range(frame_count):
        frame_motion = slice_motion_seq_to_single_frame(motion_seq, frame_idx=frame_idx)
        frame_path = os.path.join(out_dir, f"frame_{frame_idx:05d}.ply")
        _export_gaussian_model(
            model,
            infer_ctx,
            frame_motion,
            frame_path,
            export_animation_pose=True,
        )

    preview_frames = _render_sequence_frames(
        model,
        infer_ctx,
        motion_seq,
        infer_output_renderer=video_renderer,
    )
    video_path = os.path.join(out_dir, "preview.mp4")
    images_to_video(
        preview_frames,
        output_path=video_path,
        fps=video_fps,
        gradio_codec=False,
        verbose=True,
    )
    print(
        f"Saved pose-sequence package to {out_dir} "
        f"(canonical PLY + {frame_count} posed PLYs + preview video)"
    )


def setup_loaders_and_inputs(args: argparse.Namespace):
    """Load cfg, model, reference images, motion tensors, and optional shape-from-image betas.

    If ``--pose_dir`` is empty, uses a single-frame synthetic camera and neutral SMPL-X pose.
    If ``--pose_dir`` points to one SMPL-X JSON, loads that frame only (no video / mask pipeline).
    """
    _require_gs_output_model(args.model_name)
    from core.utils.app_utils import obtain_ref_imgs

    from engine.pose_estimation.pose_estimator import PoseEstimator

    device = args.device
    dtype = torch.float32

    prior_model_check(save_dir="./pretrained_models")
    model_config = MODEL_CONFIG[args.model_name]
    if args.model_path:
        model_path = args.model_path
    else:
        auto_query = AutoModelQuery(save_dir="./pretrained_models")
        model_path = auto_query.query(args.model_name)
    model_cards = {
        args.model_name: {
            "model_path": model_path,
            "model_config": model_config,
        }
    }

    _ = Accelerator()
    cfg, _ = parse_app_configs(model_cards)

    model = build_app_model(cfg)
    model.to(device)

    pose_estimator = None
    if cfg.get("use_smplx_shape_estimator", True):
        pose_estimator = PoseEstimator("./pretrained_models/human_model_files/", device="cpu")
        pose_estimator.device = device

    image_paths = _resolve_image_paths(args)
    imgs_pil = [Image.open(p) for p in image_paths]
    image_for_prepare = [(np.asarray(img),) for img in imgs_pil]

    output_abs = os.path.abspath(args.output)
    if _pose_input_mode(args) == "pose_sequence":
        os.makedirs(output_abs, exist_ok=True)
        work_base = output_abs
    else:
        work_base = os.path.dirname(output_abs) or os.getcwd()
    work_dir_path = args.work_dir or os.path.join(work_base, "debug", "tpose_gs_work")
    os.makedirs(work_dir_path, exist_ok=True)
    working_dir = SimpleNamespace()
    working_dir.name = os.path.abspath(work_dir_path)

    imgs = obtain_ref_imgs(image_for_prepare, ref_view=args.ref_view)
    sample_imgs = np.concatenate(imgs, axis=1)
    save_sample_imgs = os.path.join(working_dir.name, "raw.png")
    with Image.fromarray(sample_imgs) as img:
        img.save(save_sample_imgs)

    pose_mode = _pose_input_mode(args)
    pose_json = _resolved_pose_input(args)
    if pose_mode == "single_pose":
        assert pose_json is not None
        motion_seqs = _build_motion_seq_from_pose_json(pose_json, cfg)
    elif pose_mode == "pose_sequence":
        assert pose_json is not None
        motion_seqs = _build_motion_seq_from_pose_dir(pose_json, cfg)
    else:
        motion_seqs = _build_synthetic_motion_seq(cfg)

    if pose_estimator is not None:
        with torch.no_grad():
            with _easy_memory_manager(pose_estimator, device=device):
                shape_pose = pose_estimator(imgs[0])
        if not shape_pose.is_full_body:
            raise ValueError(f"Input image invalid for shape estimator: {shape_pose.msg}")

    img_np = np.stack(imgs) / 255.0
    ref_imgs_tensor = torch.from_numpy(img_np).permute(0, 3, 1, 2).float().to(device)
    smplx_params = motion_seqs["smplx_params"].copy()
    if pose_estimator is not None:
        smplx_params["betas"] = torch.tensor(shape_pose.beta, dtype=dtype, device=device).unsqueeze(0)
    motion_seqs["smplx_params"] = smplx_params

    return model, cfg, ref_imgs_tensor, smplx_params, motion_seqs, pose_estimator, device


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.output is None:
        args.output = default_export_output_path(args)
    pose_mode = _pose_input_mode(args)

    os.environ.update(
        {
            "APP_ENABLED": "1",
            "APP_MODEL_NAME": args.model_name,
            "APP_TYPE": "infer.human_lrm_a4o",
            "NUMBA_THREADING_LAYER": "omp",
        }
    )

    (
        model,
        cfg,
        ref_imgs_tensor,
        smplx_params,
        motion_seqs,
        pose_estimator,
        device,
    ) = setup_loaders_and_inputs(args)

    if pose_mode == "pose_sequence":
        run_pose_sequence_export(
            model,
            ref_imgs_tensor,
            motion_seqs,
            device=device,
            output_dir=os.path.abspath(args.output),
            video_fps=args.video_fps,
            video_renderer=args.video_renderer,
        )
    else:
        run_tpose_export(
            model,
            ref_imgs_tensor,
            motion_seqs,
            device=device,
            output_ply=os.path.abspath(args.output),
            export_animation_pose=pose_mode == "single_pose",
        )


if __name__ == "__main__":
    main()
