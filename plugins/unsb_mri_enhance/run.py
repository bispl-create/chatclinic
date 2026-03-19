#!/usr/bin/env python3
"""
ChatClinic Plugin — UNSB ULF MRI Enhancement

Contract:
    python3 run.py --input input.json --output output.json

Input payload example:
    {
      "question": "Enhance this low-field MRI image",
      "analysis_source": {
        "file_name": "brain_64mT.png",
        "modality": "medical-image"
      }
    }

Output payload:
    {
      "summary": "...",
      "artifacts": { ... },
      "provenance": { ... }
    }
"""

import base64
import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from PIL import Image

# ── Resolve paths relative to this plugin folder ──────────────────────
PLUGIN_DIR = Path(__file__).resolve().parent
CKPT_DIR = PLUGIN_DIR / "checkpoints"
CKPT_EPOCH = "iter_65000"

# Inference defaults
NUM_TIMESTEPS = 5
TAU = 0.01
CROP_SIZE = 256
NGF = 64

# Need torch at module level for @torch.no_grad decorator
import torch


def ensure_imports():
    """Add plugin dir to sys.path so models/, util/, options/ are importable."""
    plugin_str = str(PLUGIN_DIR)
    if plugin_str not in sys.path:
        sys.path.insert(0, plugin_str)


def make_opt(gpu_id: int = 0):
    """Create minimal opt namespace for netG loading."""

    class Opt:
        pass

    opt = Opt()
    opt.input_nc = 3
    opt.output_nc = 3
    opt.ngf = NGF
    opt.netG = "resnet_9blocks_cond"
    opt.normG = "instance"
    opt.no_dropout = True
    opt.init_type = "xavier"
    opt.init_gain = 0.02
    opt.no_antialias = False
    opt.no_antialias_up = False
    opt.gpu_ids = [gpu_id] if torch.cuda.is_available() else []
    opt.isTrain = False
    opt.num_timesteps = NUM_TIMESTEPS
    opt.tau = TAU
    opt.crop_size = CROP_SIZE
    opt.stylegan2_G_num_downsampling = 1
    opt.nce_layers = "0,4,8,12,16"
    opt.num_patches = 256
    opt.embedding_type = "positional"
    opt.n_mlp = 3
    opt.style_dim = 512
    opt.embedding_dim = 512
    return opt


def build_time_schedule(num_timesteps: int):
    T = num_timesteps
    incs = np.array([0] + [1 / (i + 1) for i in range(T - 1)])
    times = np.cumsum(incs)
    times = times / times[-1]
    times = 0.5 * times[-1] + 0.5 * times
    times = np.concatenate([np.zeros(1), times])
    return times


@torch.no_grad()
def enhance(netG, input_tensor, device):
    times = torch.tensor(build_time_schedule(NUM_TIMESTEPS)).float().to(device)
    real_A = input_tensor.to(device)
    bs = real_A.size(0)

    netG.eval()
    Xt = None
    Xt_1 = None

    for t in range(NUM_TIMESTEPS):
        if t > 0:
            delta = times[t] - times[t - 1]
            denom = times[-1] - times[t - 1]
            inter = (delta / denom).reshape(-1, 1, 1, 1)
            scale = (delta * (1 - delta / denom)).reshape(-1, 1, 1, 1)

        if t == 0:
            Xt = real_A
        else:
            Xt = (1 - inter) * Xt + inter * Xt_1.detach() + (scale * TAU).sqrt() * torch.randn_like(Xt)

        time_idx = (t * torch.ones(bs, device=device)).long()
        z = torch.randn(bs, 4 * NGF, device=device)
        Xt_1 = netG(Xt, time_idx, z)

    return Xt_1


def load_image(path: str):
    from torchvision import transforms

    img = Image.open(path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(CROP_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(CROP_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3),
    ])
    return transform(img).unsqueeze(0)


def tensor_to_pil(tensor) -> Image.Image:
    img = tensor.squeeze(0).cpu().clamp(-1, 1)
    img = ((img + 1.0) * 0.5 * 255.0).byte()
    img = img.permute(1, 2, 0).numpy()
    return Image.fromarray(img)


def make_comparison(img_in: Image.Image, img_out: Image.Image) -> Image.Image:
    h = max(img_in.height, img_out.height)
    sep = 4
    canvas = Image.new("RGB", (img_in.width + sep + img_out.width, h), (40, 40, 40))
    canvas.paste(img_in, (0, 0))
    canvas.paste(img_out, (img_in.width + sep, 0))
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    # ── Read input payload ──
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))

    # Defensive: accept multiple payload formats from orchestrator
    image_path = ""
    # Format 0: source_file_path from artifacts (runtime_uploads absolute path)
    artifacts = payload.get("analysis_artifacts") or payload.get("artifacts") or {}
    for _key in ("metadata", "source0::metadata"):
        meta_block = artifacts.get(_key)
        if isinstance(meta_block, dict):
            candidate = meta_block.get("source_file_path", "")
            if candidate and os.path.isfile(candidate):
                image_path = candidate
                break
            for item in meta_block.get("items") or []:
                candidate = (item or {}).get("source_file_path", "")
                if candidate and os.path.isfile(candidate):
                    image_path = candidate
                    break
            if image_path:
                break
    # Format 1: Plugin Guide standard  {"analysis_source": {"file_name": "..."}}
    if not image_path:
        source = payload.get("analysis_source", {})
        if isinstance(source, dict):
            image_path = source.get("file_name", "")
    # Format 2: direct file_name       {"file_name": "..."}
    if not image_path:
        image_path = payload.get("file_name", "")
    # Format 3: image_path key          {"image_path": "..."}
    if not image_path:
        image_path = payload.get("image_path", "")

    if not image_path or not os.path.isfile(image_path):
        result = {
            "summary": f"Error: input image not found — '{image_path}'",
            "artifacts": {},
            "provenance": {"tool_version": "1.0.0"},
        }
        Path(args.output).write_text(json.dumps(result), encoding="utf-8")
        return 1

    # ── Setup ──
    ensure_imports()
    from models import networks

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    opt = make_opt(gpu_id=0)

    # ── Load Generator ──
    ckpt_path = CKPT_DIR / f"{CKPT_EPOCH}_net_G.pth"
    if not ckpt_path.exists():
        result = {
            "summary": f"Error: checkpoint not found — {ckpt_path}",
            "artifacts": {},
            "provenance": {"tool_version": "1.0.0"},
        }
        Path(args.output).write_text(json.dumps(result), encoding="utf-8")
        return 1

    netG = networks.define_G(
        opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG,
        not opt.no_dropout, opt.init_type, opt.init_gain,
        opt.no_antialias, opt.no_antialias_up,
        gpu_ids=opt.gpu_ids, opt=opt,
    )
    state_dict = torch.load(str(ckpt_path), map_location="cpu")
    if isinstance(netG, torch.nn.DataParallel):
        netG.module.load_state_dict(state_dict, strict=False)
    else:
        netG.load_state_dict(state_dict, strict=False)
    netG = netG.to(device)
    netG.eval()

    # ── Inference ──
    x = load_image(image_path).to(device)
    enhanced_tensor = enhance(netG, x, device)

    # ── Save artifacts ──
    basename = Path(image_path).stem
    output_dir = Path(args.output).parent
    enhanced_path = output_dir / f"{basename}_enhanced.png"
    comparison_path = output_dir / f"{basename}_comparison.png"

    img_enhanced = tensor_to_pil(enhanced_tensor)
    img_enhanced.save(str(enhanced_path))

    img_input = Image.open(image_path).convert("RGB").resize(
        (CROP_SIZE, CROP_SIZE), Image.BICUBIC
    )
    comparison = make_comparison(img_input, img_enhanced)
    comparison.save(str(comparison_path))

    # ── Persist to runtime_uploads so files outlive the temp dir ──
    persist_dir = Path(__file__).resolve().parents[2] / "runtime_uploads" / "unsb_results"
    persist_dir.mkdir(parents=True, exist_ok=True)
    import shutil, uuid as _uuid
    tag = _uuid.uuid4().hex[:8]
    persist_enhanced = persist_dir / f"{basename}_{tag}_enhanced.png"
    persist_comparison = persist_dir / f"{basename}_{tag}_comparison.png"
    shutil.copy2(str(enhanced_path), str(persist_enhanced))
    shutil.copy2(str(comparison_path), str(persist_comparison))

    # ── Encode images as base64 data URLs for frontend ──
    import io as _io
    def _to_data_url(img: Image.Image) -> str:
        buf = _io.BytesIO()
        img.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

    enhanced_data_url = _to_data_url(img_enhanced)
    comparison_data_url = _to_data_url(comparison)

    # ── Write output payload ──
    result = {
        "summary": (
            f"ULF MRI enhancement complete. "
            f"Input 64mT image enhanced to 3T-like quality using UNSB "
            f"({NUM_TIMESTEPS}-step SDE, tau={TAU})."
        ),
        "artifacts": {
            "enhanced_image": {
                "type": "image",
                "path": str(persist_enhanced),
                "image_data_url": enhanced_data_url,
                "description": "Enhanced 3T-like MRI image",
            },
            "comparison": {
                "type": "image",
                "path": str(persist_comparison),
                "image_data_url": comparison_data_url,
                "description": "Side-by-side: 64mT input (left) vs enhanced (right)",
            },
        },
        "provenance": {
            "tool_version": "1.0.0",
            "model": "UNSB (Unpaired Neural Schrödinger Bridge)",
            "checkpoint": CKPT_EPOCH,
            "num_timesteps": NUM_TIMESTEPS,
            "tau": TAU,
        },
    }

    Path(args.output).write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())