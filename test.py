"""Inference/smoke-test entry point; intentionally computes no evaluation metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wat3r.models.wat3r import Wat3R  # noqa: E402
from wat3r.utils.load_fn import load_and_preprocess_images  # noqa: E402


def _state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError("The checkpoint must be a torch dictionary or a state dict")
    if isinstance(checkpoint.get("model"), dict):
        state = checkpoint["model"]
    elif isinstance(checkpoint.get("state_dict"), dict):
        state = checkpoint["state_dict"]
    else:
        state = checkpoint
    if state and not any(str(key).startswith("aggregator.") for key in state):
        if all(str(key).startswith("module.") for key in state):
            state = {str(key)[len("module."):]: value for key, value in state.items()}
    return state


def load_model(path: Path, device: torch.device) -> Wat3R:
    model = Wat3R(
        img_size=518,
        patch_size=14,
        enable_camera=True,
        enable_depth=True,
        enable_point=True,
        enable_track=False,
        enable_optics=True,
        dpt_plugin_type="scsa",
        dpt_plugin_gate_init=0.0,
    )
    checkpoint = torch.load(path, map_location="cpu")
    state = _state_dict(checkpoint)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError(f"Checkpoint is missing {len(missing)} model keys; first keys: {missing[:5]}")
    if unexpected:
        print(f"Ignoring {len(unexpected)} non-model checkpoint keys")
    return model.to(device).eval()


def _image_paths(directory: Path) -> list[Path]:
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in suffixes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to a separately provided model checkpoint")
    parser.add_argument("--input", type=Path, help="Directory containing RGB images")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs" / "test")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true", help="Run a synthetic forward pass without image files")
    args = parser.parse_args()

    if not args.smoke and args.input is None:
        parser.error("provide --input or use --smoke")
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)

    if args.smoke:
        images = torch.rand(1, 2, 3, 518, 518, device=device)
        image_names = []
    else:
        image_names = _image_paths(args.input)
        if len(image_names) < 2:
            raise ValueError("--input must contain at least two RGB images")
        images = load_and_preprocess_images([str(p) for p in image_names], mode="crop", target_size=518)
        images = images.unsqueeze(0).to(device)

    with torch.inference_mode():
        predictions = model(images, frames_chunk_size=8, need_camera=True, need_depth=True, need_point=True)

    args.output.mkdir(parents=True, exist_ok=True)
    for key in ("depth", "depth_conf", "world_points", "world_points_conf"):
        if key in predictions:
            torch.save(predictions[key].detach().cpu(), args.output / f"{key}.pt")
    (args.output / "metadata.json").write_text(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "device": str(device),
                "images": [str(p) for p in image_names],
                "outputs": sorted(k for k in predictions if k != "images"),
                "evaluation": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote raw predictions to {args.output}")


if __name__ == "__main__":
    main()
