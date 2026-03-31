#!/usr/bin/env python3

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from skimage import color
from torchvision import transforms

from diffusers import (
    ControlNetModel,
    DDIMScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample images from a ControlNet checkpoint saved by train_controlnet.py."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        help="Base Stable Diffusion model used during training, for example runwayml/stable-diffusion-v1-5.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help=(
            "Path to a checkpoint directory or its controlnet subdirectory. "
            "Examples: outputs/my_run/checkpoint-5000 or outputs/my_run/checkpoint-5000/controlnet."
        ),
    )
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--prompt", type=str, nargs="+", default=None)
    parser.add_argument("--negative_prompt", type=str, nargs="+", default=None)
    parser.add_argument("--conditioning_image", type=str, nargs="+", default=None)
    parser.add_argument("--l_image", type=str, nargs="+", default=None)
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=None,
        help=(
            "Dataset root or split directory for batch sampling. Expected metadata.jsonl plus "
            "images/conditioning/(optional) l_images."
        ),
    )
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--save_ground_truth", action="store_true")
    parser.add_argument("--save_conditioning_preview", action="store_true")
    parser.add_argument(
        "--conditioning_mode",
        choices=["canny", "l_canny"],
        default=None,
        help="Optional override. If omitted, the script infers it from the saved ControlNet config.",
    )
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--num_images_per_prompt", type=int, default=1)
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--controlnet_conditioning_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--scheduler", choices=["default", "unipc", "ddim", "euler", "euler_a"], default="unipc")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu", "mps"], default="auto")
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=None,
        help="CUDA GPU index to use, for example 0 or 1. Applies when --device is 'auto' or 'cuda'.",
    )
    parser.add_argument("--dtype", choices=["auto", "fp16", "bf16", "fp32"], default="auto")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--disable_progress_bar", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    args = parser.parse_args()

    if args.resolution % 8 != 0:
        raise ValueError("--resolution must be divisible by 8.")

    if args.num_images_per_prompt < 1:
        raise ValueError("--num_images_per_prompt must be at least 1.")

    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1.")

    if args.gpu_id is not None and args.gpu_id < 0:
        raise ValueError("--gpu_id must be >= 0.")

    if args.gpu_id is not None and args.device not in {"auto", "cuda"}:
        raise ValueError("--gpu_id can only be used with --device auto or --device cuda.")

    if args.dataset_root is not None:
        if args.conditioning_image is not None:
            raise ValueError("--conditioning_image cannot be used together with --dataset_root.")
        if args.l_image is not None:
            raise ValueError("--l_image cannot be used together with --dataset_root.")
        if args.prompt is not None and len(args.prompt) != 1:
            raise ValueError("--prompt must be omitted or provided once when --dataset_root is used.")
    else:
        if args.prompt is None:
            raise ValueError("--prompt is required unless --dataset_root is used.")
        if args.conditioning_image is None:
            raise ValueError("--conditioning_image is required unless --dataset_root is used.")

    return args


@dataclass
class ResolvedControlNet:
    controlnet_dir: Path
    run_dir: Path
    source_name: str


@dataclass
class SampleSpec:
    prompt: str
    conditioning_path: Path
    output_name: str
    l_path: Optional[Path] = None
    gt_path: Optional[Path] = None


def has_controlnet_weights(path: Path) -> bool:
    return path.is_dir() and (path / "config.json").is_file() and (
        (path / "diffusion_pytorch_model.safetensors").is_file()
        or (path / "diffusion_pytorch_model.bin").is_file()
    )


def resolve_controlnet_dir(checkpoint_path: str) -> ResolvedControlNet:
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    if has_controlnet_weights(path):
        if path.name == "controlnet" and re.fullmatch(r"checkpoint-\d+", path.parent.name):
            return ResolvedControlNet(controlnet_dir=path, run_dir=path.parent.parent, source_name=path.parent.name)
        if path.name == "controlnet":
            return ResolvedControlNet(controlnet_dir=path, run_dir=path.parent, source_name=path.name)
        else:
            return ResolvedControlNet(controlnet_dir=path, run_dir=path.parent, source_name=path.name)

    if path.is_dir() and path.name.startswith("checkpoint-") and has_controlnet_weights(path / "controlnet"):
        return ResolvedControlNet(controlnet_dir=path / "controlnet", run_dir=path.parent, source_name=path.name)

    raise ValueError(
        "Invalid --checkpoint_path. Point it to a checkpoint directory like "
        "'outputs/my_run/checkpoint-5000' or directly to its 'controlnet' subdirectory."
    )


def load_conditioning_mode(controlnet_dir: Path, override: Optional[str]) -> str:
    with open(controlnet_dir / "config.json", "r", encoding="utf-8") as handle:
        config = json.load(handle)

    conditioning_channels = config.get("conditioning_channels")
    inferred = "l_canny" if conditioning_channels == 2 else "canny"

    if override is not None and override != inferred:
        raise ValueError(
            f"--conditioning_mode={override} does not match the saved checkpoint "
            f"(conditioning_channels={conditioning_channels}, inferred mode={inferred})."
        )

    return override or inferred


def get_torch_device(device_arg: str, gpu_id: Optional[int]) -> torch.device:
    cuda_device = f"cuda:{gpu_id}" if gpu_id is not None else "cuda"

    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device(cuda_device)
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available.")
        return torch.device(cuda_device)
    return torch.device(device_arg)


def get_torch_dtype(dtype_arg: str, device: torch.device) -> torch.dtype:
    if dtype_arg == "fp16":
        return torch.float16
    if dtype_arg == "bf16":
        return torch.bfloat16
    if dtype_arg == "fp32":
        return torch.float32
    if device.type == "cuda":
        return torch.float16
    return torch.float32


def get_conditioning_image_transforms(resolution: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
        ]
    )


def build_reference_l_channel(
    l_image: Image.Image,
    conditioning_image_transforms: transforms.Compose,
) -> np.ndarray:
    l_tensor = conditioning_image_transforms(l_image.convert("L"))
    return (l_tensor.squeeze(0).cpu().numpy().astype(np.float32) * 100.0)


def replace_l_channel_in_generated_image(
    generated_image: Image.Image,
    reference_l_channel: np.ndarray,
) -> Image.Image:
    generated_rgb = np.asarray(generated_image.convert("RGB"), dtype=np.uint8)
    generated_lab = color.rgb2lab(generated_rgb.astype(np.float32) / 255.0).astype(np.float32)
    if generated_lab.shape[:2] != reference_l_channel.shape:
        raise ValueError(
            "Generated image shape does not match reference L channel shape: "
            f"{generated_lab.shape[:2]} vs {reference_l_channel.shape}"
        )
    generated_lab[..., 0] = reference_l_channel
    generated_rgb = np.clip(color.lab2rgb(generated_lab).astype(np.float32), 0.0, 1.0)
    return Image.fromarray((generated_rgb * 255.0 + 0.5).astype(np.uint8))


def make_conditioning_tensor(
    conditioning_image: Image.Image,
    conditioning_mode: str,
    conditioning_image_transforms: transforms.Compose,
    l_image: Optional[Image.Image] = None,
) -> torch.Tensor:
    if conditioning_mode == "canny":
        return conditioning_image_transforms(conditioning_image.convert("RGB"))
    if l_image is None:
        raise ValueError("l_image is required when conditioning_mode='l_canny'.")
    l_tensor = conditioning_image_transforms(l_image.convert("L"))
    canny_tensor = conditioning_image_transforms(conditioning_image.convert("L"))
    return torch.cat([l_tensor, canny_tensor], dim=0)


def build_conditioning_preview(
    conditioning_image: Image.Image,
    conditioning_mode: str,
    conditioning_image_transforms: transforms.Compose,
    l_image: Optional[Image.Image] = None,
) -> Image.Image:
    if conditioning_mode == "canny":
        tensor = conditioning_image_transforms(conditioning_image.convert("RGB"))
        return transforms.ToPILImage()(tensor)

    if l_image is None:
        raise ValueError("l_image is required when conditioning_mode='l_canny'.")

    l_tensor = conditioning_image_transforms(l_image.convert("L"))
    canny_tensor = conditioning_image_transforms(conditioning_image.convert("L"))
    return Image.merge(
        "RGB",
        (
            transforms.ToPILImage()(l_tensor),
            transforms.ToPILImage()(canny_tensor),
            transforms.ToPILImage()(l_tensor),
        ),
    )


def expand_to_match(values: Optional[Sequence[str]], target_length: int, name: str) -> List[Optional[str]]:
    if values is None:
        return [None] * target_length
    if len(values) == target_length:
        return list(values)
    if len(values) == 1:
        return list(values) * target_length
    raise ValueError(f"--{name} must have length 1 or match the number of prompts.")


def sanitize_filename(value: str, max_length: int = 80) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip()).strip("._")
    if not cleaned:
        cleaned = "sample"
    return cleaned[:max_length]


def configure_scheduler(pipeline: StableDiffusionControlNetPipeline, scheduler_name: str) -> None:
    if scheduler_name == "default":
        return
    if scheduler_name == "unipc":
        pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
        return
    if scheduler_name == "ddim":
        pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
        return
    if scheduler_name == "euler":
        pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)
        return
    if scheduler_name == "euler_a":
        pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)
        return
    raise ValueError(f"Unsupported scheduler: {scheduler_name}")


def make_output_dir(resolved: ResolvedControlNet, explicit_output_dir: Optional[str]) -> Path:
    if explicit_output_dir is not None:
        output_dir = Path(explicit_output_dir).expanduser().resolve()
    else:
        output_dir = (Path.cwd() / "samples" / resolved.run_dir.name / resolved.source_name).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def resolve_dataset_split_dir(dataset_root: str, split: str) -> Path:
    root = Path(dataset_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {root}")
    if (root / "metadata.jsonl").is_file():
        return root

    split_dir = root / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Dataset split directory not found: {split_dir}")
    if not (split_dir / "metadata.jsonl").is_file():
        raise FileNotFoundError(f"metadata.jsonl not found in dataset split: {split_dir}")
    return split_dir


def build_direct_sample_specs(args: argparse.Namespace, conditioning_mode: str) -> List[SampleSpec]:
    prompts = list(args.prompt)
    conditioning_images = expand_to_match(args.conditioning_image, len(prompts), "conditioning_image")

    if conditioning_mode == "l_canny" and args.l_image is None:
        raise ValueError("This checkpoint expects l_canny conditioning, so --l_image is required.")

    l_images = expand_to_match(args.l_image, len(prompts), "l_image") if args.l_image is not None else [None] * len(prompts)

    sample_specs: List[SampleSpec] = []
    for prompt_idx, (prompt, conditioning_image_path, l_image_path) in enumerate(zip(prompts, conditioning_images, l_images)):
        conditioning_path = Path(conditioning_image_path).expanduser().resolve()
        if not conditioning_path.is_file():
            raise FileNotFoundError(f"Conditioning image not found: {conditioning_path}")

        l_path = Path(l_image_path).expanduser().resolve() if l_image_path is not None else None
        if l_path is not None and not l_path.is_file():
            raise FileNotFoundError(f"L image not found: {l_path}")

        prompt_slug = sanitize_filename(f"{prompt_idx:03d}_{prompt}")
        sample_specs.append(
            SampleSpec(
                prompt=prompt,
                conditioning_path=conditioning_path,
                l_path=l_path,
                gt_path=None,
                output_name=f"{prompt_slug}.png",
            )
        )

    return sample_specs


def build_dataset_sample_specs(
    split_dir: Path,
    conditioning_mode: str,
    prompt_override: Optional[str],
    limit: Optional[int],
) -> List[SampleSpec]:
    metadata_path = split_dir / "metadata.jsonl"
    sample_specs: List[SampleSpec] = []

    with metadata_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if limit is not None and len(sample_specs) >= limit:
                break

            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            prompt = prompt_override or record.get("text")
            if not prompt:
                raise ValueError(f"Missing prompt text in {metadata_path}:{line_number}")

            conditioning_rel = record.get("conditioning_image_file_name")
            if not conditioning_rel:
                raise ValueError(f"Missing conditioning_image_file_name in {metadata_path}:{line_number}")
            conditioning_path = (split_dir / conditioning_rel).resolve()
            if not conditioning_path.is_file():
                raise FileNotFoundError(f"Conditioning image not found: {conditioning_path}")

            gt_rel = record.get("file_name")
            gt_path = (split_dir / gt_rel).resolve() if gt_rel else None
            if gt_path is not None and not gt_path.is_file():
                raise FileNotFoundError(f"Ground-truth image not found: {gt_path}")

            l_rel = record.get("l_image_file_name")
            if conditioning_mode == "l_canny" and not l_rel:
                raise ValueError(f"Missing l_image_file_name in {metadata_path}:{line_number}")
            if l_rel:
                l_path = (split_dir / l_rel).resolve()
                if not l_path.is_file():
                    raise FileNotFoundError(f"L image not found: {l_path}")
            else:
                l_path = None

            output_name = Path(gt_rel or conditioning_rel).name
            sample_specs.append(
                SampleSpec(
                    prompt=prompt,
                    conditioning_path=conditioning_path,
                    l_path=l_path,
                    gt_path=gt_path,
                    output_name=output_name,
                )
            )

    if not sample_specs:
        raise ValueError(f"No samples found in dataset split: {split_dir}")

    return sample_specs


def get_generated_output_name(output_name: str, image_idx: int, num_images_per_prompt: int) -> str:
    output_path = Path(output_name)
    suffix = output_path.suffix or ".png"
    if num_images_per_prompt == 1:
        return f"{output_path.stem}{suffix}"
    return f"{output_path.stem}_{image_idx:02d}{suffix}"


def main() -> None:
    args = parse_args()
    resolved = resolve_controlnet_dir(args.checkpoint_path)
    conditioning_mode = load_conditioning_mode(resolved.controlnet_dir, args.conditioning_mode)
    split_dir = resolve_dataset_split_dir(args.dataset_root, args.split) if args.dataset_root is not None else None

    if split_dir is not None:
        prompt_override = args.prompt[0] if args.prompt is not None else None
        sample_specs = build_dataset_sample_specs(split_dir, conditioning_mode, prompt_override, args.limit)
    else:
        sample_specs = build_direct_sample_specs(args, conditioning_mode)

    negative_prompts = expand_to_match(args.negative_prompt, len(sample_specs), "negative_prompt")

    device = get_torch_device(args.device, args.gpu_id)
    dtype = get_torch_dtype(args.dtype, device)
    output_dir = make_output_dir(resolved, args.output_dir)
    if split_dir is not None and args.output_dir is None:
        output_dir = output_dir / split_dir.name
        output_dir.mkdir(parents=True, exist_ok=True)
    conditioning_transforms = get_conditioning_image_transforms(args.resolution)
    save_conditioning_preview = args.save_conditioning_preview or split_dir is None
    conditioning_preview_dir = output_dir / "conditioning_preview" if save_conditioning_preview else None
    ground_truth_dir = output_dir / "ground_truth" if args.save_ground_truth else None
    if conditioning_preview_dir is not None:
        conditioning_preview_dir.mkdir(parents=True, exist_ok=True)
    if ground_truth_dir is not None:
        ground_truth_dir.mkdir(parents=True, exist_ok=True)

    if device.type == "cuda":
        torch.cuda.set_device(device)

    controlnet = ControlNetModel.from_pretrained(
        resolved.controlnet_dir,
        torch_dtype=dtype,
        local_files_only=args.local_files_only,
    )
    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        controlnet=controlnet,
        safety_checker=None,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=dtype,
        local_files_only=args.local_files_only,
    )
    configure_scheduler(pipeline, args.scheduler)
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=args.disable_progress_bar)

    metadata = {
        "pretrained_model_name_or_path": args.pretrained_model_name_or_path,
        "resolved_controlnet_dir": str(resolved.controlnet_dir),
        "run_dir": str(resolved.run_dir),
        "source_name": resolved.source_name,
        "conditioning_mode": conditioning_mode,
        "resolution": args.resolution,
        "scheduler": args.scheduler,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "controlnet_conditioning_scale": args.controlnet_conditioning_scale,
        "seed": args.seed,
        "device": str(device),
        "gpu_id": args.gpu_id,
        "dtype": str(dtype),
        "dataset_root": str(Path(args.dataset_root).expanduser().resolve()) if args.dataset_root is not None else None,
        "split": split_dir.name if split_dir is not None else None,
        "num_samples": len(sample_specs),
        "replace_l_channel_with_reference": True,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    generator_device = str(device) if device.type in {"cuda", "cpu"} else "cpu"
    sample_index = 0

    with (output_dir / "samples.jsonl").open("w", encoding="utf-8") as manifest:
        for sample_spec, negative_prompt in zip(sample_specs, negative_prompts):
            with Image.open(sample_spec.conditioning_path) as conditioning_image:
                reference_l_channel = None
                if sample_spec.l_path is not None:
                    with Image.open(sample_spec.l_path) as l_image:
                        reference_l_channel = build_reference_l_channel(l_image, conditioning_transforms)
                        if conditioning_mode == "l_canny":
                            control_tensor = make_conditioning_tensor(
                                conditioning_image,
                                conditioning_mode,
                                conditioning_transforms,
                                l_image=l_image,
                            )
                            preview = build_conditioning_preview(
                                conditioning_image,
                                conditioning_mode,
                                conditioning_transforms,
                                l_image=l_image,
                            )
                        else:
                            control_tensor = make_conditioning_tensor(
                                conditioning_image,
                                conditioning_mode,
                                conditioning_transforms,
                            )
                            preview = build_conditioning_preview(
                                conditioning_image,
                                conditioning_mode,
                                conditioning_transforms,
                            )
                else:
                    control_tensor = make_conditioning_tensor(
                        conditioning_image,
                        conditioning_mode,
                        conditioning_transforms,
                    )
                    preview = build_conditioning_preview(
                        conditioning_image,
                        conditioning_mode,
                        conditioning_transforms,
                    )

            if conditioning_preview_dir is not None:
                preview.save(conditioning_preview_dir / sample_spec.output_name)

            if ground_truth_dir is not None and sample_spec.gt_path is not None:
                shutil.copy2(sample_spec.gt_path, ground_truth_dir / Path(sample_spec.output_name).name)

            control_tensor = control_tensor.unsqueeze(0).to(device=device, dtype=dtype)

            for image_idx in range(args.num_images_per_prompt):
                generator = None
                if args.seed is not None:
                    generator = torch.Generator(device=generator_device).manual_seed(args.seed + sample_index)

                result = pipeline(
                    prompt=sample_spec.prompt,
                    negative_prompt=negative_prompt,
                    image=control_tensor,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    controlnet_conditioning_scale=args.controlnet_conditioning_scale,
                    generator=generator,
                    height=args.resolution,
                    width=args.resolution,
                )
                image = result.images[0]
                if reference_l_channel is not None:
                    image = replace_l_channel_in_generated_image(image, reference_l_channel)
                output_name = get_generated_output_name(sample_spec.output_name, image_idx, args.num_images_per_prompt)
                output_path = output_dir / output_name
                image.save(output_path)
                manifest.write(
                    json.dumps(
                        {
                            "prompt": sample_spec.prompt,
                            "negative_prompt": negative_prompt,
                            "conditioning_path": str(sample_spec.conditioning_path),
                            "l_path": str(sample_spec.l_path) if sample_spec.l_path is not None else None,
                            "ground_truth_path": str(sample_spec.gt_path) if sample_spec.gt_path is not None else None,
                            "output_path": str(output_path),
                        }
                    )
                    + "\n"
                )
                sample_index += 1

    print(f"Saved samples to {output_dir}")
    print(f"Loaded ControlNet from {resolved.controlnet_dir}")


if __name__ == "__main__":
    main()
