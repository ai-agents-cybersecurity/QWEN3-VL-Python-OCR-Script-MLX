"""Image to text pipeline using an MLX vision-language model."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

try:
    from mlx_vlm import load, generate  # type: ignore
    USING_MLX_VLM = True
except ModuleNotFoundError:
    USING_MLX_VLM = False
    try:
        from mlx_lm import load, generate
    except ModuleNotFoundError as exc:  # pragma: no cover - fails fast if deps missing
        raise SystemExit(
            "mlx_lm (or mlx_vlm) is required. Install dependencies from requirements.txt first."
        ) from exc

try:
    from huggingface_hub import snapshot_download
except ModuleNotFoundError as exc:  # pragma: no cover - fails fast if deps missing
    raise SystemExit(
        "huggingface_hub is required. Install dependencies from requirements.txt first."
    ) from exc

try:
    from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE as _HF_CACHE
except Exception:  # pragma: no cover - best-effort fallback for older hubs
    _HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"

HUGGINGFACE_HUB_CACHE = Path(_HF_CACHE).expanduser()


DEFAULT_SYSTEM_PROMPT = (
    "You are a meticulous assistant that describes images clearly and concisely."
)
DEFAULT_USER_PROMPT = "Describe the contents of this image."
DEFAULT_REPO_ID = "lmstudio-community/Qwen3-VL-30B-A3B-Instruct-MLX-8bit"


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run image-to-text captioning over all image files (PNG, JPG, JPEG) in a folder using an MLX VLM.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Path to the local MLX model directory. Defaults to ~/.cache/huggingface/hub/models--<repo-id>.",
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help="Hugging Face repo id to download if the model directory is missing.",
    )
    parser.add_argument(
        "--input-dir",
        default="input",
        help="Directory containing PNG files to process (default: input)",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory for writing per-image JSON reports (default: output)",
    )
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt injected ahead of each request.",
    )
    parser.add_argument(
        "--user-prompt",
        default=DEFAULT_USER_PROMPT,
        help="Prompt sent alongside each image.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum new tokens to sample per image (default: 512)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for generation (default: 0.0)",
    )
    return parser.parse_args(argv)


def iter_images(input_dir: Path) -> Iterable[Path]:
    """Iterate over all supported image files (PNG, JPG, JPEG) in the input directory."""
    candidates = []
    for pattern in ("*.png", "*.PNG", "*.jpg", "*.JPG", "*.jpeg", "*.JPEG"):
        candidates.extend(input_dir.glob(pattern))
    
    unique_paths = sorted({p.resolve() for p in candidates}, key=lambda p: p.name.lower())
    logging.info(f"Found {len(unique_paths)} image(s) in {input_dir}")
    
    for path in unique_paths:
        if path.is_file():
            logging.debug(f"Yielding image: {path.name}")
            yield path


def load_image_for_validation(path: Path) -> None:
    # Fail fast if the file is not a valid image.
    with Image.open(path) as img:
        img.verify()


REQUIRED_FILES = ("params.json", "config.json")


def ensure_model_available(model_path: Path, repo_id: str) -> Path:
    if any((model_path / candidate).exists() for candidate in REQUIRED_FILES):
        return model_path

    print(
        f"Model files not found at {model_path}. Initiating download for '{repo_id}'."
    )
    try:
        model_path.mkdir(parents=True, exist_ok=True)
    except PermissionError as exc:
        raise SystemExit(
            f"Cannot create model directory {model_path}: {exc}. Use --model to choose a writable location."
        ) from exc
    try:
        snapshot_download(
            repo_id,
            local_dir=str(model_path),
        )
    except Exception as exc:  # pragma: no cover - network failures, auth issues, etc.
        raise SystemExit(
            f"Failed to download model '{repo_id}' to {model_path}: {exc}"
        ) from exc

    if not any((model_path / candidate).exists() for candidate in REQUIRED_FILES):
        expected = ", ".join(REQUIRED_FILES)
        raise SystemExit(
            f"Download finished but none of the expected metadata files ({expected}) found in {model_path}."
        )

    return model_path


def resolve_model_path(model_arg: str | None, repo_id: str) -> Path:
    if model_arg:
        return Path(model_arg).expanduser()

    safe_repo = repo_id.replace("/", "--")
    default_hf_cache = HUGGINGFACE_HUB_CACHE / f"models--{safe_repo}"
    return default_hf_cache


def build_prompt(tokenizer, image_path: Path, system_prompt: str, user_prompt: str) -> str:
    """Build a chat prompt with placeholders for the image."""
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def generate_caption(
    model,
    tokenizer,
    image_path: Path,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
) -> str:
    prompt = build_prompt(tokenizer, image_path, system_prompt, user_prompt)
    kwargs = {
        "max_tokens": max_tokens,
        "image": str(image_path),
    }
    if USING_MLX_VLM:
        kwargs["temperature"] = temperature
    else:
        if temperature not in (0.0, 0):
            print("Warning: temperature control requires mlx_vlm; using greedy decoding.")
        kwargs["temperature"] = 0.0
    result = generate(
        model,
        tokenizer,
        prompt,
        **kwargs,
    )
    # Extract text from GenerationResult object (mlx_vlm returns an object, not a string)
    if isinstance(result, str):
        return result
    elif hasattr(result, 'text'):
        return result.text
    else:
        # Fallback: convert to string
        return str(result)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise SystemExit(f"Input directory not found: {input_dir}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output directory: {output_dir}")

    image_files = list(iter_images(input_dir))
    if not image_files:
        raise SystemExit(f"No image files (PNG, JPG, JPEG) found in {input_dir}")

    model_path = resolve_model_path(args.model, args.repo_id)
    model_path = ensure_model_available(model_path, args.repo_id)

    logging.info(f"Loading model from {model_path}...")
    loaded = load(str(model_path), lazy=True)
    if not isinstance(loaded, tuple):
        raise SystemExit("Unexpected response from mlx_lm.load: expected tuple")
    model, tokenizer = loaded[:2]
    logging.info("Model loaded successfully")

    if not USING_MLX_VLM:
        raise SystemExit(
            "This checkpoint is multimodal and requires the `mlx-vlm` package. "
            "Install it (pip install mlx-vlm) and rerun."
        )

    results = []
    for idx, image_path in enumerate(image_files, 1):
        logging.info(f"[{idx}/{len(image_files)}] Processing {image_path.name}...")
        try:
            load_image_for_validation(image_path)
        except Exception as e:
            logging.error(f"Failed to validate {image_path.name}: {e}")
            continue
        try:
            caption = generate_caption(
                model=model,
                tokenizer=tokenizer,
                image_path=image_path,
                system_prompt=args.system_prompt,
                user_prompt=args.user_prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            record = {
                "image": image_path.name,
                "prompt": args.user_prompt,
                "caption": caption.strip(),
            }
            results.append(record)
            output_file = output_dir / f"{image_path.stem}.json"
            output_file.write_text(
                json.dumps(record, indent=2),
                encoding="utf-8",
            )
            logging.info(f"✓ Saved caption to {output_file.name}")
        except Exception as e:
            logging.error(f"Failed to process {image_path.name}: {e}")
            continue

    summary_path = output_dir / "captions.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {len(results)} caption files to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
