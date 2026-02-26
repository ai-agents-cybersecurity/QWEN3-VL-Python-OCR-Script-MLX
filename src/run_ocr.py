
#!/usr/bin/env python3
"""
drun_ocr.py— "Make it awesome" edition

What this does better than direct_ocr.py:
1) **Smarter keyframe extraction** (captures distinct screens more reliably)
   - Hybrid change detector combining SSIM, perceptual hash (pHash), and HSV histogram distance.
   - Ignores tiny changes (e.g., a blinking cursor) with a "changed area" threshold.
   - Guarantees a keyframe at least every `--max-interval-seconds` so slow scrolls aren't missed.

2) **No duplicate code in OCR output**
   - Multi-stage de-duplication:
     a) Line-number aware merge (if numbers detected).
     b) Fuzzy section de-duplication (difflib) to avoid repeated blocks between frames.
     c) Optional unique-lines pass to eliminate identical code lines across frames.
     d) Optional n-gram de-dup for near duplicates even when line numbers are missing.

3) **Drop-in**: Reuses your existing OCR stack from `direct_ocr.py` (models, prompts, configs).
   - We dynamically import `direct_ocr.py` to call: `load_api_config` and `extract_text_from_image`.

Usage examples:
    python drun_ocr.py--video input/foo.mov --model gemini-2.0-flash --output-type md
    python drun_ocr.py--input_dir input --output_dir output --max-interval-seconds 4

Requirements: opencv-python, numpy, scikit-image, Pillow, json5, requests, psutil (same as your project).
"""

import os
import sys
import time
import math
import json
import logging
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import cv2
from PIL import Image
from io import BytesIO

# Change detectors
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize as sk_resize

# We'll import your existing module dynamically
from importlib.machinery import SourceFileLoader

# MLX vision-language model support
try:
    from mlx_vlm import load, generate  # type: ignore
    USING_MLX_VLM = True
except ModuleNotFoundError:
    USING_MLX_VLM = False

try:
    from huggingface_hub import snapshot_download
    from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE as _HF_CACHE
except Exception:
    _HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"

HUGGINGFACE_HUB_CACHE = Path(_HF_CACHE).expanduser()
DEFAULT_QWEN_REPO_ID = "mlx-community/Qwen3-VL-30B-A3B-Instruct-bf16"
DEFAULT_SYSTEM_PROMPT = (
    "You are a meticulous document analysis assistant. You extract and reproduce all visible "
    "content from images with perfect fidelity. Output raw Markdown directly — never wrap "
    "your output in ```markdown code fences."
)
DEFAULT_USER_PROMPT = """\
Analyze this image and extract all content into Markdown, following these rules:

1. **Text**: Reproduce all visible text exactly, preserving hierarchy with Markdown headings.
2. **Tables**: Render every table as a proper Markdown table with aligned columns. Preserve all cell values, headers, and merged cells.
3. **Diagrams/Flowcharts**: Convert each diagram into a Mermaid.js code block (```mermaid). Use the appropriate Mermaid diagram type (flowchart, sequenceDiagram, stateDiagram-v2, classDiagram, etc.). IMPORTANT: Always quote node labels that contain parentheses or special characters using double quotes, e.g. D["Human approval (ASR reviewer)"] not D[Human approval (ASR reviewer)]. Then provide a detailed written explanation of the diagram below the code block.
4. **Charts/Graphs**: Describe the chart type, axes, data series, and key data points in detail.
5. **Code**: Reproduce any visible code in fenced code blocks with the correct language tag.
6. **Images/Icons**: Briefly describe any non-text visual elements in italics.

IMPORTANT:
- Output raw Markdown directly. Do NOT wrap output in ```markdown fences.
- Ignore UI chrome: toolbars, buttons, navigation menus, input placeholders, formatting icons, and browser elements are NOT content — skip them entirely.
- If content is partially visible (cut off at the edge of the screen), extract only what is fully readable. Do not guess or hallucinate missing content.
- Do not repeat content that appears identical or near-identical to content you already output.\
"""

# ---------- Logging ----------
logger = logging.getLogger("direct_ocr_awesome")
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%H:%M:%S"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)


# ---------- Helpers ----------

def _to_gray_small(bgr: np.ndarray, long_side: int = 640) -> np.ndarray:
    """Convert to grayscale and shrink to speed up comparisons."""
    h, w = bgr.shape[:2]
    if max(h, w) > long_side:
        if h >= w:
            new_h = long_side
            new_w = int(round(w * (long_side / h)))
        else:
            new_w = long_side
            new_h = int(round(h * (long_side / w)))
        bgr_small = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        bgr_small = bgr
    gray = cv2.cvtColor(bgr_small, cv2.COLOR_BGR2GRAY)
    # Mild blur reduces sensitivity to flicker / cursor blinking
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return gray


def _phash(gray: np.ndarray) -> np.ndarray:
    """Compute a simple perceptual hash (pHash) via DCT (64-bit)."""
    # resize to 32x32, DCT, take top-left 8x8 (skip DC), compare to median
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    small = np.float32(small)
    dct = cv2.dct(small)  # 32x32
    dct_low = dct[:8, :8]  # 8x8
    median = np.median(dct_low[1:, 1:])  # ignore DC
    bits = (dct_low > median).astype(np.uint8)
    return bits  # 8x8 bit array


def _phash_distance(bits_a: np.ndarray, bits_b: np.ndarray) -> int:
    """Hamming distance between two 8x8 bit arrays."""
    x = bits_a.flatten() ^ bits_b.flatten()
    # When dtype is uint8 0/1, xor yields 0/1 difference; sum is distance
    return int(np.sum(x))


def _hsv_hist_corr(bgr_a: np.ndarray, bgr_b: np.ndarray, bins=(24, 32)) -> float:
    """Return correlation of HSV histograms in [0..1]. Lower => more different."""
    def hist(bgr):
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv], [0, 1], None, bins, [0, 180, 0, 256])
        h = cv2.normalize(h, h).flatten()
        return h
    h1, h2 = hist(bgr_a), hist(bgr_b)
    # correlation in [-1..1]; rescale to [0..1]
    corr = cv2.compareHist(h1.astype('float32'), h2.astype('float32'), cv2.HISTCMP_CORREL)
    return float(max(0.0, min(1.0, (corr + 1.0) / 2.0)))


def _ssim_and_changed_area(gray_a: np.ndarray, gray_b: np.ndarray, area_thresh: float) -> Tuple[float, float]:
    """Compute SSIM and fraction of pixels that changed more than a small epsilon."""
    s, diff = ssim(gray_a, gray_b, full=True)
    # diff is similarity map in [0..1]; convert to "difference" (1 - ssimmap)
    delta = 1.0 - diff
    # threshold small changes
    mask = (delta > 0.15).astype(np.uint8)
    changed_fraction = float(mask.mean())
    return float(s), changed_fraction


# ---------- Keyframe Extraction (Hybrid) ----------

def extract_keyframes_hybrid(
    video_path: str,
    ssim_threshold: float = 0.97,
    phash_threshold: int = 8,
    hist_corr_threshold: float = 0.90,
    min_interval_seconds: float = 0.5,
    max_interval_seconds: float = 4.0,
    min_changed_area: float = 0.02,
) -> List[Tuple[float, np.ndarray]]:
    """
    Return a list of (timestamp_seconds, frame_bgr) keyframes.
    We declare a new keyframe if ANY of these holds:
      - SSIM < ssim_threshold **AND** changed area >= min_changed_area
      - pHash distance >= phash_threshold
      - HSV histogram correlation <= hist_corr_threshold
    We always emit at least one frame every `max_interval_seconds`.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video: {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total_frames / max(fps, 1e-6) if total_frames else 0

    logger.info(f"Scanning {video_path} ({duration:.1f}s, {fps:.1f} fps) for keyframes…")

    keyframes: List[Tuple[float, np.ndarray]] = []
    last_emit_ts = -1e9
    last_gray: Optional[np.ndarray] = None
    last_bgr_small: Optional[np.ndarray] = None
    last_ph: Optional[np.ndarray] = None

    frame_idx = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        ts = frame_idx / max(fps, 1e-6)  # seconds

        gray = _to_gray_small(frame_bgr)
        should_emit = False

        # Always emit first frame
        if last_gray is None:
            should_emit = True
        else:
            # SSIM + changed area
            try:
                ssim_val, changed_fraction = _ssim_and_changed_area(last_gray, gray, min_changed_area)
            except Exception:
                ssim_val, changed_fraction = 1.0, 0.0

            # pHash
            try:
                ph = _phash(gray)
                ph_dist = _phash_distance(ph, last_ph) if last_ph is not None else 0
            except Exception:
                ph, ph_dist = None, 0

            # Histogram correlation
            try:
                corr = _hsv_hist_corr(last_bgr_small if last_bgr_small is not None else frame_bgr, frame_bgr)
            except Exception:
                corr = 1.0

            # Decision
            big_change = (ssim_val < ssim_threshold and changed_fraction >= min_changed_area)
            phash_change = (ph is not None and last_ph is not None and ph_dist >= phash_threshold)
            hist_change  = (corr <= hist_corr_threshold)

            time_gate = (ts - last_emit_ts) >= min_interval_seconds
            force_periodic = (ts - last_emit_ts) >= max_interval_seconds

            if (time_gate and (big_change or phash_change or hist_change)) or force_periodic:
                should_emit = True

        if should_emit:
            keyframes.append((ts, frame_bgr.copy()))
            last_emit_ts = ts
            last_gray = gray
            last_ph = _phash(gray)
            # keep a smaller BGR to reduce histogram cost
            h, w = frame_bgr.shape[:2]
            scale = 640.0 / max(h, w)
            if scale < 1.0:
                last_bgr_small = cv2.resize(frame_bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
            else:
                last_bgr_small = frame_bgr

        frame_idx += 1

    cap.release()
    logger.info(f"Keyframes selected: {len(keyframes)}")
    return keyframes


# ---------- Local QWEN Model Support ----------

def resolve_model_path(model_arg: Optional[str], repo_id: str) -> Path:
    """Resolve the path to the local QWEN model."""
    if model_arg:
        return Path(model_arg).expanduser()
    
    safe_repo = repo_id.replace("/", "--")
    default_hf_cache = HUGGINGFACE_HUB_CACHE / f"models--{safe_repo}"
    return default_hf_cache


def ensure_model_available(model_path: Path, repo_id: str) -> Path:
    """Ensure model is downloaded and available."""
    REQUIRED_FILES = ("params.json", "config.json")
    
    if any((model_path / candidate).exists() for candidate in REQUIRED_FILES):
        return model_path
    
    logger.info(f"Model files not found at {model_path}. Downloading '{repo_id}'...")
    try:
        model_path.mkdir(parents=True, exist_ok=True)
    except PermissionError as exc:
        raise SystemExit(f"Cannot create model directory {model_path}: {exc}") from exc
    
    try:
        snapshot_download(repo_id, local_dir=str(model_path))
    except Exception as exc:
        raise SystemExit(f"Failed to download model '{repo_id}' to {model_path}: {exc}") from exc
    
    if not any((model_path / candidate).exists() for candidate in REQUIRED_FILES):
        raise SystemExit(f"Download finished but required files not found in {model_path}")
    
    return model_path


def build_ocr_prompt(tokenizer, image_path: str, system_prompt: str, user_prompt: str) -> str:
    """Build a chat prompt for OCR with the QWEN model."""
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def ocr_with_qwen(
    model,
    tokenizer,
    frame_bgr: np.ndarray,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    user_prompt: str = DEFAULT_USER_PROMPT,
    max_tokens: int = 2048,
    temperature: float = 0.0,
) -> str:
    """Extract text from image using local QWEN model."""
    # Convert BGR to RGB and save temporarily
    import tempfile
    pil_image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
        pil_image.save(tmp_path)
    
    try:
        prompt = build_ocr_prompt(tokenizer, tmp_path, system_prompt, user_prompt)
        kwargs = {
            "max_tokens": max_tokens,
            "image": tmp_path,
            "temperature": temperature,
        }
        
        result = generate(model, tokenizer, prompt, **kwargs)
        
        # Extract text from result
        if isinstance(result, str):
            return result
        elif hasattr(result, 'text'):
            return result.text
        else:
            return str(result)
    finally:
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except:
            pass


# ---------- OCR + De-duplication ----------

def parse_line_numbers(text: str) -> List[Tuple[Optional[int], str]]:
    """
    Return list of (line_number_or_None, content).
    Matches patterns like: '123:', '123 |', '  123  code', '1\tcode'.
    """
    import re
    results = []
    for raw in text.splitlines():
        m = re.match(r'^\s*(\d+)\s*[:|]\s*(.*)$', raw) or \
            re.match(r'^\s*(\d+)\s{2,}(.*)$', raw) or \
            re.match(r'^\s*(\d+)\s*\t\s*(.*)$', raw)
        if m:
            num = int(m.group(1))
            content = m.group(2)
            results.append((num, content))
        else:
            results.append((None, raw))
    return results


def dedupe_by_line_numbers(new_text: str, seen_ranges: List[Tuple[int, int]]) -> Tuple[str, List[Tuple[int, int]]]:
    """
    Keep only lines with line numbers we haven't seen yet.
    `seen_ranges` is a list of inclusive ranges (start, end).
    """
    parsed = parse_line_numbers(new_text)
    new_ranges = []
    kept_lines = []
    # Build a set of seen line numbers
    seen = set()
    for s, e in seen_ranges:
        seen.update(range(s, e+1))

    import re
    # Track contiguous number ranges we add
    current_start = current_end = None
    for num, content in parsed:
        if num is not None:
            if num not in seen:
                kept_lines.append(content)
                seen.add(num)
                if current_start is None:
                    current_start = current_end = num
                elif num == current_end + 1:
                    current_end = num
                else:
                    new_ranges.append((current_start, current_end))
                    current_start = current_end = num
        else:
            kept_lines.append(content)

    if current_start is not None:
        new_ranges.append((current_start, current_end))

    return ("\n".join(kept_lines).strip(), new_ranges)


def normalize_for_hash(text: str, strip_line_numbers: bool = True) -> str:
    """Normalize text to compare across frames."""
    # strip Windows/Mac line endings
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    # collapse trailing/leading blanks per line
    t = "\n".join(line.rstrip() for line in t.split("\n"))
    if strip_line_numbers:
        # remove leading numbers + separators
        import re
        t = "\n".join(re.sub(r'^\s*\d+\s*[:|]?\s*', '', ln) for ln in t.splitlines())
    return t.strip()


def fuzzy_section_is_dup(section: str, prior_sections: List[str], threshold: float) -> bool:
    """Fuzzy test using difflib to decide if a section was already captured."""
    import difflib
    norm = section.strip()
    if not norm:
        return True
    for prev in prior_sections:
        if not prev.strip():
            continue
        s = difflib.SequenceMatcher(None, norm, prev.strip()).ratio()
        if s >= threshold:
            return True
    return False


def ngram_signature(lines: List[str], n: int = 3) -> set:
    """Return a set of n-gram tuples for approximate duplicate detection."""
    sig = set()
    for i in range(0, max(0, len(lines) - n + 1)):
        sig.add(tuple(lines[i:i+n]))
    return sig


def dedupe_sections(
    text: str,
    collected_sections: List[str],
    strategy: str = "auto",
    similarity: float = 0.95,
    unique_lines: bool = False,
    min_line_len: int = 2,
    ngram_n: int = 3
) -> List[str]:
    """
    Deduplicate incoming text against previously collected sections.
    Returns a list of **new** sections to add.
    Strategies:
      - auto: try line-number aware first (if detected), then fuzzy
      - exact: exact string match only
      - fuzzy: difflib ratio >= similarity => duplicate
      - lines: enforce unique lines across frames
      - ngram: near-duplicate detection via line n-grams
    """
    new_sections: List[str] = []

    # If text is large, split into logical sections (blank lines delimiters)
    sections = [blk for blk in text.split('\n\n') if blk.strip()]

    # Quick exact de-dup
    if strategy == "exact":
        for s in sections:
            if s not in collected_sections:
                new_sections.append(s)
        return new_sections

    # Line-number aware path first
    def has_line_numbers(s: str) -> bool:
        import re
        return bool(re.search(r'^\s*\d+\s*[:|\t]|\s{2,}\d+\s', s, flags=re.MULTILINE))

    if strategy in ("auto", "lines"):
        # If any section has numbered lines, use the specialized merge
        if any(has_line_numbers(s) for s in sections):
            # Build a union of already seen ranges by scanning collected sections
            seen_ranges: List[Tuple[int, int]] = []
            for prev in collected_sections:
                # Extract ranges from prior text
                parsed = parse_line_numbers(prev)
                nums = [n for n, _ in parsed if n is not None]
                if nums:
                    nums.sort()
                    # compress to ranges
                    start = end = nums[0]
                    for n in nums[1:]:
                        if n == end + 1:
                            end = n
                        else:
                            seen_ranges.append((start, end))
                            start = end = n
                    seen_ranges.append((start, end))

            # Perform dedupe for the new text
            merged, new_ranges = dedupe_by_line_numbers("\n\n".join(sections), seen_ranges)
            if merged.strip():
                sections = [merged]

    # Fuzzy / ngram stages
    if strategy in ("auto", "fuzzy"):
        for s in sections:
            if not fuzzy_section_is_dup(s, collected_sections, similarity):
                new_sections.append(s)

    elif strategy == "ngram":
        # Build a signature of existing content
        existing_lines = []
        for prev in collected_sections:
            existing_lines.extend([ln for ln in prev.splitlines() if ln.strip()])
        existing_sig = ngram_signature(existing_lines, n=ngram_n)

        for s in sections:
            lines = [ln for ln in s.splitlines() if ln.strip()]
            sig = ngram_signature(lines, n=ngram_n)
            jaccard = 0.0
            if sig or existing_sig:
                inter = len(sig & existing_sig)
                union = len(sig | existing_sig)
                jaccard = inter / union if union else 0.0
            # If overlap is below threshold => new content
            if jaccard < (1.0 - (1.0 - similarity) * 2):  # map similarity roughly to n-gram
                new_sections.append(s)

    # Optional unique-lines pass
    if unique_lines and new_sections:
        seen_lines = set()
        for prev in collected_sections:
            for ln in prev.splitlines():
                key = ln.strip()
                if len(key) >= min_line_len:
                    seen_lines.add(key)

        filtered = []
        for s in new_sections:
            kept = []
            for ln in s.splitlines():
                key = ln.strip()
                if len(key) < min_line_len:
                    continue
                if key in seen_lines:
                    continue
                kept.append(ln)
                seen_lines.add(key)
            text_kept = "\n".join(kept).strip()
            if text_kept:
                filtered.append(text_kept)
        new_sections = filtered

    return new_sections


# ---------- Main pipeline ----------

def process_video(
    video_path: str,
    output_path: str,
    model_name: str = "gpt-5-mini",
    output_type: str = "txt",
    private: bool = False,
    use_local_qwen: bool = False,
    qwen_model_path: Optional[str] = None,
    qwen_repo_id: str = DEFAULT_QWEN_REPO_ID,
    qwen_max_tokens: int = 2048,
    qwen_temperature: float = 0.0,
    ssim_threshold: float = 0.97,
    phash_threshold: int = 8,
    hist_corr_threshold: float = 0.90,
    min_interval_seconds: float = 0.5,
    max_interval_seconds: float = 4.0,
    min_changed_area: float = 0.02,
    dedupe_strategy: str = "auto",
    similarity: float = 0.95,
    unique_lines: bool = True,
    ngram_n: int = 3,
) -> str:
    """
    Improved pipeline that supports API-based OCR and local QWEN model.
    """
    # Setup OCR method
    qwen_model = None
    qwen_tokenizer = None
    original = None
    api_cfg = None
    
    if use_local_qwen:
        if not USING_MLX_VLM:
            raise RuntimeError(
                "Local QWEN model requires mlx-vlm package. "
                "Install it with: pip install mlx-vlm"
            )
        
        logger.info(f"Loading local QWEN model from {qwen_repo_id}...")
        model_path = resolve_model_path(qwen_model_path, qwen_repo_id)
        model_path = ensure_model_available(model_path, qwen_repo_id)
        
        loaded = load(str(model_path), lazy=True)
        if not isinstance(loaded, tuple):
            raise SystemExit("Unexpected response from mlx_vlm.load: expected tuple")
        qwen_model, qwen_tokenizer = loaded[:2]
        logger.info("QWEN model loaded successfully")
    else:
        # Try to import the original module for API-based OCR
        try:
            original = SourceFileLoader("direct_ocr", str(Path(__file__).with_name("direct_ocr.py"))).load_module()
            api_cfg = original.load_api_config(model_name)
            if not api_cfg:
                raise RuntimeError(f"Could not load API config for model: {model_name}")
        except Exception as e:
            raise RuntimeError(f"Could not load API config: {e}. Use --use-local-qwen for local inference.")

    keyframes = extract_keyframes_hybrid(
        video_path,
        ssim_threshold=ssim_threshold,
        phash_threshold=phash_threshold,
        hist_corr_threshold=hist_corr_threshold,
        min_interval_seconds=min_interval_seconds,
        max_interval_seconds=max_interval_seconds,
        min_changed_area=min_changed_area,
    )

    sections: List[str] = []
    t0 = time.time()

    for idx, (ts, frame_bgr) in enumerate(keyframes, 1):
        t1 = time.time()
        
        # Choose OCR method
        if use_local_qwen:
            text = ocr_with_qwen(
                qwen_model,
                qwen_tokenizer,
                frame_bgr,
                system_prompt=DEFAULT_SYSTEM_PROMPT,
                user_prompt=DEFAULT_USER_PROMPT,
                max_tokens=qwen_max_tokens,
                temperature=qwen_temperature,
            )
        else:
            text = original.extract_text_from_image(frame_bgr, api_cfg, output_type=output_type) if original and hasattr(original, 'extract_text_from_image') else ''
        
        dt = time.time() - t1
        logger.info(f"OCR {idx}/{len(keyframes)} at {ts:.2f}s -> {len(text)} chars in {dt:.2f}s")

        # De-duplicate before appending
        new_secs = dedupe_sections(
            text,
            sections,
            strategy=dedupe_strategy,
            similarity=similarity,
            unique_lines=unique_lines,
            ngram_n=ngram_n
        )
        if new_secs:
            sections.extend(new_secs)
        else:
            logger.info("Skipped frame: duplicate content.")

    elapsed = time.time() - t0
    logger.info(f"OCR complete in {elapsed:.1f}s — writing output…")

    # Combine sections
    if output_type == "json":
        data = [{"section": i+1, "text": s} for i, s in enumerate(sections)]
        Path(output_path).write_text(json.dumps(data, indent=2), encoding="utf-8")
    elif output_type == "md":
        out = []
        for i, s in enumerate(sections, 1):
            out.append(f"\n\n<!-- Frame Section {i} -->\n\n{s}")
        Path(output_path).write_text("".join(out).lstrip(), encoding="utf-8")
    else:  # txt
        Path(output_path).write_text("\n\n".join(sections).strip(), encoding="utf-8")

    return output_path


def find_videos(dir_path: str) -> List[str]:
    videos = []
    for root, _, files in os.walk(dir_path):
        for f in files:
            if f.lower().endswith((".mp4", ".mov", ".mkv")):
                videos.append(os.path.join(root, f))
    return videos


def main():
    p = argparse.ArgumentParser(description="Improved OCR from screen recordings (hybrid keyframes + de-dup)")
    p.add_argument("--video", "-v", help="Single video to process")
    p.add_argument("--input_dir", default="input/", help="Directory to scan for videos")
    p.add_argument("--output_dir", default="output/", help="Directory to write outputs")
    p.add_argument("--model", default="gpt-5-mini", help="Model to use (same names as your original script)")
    p.add_argument("--output-type", default="md", choices=["txt", "json", "md"])
    p.add_argument("--private", action="store_true", help="Use your script's private OCR if configured")
    # Local QWEN model options
    p.add_argument("--use-local-qwen", action="store_true", help="Use local QWEN model instead of API")
    p.add_argument("--qwen-model-path", help="Path to local QWEN model directory")
    p.add_argument("--qwen-repo-id", default=DEFAULT_QWEN_REPO_ID, help="Hugging Face repo ID for QWEN model")
    p.add_argument("--qwen-max-tokens", type=int, default=2048, help="Max tokens for QWEN generation")
    p.add_argument("--qwen-temperature", type=float, default=0.0, help="Temperature for QWEN generation")
    # Hybrid change detection
    p.add_argument("--ssim-threshold", type=float, default=0.97)
    p.add_argument("--phash-threshold", type=int, default=8)
    p.add_argument("--hist-corr-threshold", type=float, default=0.90)
    p.add_argument("--min-interval-seconds", type=float, default=0.5)
    p.add_argument("--max-interval-seconds", type=float, default=4.0)
    p.add_argument("--min-changed-area", type=float, default=0.02)
    # De-duplication
    p.add_argument("--dedupe-strategy", default="auto", choices=["auto", "exact", "fuzzy", "lines", "ngram"])
    p.add_argument("--similarity", type=float, default=0.95, help="Fuzzy similarity for duplicate detection [0..1]")
    p.add_argument("--unique-lines", action="store_true", help="Enforce globally unique lines across frames")
    p.add_argument("--ngram-n", type=int, default=3, help="n-gram size for near-duplicate detection")
    args = p.parse_args()

    # Decide which videos to process
    videos = [args.video] if args.video else find_videos(args.input_dir)
    if not videos:
        print(f"No videos found under {args.input_dir}")
        sys.exit(2)

    os.makedirs(args.output_dir, exist_ok=True)

    for vp in videos:
        rel = os.path.relpath(vp, args.input_dir) if args.video is None else os.path.basename(vp)
        stem = os.path.splitext(rel)[0]
        out_ext = args.output_type
        out_path = os.path.join(args.output_dir, f"{stem}.{out_ext}")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        logger.info(f"Processing: {vp}")
        process_video(
            vp,
            out_path,
            model_name=args.model,
            output_type=args.output_type,
            private=args.private,
            use_local_qwen=args.use_local_qwen,
            qwen_model_path=args.qwen_model_path,
            qwen_repo_id=args.qwen_repo_id,
            qwen_max_tokens=args.qwen_max_tokens,
            qwen_temperature=args.qwen_temperature,
            ssim_threshold=args.ssim_threshold,
            phash_threshold=args.phash_threshold,
            hist_corr_threshold=args.hist_corr_threshold,
            min_interval_seconds=args.min_interval_seconds,
            max_interval_seconds=args.max_interval_seconds,
            min_changed_area=args.min_changed_area,
            dedupe_strategy=args.dedupe_strategy,
            similarity=args.similarity,
            unique_lines=args.unique_lines,
            ngram_n=args.ngram_n,
        )
        print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
