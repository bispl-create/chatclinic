from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote
from uuid import uuid4

try:
    from PIL import Image
except Exception:
    Image = None

PLUGIN_DIR = Path(__file__).resolve().parent
DEFAULT_AYQ_ROOT = (PLUGIN_DIR.parents[2] / "AYQ").resolve()
DEFAULT_CONFIG_PATH = (PLUGIN_DIR / "configs" / "dino_ayq_config.py").resolve()
DEFAULT_WEIGHTS_PATH = (PLUGIN_DIR / "checkpoints" / "dino_ayq.pth").resolve()
DEFAULT_TARGET_CLASS = "Aortic_enlargement_in_CXR"
DEFAULT_PRED_SCORE_THR = 0.9
DEFAULT_MAX_DETECTIONS = 20
SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

CLASS_NAMES = [
    "Aortic enlargement in CXR",
    "Atelectasis in CXR",
    "Calcification in CXR",
    "Cardiomegaly in CXR",
    "Consolidation in CXR",
    "ILD in CXR",
    "Infiltration in CXR",
    "Lung Opacity in CXR",
    "Nodule/Mass in CXR",
    "Other lesion in CXR",
    "Pleural effusion in CXR",
    "Pleural thickening in CXR",
    "Pneumothorax in CXR",
    "Pulmonary fibrosis in CXR",
    "Brain tumor in MRI",
    "Epithelial in Pathology (H&E stain)",
    "Lymphocyte in Pathology (H&E stain)",
    "Neutrophil in Pathology (H&E stain)",
    "Macrophage in Pathology (H&E stain)",
    "Left heart ventricle in cardiac MRI",
    "Myocardium in cardiac MRI",
    "Right heart ventricle in cardiac MRI",
    "COVID-19 infection in lung CT",
    "Nodule in lung CT",
    "Neoplastic polyp in colon endoscope",
    "Polyp in colon endoscope",
    "Non-neoplastic polyp in colon endoscope",
]

HUMAN_CLASS_TO_STEM = {
    "Aortic enlargement in CXR": "Aortic_enlargement_in_CXR",
    "Atelectasis in CXR": "Atelectasis_in_CXR",
    "Calcification in CXR": "Calcification_in_CXR",
    "Cardiomegaly in CXR": "Cardiomegaly_in_CXR",
    "Consolidation in CXR": "Consolidation_in_CXR",
    "ILD in CXR": "ILD_in_CXR",
    "Infiltration in CXR": "Infiltration_in_CXR",
    "Lung Opacity in CXR": "Lung_Opacity_in_CXR",
    "Nodule/Mass in CXR": "Nodule-Mass_in_CXR",
    "Other lesion in CXR": "Other_lesion_in_CXR",
    "Pleural effusion in CXR": "Pleural_effusion_in_CXR",
    "Pleural thickening in CXR": "Pleural_thickening_in_CXR",
    "Pneumothorax in CXR": "Pneumothorax_in_CXR",
    "Pulmonary fibrosis in CXR": "Pulmonary_fibrosis_in_CXR",
    "Brain tumor in MRI": "Brain_tumor_in_MRI",
    "Epithelial in Pathology (H&E stain)": "Epithelial_in_Pathology_HandE_stain",
    "Lymphocyte in Pathology (H&E stain)": "Lymphocyte_in_Pathology_HandE_stain",
    "Neutrophil in Pathology (H&E stain)": "Neutrophil_in_Pathology_HandE_stain",
    "Macrophage in Pathology (H&E stain)": "Macrophage_in_Pathology_HandE_stain",
    "Left heart ventricle in cardiac MRI": "Left_heart_ventricle_in_cardiac_MRI",
    "Myocardium in cardiac MRI": "Myocardium_in_cardiac_MRI",
    "Right heart ventricle in cardiac MRI": "Right_heart_ventricle_in_cardiac_MRI",
    "COVID-19 infection in lung CT": "COVID-19_infection_in_lung_CT",
    "Nodule in lung CT": "Nodule_in_lung_CT",
    "Neoplastic polyp in colon endoscope": "Neoplastic_polyp_in_colon_endoscope",
    "Polyp in colon endoscope": "Polyp_in_colon_endoscope",
    "Non-neoplastic polyp in colon endoscope": "Non-neoplastic_polyp_in_colon_endoscope",
}

STEM_TO_HUMAN = {stem: human for human, stem in HUMAN_CLASS_TO_STEM.items()}


def _normalize_class_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _candidate_aliases_for_stem(stem: str) -> list[str]:
    spaced = stem.replace("_", " ").replace("-", " ")
    aliases = [
        stem,
        spaced,
        spaced.replace("hande", "h&e"),
        spaced.replace("hande", "h and e"),
    ]
    return aliases


def _build_target_class_alias_map(available_classes: list[str]) -> dict[str, str]:
    alias_map: dict[str, str] = {}
    available_set = set(available_classes)

    for stem in available_classes:
        for alias in _candidate_aliases_for_stem(stem):
            normalized = _normalize_class_key(alias)
            if normalized:
                alias_map.setdefault(normalized, stem)

    for human_name, stem in HUMAN_CLASS_TO_STEM.items():
        if stem in available_set:
            normalized = _normalize_class_key(human_name)
            if normalized:
                alias_map[normalized] = stem

    return alias_map


def _resolve_target_candidate(candidate: str, available_classes: list[str], alias_map: dict[str, str]) -> str | None:
    stripped = candidate.strip()
    if not stripped:
        return None

    normalized_available = {item.lower(): item for item in available_classes}
    lowered = stripped.lower()
    if lowered in normalized_available:
        return normalized_available[lowered]

    normalized = _normalize_class_key(stripped)
    if normalized in alias_map:
        return alias_map[normalized]

    return None


def _extract_requested_target_class(payload: dict[str, Any], available_classes: list[str]) -> str | None:
    alias_map = _build_target_class_alias_map(available_classes)

    explicit = payload.get("target_class")
    if isinstance(explicit, str):
        resolved = _resolve_target_candidate(explicit, available_classes, alias_map)
        if resolved:
            return resolved

    question = str(payload.get("question") or "")

    quoted_match = re.search(r"target[-_ ]class\s*[:=]\s*['\"]([^'\"]+)['\"]", question, flags=re.IGNORECASE)
    if quoted_match:
        resolved = _resolve_target_candidate(quoted_match.group(1), available_classes, alias_map)
        if resolved:
            return resolved

    bare_match = re.search(r"target[-_ ]class\s*[:=]\s*([A-Za-z0-9_\-()&/ ]+)", question, flags=re.IGNORECASE)
    if bare_match:
        resolved = _resolve_target_candidate(bare_match.group(1), available_classes, alias_map)
        if resolved:
            return resolved

    direct = _resolve_target_candidate(question, available_classes, alias_map)
    if direct:
        return direct

    normalized_question = _normalize_class_key(question)
    if normalized_question:
        wrapped = f" {normalized_question} "
        for normalized_alias, stem in sorted(alias_map.items(), key=lambda item: len(item[0]), reverse=True):
            if normalized_alias and f" {normalized_alias} " in wrapped:
                return stem

    return None


def _ordered_class_options(available_classes: list[str]) -> list[str]:
    available_set = set(available_classes)
    options: list[str] = []

    for class_name in CLASS_NAMES:
        stem = HUMAN_CLASS_TO_STEM.get(class_name)
        if stem and stem in available_set:
            options.append(class_name)

    for stem in available_classes:
        if stem not in STEM_TO_HUMAN:
            options.append(stem.replace("_", " "))

    if options:
        return options

    return list(CLASS_NAMES)


def _build_selection_result(
    question: str,
    available_classes: list[str],
    score_threshold: float,
) -> dict[str, Any]:
    options = _ordered_class_options(available_classes)
    option_lines = "\n".join(f"- {item}" for item in options)
    # stripped_question = question.strip()

    intro = "AYQ detection is ready."
    # if stripped_question:
    #     intro += f" I could not match `{stripped_question}` to a valid class."

    summary = (
        f"{intro}\n\n"
        "Choose a class to detect from the list below:\n\n"
        f"{option_lines}\n\n"
        "Example: `Aortic enlargement in CXR`"
    )

    return {
        "summary": summary,
        "artifacts": {
            "report_sections": {
                "exam": "AYQ detection class selection",
                "findings": ["Select one target class to proceed."],
                "impression": ["Awaiting target class selection."],
            },
        },
        "provenance": {
            "tool": "detection_ayq_tool",
            "version": "0.1.0",
            "stage": "awaiting-target-class-selection",
            "accepted_input_format": "One class name from target_class_options",
            "available_target_class_count": len(options),
        },
    }


def _iter_path_candidates(node: Any) -> Iterable[str]:
    if isinstance(node, dict):
        for key, value in node.items():
            if key in {"source_file_path", "source_path", "image_path", "img_path"} and isinstance(value, str):
                stripped = value.strip()
                if stripped:
                    yield stripped
            yield from _iter_path_candidates(value)
        return
    if isinstance(node, list):
        for item in node:
            yield from _iter_path_candidates(item)


def _dedupe_preserve_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _collect_image_candidates(payload: dict[str, Any]) -> list[Path]:
    candidates: list[str] = []
    candidates.extend(_iter_path_candidates(payload.get("active_artifact") or {}))
    candidates.extend(_iter_path_candidates(payload.get("analysis_artifacts") or {}))

    resolved: list[Path] = []
    for raw in _dedupe_preserve_order(candidates):
        path = Path(raw).expanduser()
        if not path.exists() or not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix and suffix not in SUPPORTED_IMAGE_SUFFIXES:
            continue
        resolved.append(path)
    return resolved


def _available_target_classes(ayq_root: Path) -> list[str]:
    embedding_dir = ayq_root / "text_embeddings" / "single_class_embeddings_openclip"
    if not embedding_dir.exists():
        return []
    return sorted(path.stem for path in embedding_dir.glob("*.npy"))


def _select_device(execution_context: dict[str, Any]) -> str:
    selected_accelerator = str((execution_context.get("selected_accelerator") or "")).strip().lower()
    if selected_accelerator == "cpu":
        return "cpu"
    override = str(os.getenv("AYQ_DEVICE", "")).strip()
    if override:
        return override
    return "cuda:0"


def _build_env(ayq_root: Path, device: str) -> dict[str, str]:
    env = os.environ.copy()
    existing_pythonpath = str(env.get("PYTHONPATH", "")).strip()
    pythonpath_parts = [str(ayq_root)]
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

    if device.startswith("cuda"):
        env.setdefault("CUDA_VISIBLE_DEVICES", str(os.getenv("AYQ_CUDA_VISIBLE_DEVICES", "3")))
    else:
        env.pop("CUDA_VISIBLE_DEVICES", None)
    return env


def _build_command(
    python_executable: Path,
    image_path: Path,
    config_path: Path,
    weights_path: Path,
    out_dir: Path,
    target_class: str,
    device: str,
    score_threshold: float,
) -> list[str]:
    command = [
        str(python_executable.resolve()),
        str((PLUGIN_DIR / "demo" / "image_demo.py").resolve()),
        str(image_path.resolve()),
        str(config_path.resolve()),
        "--weights",
        str(weights_path.resolve()),
        "--target-class",
        target_class,
        "--out-dir",
        str(out_dir.resolve()),
        "--pred-score-thr",
        str(score_threshold),
        "--device",
        device,
    ]
    return command


def _read_latest_prediction(pred_dir: Path) -> tuple[Path, dict[str, Any]]:
    if not pred_dir.exists():
        raise RuntimeError(f"Prediction directory was not created: {pred_dir}")
    prediction_files = sorted(pred_dir.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not prediction_files:
        raise RuntimeError(f"No prediction JSON was generated in: {pred_dir}")
    prediction_path = prediction_files[0]
    prediction_payload = json.loads(prediction_path.read_text(encoding="utf-8"))
    return prediction_path, prediction_payload


def _read_latest_visualization(vis_dir: Path) -> Path | None:
    if not vis_dir.exists():
        return None
    candidates = sorted(
        [path for path in vis_dir.iterdir() if path.is_file()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return None
    return candidates[0]


def _to_data_url(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    mime_type, _ = mimetypes.guess_type(str(path))
    if not mime_type:
        mime_type = "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _parse_int_env(name: str, default: int, min_value: int, max_value: int) -> int:
    raw = str(os.getenv(name, str(default))).strip()
    try:
        value = int(raw)
    except Exception:
        return default
    if value < min_value:
        return min_value
    if value > max_value:
        return max_value
    return value


def _build_resized_preview_image(visualization_path: Path | None) -> Path | None:
    if visualization_path is None or not visualization_path.exists() or not visualization_path.is_file():
        return visualization_path
    if Image is None:
        return visualization_path

    max_width = _parse_int_env("AYQ_PREVIEW_MAX_WIDTH", 420, 64, 4096)
    jpeg_quality = _parse_int_env("AYQ_PREVIEW_JPEG_QUALITY", 80, 40, 100)

    try:
        with Image.open(visualization_path) as image:
            width, _height = image.size
            if width <= max_width:
                return visualization_path

            preview = image.copy()
            preview.thumbnail((max_width, max_width * 16))

            if preview.mode not in {"RGB", "L"}:
                preview = preview.convert("RGB")

            preview_path = visualization_path.with_name(f"{visualization_path.stem}_preview.jpg")
            preview.save(preview_path, format="JPEG", quality=jpeg_quality, optimize=True)
            return preview_path
    except Exception:
        return visualization_path


def _parse_preview_port() -> int:
    raw = str(os.getenv("AYQ_PREVIEW_SERVER_PORT", "8899")).strip()
    try:
        port = int(raw)
    except Exception:
        return 8899
    if 1 <= port <= 65535:
        return port
    return 8899


def _ensure_preview_server(served_dir: Path) -> tuple[str, int, bool]:
    host = str(os.getenv("AYQ_PREVIEW_SERVER_HOST", "127.0.0.1")).strip() or "127.0.0.1"
    port = _parse_preview_port()

    command = [
        str(Path(sys.executable).resolve()),
        "-m",
        "http.server",
        str(port),
        "--bind",
        host,
        "--directory",
        str(served_dir.resolve()),
    ]

    try:
        subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            cwd=str(served_dir.resolve()),
        )
    except Exception:
        # Even if spawn fails (e.g. server already running), still return URL.
        # The frontend can load from an already-running preview server.
        return host, port, True

    return host, port, True


def _build_preview_http_url(visualization_path: Path | None, served_dir: Path) -> str | None:
    if visualization_path is None or not visualization_path.exists():
        return None

    try:
        relative = visualization_path.resolve().relative_to(served_dir.resolve())
    except Exception:
        return None

    host, port, ready = _ensure_preview_server(served_dir)
    if not ready:
        return None

    relative_url = "/".join(quote(part) for part in relative.parts)
    public_base = str(os.getenv("AYQ_PREVIEW_PUBLIC_BASE_URL", "")).strip()
    if public_base:
        return f"{public_base.rstrip('/')}/{relative_url}"
    return f"http://{host}:{port}/{relative_url}"


def _label_name(label_id: int) -> str:
    if 0 <= label_id < len(CLASS_NAMES):
        return CLASS_NAMES[label_id]
    return f"class_{label_id}"


def _extract_detection_candidates(
    prediction_payload: dict[str, Any], score_threshold: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    labels = list(prediction_payload.get("labels") or [])
    scores = list(prediction_payload.get("scores") or [])
    bboxes = list(prediction_payload.get("bboxes") or [])

    total = min(len(labels), len(scores), len(bboxes))
    all_candidates: list[dict[str, Any]] = []
    for index in range(total):
        try:
            label_id = int(labels[index])
            confidence = float(scores[index])
            bbox_values = [float(value) for value in list(bboxes[index])[:4]]
        except Exception:
            continue
        all_candidates.append(
            {
                "label": _label_name(label_id),
                "class_id": label_id,
                "confidence": round(confidence, 6),
                "bbox": [round(value, 3) for value in bbox_values],
            }
        )

    all_candidates.sort(key=lambda item: float(item.get("confidence", 0.0)), reverse=True)
    high_confidence = [item for item in all_candidates if float(item.get("confidence", 0.0)) >= score_threshold]
    # Strict threshold behavior:
    # only keep candidates at or above the threshold.
    selected = high_confidence[:DEFAULT_MAX_DETECTIONS]
    return all_candidates, high_confidence, selected


def _format_summary(
    image_path: Path,
    target_class: str,
    device: str,
    score_threshold: float,
    high_confidence: list[dict[str, Any]],
    visualization_url: str | None = None,
) -> str:
    image_markdown = ""
    if visualization_url:
        image_markdown = (
            f"\n\nDetection preview:\n\n"
            f"![AYQ detection result]({visualization_url})\n\n"
            # f"Direct link: {visualization_url}"
        )

    if high_confidence:
        top = high_confidence[0]
        return (
            f"AYQ detection completed on `{image_path.name}` for target class `{target_class}`. "
            f"Found {len(high_confidence)} high-confidence candidate(s); "
            f"top candidate is **{top['label']}** at confidence **{top['confidence']:.3f}**. "
            f"Inference device: `{device}`."
            f"{image_markdown}"
        )
    return (
        f"AYQ detection completed on `{image_path.name}` for target class `{target_class}`, "
        f"but no candidate was found. Inference device: `{device}`."
        f"{image_markdown}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    execution_context = dict(payload.get("execution_context") or {})

    ayq_root = Path(os.getenv("AYQ_ROOT", str(DEFAULT_AYQ_ROOT))).expanduser().resolve()
    if not ayq_root.exists():
        raise RuntimeError(
            f"AYQ root was not found at `{ayq_root}`. Set AYQ_ROOT to the AYQ repository path before running this tool."
        )

    config_path = DEFAULT_CONFIG_PATH
    weights_path = DEFAULT_WEIGHTS_PATH
    python_executable = Path(os.getenv("AYQ_PYTHON_EXECUTABLE", sys.executable)).expanduser().resolve()
    if not config_path.exists():
        raise RuntimeError(f"Missing config file: {config_path}")
    if not weights_path.exists():
        raise RuntimeError(f"Missing checkpoint file: {weights_path}")
    if not python_executable.exists():
        raise RuntimeError(
            f"AYQ python executable was not found: {python_executable}. "
            "Set AYQ_PYTHON_EXECUTABLE to a valid Python interpreter with torch/mmcv/mmengine installed."
        )

    image_candidates = _collect_image_candidates(payload)
    if not image_candidates:
        raise RuntimeError(
            "detection_ayq_tool could not locate an uploaded image path from analysis artifacts. "
            "Please upload at least one raster image or DICOM source first."
        )
    image_path = image_candidates[0]

    score_threshold = float(os.getenv("AYQ_PRED_SCORE_THR", str(DEFAULT_PRED_SCORE_THR)))
    available_classes = _available_target_classes(ayq_root)
    target_class = _extract_requested_target_class(payload, available_classes)

    if not target_class:
        selection_result = _build_selection_result(
            question=str(payload.get("question") or ""),
            available_classes=available_classes,
            score_threshold=score_threshold,
        )
        Path(args.output).write_text(json.dumps(selection_result, ensure_ascii=False, indent=2), encoding="utf-8")
        return 0

    run_id = uuid4().hex[:12]
    run_dir = (PLUGIN_DIR / "runtime_outputs" / run_id).resolve()
    out_dir = run_dir / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = _select_device(execution_context)
    env = _build_env(ayq_root, device)
    command = _build_command(
        python_executable=python_executable,
        image_path=image_path,
        config_path=config_path,
        weights_path=weights_path,
        out_dir=out_dir,
        target_class=target_class,
        device=device,
        score_threshold=score_threshold,
    )

    completed = subprocess.run(
        command,
        cwd=str(ayq_root),
        capture_output=True,
        text=True,
        env=env,
        timeout=int(os.getenv("AYQ_INFERENCE_TIMEOUT_SEC", "900")),
        check=False,
    )

    used_device = device
    used_cuda_visible_devices = env.get("CUDA_VISIBLE_DEVICES")

    # Prefer matching AYQ inference flow (GPU), then fallback to CPU when unavailable.
    if completed.returncode != 0 and device != "cpu":
        cpu_device = "cpu"
        cpu_env = _build_env(ayq_root, cpu_device)
        cpu_command = _build_command(
            python_executable=python_executable,
            image_path=image_path,
            config_path=config_path,
            weights_path=weights_path,
            out_dir=out_dir,
            target_class=target_class,
            device=cpu_device,
            score_threshold=score_threshold,
        )
        retry = subprocess.run(
            cpu_command,
            cwd=str(ayq_root),
            capture_output=True,
            text=True,
            env=cpu_env,
            timeout=int(os.getenv("AYQ_INFERENCE_TIMEOUT_SEC", "900")),
            check=False,
        )
        if retry.returncode == 0:
            completed = retry
            command = cpu_command
            used_device = cpu_device
            used_cuda_visible_devices = cpu_env.get("CUDA_VISIBLE_DEVICES")

    if completed.returncode != 0:
        raise RuntimeError(
            "AYQ inference command failed.\n"
            f"command={' '.join(command)}\n"
            f"returncode={completed.returncode}\n"
            f"stdout={completed.stdout}\n"
            f"stderr={completed.stderr}"
        )

    prediction_path, prediction_payload = _read_latest_prediction(out_dir / "preds")
    visualization_path = _read_latest_visualization(out_dir / "vis")
    preview_visualization_path = _build_resized_preview_image(visualization_path)
    visualization_url = _build_preview_http_url(preview_visualization_path, PLUGIN_DIR)
    visualization_data_url = _to_data_url(preview_visualization_path) if preview_visualization_path else None

    all_candidates, high_confidence, selected = _extract_detection_candidates(prediction_payload, score_threshold)
    summary = _format_summary(
        image_path=image_path,
        target_class=target_class,
        device=used_device,
        score_threshold=score_threshold,
        high_confidence=high_confidence,
        visualization_url=visualization_url,
    )

    findings = [
        f"{item['label']} (confidence {float(item['confidence']):.3f}) at bbox {item['bbox']}"
        for item in selected[:10]
    ]
    if not findings:
        findings = [f"No detection candidates met threshold {score_threshold:.2f}."]

    if high_confidence:
        impression = [
            f"Detected {len(high_confidence)} candidate(s) at or above threshold {score_threshold:.2f} for target class `{target_class}`."
        ]
    else:
        impression = [
            f"No candidate met threshold {score_threshold:.2f}."
        ]

    result = {
        "summary": summary,
        "artifacts": {
            "report_sections": {
                "exam": f"AYQ detection inference on {image_path.name}",
                "findings": findings,
                "impression": impression,
            },
            "detections": selected,
        },
        "provenance": {
            "tool": "detection_ayq_tool",
            "version": "0.1.0",
            "target_class": target_class,
            "device": used_device,
            "python_executable": str(python_executable),
        },
    }

    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
