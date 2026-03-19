from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path
from typing import Any


def _open_image(raw: bytes) -> tuple[Any, str]:
    """Return (PIL.Image, format_str). Raises ImportError if Pillow not available."""
    from PIL import Image  # type: ignore[import]

    img = Image.open(io.BytesIO(raw))
    img.load()  # fully decode
    return img, (img.format or "PNG")


def _thumbnail_data_url(img: Any) -> str:
    img = img.copy()
    img.thumbnail((512, 512))
    # Convert palette / RGBA with transparency for safe PNG encode
    if img.mode not in {"RGB", "L", "RGBA"}:
        img = img.convert("RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _process_file(file_name: str, raw: bytes, suffix: str, source_path: str | None) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "width": "not available",
        "height": "not available",
        "mode": "not available",
        "format": suffix.upper(),
        "file_size_bytes": len(raw),
        "source_file_path": source_path,
    }
    preview: dict[str, Any] = {"available": False, "image_data_url": None}

    try:
        img, fmt = _open_image(raw)
        meta["width"] = img.width
        meta["height"] = img.height
        meta["mode"] = img.mode
        meta["format"] = fmt

        # EXIF (JPEG)
        exif_summary: dict[str, str] = {}
        try:
            exif_data = img._getexif()  # type: ignore[attr-defined]
            if exif_data:
                from PIL.ExifTags import TAGS  # type: ignore[import]

                for tag_id, value in exif_data.items():
                    tag = TAGS.get(tag_id, str(tag_id))
                    if tag in {"Make", "Model", "DateTime", "ImageDescription", "Software"}:
                        exif_summary[tag] = str(value)
        except Exception:
            pass
        if exif_summary:
            meta["exif"] = exif_summary

        preview = {
            "available": True,
            "image_data_url": _thumbnail_data_url(img),
        }
    except Exception as exc:
        meta["parse_error"] = str(exc)

    return {
        "file_name": file_name,
        "suffix": suffix,
        "source_file_path": source_path,
        "meta": meta,
        "preview": preview,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    files = payload.get("files") or []
    if not files:
        raise ValueError("png_intake_tool requires one or more files")

    processed = []
    for item in files:
        raw = base64.b64decode(item["raw_base64"])
        processed.append(
            _process_file(
                file_name=str(item.get("file_name", "uploaded-file")),
                raw=raw,
                suffix=str(item.get("suffix", "png")),
                source_path=item.get("source_path"),
            )
        )

    # Build aggregate summary
    if len(processed) == 1:
        p = processed[0]
        m = p["meta"]
        summary = (
            f"Raster image '{p['file_name']}' received. "
            f"Format: {m['format']}, size: {m.get('width', '?')} x {m.get('height', '?')} px, "
            f"color mode: {m['mode']}. File size: {m['file_size_bytes']:,} bytes."
        )
        file_label = p["file_name"]
        file_type = p["suffix"]
        total_bytes = m["file_size_bytes"]
    else:
        total_bytes = sum(p["meta"]["file_size_bytes"] for p in processed)
        summary = (
            f"{len(processed)} raster image(s) received. "
            f"Total size: {total_bytes:,} bytes. "
            f"Formats: {', '.join(sorted({p['meta']['format'] for p in processed}))}."
        )
        file_label = f"{len(processed)} images"
        file_type = processed[0]["suffix"] if processed else "png"

    result = {
        "source": {
            "file_name": file_label,
            "file_type": file_type,
            "modality": "medical-image",
            "size_bytes": total_bytes,
            "status": "parsed",
        },
        "grounded_summary": summary,
        "studio_cards": [
            {"id": "metadata", "title": "Image Review", "subtitle": "Metadata and preview"}
        ],
        "artifacts": {
            "metadata": {
                "items": [p["meta"] for p in processed],
                "source_file_paths": [p["source_file_path"] for p in processed],
            },
            "preview": {
                "items": [
                    {"file_name": p["file_name"], **p["preview"]}
                    for p in processed
                ]
            },
            "qc": {
                "file_count": len(processed),
                "total_bytes": total_bytes,
                "formats": sorted({p["meta"]["format"] for p in processed}),
            },
        },
        "sources": [],
        "used_tools": ["png_intake_tool"],
    }

    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
