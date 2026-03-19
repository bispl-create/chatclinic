from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

from app.main import _summarize_fhir_json, _summarize_fhir_ndjson_group, _summarize_fhir_xml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    execution_context = payload.get("execution_context") or {}
    files = payload.get("files") or []
    if not files:
        raise ValueError("fhir_browser_tool requires one or more files")

    normalized = []
    for item in files:
        normalized.append(
            (
                str(item.get("file_name", "uploaded-file")),
                base64.b64decode(item["raw_base64"]),
                str(item.get("suffix", "json")),
            )
        )

    if len(normalized) > 1 or normalized[0][2] == "ndjson":
        response = _summarize_fhir_ndjson_group(normalized)
    else:
        file_name, raw, suffix = normalized[0]
        if suffix == "xml":
            response = _summarize_fhir_xml(file_name, raw, suffix)
        else:
            response = _summarize_fhir_json(file_name, raw, suffix)

    result = response.model_dump()
    result["used_tools"] = ["fhir_browser_tool"]
    result["execution_context"] = execution_context
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
