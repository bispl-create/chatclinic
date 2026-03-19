from __future__ import annotations

import base64
import csv
import io
import json
import math
import os
import re
import site
import statistics
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.services.skill_orchestrator import initial_chat_prompt, suggest_tool
from app.services.tool_runner import discover_tools, run_tool

_VENV_SITE_PACKAGES = Path(__file__).resolve().parents[1] / ".venv" / "lib" / "python3.9" / "site-packages"
if _VENV_SITE_PACKAGES.exists():
    site.addsitedir(str(_VENV_SITE_PACKAGES))

try:
    import pydicom  # type: ignore
except Exception:  # pragma: no cover
    pydicom = None

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None

try:
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover
    Image = None


UPLOAD_CACHE_DIR = Path(__file__).resolve().parents[1] / "runtime_uploads"
UPLOAD_CACHE_DIR.mkdir(parents=True, exist_ok=True)


class UploadedSourceSummary(BaseModel):
    file_name: str
    file_type: str
    modality: str
    size_bytes: int
    status: str


def _safe_filename(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return cleaned or "uploaded-file"


def _persist_uploaded_file(file_name: str, raw: bytes) -> str:
    target_dir = UPLOAD_CACHE_DIR / uuid4().hex
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / _safe_filename(file_name)
    target_path.write_bytes(raw)
    return str(target_path)


class IntakeSummaryResponse(BaseModel):
    source: UploadedSourceSummary
    grounded_summary: str
    studio_cards: list[dict[str, Any]]
    artifacts: dict[str, Any]
    sources: list[UploadedSourceSummary] = []
    used_tools: list[str] = []


class ChatTurn(BaseModel):
    role: str
    content: str


class ArtifactChatRequest(BaseModel):
    question: str
    analysis: IntakeSummaryResponse
    history: list[ChatTurn] = []
    active_view: Optional[str] = None
    active_card: Optional[dict[str, Any]] = None
    active_artifact: Optional[dict[str, Any]] = None


class ArtifactChatResponse(BaseModel):
    answer: str


class ToolInfo(BaseModel):
    name: str
    team: Optional[str] = None
    task_type: Optional[str] = None
    modality: Optional[str] = None
    approval_required: bool = True
    description: Optional[str] = None
    runtime: Optional[dict[str, Any]] = None
    execution: Optional[dict[str, Any]] = None


class ToolListResponse(BaseModel):
    tools: list[ToolInfo]


class ToolRunRequest(BaseModel):
    tool_name: str
    analysis: IntakeSummaryResponse
    active_view: Optional[str] = None
    active_card: Optional[dict[str, Any]] = None
    active_artifact: Optional[dict[str, Any]] = None
    question: Optional[str] = None


class ToolSuggestRequest(BaseModel):
    question: str
    analysis: Optional[IntakeSummaryResponse] = None
    active_view: Optional[str] = None
    active_card: Optional[dict[str, Any]] = None
    active_artifact: Optional[dict[str, Any]] = None


class ToolSuggestionResponse(BaseModel):
    tool: Optional[ToolInfo] = None
    rationale: Optional[str] = None


class UiBootstrapResponse(BaseModel):
    initial_chat_prompt: str


class ToolRunResponse(BaseModel):
    tool: ToolInfo
    summary: str
    artifacts: dict[str, Any] = {}
    provenance: dict[str, Any] = {}
    stdout: str = ""
    stderr: str = ""


def _wants_korean(question: str) -> bool:
    lowered = question.lower()
    if re.search(r"[가-힣]", question):
        return True
    return any(
        token in lowered
        for token in [
            "한국어",
            "한글",
            "korean",
            "kor",
        ]
    )


def _contains_any(text: str, tokens: list[str]) -> bool:
    return any(token in text for token in tokens)


def _is_generic_explanation_request(question: str, original_question: str) -> bool:
    lowered_original = original_question.lower()
    return _contains_any(
        question,
        [
            "explain",
            "what",
            "meaning",
            "interpret",
            "detail",
            "summary",
            "translate",
            "korean",
        ],
    ) or _contains_any(
        original_question,
        [
            "설명",
            "의미",
            "무슨 뜻",
            "뭐야",
            "자세히",
            "요약",
            "해석",
            "번역",
            "한국어",
            "한글",
        ],
    ) or _contains_any(
        lowered_original,
        [
            "in korean",
        ],
    )


def _korean_analysis_summary(analysis: IntakeSummaryResponse) -> str:
    if analysis.sources:
        lines = ["현재 업로드에는 여러 patient-linked source가 포함되어 있습니다."]
        for index, source in enumerate(analysis.sources):
            prefix = f"source{index}::"
            if source.modality == "clinical-table":
                qc = analysis.artifacts.get(prefix + "qc") or {}
                intake = analysis.artifacts.get(prefix + "intake") or {}
                schema = analysis.artifacts.get(prefix + "schema") or {}
                profiles = (schema.get("profiles") if isinstance(schema, dict) else None) or []
                profile_text = ", ".join(
                    f"{item['name']} ({item['inferred_type']})" for item in profiles[:4]
                )
                lines.append(
                    f"- {source.file_name} (clinical-table): 행 {qc.get('row_count', 'n/a')}개, 열 {qc.get('column_count', 'n/a')}개이며 "
                    f"`{intake.get('analysis_mode', 'n/a')}`로 분류되었고, 대표 변수는 {profile_text or 'n/a'} 입니다."
                )
            elif source.modality == "clinical-message":
                message = analysis.artifacts.get(prefix + "message") or {}
                if isinstance(message, dict) and message.get("format") == "FHIR JSON":
                    lines.append(
                        f"- {source.file_name} (FHIR): resource_type {message.get('resource_type', 'n/a')}, id {message.get('id', 'n/a')} 입니다."
                    )
                else:
                    lines.append(
                        f"- {source.file_name} (HL7): message_type {message.get('message_type', 'n/a')}, control_id {message.get('control_id', 'n/a')}, version {message.get('version', 'n/a')} 입니다."
                    )
            elif source.modality == "medical-image":
                metadata = analysis.artifacts.get(prefix + "metadata") or {}
                if isinstance(metadata, dict):
                    if "items" in metadata:
                        items = metadata.get("items") or []
                        first = items[0] if items else {}
                        lines.append(
                            f"- {source.file_name} (DICOM): modality {first.get('modality', 'n/a')}, study {first.get('study_description', 'n/a')}, matrix {first.get('rows', 'n/a')} x {first.get('columns', 'n/a')} 입니다."
                        )
                    else:
                        lines.append(
                            f"- {source.file_name} (DICOM): modality {metadata.get('modality', 'n/a')}, study {metadata.get('study_description', 'n/a')}, matrix {metadata.get('rows', 'n/a')} x {metadata.get('columns', 'n/a')} 입니다."
                        )
            elif source.modality == "clinical-note":
                note = analysis.artifacts.get(prefix + "note") or {}
                lines.append(
                    f"- {source.file_name} (clinical-note): headline {note.get('headline', 'n/a')}, line_count {note.get('line_count', 'n/a')}, word_count {note.get('word_count', 'n/a')} 입니다."
                )
            else:
                lines.append(f"- {source.file_name}: modality {source.modality}")
        lines.append("원하시면 특정 source 또는 현재 선택한 Studio card 기준으로 더 자세히 설명드릴 수 있습니다.")
        return "\n".join(lines)

    source = analysis.source
    artifacts = analysis.artifacts
    if source.modality == "clinical-table":
        qc = artifacts.get("qc") or {}
        intake = artifacts.get("intake") or {}
        schema = artifacts.get("schema") or {}
        profiles = (schema.get("profiles") if isinstance(schema, dict) else None) or []
        profile_text = ", ".join(f"{item['name']} ({item['inferred_type']})" for item in profiles[:4])
        return (
            f"현재 source는 `{source.file_name}` (clinical-table) 입니다.\n\n"
            f"- 분류: {intake.get('analysis_mode', 'n/a')}\n"
            f"- 행 수: {qc.get('row_count', 'n/a')}\n"
            f"- 열 수: {qc.get('column_count', 'n/a')}\n"
            f"- 대표 변수: {profile_text or 'n/a'}"
        )
    if source.modality == "clinical-message":
        message = artifacts.get("message") or {}
        if isinstance(message, dict) and message.get("format") == "FHIR JSON":
            return (
                f"현재 source는 `{source.file_name}` (FHIR) 입니다.\n\n"
                f"- resource_type: {message.get('resource_type', 'n/a')}\n"
                f"- id: {message.get('id', 'n/a')}"
            )
        return (
            f"현재 source는 `{source.file_name}` (HL7) 입니다.\n\n"
            f"- message_type: {message.get('message_type', 'n/a')}\n"
            f"- control_id: {message.get('control_id', 'n/a')}\n"
            f"- version: {message.get('version', 'n/a')}"
        )
    if source.modality == "medical-image":
        metadata = artifacts.get("metadata") or {}
        if isinstance(metadata, dict):
            return (
                f"현재 source는 `{source.file_name}` (DICOM) 입니다.\n\n"
                f"- modality: {metadata.get('modality', 'n/a')}\n"
                f"- study_description: {metadata.get('study_description', 'n/a')}\n"
                f"- matrix: {metadata.get('rows', 'n/a')} x {metadata.get('columns', 'n/a')}"
            )
    if source.modality == "clinical-note":
        note = artifacts.get("note") or {}
        return (
            f"현재 source는 `{source.file_name}` (clinical-note) 입니다.\n\n"
            f"- headline: {note.get('headline', 'n/a')}\n"
            f"- line_count: {note.get('line_count', 'n/a')}\n"
            f"- word_count: {note.get('word_count', 'n/a')}"
        )
    return (
        f"현재 source는 `{source.file_name}` ({source.modality}) 입니다.\n\n"
        "현재 결과를 한국어로 요약했으며, 더 구체적인 card를 지정하면 자세히 설명드릴 수 있습니다."
    )


WINDOW_PRESETS: list[dict[str, Any]] = [
    {"id": "default", "label": "Default", "width": None, "center": None},
    {"id": "soft", "label": "Soft Tissue", "width": 400, "center": 40},
    {"id": "lung", "label": "Lung", "width": 1500, "center": -600},
    {"id": "bone", "label": "Bone", "width": 2000, "center": 300},
    {"id": "brain", "label": "Brain", "width": 80, "center": 40},
]


app = FastAPI(title="ChatClinic")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:3010", "http://localhost:3010"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _load_env_file() -> None:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


_load_env_file()


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "service": "ChatClinic backend",
        "status": "ok",
        "available_endpoints": [
            "/health",
            "/api/v1/intake/upload",
            "/api/v1/chat/artifacts",
            "/api/v1/ui/bootstrap",
            "/api/v1/tools",
            "/api/v1/tools/suggest",
            "/api/v1/tools/run",
        ],
        "note": "Use POST /api/v1/intake/upload for file intake, POST /api/v1/chat/artifacts for grounded follow-up chat, and the /api/v1/tools endpoints for classroom plugins and orchestrated tool suggestions.",
    }


@app.get("/api/v1/ui/bootstrap", response_model=UiBootstrapResponse)
def ui_bootstrap() -> UiBootstrapResponse:
    return UiBootstrapResponse(initial_chat_prompt=initial_chat_prompt())


@app.get("/api/v1/tools", response_model=ToolListResponse)
def list_tools() -> ToolListResponse:
    tools = [
        ToolInfo(
            name=str(tool.get("name", "")),
            team=tool.get("team"),
            task_type=tool.get("task_type"),
            modality=tool.get("modality"),
            approval_required=bool(tool.get("approval_required", True)),
            description=tool.get("description"),
            runtime=dict(tool.get("runtime", {}) or {}),
        )
        for tool in discover_tools()
        if tool.get("name")
    ]
    return ToolListResponse(tools=tools)


@app.post("/api/v1/tools/suggest", response_model=ToolSuggestionResponse)
def suggest_registered_tool(request: ToolSuggestRequest) -> ToolSuggestionResponse:
    suggestion = suggest_tool(
        question=request.question,
        analysis=request.analysis.model_dump() if request.analysis else {},
        active_view=request.active_view,
    )
    if not suggestion:
        return ToolSuggestionResponse()
    tool_payload = suggestion.get("tool") or {}
    return ToolSuggestionResponse(
        tool=ToolInfo(
            name=str(tool_payload.get("name", "")),
            team=tool_payload.get("team"),
            task_type=tool_payload.get("task_type"),
            modality=tool_payload.get("modality"),
            approval_required=bool(tool_payload.get("approval_required", True)),
            description=tool_payload.get("description"),
            runtime=dict(tool_payload.get("runtime", {}) or {}),
        ),
        rationale=suggestion.get("rationale"),
    )


@app.post("/api/v1/tools/run", response_model=ToolRunResponse)
def execute_tool(request: ToolRunRequest) -> ToolRunResponse:
    payload = {
        "question": request.question or "",
        "analysis_source": request.analysis.source.model_dump(),
        "analysis_sources": [source.model_dump() for source in request.analysis.sources],
        "analysis_artifacts": request.analysis.artifacts,
        "grounded_summary": request.analysis.grounded_summary,
        "active_view": request.active_view,
        "active_card": request.active_card or {},
        "active_artifact": request.active_artifact or {},
    }
    try:
        tool_result = run_tool(request.tool_name, payload)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    tool_meta = tool_result.get("tool", {})
    result_payload = tool_result.get("result", {})
    return ToolRunResponse(
        tool=ToolInfo(
            name=str(tool_meta.get("name", request.tool_name)),
            team=tool_meta.get("team"),
            task_type=tool_meta.get("task_type"),
            modality=tool_meta.get("modality"),
            approval_required=bool(tool_meta.get("approval_required", True)),
            description=None,
            runtime=dict(tool_meta.get("runtime", {}) or {}),
            execution=dict(tool_meta.get("execution", {}) or {}),
        ),
        summary=str(result_payload.get("summary", "Tool completed.")),
        artifacts=dict(result_payload.get("artifacts", {}) or {}),
        provenance=dict(result_payload.get("provenance", {}) or {}),
        stdout=str(tool_result.get("stdout", "")),
        stderr=str(tool_result.get("stderr", "")),
    )


def _guess_modality(filename: str) -> tuple[str, str]:
    suffix = Path(filename).suffix.lower()
    if suffix in {".csv", ".tsv", ".xlsx", ".xlsm", ".xls"}:
        return "clinical-table", suffix.lstrip(".")
    if suffix in {".dcm", ".dicom"}:
        return "medical-image", suffix.lstrip(".")
    if suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
        return "medical-image", suffix.lstrip(".")
    if suffix in {".json", ".xml", ".hl7", ".ndjson"}:
        return "clinical-message", suffix.lstrip(".")
    if suffix in {".txt"}:
        lowered = filename.lower()
        if "hl7" in lowered or "fhir" in lowered:
            return "clinical-message", suffix.lstrip(".")
        return "clinical-note", suffix.lstrip(".")
    if suffix in {".md", ".text"}:
        return "clinical-note", suffix.lstrip(".")
    if suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
        return "medical-image", suffix.lstrip(".")
    return "unknown", suffix.lstrip(".") or "unknown"


def _is_dicom_suffix(suffix: str) -> bool:
    return suffix.lower() in {"dcm", "dicom"}


def _is_raster_image_suffix(suffix: str) -> bool:
    return suffix.lower() in {"png", "jpg", "jpeg", "tif", "tiff"}


def _is_dicom_suffix(suffix: str) -> bool:
    return suffix.lower() in {"dcm", "dicom"}


def _is_raster_image_suffix(suffix: str) -> bool:
    return suffix.lower() in {"png", "jpg", "jpeg", "tif", "tiff"}

def _looks_like_hl7_v2(decoded: str) -> bool:
    stripped = decoded.lstrip()
    return stripped.startswith("MSH|") or "\nMSH|" in stripped or "\rMSH|" in stripped


def _looks_like_fhir_json(decoded: str) -> bool:
    try:
        payload = json.loads(decoded)
    except Exception:
        return False
    return isinstance(payload, dict) and isinstance(payload.get("resourceType"), str)


def _looks_like_fhir_xml(decoded: str) -> bool:
    stripped = decoded.lstrip()
    if not stripped.startswith("<"):
        return False
    return 'http://hl7.org/fhir' in stripped or "<Patient" in stripped or "<Bundle" in stripped


def _local_name(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[-1]
    return tag


def _find_child(element: ET.Element, name: str) -> ET.Element | None:
    for child in list(element):
        if _local_name(child.tag) == name:
            return child
    return None


def _find_children(element: ET.Element, name: str) -> list[ET.Element]:
    return [child for child in list(element) if _local_name(child.tag) == name]


def _attr_value(element: ET.Element | None) -> str:
    if element is None:
        return "n/a"
    return str(element.attrib.get("value", "n/a"))


def _patient_browser_from_json(payload: dict[str, Any]) -> dict[str, Any]:
    names = payload.get("name") or []
    full_name = "n/a"
    if names and isinstance(names, list) and isinstance(names[0], dict):
        given = names[0].get("given") or []
        family = names[0].get("family") or ""
        parts = []
        if isinstance(given, list):
            parts.extend(str(item) for item in given)
        if family:
            parts.append(str(family))
        full_name = " ".join(part for part in parts if part).strip() or "n/a"

    identifiers = []
    for item in payload.get("identifier") or []:
        if not isinstance(item, dict):
            continue
        identifiers.append(
            {
                "system": str(item.get("system", "n/a")),
                "value": str(item.get("value", "n/a")),
                "use": str(item.get("use", "n/a")),
            }
        )

    telecom = []
    for item in payload.get("telecom") or []:
        if not isinstance(item, dict):
            continue
        telecom.append(
            {
                "system": str(item.get("system", "n/a")),
                "value": str(item.get("value", "n/a")),
                "use": str(item.get("use", "n/a")),
            }
        )

    addresses = []
    for item in payload.get("address") or []:
        if not isinstance(item, dict):
            continue
        line = item.get("line") or []
        line_text = ", ".join(str(part) for part in line) if isinstance(line, list) else str(line)
        addresses.append(
            {
                "line": line_text or "n/a",
                "city": str(item.get("city", "n/a")),
                "state": str(item.get("state", "n/a")),
                "postalCode": str(item.get("postalCode", "n/a")),
                "country": str(item.get("country", "n/a")),
            }
        )

    return {
        "resource_type": str(payload.get("resourceType", "Unknown")),
        "id": str(payload.get("id", "n/a")),
        "full_name": full_name,
        "gender": str(payload.get("gender", "n/a")),
        "birth_date": str(payload.get("birthDate", "n/a")),
        "active": str(payload.get("active", "n/a")),
        "identifiers": identifiers,
        "telecom": telecom,
        "addresses": addresses,
        "managing_organization": str(((payload.get("managingOrganization") or {}).get("reference")) or "n/a"),
    }


def _resolve_fhir_resources_json(payload: dict[str, Any], resource_type: str) -> list[dict[str, Any]]:
    if str(payload.get("resourceType")) == resource_type:
        return [payload]
    if str(payload.get("resourceType")) != "Bundle":
        return []
    resources: list[dict[str, Any]] = []
    for entry in payload.get("entry") or []:
        if not isinstance(entry, dict):
            continue
        resource = entry.get("resource")
        if isinstance(resource, dict) and str(resource.get("resourceType")) == resource_type:
            resources.append(resource)
    return resources


def _first_fhir_patient_json(payload: dict[str, Any]) -> dict[str, Any]:
    patient_resources = _resolve_fhir_resources_json(payload, "Patient")
    if patient_resources:
        return patient_resources[0]
    return payload


def _fhir_code_display_json(resource: dict[str, Any], field_name: str = "code") -> str:
    node = resource.get(field_name)
    if not isinstance(node, dict):
        return "n/a"
    coding = node.get("coding") or []
    if coding and isinstance(coding[0], dict):
        return str(coding[0].get("display") or coding[0].get("code") or "n/a")
    return str(node.get("text") or "n/a")


def _observation_numeric_json(obs: dict[str, Any]) -> tuple[float | None, str, float | None, float | None]:
    quantity = obs.get("valueQuantity")
    if not isinstance(quantity, dict):
        return None, "n/a", None, None
    try:
        numeric = float(quantity.get("value"))
    except Exception:
        numeric = None
    unit = str(quantity.get("unit") or quantity.get("code") or "n/a")
    low = None
    high = None
    ranges = obs.get("referenceRange") or []
    if ranges and isinstance(ranges[0], dict):
        low_node = ranges[0].get("low") or {}
        high_node = ranges[0].get("high") or {}
        try:
            low = float(low_node.get("value")) if isinstance(low_node, dict) and low_node.get("value") is not None else None
        except Exception:
            low = None
        try:
            high = float(high_node.get("value")) if isinstance(high_node, dict) and high_node.get("value") is not None else None
        except Exception:
            high = None
    return numeric, unit, low, high


def _observation_category_json(obs: dict[str, Any]) -> str:
    categories = obs.get("category") or []
    if not isinstance(categories, list):
        return "n/a"
    for category in categories:
        if not isinstance(category, dict):
            continue
        coding = category.get("coding") or []
        if coding and isinstance(coding[0], dict):
            return str(coding[0].get("code") or coding[0].get("display") or "n/a")
    return "n/a"


def _blood_pressure_value_json(obs: dict[str, Any]) -> str:
    components = obs.get("component") or []
    systolic = None
    diastolic = None
    unit = "mmHg"
    for component in components:
        if not isinstance(component, dict):
            continue
        label = _fhir_code_display_json(component)
        quantity = component.get("valueQuantity") if isinstance(component.get("valueQuantity"), dict) else {}
        value = quantity.get("value")
        unit = str(quantity.get("unit") or unit)
        if "systolic" in label.lower():
            systolic = value
        if "diastolic" in label.lower():
            diastolic = value
    if systolic is not None or diastolic is not None:
        return f"{systolic or '?'} / {diastolic or '?'} {unit}".strip()
    return "n/a"


def _observation_viewer_from_json(payload: dict[str, Any]) -> dict[str, Any]:
    observations = _resolve_fhir_resources_json(payload, "Observation")
    items: list[dict[str, Any]] = []
    for obs in observations[:24]:
        code = _fhir_code_display_json(obs)
        value = "n/a"
        numeric_value, unit, ref_low, ref_high = _observation_numeric_json(obs)
        if numeric_value is not None:
            value = f"{numeric_value} {unit}".strip()
        elif "valueString" in obs:
            value = str(obs.get("valueString"))
        elif "valueCodeableConcept" in obs and isinstance(obs.get("valueCodeableConcept"), dict):
            concepts = (obs.get("valueCodeableConcept") or {}).get("coding") or []
            if concepts and isinstance(concepts[0], dict):
                value = str(concepts[0].get("display") or concepts[0].get("code") or "n/a")
        elif (obs.get("component") or []) and "blood pressure" in code.lower():
            value = _blood_pressure_value_json(obs)
        items.append(
            {
                "code": code,
                "value": value,
                "status": str(obs.get("status", "n/a")),
                "effective": str(obs.get("effectiveDateTime", obs.get("issued", "n/a"))),
                "category": _observation_category_json(obs),
                "numeric_value": numeric_value,
                "unit": unit,
                "reference_low": ref_low,
                "reference_high": ref_high,
            }
        )
    return {"count": len(observations), "items": items}


def _medication_timeline_from_json(payload: dict[str, Any]) -> dict[str, Any]:
    meds = _resolve_fhir_resources_json(payload, "MedicationRequest") + _resolve_fhir_resources_json(payload, "MedicationStatement")
    items: list[dict[str, Any]] = []
    for med in meds[:24]:
        med_name = "n/a"
        concept = med.get("medicationCodeableConcept")
        if isinstance(concept, dict):
            coding = concept.get("coding") or []
            if coding and isinstance(coding[0], dict):
                med_name = str(coding[0].get("display") or coding[0].get("code") or "n/a")
            elif concept.get("text"):
                med_name = str(concept.get("text"))
        status = str(med.get("status", "n/a"))
        intent = str(med.get("intent", "n/a"))
        authored = str(med.get("authoredOn", med.get("effectiveDateTime", "n/a")))
        dosage = "n/a"
        dosage_list = med.get("dosageInstruction") or []
        if dosage_list and isinstance(dosage_list[0], dict):
            dosage = str(dosage_list[0].get("text") or "n/a")
        start = str(
            med.get("authoredOn")
            or ((med.get("effectivePeriod") or {}).get("start") if isinstance(med.get("effectivePeriod"), dict) else None)
            or med.get("effectiveDateTime")
            or "n/a"
        )
        end = str(
            ((med.get("dispenseRequest") or {}).get("validityPeriod") or {}).get("end")
            if isinstance((med.get("dispenseRequest") or {}).get("validityPeriod"), dict)
            else (
                (med.get("effectivePeriod") or {}).get("end")
                if isinstance(med.get("effectivePeriod"), dict)
                else "n/a"
            )
        )
        duration_days = None
        duration = (med.get("dispenseRequest") or {}).get("expectedSupplyDuration") if isinstance(med.get("dispenseRequest"), dict) else None
        if isinstance(duration, dict):
            try:
                duration_days = float(duration.get("value"))
            except Exception:
                duration_days = None
        current = status.lower() in {"active", "in-progress", "on-hold"}
        items.append(
            {
                "medication": med_name,
                "status": status,
                "intent": intent,
                "date": authored,
                "dosage": dosage,
                "start": start,
                "end": end,
                "duration_days": duration_days,
                "current": current,
            }
        )
    return {"count": len(meds), "items": items}


def _patient_browser_from_xml(root: ET.Element) -> dict[str, Any]:
    resource_type = _local_name(root.tag)
    names = _find_children(root, "name")
    full_name = "n/a"
    if names:
        given_names = [_attr_value(child) for child in _find_children(names[0], "given")]
        family = _attr_value(_find_child(names[0], "family"))
        full_name = " ".join(part for part in [*given_names, family] if part and part != "n/a") or "n/a"

    identifiers = []
    for identifier in _find_children(root, "identifier"):
        identifiers.append(
            {
                "system": _attr_value(_find_child(identifier, "system")),
                "value": _attr_value(_find_child(identifier, "value")),
                "use": _attr_value(_find_child(identifier, "use")),
            }
        )

    telecom = []
    for item in _find_children(root, "telecom"):
        telecom.append(
            {
                "system": _attr_value(_find_child(item, "system")),
                "value": _attr_value(_find_child(item, "value")),
                "use": _attr_value(_find_child(item, "use")),
            }
        )

    addresses = []
    for item in _find_children(root, "address"):
        line_values = [_attr_value(child) for child in _find_children(item, "line")]
        addresses.append(
            {
                "line": ", ".join(value for value in line_values if value != "n/a") or "n/a",
                "city": _attr_value(_find_child(item, "city")),
                "state": _attr_value(_find_child(item, "state")),
                "postalCode": _attr_value(_find_child(item, "postalCode")),
                "country": _attr_value(_find_child(item, "country")),
            }
        )

    managing_org = _find_child(root, "managingOrganization")
    return {
        "resource_type": resource_type,
        "id": _attr_value(_find_child(root, "id")),
        "full_name": full_name,
        "gender": _attr_value(_find_child(root, "gender")),
        "birth_date": _attr_value(_find_child(root, "birthDate")),
        "active": _attr_value(_find_child(root, "active")),
        "identifiers": identifiers,
        "telecom": telecom,
        "addresses": addresses,
        "managing_organization": _attr_value(_find_child(managing_org, "reference")) if managing_org is not None else "n/a",
    }


def _resolve_fhir_resources_xml(root: ET.Element, resource_type: str) -> list[ET.Element]:
    if _local_name(root.tag) == resource_type:
        return [root]
    if _local_name(root.tag) != "Bundle":
        return []
    resources: list[ET.Element] = []
    for entry in _find_children(root, "entry"):
        resource_container = _find_child(entry, "resource")
        if resource_container is None:
            continue
        for child in list(resource_container):
            if _local_name(child.tag) == resource_type:
                resources.append(child)
    return resources


def _first_fhir_patient_xml(root: ET.Element) -> ET.Element:
    resources = _resolve_fhir_resources_xml(root, "Patient")
    if resources:
        return resources[0]
    return root


def _observation_viewer_from_xml(root: ET.Element) -> dict[str, Any]:
    observations = _resolve_fhir_resources_xml(root, "Observation")
    items: list[dict[str, Any]] = []
    for obs in observations[:24]:
        code = "n/a"
        code_node = _find_child(obs, "code")
        if code_node is not None:
            coding = _find_children(code_node, "coding")
            if coding:
                code = _attr_value(_find_child(coding[0], "display"))
                if code == "n/a":
                    code = _attr_value(_find_child(coding[0], "code"))
        value = "n/a"
        quantity = _find_child(obs, "valueQuantity")
        if quantity is not None:
            value = f"{_attr_value(_find_child(quantity, 'value'))} {_attr_value(_find_child(quantity, 'unit'))}".strip()
        else:
            value = _attr_value(_find_child(obs, "valueString"))
        items.append(
            {
                "code": code,
                "value": value,
                "status": _attr_value(_find_child(obs, "status")),
                "effective": _attr_value(_find_child(obs, "effectiveDateTime")),
                "category": "n/a",
            }
        )
    return {"count": len(observations), "items": items}


def _medication_timeline_from_xml(root: ET.Element) -> dict[str, Any]:
    meds = _resolve_fhir_resources_xml(root, "MedicationRequest") + _resolve_fhir_resources_xml(root, "MedicationStatement")
    items: list[dict[str, Any]] = []
    for med in meds[:24]:
        med_name = "n/a"
        concept = _find_child(med, "medicationCodeableConcept")
        if concept is not None:
            coding = _find_children(concept, "coding")
            if coding:
                med_name = _attr_value(_find_child(coding[0], "display"))
                if med_name == "n/a":
                    med_name = _attr_value(_find_child(coding[0], "code"))
            if med_name == "n/a":
                med_name = _attr_value(_find_child(concept, "text"))
        items.append(
            {
                "medication": med_name,
                "status": _attr_value(_find_child(med, "status")),
                "intent": _attr_value(_find_child(med, "intent")),
                "date": _attr_value(_find_child(med, "authoredOn")),
                "dosage": "n/a",
                "start": _attr_value(_find_child(med, "authoredOn")),
                "end": "n/a",
                "duration_days": None,
                "current": _attr_value(_find_child(med, "status")) in {"active", "in-progress", "on-hold"},
            }
        )
    return {"count": len(meds), "items": items}


def _allergy_summary_from_json(payload: dict[str, Any]) -> dict[str, Any]:
    allergies = _resolve_fhir_resources_json(payload, "AllergyIntolerance")
    items: list[dict[str, Any]] = []
    for allergy in allergies[:12]:
        items.append(
            {
                "substance": _fhir_code_display_json(allergy),
                "criticality": str(allergy.get("criticality", "n/a")),
                "clinical_status": _fhir_code_display_json(allergy, "clinicalStatus"),
                "verification_status": _fhir_code_display_json(allergy, "verificationStatus"),
            }
        )
    return {"count": len(allergies), "items": items}


def _allergy_summary_from_xml(root: ET.Element) -> dict[str, Any]:
    allergies = _resolve_fhir_resources_xml(root, "AllergyIntolerance")
    items: list[dict[str, Any]] = []
    for allergy in allergies[:12]:
        code_node = _find_child(allergy, "code")
        substance = "n/a"
        if code_node is not None:
            coding = _find_children(code_node, "coding")
            if coding:
                substance = _attr_value(_find_child(coding[0], "display"))
                if substance == "n/a":
                    substance = _attr_value(_find_child(coding[0], "code"))
        items.append(
            {
                "substance": substance,
                "criticality": _attr_value(_find_child(allergy, "criticality")),
                "clinical_status": "n/a",
                "verification_status": "n/a",
            }
        )
    return {"count": len(allergies), "items": items}


def _vital_summary_from_observations(observation_artifact: dict[str, Any]) -> dict[str, Any]:
    wanted = [
        ("blood pressure", "Blood pressure"),
        ("body weight", "Weight"),
        ("glucose", "Glucose"),
        ("heart rate", "Heart rate"),
        ("temperature", "Temperature"),
        ("oxygen saturation", "O2 saturation"),
    ]
    latest: list[dict[str, Any]] = []
    items = observation_artifact.get("items") or []
    for needle, label in wanted:
        matches = [item for item in items if needle in str(item.get("code", "")).lower()]
        if matches:
            latest.append(
                {
                    "label": label,
                    "value": matches[0].get("value", "n/a"),
                    "effective": matches[0].get("effective", "n/a"),
                    "status": matches[0].get("status", "n/a"),
                }
            )
    return {"items": latest}


def _timeline_events_from_json(payload: dict[str, Any]) -> dict[str, Any]:
    encounters = _resolve_fhir_resources_json(payload, "Encounter")
    procedures = _resolve_fhir_resources_json(payload, "Procedure")
    events: list[dict[str, Any]] = []
    for encounter in encounters[:12]:
        period = encounter.get("period") if isinstance(encounter.get("period"), dict) else {}
        events.append(
            {
                "type": "Encounter",
                "label": _fhir_code_display_json(encounter, "type"),
                "start": str(period.get("start") or encounter.get("actualPeriod", {}).get("start") if isinstance(encounter.get("actualPeriod"), dict) else period.get("start") or "n/a"),
                "end": str(period.get("end") or encounter.get("actualPeriod", {}).get("end") if isinstance(encounter.get("actualPeriod"), dict) else period.get("end") or "n/a"),
                "status": str(encounter.get("status", "n/a")),
            }
        )
    for procedure in procedures[:12]:
        performed = procedure.get("performedPeriod") if isinstance(procedure.get("performedPeriod"), dict) else {}
        events.append(
            {
                "type": "Procedure",
                "label": _fhir_code_display_json(procedure),
                "start": str(performed.get("start") or procedure.get("performedDateTime") or "n/a"),
                "end": str(performed.get("end") or "n/a"),
                "status": str(procedure.get("status", "n/a")),
            }
        )
    return {"events": events}


def _timeline_events_from_xml(root: ET.Element) -> dict[str, Any]:
    encounters = _resolve_fhir_resources_xml(root, "Encounter")
    procedures = _resolve_fhir_resources_xml(root, "Procedure")
    events: list[dict[str, Any]] = []
    for encounter in encounters[:12]:
        events.append(
            {
                "type": "Encounter",
                "label": "Encounter",
                "start": _attr_value(_find_child(_find_child(encounter, "period") or encounter, "start")),
                "end": _attr_value(_find_child(_find_child(encounter, "period") or encounter, "end")),
                "status": _attr_value(_find_child(encounter, "status")),
            }
        )
    for procedure in procedures[:12]:
        events.append(
            {
                "type": "Procedure",
                "label": _attr_value(_find_child(_find_child(procedure, "code"), "text")),
                "start": _attr_value(_find_child(_find_child(procedure, "performedPeriod") or procedure, "start")),
                "end": _attr_value(_find_child(_find_child(procedure, "performedPeriod") or procedure, "end")),
                "status": _attr_value(_find_child(procedure, "status")),
            }
        )
    return {"events": events}


def _lab_trends_from_observations(observation_artifact: dict[str, Any]) -> dict[str, Any]:
    series_map: dict[str, list[dict[str, Any]]] = {}
    for item in observation_artifact.get("items") or []:
        numeric_value = item.get("numeric_value")
        if numeric_value is None:
            continue
        key = str(item.get("code") or "Unknown")
        series_map.setdefault(key, []).append(
            {
                "date": str(item.get("effective", "n/a")),
                "value": numeric_value,
                "unit": str(item.get("unit", "n/a")),
                "low": item.get("reference_low"),
                "high": item.get("reference_high"),
            }
        )
    series = [{"label": label, "points": points[:16]} for label, points in list(series_map.items())[:6]]
    latest = []
    for item in series[:4]:
        point = item["points"][0] if item["points"] else {}
        latest.append(
            {
                "label": item["label"],
                "value": point.get("value", "n/a"),
                "unit": point.get("unit", "n/a"),
                "low": point.get("low"),
                "high": point.get("high"),
            }
        )
    return {"series": series, "latest": latest}


def _care_team_from_json(payload: dict[str, Any]) -> dict[str, Any]:
    practitioners = _resolve_fhir_resources_json(payload, "Practitioner")
    organizations = _resolve_fhir_resources_json(payload, "Organization")
    practitioner_cards: list[dict[str, Any]] = []
    for practitioner in practitioners[:12]:
        name = _patient_browser_from_json(practitioner).get("full_name", "n/a")
        telecom = practitioner.get("telecom") or []
        practitioner_cards.append(
            {
                "name": name,
                "role": "Practitioner",
                "contact": str((telecom[0] or {}).get("value")) if telecom and isinstance(telecom[0], dict) else "n/a",
                "organization": "n/a",
            }
        )
    organization_cards: list[dict[str, Any]] = []
    for org in organizations[:12]:
        telecom = org.get("telecom") or []
        organization_cards.append(
            {
                "name": str(org.get("name", "n/a")),
                "contact": str((telecom[0] or {}).get("value")) if telecom and isinstance(telecom[0], dict) else "n/a",
            }
        )
    return {"practitioners": practitioner_cards, "organizations": organization_cards}


def _care_team_from_xml(root: ET.Element) -> dict[str, Any]:
    practitioners = _resolve_fhir_resources_xml(root, "Practitioner")
    organizations = _resolve_fhir_resources_xml(root, "Organization")
    practitioner_cards: list[dict[str, Any]] = []
    for practitioner in practitioners[:12]:
        practitioner_cards.append(
            {
                "name": _patient_browser_from_xml(practitioner).get("full_name", "n/a"),
                "role": "Practitioner",
                "contact": "n/a",
                "organization": "n/a",
            }
        )
    organization_cards: list[dict[str, Any]] = []
    for org in organizations[:12]:
        organization_cards.append(
            {
                "name": _attr_value(_find_child(org, "name")),
                "contact": "n/a",
            }
        )
    return {"practitioners": practitioner_cards, "organizations": organization_cards}


def _summarize_clinical_note(file_name: str, raw: bytes, suffix: str) -> IntakeSummaryResponse:
    decoded = raw.decode("utf-8", errors="replace")
    lines = [line.strip() for line in decoded.splitlines() if line.strip()]
    words = re.findall(r"\b\w+\b", decoded)
    headline = lines[0][:160] if lines else "No readable headline"
    summary = (
        f"This clinical note contains {len(lines)} non-empty line(s) and approximately {len(words)} word(s). "
        f"The opening line is: {headline}. "
        "This deterministic summary should be followed by section parsing, entity extraction, and clinical interpretation review."
    )
    studio_cards = [
        {"id": "qc", "title": "Note QC", "subtitle": "Length and readability checks"},
        {"id": "note", "title": "Clinical Note", "subtitle": "Text preview and note metadata"},
    ]
    artifacts = {
        "qc": {
            "file_size_bytes": len(raw),
            "line_count": len(lines),
            "word_count": len(words),
        },
        "note": {
            "headline": headline,
            "line_count": len(lines),
            "word_count": len(words),
            "preview": "\n".join(lines[:12]),
        },
    }
    return IntakeSummaryResponse(
        source=UploadedSourceSummary(
            file_name=file_name,
            file_type=suffix,
            modality="clinical-note",
            size_bytes=len(raw),
            status="parsed",
        ),
        grounded_summary=summary,
        studio_cards=studio_cards,
        artifacts=artifacts,
        sources=[],
    )


def _summarize_fhir_json(file_name: str, raw: bytes, suffix: str) -> IntakeSummaryResponse:
    payload = json.loads(raw.decode("utf-8", errors="replace"))
    resource_type = payload.get("resourceType", "Unknown")
    resource_id = payload.get("id", "n/a")
    top_level_keys = sorted(payload.keys())

    identity_bits: list[str] = []
    if resource_type == "Patient":
        names = payload.get("name") or []
        first_name = ""
        if names and isinstance(names, list) and isinstance(names[0], dict):
            given = names[0].get("given") or []
            family = names[0].get("family") or ""
            first_name = " ".join([*([str(x) for x in given] if isinstance(given, list) else []), str(family)]).strip()
        gender = payload.get("gender", "n/a")
        birth_date = payload.get("birthDate", "n/a")
        if first_name:
            identity_bits.append(f"name {first_name}")
        identity_bits.append(f"gender {gender}")
        identity_bits.append(f"birthDate {birth_date}")

    summary = (
        f"This clinical message was recognized as FHIR JSON. "
        f"The top-level resourceType is {resource_type} with id {resource_id}. "
        f"Available top-level keys include {', '.join(top_level_keys[:8]) if top_level_keys else 'none'}. "
        f"{'Key patient-style fields include ' + ', '.join(identity_bits) + '. ' if identity_bits else ''}"
        "This deterministic summary should be followed by resource validation, profile/conformance checks, and field-level review."
    )

    patient_browser = _patient_browser_from_json(_first_fhir_patient_json(payload))
    observation_viewer = _observation_viewer_from_json(payload)
    medication_timeline = _medication_timeline_from_json(payload)
    allergy_summary = _allergy_summary_from_json(payload)
    vital_summary = _vital_summary_from_observations(observation_viewer)
    timeline_events = _timeline_events_from_json(payload)
    lab_trends = _lab_trends_from_observations(observation_viewer)
    care_team = _care_team_from_json(payload)

    studio_cards = [
        {"id": "fhir_browser", "title": "FHIR Browser", "subtitle": "Patient, observations, and medications in one review"},
    ]
    nested_counts = {}
    for key, value in payload.items():
        if isinstance(value, list):
            nested_counts[key] = len(value)
        elif isinstance(value, dict):
            nested_counts[key] = len(value.keys())

    artifacts = {
        "qc": {
            "file_size_bytes": len(raw),
            "message_type": "FHIR",
            "resource_type": resource_type,
        },
        "message": {
            "format": "FHIR JSON",
            "resource_type": resource_type,
            "id": str(resource_id),
            "top_level_keys": top_level_keys,
        },
        "patient": patient_browser,
        "allergies": allergy_summary,
        "vitals": vital_summary,
        "observations": observation_viewer,
        "medications": medication_timeline,
        "timeline": timeline_events,
        "labs": lab_trends,
        "care_team": care_team,
        "resources": {
            "resource_type": resource_type,
            "top_level_keys": top_level_keys,
            "nested_counts": nested_counts,
            "sample": {key: payload.get(key) for key in top_level_keys[:8]},
        },
    }
    return IntakeSummaryResponse(
        source=UploadedSourceSummary(
            file_name=file_name,
            file_type=suffix,
            modality="clinical-message",
            size_bytes=len(raw),
            status="parsed",
        ),
        grounded_summary=summary,
        studio_cards=studio_cards,
        artifacts=artifacts,
        sources=[],
    )


def _summarize_fhir_xml(file_name: str, raw: bytes, suffix: str) -> IntakeSummaryResponse:
    root = ET.fromstring(raw.decode("utf-8", errors="replace"))
    resource_type = _local_name(root.tag)
    resource_id = _attr_value(_find_child(root, "id"))
    top_level_keys = sorted(_local_name(child.tag) for child in list(root))
    patient_browser = _patient_browser_from_xml(_first_fhir_patient_xml(root))
    observation_viewer = _observation_viewer_from_xml(root)
    medication_timeline = _medication_timeline_from_xml(root)
    allergy_summary = _allergy_summary_from_xml(root)
    vital_summary = _vital_summary_from_observations(observation_viewer)
    timeline_events = _timeline_events_from_xml(root)
    lab_trends = _lab_trends_from_observations(observation_viewer)
    care_team = _care_team_from_xml(root)

    identity_bits: list[str] = []
    if patient_browser.get("full_name") not in {None, "", "n/a"}:
        identity_bits.append(f"name {patient_browser.get('full_name')}")
    if patient_browser.get("gender") not in {None, "", "n/a"}:
        identity_bits.append(f"gender {patient_browser.get('gender')}")
    if patient_browser.get("birth_date") not in {None, "", "n/a"}:
        identity_bits.append(f"birthDate {patient_browser.get('birth_date')}")

    summary = (
        f"This clinical message was recognized as FHIR XML. "
        f"The top-level resourceType is {resource_type} with id {resource_id}. "
        f"Available top-level elements include {', '.join(top_level_keys[:8]) if top_level_keys else 'none'}. "
        f"{'Key patient-style fields include ' + ', '.join(identity_bits) + '. ' if identity_bits else ''}"
        "This deterministic summary should be followed by resource validation, profile/conformance checks, and field-level review."
    )

    nested_counts = {}
    for child in list(root):
        child_name = _local_name(child.tag)
        nested_counts[child_name] = len(list(child))

    studio_cards = [
        {"id": "fhir_browser", "title": "FHIR Browser", "subtitle": "Patient, observations, and medications in one review"},
    ]
    artifacts = {
        "qc": {
            "file_size_bytes": len(raw),
            "message_type": "FHIR",
            "resource_type": resource_type,
        },
        "message": {
            "format": "FHIR XML",
            "resource_type": resource_type,
            "id": str(resource_id),
            "top_level_keys": top_level_keys,
        },
        "patient": patient_browser,
        "allergies": allergy_summary,
        "vitals": vital_summary,
        "observations": observation_viewer,
        "medications": medication_timeline,
        "timeline": timeline_events,
        "labs": lab_trends,
        "care_team": care_team,
        "resources": {
            "resource_type": resource_type,
            "top_level_keys": top_level_keys,
            "nested_counts": nested_counts,
            "sample": {child_name: ET.tostring(child, encoding="unicode")[:300] for child_name, child in [(_local_name(child.tag), child) for child in list(root)[:6]]},
        },
    }
    return IntakeSummaryResponse(
        source=UploadedSourceSummary(
            file_name=file_name,
            file_type=suffix,
            modality="clinical-message",
            size_bytes=len(raw),
            status="parsed",
        ),
        grounded_summary=summary,
        studio_cards=studio_cards,
        artifacts=artifacts,
        sources=[],
    )


def _fhir_bundle_from_ndjson_files(files: list[tuple[str, bytes, str]]) -> tuple[dict[str, Any], dict[str, int]]:
    entries: list[dict[str, Any]] = []
    resource_counts: dict[str, int] = {}
    for file_name, raw, _suffix in files:
        decoded = raw.decode("utf-8", errors="replace")
        for line in decoded.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            try:
                resource = json.loads(stripped)
            except Exception:
                continue
            if not isinstance(resource, dict):
                continue
            resource_type = str(resource.get("resourceType", "Unknown"))
            resource_counts[resource_type] = resource_counts.get(resource_type, 0) + 1
            entries.append(
                {
                    "fullUrl": f"urn:chatclinic:{file_name}:{resource_type}:{resource_counts[resource_type]}",
                    "resource": resource,
                }
            )
    bundle = {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": entries,
    }
    return bundle, resource_counts


def _summarize_fhir_ndjson(file_name: str, raw: bytes, suffix: str) -> IntakeSummaryResponse:
    return _summarize_fhir_ndjson_group([(file_name, raw, suffix)])


def _summarize_fhir_ndjson_group(files: list[tuple[str, bytes, str]]) -> IntakeSummaryResponse:
    bundle, resource_counts = _fhir_bundle_from_ndjson_files(files)
    top_resource_types = ", ".join(
        f"{resource_type}={count}" for resource_type, count in sorted(resource_counts.items(), key=lambda item: (-item[1], item[0]))[:8]
    ) or "none"
    patient_browser = _patient_browser_from_json(_first_fhir_patient_json(bundle))
    observation_viewer = _observation_viewer_from_json(bundle)
    medication_timeline = _medication_timeline_from_json(bundle)
    allergy_summary = _allergy_summary_from_json(bundle)
    vital_summary = _vital_summary_from_observations(observation_viewer)
    timeline_events = _timeline_events_from_json(bundle)
    lab_trends = _lab_trends_from_observations(observation_viewer)
    care_team = _care_team_from_json(bundle)

    summary = (
        f"This upload was recognized as bulk FHIR NDJSON. "
        f"It contains {sum(resource_counts.values())} resource rows across {len(resource_counts)} resource type(s): {top_resource_types}. "
        f"Parsed {resource_counts.get('Patient', 0)} Patient, {resource_counts.get('Observation', 0)} Observation, "
        f"{resource_counts.get('MedicationRequest', 0) + resource_counts.get('MedicationStatement', 0)} medication, "
        f"{resource_counts.get('Encounter', 0)} Encounter, and {resource_counts.get('AllergyIntolerance', 0)} AllergyIntolerance records. "
        "This deterministic summary should be followed by cohort review, patient-level drilldown, and quality checks for missing links across resource types."
    )

    artifacts = {
        "qc": {
            "file_count": len(files),
            "resource_row_count": sum(resource_counts.values()),
            "resource_type_count": len(resource_counts),
            "resource_counts": resource_counts,
        },
        "message": {
            "format": "FHIR NDJSON",
            "resource_type": "Bundle",
            "id": "bulk-ndjson",
            "top_level_keys": ["entry", "resourceType", "type"],
        },
        "patient": patient_browser,
        "allergies": allergy_summary,
        "vitals": vital_summary,
        "observations": observation_viewer,
        "medications": medication_timeline,
        "timeline": timeline_events,
        "labs": lab_trends,
        "care_team": care_team,
        "resources": {
            "resource_type": "Bundle",
            "top_level_keys": ["entry", "resourceType", "type"],
            "nested_counts": resource_counts,
            "sample": {
                "files": [item[0] for item in files[:10]],
                "resource_counts": resource_counts,
            },
        },
    }
    return IntakeSummaryResponse(
        source=UploadedSourceSummary(
            file_name=f"{len(files)} bulk FHIR file(s)" if len(files) > 1 else files[0][0],
            file_type="ndjson",
            modality="clinical-message",
            size_bytes=sum(len(item[1]) for item in files),
            status="parsed",
        ),
        grounded_summary=summary,
        studio_cards=[{"id": "fhir_browser", "title": "FHIR Browser", "subtitle": "Bulk patient, observations, and medications review"}],
        artifacts=artifacts,
        sources=[],
    )


def _summarize_hl7_v2(file_name: str, raw: bytes, suffix: str) -> IntakeSummaryResponse:
    decoded = raw.decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
    segments = [line.strip() for line in decoded.split("\n") if line.strip()]
    segment_names = [segment.split("|", 1)[0] for segment in segments]
    msh_fields = segments[0].split("|") if segments else []
    message_type = msh_fields[8] if len(msh_fields) > 8 else "n/a"
    control_id = msh_fields[9] if len(msh_fields) > 9 else "n/a"
    version = msh_fields[11] if len(msh_fields) > 11 else "n/a"

    patient_id = "n/a"
    patient_name = "n/a"
    visit_class = "n/a"
    for segment in segments:
        fields = segment.split("|")
        if fields[0] == "PID":
            if len(fields) > 3:
                patient_id = fields[3] or "n/a"
            if len(fields) > 5:
                patient_name = fields[5] or "n/a"
        if fields[0] == "PV1" and len(fields) > 2:
            visit_class = fields[2] or "n/a"

    counts: dict[str, int] = {}
    for name in segment_names:
        counts[name] = counts.get(name, 0) + 1

    summary = (
        f"This clinical message was recognized as HL7 v2 text. "
        f"The message type is {message_type}, control id is {control_id}, and version is {version}. "
        f"Detected {len(segments)} segment(s) with segment types {', '.join(sorted(counts.keys()))}. "
        f"Patient identifier is {patient_id}, patient name is {patient_name}, and visit class is {visit_class}. "
        "This deterministic summary should be followed by segment validation, mapping review, and downstream transformation checks."
    )

    studio_cards = [
        {"id": "qc", "title": "Message QC", "subtitle": "Segment count and format checks"},
        {"id": "message", "title": "HL7 Review", "subtitle": "Message header and patient metadata"},
        {"id": "resources", "title": "Segment Structure", "subtitle": "Segment inventory and sample lines"},
    ]
    artifacts = {
        "qc": {
            "file_size_bytes": len(raw),
            "message_type": message_type,
            "segment_count": len(segments),
            "version": version,
        },
        "message": {
            "format": "HL7 v2",
            "message_type": message_type,
            "control_id": control_id,
            "version": version,
            "patient_id": patient_id,
            "patient_name": patient_name,
            "visit_class": visit_class,
        },
        "resources": {
            "segment_counts": counts,
            "segments": segments[:12],
        },
    }
    return IntakeSummaryResponse(
        source=UploadedSourceSummary(
            file_name=file_name,
            file_type=suffix,
            modality="clinical-message",
            size_bytes=len(raw),
            status="parsed",
        ),
        grounded_summary=summary,
        studio_cards=studio_cards,
        artifacts=artifacts,
        sources=[],
    )


def _summarize_clinical_message(file_name: str, raw: bytes, suffix: str) -> IntakeSummaryResponse:
    if suffix == "ndjson":
        return _summarize_fhir_ndjson(file_name, raw, suffix)
    decoded = raw.decode("utf-8", errors="replace")
    if _looks_like_fhir_json(decoded):
        return _summarize_fhir_json(file_name, raw, suffix)
    if _looks_like_fhir_xml(decoded):
        return _summarize_fhir_xml(file_name, raw, suffix)
    if _looks_like_hl7_v2(decoded):
        return _summarize_hl7_v2(file_name, raw, suffix)

    summary = (
        "The uploaded source looks like a clinical message file, but ChatClinic could not confidently classify it as "
        "FHIR JSON or HL7 v2 text. Provide a JSON FHIR resource or an HL7 message beginning with MSH| for deterministic review."
    )
    return IntakeSummaryResponse(
        source=UploadedSourceSummary(
            file_name=file_name,
            file_type=suffix,
            modality="clinical-message",
            size_bytes=len(raw),
            status="unsupported",
        ),
        grounded_summary=summary,
        studio_cards=[],
        artifacts={},
        sources=[],
    )


def _is_int_like(value: str) -> bool:
    try:
        int(value)
        return True
    except Exception:
        return False


def _is_float_like(value: str) -> bool:
    try:
        float(value)
        return True
    except Exception:
        return False


def _is_date_like(value: str) -> bool:
    value = value.strip()
    if not value:
        return False
    date_patterns = (
        r"^\d{4}-\d{2}-\d{2}$",
        r"^\d{4}/\d{2}/\d{2}$",
        r"^\d{2}/\d{2}/\d{4}$",
        r"^\d{2}-\d{2}-\d{4}$",
    )
    import re

    return any(re.match(pattern, value) for pattern in date_patterns)


def _infer_column_profile(name: str, values: list[str]) -> dict[str, Any]:
    non_empty = [value.strip() for value in values if value is not None and value.strip()]
    missing_count = len(values) - len(non_empty)
    unique_values = sorted(set(non_empty))
    unique_count = len(unique_values)
    sample_values = unique_values[:5]
    if not non_empty:
        return {
            "name": name,
            "inferred_type": "empty",
            "non_empty_count": 0,
            "missing_count": missing_count,
            "missing_rate": 1.0 if values else 0.0,
            "unique_count": 0,
            "sample_values": [],
        }

    int_like = all(_is_int_like(value) for value in non_empty)
    float_like = all(_is_float_like(value) for value in non_empty)
    date_like = all(_is_date_like(value) for value in non_empty)

    inferred_type = "categorical"
    numeric_summary: dict[str, Any] | None = None
    if int_like or float_like:
        inferred_type = "integer" if int_like else "float"
        numeric_values = [float(value) for value in non_empty]
        numeric_summary = {
            "min": min(numeric_values),
            "max": max(numeric_values),
            "mean": round(statistics.fmean(numeric_values), 3),
        }
    elif date_like:
        inferred_type = "date-like"
    elif unique_count <= max(12, math.ceil(len(non_empty) * 0.2)):
        inferred_type = "categorical"
    else:
        inferred_type = "free-text"

    return {
        "name": name,
        "inferred_type": inferred_type,
        "non_empty_count": len(non_empty),
        "missing_count": missing_count,
        "missing_rate": missing_count / max(len(values), 1),
        "unique_count": unique_count,
        "sample_values": sample_values,
        "numeric_summary": numeric_summary,
    }


def _build_table_profiles(columns: list[str], rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    profiles: list[dict[str, Any]] = []
    for column in columns:
        values = [str(row.get(column, "") or "") for row in rows]
        profiles.append(_infer_column_profile(column, values))
    return profiles


def _excel_column_index(ref: str) -> int:
    letters = "".join(char for char in ref if char.isalpha()).upper()
    value = 0
    for char in letters:
        value = (value * 26) + (ord(char) - 64)
    return max(value - 1, 0)


def _xlsx_read_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    try:
        root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    except Exception:
        return []
    values: list[str] = []
    for child in root:
        if _local_name(child.tag) != "si":
            continue
        text_parts: list[str] = []
        for descendant in child.iter():
            if _local_name(descendant.tag) == "t" and descendant.text:
                text_parts.append(descendant.text)
        values.append("".join(text_parts))
    return values


def _xlsx_sheet_targets(archive: zipfile.ZipFile) -> list[tuple[str, str]]:
    try:
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    except Exception:
        return []
    rel_map: dict[str, str] = {}
    for rel in rels:
        rel_id = rel.attrib.get("Id")
        target = rel.attrib.get("Target")
        if rel_id and target:
            rel_map[rel_id] = target
    targets: list[tuple[str, str]] = []
    sheets = _find_child(workbook, "sheets")
    if sheets is None:
        return []
    for sheet in list(sheets):
        if _local_name(sheet.tag) != "sheet":
            continue
        name = sheet.attrib.get("name", "Sheet")
        rel_id = sheet.attrib.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
        target = rel_map.get(rel_id or "", "")
        if not target:
            continue
        normalized = target if target.startswith("xl/") else f"xl/{target.lstrip('/')}"
        targets.append((name, normalized))
    return targets


def _xlsx_cell_text(cell: ET.Element, shared_strings: list[str]) -> str:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        parts = [desc.text or "" for desc in cell.iter() if _local_name(desc.tag) == "t"]
        return "".join(parts).strip()
    value_node = _find_child(cell, "v")
    if value_node is None or value_node.text is None:
        return ""
    raw_value = value_node.text.strip()
    if cell_type == "s":
        try:
            return shared_strings[int(raw_value)]
        except Exception:
            return raw_value
    return raw_value


def _xlsx_sheet_rows(archive: zipfile.ZipFile, target: str, shared_strings: list[str]) -> list[list[str]]:
    try:
        root = ET.fromstring(archive.read(target))
    except Exception:
        return []
    sheet_data = _find_child(root, "sheetData")
    if sheet_data is None:
        return []
    rows: list[list[str]] = []
    for row in list(sheet_data):
        if _local_name(row.tag) != "row":
            continue
        row_values: list[str] = []
        for cell in list(row):
            if _local_name(cell.tag) != "c":
                continue
            cell_ref = cell.attrib.get("r", "")
            column_index = _excel_column_index(cell_ref)
            while len(row_values) <= column_index:
                row_values.append("")
            row_values[column_index] = _xlsx_cell_text(cell, shared_strings)
        if any(value.strip() for value in row_values):
            rows.append(row_values)
    return rows


def _normalize_headers(values: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    headers: list[str] = []
    for index, value in enumerate(values):
        base = value.strip() or f"column_{index + 1}"
        count = seen.get(base, 0)
        seen[base] = count + 1
        headers.append(base if count == 0 else f"{base}_{count + 1}")
    return headers


def _sheet_domain_name(sheet_name: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", sheet_name.lower()).strip("_")
    return normalized or "sheet"


def _merge_sheet_values(existing: str, incoming: str) -> str:
    existing = str(existing or "").strip()
    incoming = str(incoming or "").strip()
    if not incoming:
        return existing
    if not existing:
        return incoming
    if incoming == existing:
        return existing
    existing_parts = [part.strip() for part in existing.split(" | ") if part.strip()]
    if incoming in existing_parts:
        return existing
    return f"{existing} | {incoming}"


def _parse_xlsx_rows(raw: bytes, suffix: str) -> tuple[list[dict[str, str]], list[str], dict[str, Any]]:
    if suffix == "xls":
        raise ValueError("Legacy .xls parsing is not supported in this scaffold. Please convert the eCRF workbook to .xlsx.")
    archive = zipfile.ZipFile(io.BytesIO(raw))
    shared_strings = _xlsx_read_shared_strings(archive)
    sheet_targets = _xlsx_sheet_targets(archive)
    if not sheet_targets and "xl/worksheets/sheet1.xml" in archive.namelist():
        sheet_targets = [("Sheet1", "xl/worksheets/sheet1.xml")]
    sheet_names = [item[0] for item in sheet_targets]
    selected_sheet = "Sheet1"
    selected_rows: list[list[str]] = []
    sheet_tables: list[dict[str, Any]] = []
    for sheet_name, target in sheet_targets:
        rows = _xlsx_sheet_rows(archive, target, shared_strings)
        if not rows:
            continue
        headers = _normalize_headers(rows[0])
        records: list[dict[str, str]] = []
        for raw_row in rows[1:]:
            padded = raw_row + [""] * max(len(headers) - len(raw_row), 0)
            records.append({headers[index]: str(padded[index] or "") for index in range(len(headers))})
        if records and not selected_rows:
            selected_sheet = sheet_name
            selected_rows = rows
        sheet_tables.append(
            {
                "sheet_name": sheet_name,
                "columns": headers,
                "rows": records,
            }
        )
    if not sheet_tables:
        return [], [], {"workbook_format": suffix, "sheet_names": sheet_names, "selected_sheet": selected_sheet, "sheet_tables": []}

    first_table = sheet_tables[0]
    return (
        first_table["rows"],
        first_table["columns"],
        {
            "workbook_format": suffix,
            "sheet_names": sheet_names,
            "selected_sheet": first_table["sheet_name"],
            "sheet_tables": sheet_tables,
        },
    )


def _parse_table_records(raw: bytes, suffix: str) -> tuple[list[dict[str, str]], list[str], dict[str, Any]]:
    if suffix in {"csv", "tsv"}:
        delimiter = "\t" if suffix == "tsv" else ","
        decoded = raw.decode("utf-8", errors="replace")
        reader = csv.DictReader(io.StringIO(decoded), delimiter=delimiter)
        rows = list(reader)
        return rows, reader.fieldnames or [], {"workbook_format": suffix}
    if suffix in {"xlsx", "xlsm", "xls"}:
        return _parse_xlsx_rows(raw, suffix)
    raise ValueError(f"Unsupported table suffix: {suffix}")


def _name_matches(name: str, patterns: tuple[str, ...]) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return any(pattern in normalized for pattern in patterns)


def _infer_table_roles(columns: list[str], profiles: list[dict[str, Any]]) -> dict[str, list[str]]:
    role_map = {
        "subject_id_columns": ("subject", "patient_id", "participant", "screening", "mrn", "person_id"),
        "visit_columns": ("visit", "timepoint", "epoch", "cycle"),
        "site_columns": ("site", "center", "hospital"),
        "arm_columns": ("arm", "group", "cohort", "treatment"),
        "date_columns": ("date", "time", "datetime"),
        "outcome_columns": ("outcome", "response", "status", "grade", "severity"),
    }
    by_name = {profile["name"]: profile for profile in profiles}
    roles: dict[str, list[str]] = {key: [] for key in role_map}
    for column in columns:
        profile = by_name.get(column, {})
        inferred = str(profile.get("inferred_type", ""))
        for role, patterns in role_map.items():
            if _name_matches(column, patterns):
                if role == "date_columns" and inferred not in {"date-like", "integer", "float", "categorical"}:
                    continue
                roles[role].append(column)
    return roles


def _classify_table_mode(
    file_name: str,
    rows: list[dict[str, str]],
    columns: list[str],
    profiles: list[dict[str, Any]],
    roles: dict[str, list[str]],
    suffix: str,
) -> dict[str, Any]:
    row_count = len(rows)
    cohort_score = 0
    single_score = 0
    rationale: list[str] = []
    subject_column = _pick_subject_preview_column(rows, roles, columns)
    subject_unique = 0
    if subject_column:
        subject_values = {str(row.get(subject_column, "") or "").strip() for row in rows if str(row.get(subject_column, "") or "").strip()}
        subject_unique = len(subject_values)
        if subject_unique > 1:
            cohort_score += 5
            rationale.append(f"`{subject_column}` has {subject_unique} unique patient/subject identifiers.")
        elif subject_unique == 1:
            single_score += 3
            rationale.append(f"`{subject_column}` carries a single patient/subject identifier.")

    if row_count > 20:
        cohort_score += 3
        rationale.append(f"Table has {row_count} rows, which is more consistent with cohort-style eCRF data.")
    elif row_count <= 3:
        single_score += 3
        rationale.append(f"Table has only {row_count} row(s), which is more consistent with a single-patient worksheet.")

    if any(roles[key] for key in ("visit_columns", "site_columns", "arm_columns")):
        cohort_score += 3
        present = [label for label in ("visit_columns", "site_columns", "arm_columns") if roles[label]]
        rationale.append(f"Detected cohort-style organizational columns: {', '.join(present)}.")

    free_text_count = sum(1 for item in profiles if item["inferred_type"] == "free-text")
    if free_text_count >= max(2, math.ceil(len(profiles) * 0.4)) and row_count <= 10:
        single_score += 2
        rationale.append("The sheet has several free-text columns with a small row count, suggesting patient-level abstraction.")

    if suffix in {"xlsx", "xlsm", "xls"}:
        cohort_score += 1
        rationale.append("Excel workbook intake slightly favors cohort/eCRF interpretation.")

    analysis_mode = "ambiguous"
    if cohort_score >= single_score + 2:
        analysis_mode = "cohort"
    elif single_score >= cohort_score + 2:
        analysis_mode = "single-patient"

    return {
        "analysis_mode": analysis_mode,
        "cohort_score": cohort_score,
        "single_patient_score": single_score,
        "subject_column": subject_column,
        "subject_unique_count": subject_unique,
        "visit_columns": roles["visit_columns"],
        "site_columns": roles["site_columns"],
        "arm_columns": roles["arm_columns"],
        "rationale": rationale,
    }


def _missingness_summary(profiles: list[dict[str, Any]]) -> dict[str, Any]:
    ranked = sorted(profiles, key=lambda item: item["missing_rate"], reverse=True)
    return {
        "top_missing_columns": [
            {
                "column": item["name"],
                "missing_rate": item["missing_rate"],
                "missing_count": item["missing_count"],
                "non_empty_count": item["non_empty_count"],
            }
            for item in ranked[:8]
        ]
    }


def _pick_subject_preview_column(rows: list[dict[str, str]], roles: dict[str, list[str]], columns: list[str] | None = None) -> str | None:
    if roles["subject_id_columns"]:
        return roles["subject_id_columns"][0]
    search_columns = columns or (list(rows[0].keys()) if rows else [])
    preferred = ("subject", "patient", "participant", "person", "mrn", "screen", "id")
    for column in search_columns:
        normalized = _normalize_name(column)
        if any(token in normalized for token in preferred):
            return column
    best_column: str | None = None
    best_unique = 0
    for column in search_columns:
        values = [str(row.get(column, "") or "").strip() for row in rows]
        non_empty = [value for value in values if value]
        if len(non_empty) < 2:
            continue
        unique_count = len(set(non_empty))
        if 1 < unique_count <= max(250, len(rows)) and unique_count > best_unique:
            best_unique = unique_count
            best_column = column
    return best_column


def _value_counts(rows: list[dict[str, str]], column: str, limit: int = 8) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(column, "") or "").strip() or "(missing)"
        counts[value] = counts.get(value, 0) + 1
    return [{"label": label, "count": count} for label, count in sorted(counts.items(), key=lambda item: item[1], reverse=True)[:limit]]


def _histogram(values: list[float], bins: int = 6) -> list[dict[str, Any]]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    if high <= low:
        return [{"label": f"{low:g}", "count": len(values)}]
    width = (high - low) / bins
    counts = [0 for _ in range(bins)]
    for value in values:
        index = min(int((value - low) / width), bins - 1)
        counts[index] += 1
    histogram: list[dict[str, Any]] = []
    for index, count in enumerate(counts):
        start = low + (index * width)
        end = low + ((index + 1) * width)
        histogram.append({"label": f"{start:.1f}-{end:.1f}", "count": count})
    return histogram


def _build_subject_preview(rows: list[dict[str, str]], roles: dict[str, list[str]]) -> list[dict[str, Any]]:
    subject_column = _pick_subject_preview_column(rows, roles)
    visit_column = roles["visit_columns"][0] if roles["visit_columns"] else None
    site_column = roles["site_columns"][0] if roles["site_columns"] else None
    arm_column = roles["arm_columns"][0] if roles["arm_columns"] else None
    outcome_column = roles["outcome_columns"][0] if roles["outcome_columns"] else None
    if not subject_column:
        return []
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        subject = str(row.get(subject_column, "") or "").strip() or "(missing)"
        grouped.setdefault(subject, []).append(row)
    preview: list[dict[str, Any]] = []
    for subject, subject_rows in list(grouped.items())[:12]:
        latest_row = subject_rows[-1]
        visits = []
        if visit_column:
            visits = sorted({str(item.get(visit_column, "") or "").strip() for item in subject_rows if str(item.get(visit_column, "") or "").strip()})
        preview.append(
            {
                "subject_id": subject,
                "record_count": len(subject_rows),
                "site": str(latest_row.get(site_column, "") or "n/a") if site_column else "n/a",
                "arm": str(latest_row.get(arm_column, "") or "n/a") if arm_column else "n/a",
                "latest_outcome": str(latest_row.get(outcome_column, "") or "n/a") if outcome_column else "n/a",
                "visits": visits,
            }
        )
    return preview


def _build_cohort_browser_artifact(
    rows: list[dict[str, str]],
    columns: list[str],
    profiles: list[dict[str, Any]],
    roles: dict[str, list[str]],
    intake: dict[str, Any],
    cohort: dict[str, Any],
    missingness: dict[str, Any],
) -> dict[str, Any]:
    subject_column = _pick_subject_preview_column(rows, roles, columns)
    visit_column = roles["visit_columns"][0] if roles["visit_columns"] else None
    site_column = roles["site_columns"][0] if roles["site_columns"] else None
    arm_column = roles["arm_columns"][0] if roles["arm_columns"] else None
    outcome_column = roles["outcome_columns"][0] if roles["outcome_columns"] else None

    subject_values = {str(row.get(subject_column, "") or "").strip() for row in rows if subject_column and str(row.get(subject_column, "") or "").strip()}
    visit_values = {str(row.get(visit_column, "") or "").strip() for row in rows if visit_column and str(row.get(visit_column, "") or "").strip()}

    age_profile = next((profile for profile in profiles if _name_matches(profile["name"], ("age",)) and profile["inferred_type"] in {"integer", "float"}), None)
    age_values: list[float] = []
    if age_profile:
        for row in rows:
            value = str(row.get(age_profile["name"], "") or "").strip()
            if _is_float_like(value):
                age_values.append(float(value))

    composition = {
        "site_distribution": _value_counts(rows, site_column, limit=10) if site_column else [],
        "arm_distribution": _value_counts(rows, arm_column, limit=10) if arm_column else [],
        "outcome_distribution": _value_counts(rows, outcome_column, limit=10) if outcome_column else [],
        "age_histogram": _histogram(age_values, bins=6),
    }

    overview = {
        "row_count": len(rows),
        "column_count": len(columns),
        "subject_count": len(subject_values) if subject_values else cohort.get("record_count", len(rows)),
        "visit_count": len(visit_values),
        "site_count": len({entry["label"] for entry in composition["site_distribution"] if entry["label"] != "(missing)"}),
        "arm_count": len({entry["label"] for entry in composition["arm_distribution"] if entry["label"] != "(missing)"}),
        "completeness_rate": round(100 * (1 - sum(item["missing_rate"] for item in profiles) / max(len(profiles), 1)), 1) if profiles else 0.0,
        "analysis_mode": intake.get("analysis_mode", "n/a"),
        "selected_sheet": (intake.get("table_meta") or {}).get("selected_sheet", "n/a"),
    }
    domains = [
        {
            "sheet_name": item.get("sheet_name", "Sheet"),
            "domain": item.get("domain", _sheet_domain_name(str(item.get("sheet_name", "Sheet")))),
            "row_count": item.get("row_count", 0),
            "subject_count": item.get("subject_count", 0),
            "subject_column": item.get("subject_column", "n/a"),
            "visit_columns": item.get("visit_columns", []),
            "date_columns": item.get("date_columns", []),
        }
        for item in ((intake.get("table_meta") or {}).get("sheet_details") or [])
    ]

    return {
        "overview": overview,
        "intake": intake,
        "composition": composition,
        "domains": domains,
        "subjects": _build_subject_preview(rows, roles),
        "grid": {
            "columns": columns,
            "rows": rows,
            "row_count": len(rows),
        },
        "schema_highlights": [
            {
                "name": item["name"],
                "inferred_type": item["inferred_type"],
                "missing_count": item["missing_count"],
                "unique_count": item["unique_count"],
            }
            for item in profiles[:8]
        ],
        "roles": roles,
        "missingness": missingness,
    }


def _cohort_summary_from_profiles(rows: list[dict[str, str]], profiles: list[dict[str, Any]]) -> dict[str, Any]:
    categorical_candidates = [item for item in profiles if item["inferred_type"] == "categorical" and item["unique_count"] > 0]
    numeric_candidates = [item for item in profiles if item["inferred_type"] in {"integer", "float"}]
    best_categorical = sorted(categorical_candidates, key=lambda item: item["unique_count"])[0:3]
    best_numeric = numeric_candidates[0:3]

    category_breakdowns = []
    for profile in best_categorical:
        counts: dict[str, int] = {}
        for row in rows:
            value = str(row.get(profile["name"], "") or "").strip() or "(missing)"
            counts[value] = counts.get(value, 0) + 1
        top_values = sorted(counts.items(), key=lambda item: item[1], reverse=True)[:5]
        category_breakdowns.append(
            {
                "column": profile["name"],
                "top_values": [{"label": label, "count": count} for label, count in top_values],
            }
        )

    numeric_breakdowns = []
    for profile in best_numeric:
        numeric_breakdowns.append(
            {
                "column": profile["name"],
                "summary": profile["numeric_summary"],
            }
        )

    return {
        "record_count": len(rows),
        "field_count": len(profiles),
        "categorical_breakdowns": category_breakdowns,
        "numeric_breakdowns": numeric_breakdowns,
    }


def _summarize_table(file_name: str, raw: bytes, suffix: str) -> IntakeSummaryResponse:
    rows, columns, table_meta = _parse_table_records(raw, suffix)
    sheet_tables = (table_meta.get("sheet_tables") or []) if isinstance(table_meta, dict) else []
    normalized_sheet_tables = list(sheet_tables)
    if suffix in {"xlsx", "xlsm", "xls"} and not normalized_sheet_tables:
        normalized_sheet_tables = [
            {
                "sheet_name": str((table_meta or {}).get("selected_sheet") or Path(file_name).stem or "Sheet1"),
                "columns": columns,
                "rows": rows,
            }
        ]
    if suffix in {"xlsx", "xlsm", "xls"} and normalized_sheet_tables:
        try:
            tool_payload = {
                "file_name": file_name,
                "suffix": suffix,
                "sheet_tables": normalized_sheet_tables,
                "table_meta": table_meta,
                "analysis_source": {
                    "file_name": file_name,
                    "file_type": suffix,
                    "modality": "clinical-table",
                    "size_bytes": len(raw),
                    "status": "parsed",
                },
            }
            tool_result = run_tool("cohort_analysis_tool", tool_payload)
            tool_payload_result = tool_result.get("result", {})
            studio_cards = tool_payload_result.get("studio_cards") or []
            artifacts = tool_payload_result.get("artifacts") or {}
            summary = str(tool_payload_result.get("summary") or "The workbook was analyzed as one or more cohort views.")
            if studio_cards and isinstance(artifacts, dict):
                return IntakeSummaryResponse(
                    source=UploadedSourceSummary(
                        file_name=file_name,
                        file_type=suffix,
                        modality="clinical-table",
                        size_bytes=len(raw),
                        status="parsed",
                    ),
                    grounded_summary=summary,
                    studio_cards=studio_cards,
                    artifacts=artifacts,
                    sources=[],
                    used_tools=["cohort_analysis_tool"],
                )
        except Exception:
            pass

        sheet_cards: list[dict[str, Any]] = []
        sheet_artifacts: dict[str, Any] = {}
        summary_lines = [
            f"This workbook contains **{len(normalized_sheet_tables)}** non-empty sheet(s), each analyzed as an individual cohort view.",
            "",
            "## Sheet overview",
            "",
        ]
        first_sheet_rows: list[dict[str, str]] = rows
        first_sheet_columns: list[str] = columns
        selected_sheet_name = table_meta.get("selected_sheet", "Sheet1")
        for sheet_index, sheet_table in enumerate(normalized_sheet_tables):
            sheet_rows = sheet_table["rows"]
            sheet_columns = sheet_table["columns"]
            sheet_name = sheet_table["sheet_name"]
            if sheet_index == 0:
                first_sheet_rows = sheet_rows
                first_sheet_columns = sheet_columns
                selected_sheet_name = sheet_name
            profiles = _build_table_profiles(sheet_columns, sheet_rows)
            roles = _infer_table_roles(sheet_columns, profiles)
            intake = _classify_table_mode(file_name, sheet_rows, sheet_columns, profiles, roles, suffix)
            cohort = _cohort_summary_from_profiles(sheet_rows, profiles)
            missingness = _missingness_summary(profiles)
            missing_cells = 0
            for row in sheet_rows:
                missing_cells += sum(1 for value in row.values() if value is None or str(value).strip() == "")
            missing_rate = (missing_cells / (max(len(sheet_rows), 1) * max(len(sheet_columns), 1))) if sheet_columns else 0.0
            sheet_table_meta = {
                "workbook_format": suffix,
                "sheet_names": table_meta.get("sheet_names", []),
                "selected_sheet": sheet_name,
                "sheet_name": sheet_name,
            }
            card_id = f"sheet::{sheet_name}::cohort_browser"
            sheet_cards.append(
                {
                    "id": card_id,
                    "title": sheet_name,
                    "subtitle": "Cohort Browser",
                    "base_id": "cohort_browser",
                }
            )
            sheet_artifacts[card_id] = _build_cohort_browser_artifact(
                sheet_rows,
                sheet_columns,
                profiles,
                roles,
                {**intake, "row_count": len(sheet_rows), "column_count": len(sheet_columns), "table_meta": sheet_table_meta},
                cohort,
                missingness,
            )
            subject_count = intake.get("subject_unique_count", "n/a")
            summary_lines.append(
                f"- **{sheet_name}**: {len(sheet_rows)} row(s), {len(sheet_columns)} column(s), mode `{intake['analysis_mode']}`, subjects {subject_count}."
            )

        selected_summary = (
            "\n".join(
                [
                    "",
                    f"**Selected sheet:** `{selected_sheet_name}`",
                    "",
                    "Ask about a specific sheet by name to inspect its cohort profile in detail.",
                ]
            )
        )
        return IntakeSummaryResponse(
            source=UploadedSourceSummary(
                file_name=file_name,
                file_type=suffix,
                modality="clinical-table",
                size_bytes=len(raw),
                status="parsed",
            ),
            grounded_summary="\n".join(summary_lines + [selected_summary]),
            studio_cards=sheet_cards,
            artifacts=sheet_artifacts,
            sources=[],
        )

    profiles = _build_table_profiles(columns, rows)
    roles = _infer_table_roles(columns, profiles)
    intake = _classify_table_mode(file_name, rows, columns, profiles, roles, suffix)
    cohort = _cohort_summary_from_profiles(rows, profiles)
    missingness = _missingness_summary(profiles)
    missing_cells = 0
    for row in rows:
        missing_cells += sum(1 for value in row.values() if value is None or str(value).strip() == "")
    missing_rate = (missing_cells / (max(len(rows), 1) * max(len(columns), 1))) if columns else 0.0

    profile_labels = ", ".join(
        f"{item['name']} ({item['inferred_type']})" for item in profiles[:6]
    )
    cohort_bits: list[str] = []
    if cohort["categorical_breakdowns"]:
        lead = cohort["categorical_breakdowns"][0]
        if lead["top_values"]:
            cohort_bits.append(
                f"{lead['column']} is a useful cohort splitter, led by {lead['top_values'][0]['label']} ({lead['top_values'][0]['count']})"
            )
    if cohort["numeric_breakdowns"]:
        lead_numeric = cohort["numeric_breakdowns"][0]
        summary = lead_numeric["summary"] or {}
        if summary:
            cohort_bits.append(
                f"{lead_numeric['column']} spans {summary.get('min')} to {summary.get('max')} with mean {summary.get('mean')}"
            )

    workbook_context = ""
    if table_meta.get("selected_sheet"):
        workbook_context = f" Selected sheet is {table_meta['selected_sheet']}."
    summary = (
        f"This clinical table contains {len(rows)} row(s) and {len(columns)} column(s). "
        f"It was classified as `{intake['analysis_mode']}` with cohort score {intake['cohort_score']} and single-patient score {intake['single_patient_score']}. "
        f"The detected schema includes {profile_labels if profile_labels else 'no readable columns'}. "
        f"Current table completeness is approximately {(1 - missing_rate) * 100:.1f}% based on non-empty cells.{workbook_context} "
        f"{' '.join(cohort_bits) if cohort_bits else ''} "
        "This first-pass summary is deterministic and should be followed by field-level validation, table-mode confirmation, and downstream cohort or patient-level review."
    )

    studio_cards = [
        {"id": "intake", "title": "Table Intake", "subtitle": "Single-patient vs cohort classification"},
        {"id": "schema", "title": "Schema Review", "subtitle": "Detected variables and types"},
    ]
    if intake["analysis_mode"] == "cohort":
        studio_cards = [
            {
                "id": "cohort_browser",
                "title": "Cohort Browser",
                "subtitle": "Intake, schema, cohort, roles, and missingness in one review",
            }
        ]
    else:
        studio_cards.extend(
            [
                {"id": "qc", "title": "Clinical QC", "subtitle": "Rows, columns, completeness"},
                {"id": "cohort", "title": "Table Summary", "subtitle": "Detected distributions in the current sheet"},
            ]
        )
    artifacts = {
        "intake": {
            **intake,
            "row_count": len(rows),
            "column_count": len(columns),
            "table_meta": table_meta,
        },
        "qc": {
            "row_count": len(rows),
            "column_count": len(columns),
            "missing_rate": missing_rate,
            "selected_sheet": table_meta.get("selected_sheet"),
            "sheet_names": table_meta.get("sheet_names", []),
        },
        "schema": {
            "columns": columns,
            "profiles": profiles,
            "sample_rows": rows[:3],
        },
        "cohort": cohort,
        "roles": roles,
        "missingness": missingness,
        "cohort_browser": _build_cohort_browser_artifact(rows, columns, profiles, roles, {**intake, "table_meta": table_meta}, cohort, missingness),
    }
    return IntakeSummaryResponse(
        source=UploadedSourceSummary(
            file_name=file_name,
            file_type=suffix,
            modality="clinical-table",
            size_bytes=len(raw),
            status="parsed",
        ),
        grounded_summary=summary,
        studio_cards=studio_cards,
        artifacts=artifacts,
        sources=[],
    )


def _summarize_dicom(file_name: str, raw: bytes, suffix: str, source_path: Optional[str] = None) -> IntakeSummaryResponse:
    meta: dict[str, Any] = {
        "patient_id": "not available",
        "study_description": "not available",
        "modality": "not available",
        "rows": "not available",
        "columns": "not available",
    }
    preview: dict[str, Any] = {"available": False, "image_data_url": None, "message": "Preview not available"}
    preview_presets: dict[str, Any] = {}
    if pydicom is not None:
        try:
            dataset = pydicom.dcmread(io.BytesIO(raw), stop_before_pixels=True, force=True)
            meta = {
                "patient_id": str(getattr(dataset, "PatientID", "not available")),
                "study_description": str(getattr(dataset, "StudyDescription", "not available")),
                "modality": str(getattr(dataset, "Modality", "not available")),
                "rows": str(getattr(dataset, "Rows", "not available")),
                "columns": str(getattr(dataset, "Columns", "not available")),
            }
        except Exception:
            pass
        preview = _build_dicom_preview(raw)
        preview_presets = _build_dicom_preview_presets(raw)

    summary = (
        f"This imaging source was recognized as a DICOM-style file. "
        f"Detected modality is {meta['modality']}, study description is {meta['study_description']}, "
        f"and matrix size is {meta['rows']} x {meta['columns']}. "
        "This first-pass summary should be followed by series-level QC, metadata validation, and visual review."
    )

    studio_cards = [
        {"id": "metadata", "title": "DICOM Review", "subtitle": "Metadata and image preview"},
    ]
    artifacts = {
        "qc": {
            "file_size_bytes": len(raw),
            "dicom_detected": True,
        },
        "metadata": {
            **meta,
            "preview": preview,
            "preview_presets": preview_presets,
            "source_file_path": source_path,
        },
        "series": {
            "note": "Series-level review will expand when multi-file studies are supported.",
            "source_file_path": source_path,
        },
    }
    return IntakeSummaryResponse(
        source=UploadedSourceSummary(
            file_name=file_name,
            file_type=suffix,
            modality="medical-image",
            size_bytes=len(raw),
            status="parsed",
        ),
        grounded_summary=summary,
        studio_cards=studio_cards,
        artifacts=artifacts,
        sources=[],
    )


def _infer_raster_modality_hint(file_name: str, suffix: str) -> tuple[str, list[str]]:
    lowered = file_name.lower()
    if any(token in lowered for token in ["cxr", "xray", "x-ray", "chest"]):
        return "chest-xray-like", ["cxr_classification_tool", "cxr_segmentation_tool"]
    if any(token in lowered for token in ["fundus", "retina", "oct"]):
        return "ophthalmic-image-like", ["fundus_analysis_tool"]
    if any(token in lowered for token in ["path", "slide", "wsi", "histo", "micro"]):
        return "pathology-or-microscopy-like", ["pathology_review_tool"]
    if suffix.lower() in {"tif", "tiff"}:
        return "pathology-or-microscopy-like", ["pathology_review_tool"]
    if any(token in lowered for token in ["ultrasound", "us", "echo"]):
        return "ultrasound-like", ["ultrasound_review_tool"]
    return "general-raster-medical-image", []


def _build_raster_preview(raw: bytes) -> dict[str, Any]:
    if Image is None:
        return {"available": False, "image_data_url": None, "message": "Preview not available"}
    try:
        with Image.open(io.BytesIO(raw)) as image:
            image.load()
            preview_image = image.copy()
            if preview_image.mode not in {"L", "RGB"}:
                preview_image = preview_image.convert("RGB")
            preview_image.thumbnail((1024, 1024))
            buffer = io.BytesIO()
            preview_image.save(buffer, format="PNG")
            encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
            return {
                "available": True,
                "image_data_url": f"data:image/png;base64,{encoded}",
                "message": "Preview generated",
            }
    except Exception as exc:
        return {
            "available": False,
            "image_data_url": None,
            "message": f"Preview not available: {exc}",
        }


def _read_raster_metadata(file_name: str, raw: bytes, suffix: str, source_path: Optional[str] = None) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "file_name": file_name,
        "file_format": suffix.upper(),
        "width": "not available",
        "height": "not available",
        "channels": "not available",
        "mode": "not available",
        "color_space": "not available",
        "source_file_path": source_path,
        "preview": {"available": False, "image_data_url": None, "message": "Preview not available"},
    }
    preview = _build_raster_preview(raw)
    metadata["preview"] = preview
    if Image is None:
        return metadata
    try:
        with Image.open(io.BytesIO(raw)) as image:
            image.load()
            bands = list(image.getbands() or [])
            metadata.update(
                {
                    "width": image.width,
                    "height": image.height,
                    "channels": len(bands) if bands else "not available",
                    "mode": image.mode,
                    "color_space": "grayscale" if image.mode == "L" else "color",
                }
            )
    except Exception:
        pass
    return metadata


def _summarize_raster_image(file_name: str, raw: bytes, suffix: str, source_path: Optional[str] = None) -> IntakeSummaryResponse:
    metadata = _read_raster_metadata(file_name, raw, suffix, source_path=source_path)
    modality_hint, next_tools = _infer_raster_modality_hint(file_name, suffix)
    width = metadata.get("width", "not available")
    height = metadata.get("height", "not available")
    color_space = metadata.get("color_space", "not available")
    summary = (
        f"This imaging source was recognized as a raster medical image ({suffix.upper()}). "
        f"Detected canvas size is {width} x {height}, color space is {color_space}, and the current modality hint is {modality_hint}. "
        "This first-pass review should be followed by domain-specific image analysis or interpretation tools."
    )
    return IntakeSummaryResponse(
        source=UploadedSourceSummary(
            file_name=file_name,
            file_type=suffix,
            modality="medical-image",
            size_bytes=len(raw),
            status="parsed",
        ),
        grounded_summary=summary,
        studio_cards=[
            {
                "id": "image_review",
                "title": "Image Review",
                "subtitle": "Raster metadata and preview",
            }
        ],
        artifacts={
            "qc": {
                "file_size_bytes": len(raw),
                "readable": bool(metadata.get("preview", {}).get("available", False)),
                "color_space": metadata.get("color_space", "not available"),
            },
            "metadata": metadata,
            "image_review": {
                "modality_hint": modality_hint,
                "next_tools": next_tools,
                "preview": metadata.get("preview"),
                "metadata": metadata,
            },
        },
        sources=[],
    )


def _summarize_raster_image_group(files: list[tuple[str, bytes, str, str]]) -> IntakeSummaryResponse:
    items = [_read_raster_metadata(file_name, raw, suffix, source_path) for file_name, raw, suffix, source_path in files]
    lead = items[0] if items else {}
    hints = [_infer_raster_modality_hint(item["file_name"], str(item["file_format"]).lower())[0] for item in items]
    next_tools: list[str] = []
    for item in items:
        _, suggested = _infer_raster_modality_hint(item["file_name"], str(item["file_format"]).lower())
        for tool_name in suggested:
            if tool_name not in next_tools:
                next_tools.append(tool_name)
    summary = (
        f"This upload includes {len(files)} raster medical image file(s). "
        f"The lead format is {lead.get('file_format', 'n/a')}, and the dominant modality hint is {hints[0] if hints else 'general-raster-medical-image'}. "
        "Use the Image Review card to inspect preview state and per-file metadata before running downstream tools."
    )
    return IntakeSummaryResponse(
        source=UploadedSourceSummary(
            file_name=f"{len(files)} raster images",
            file_type="mixed-image",
            modality="medical-image",
            size_bytes=sum(len(raw) for _, raw, _, _ in files),
            status="parsed",
        ),
        grounded_summary=summary,
        studio_cards=[
            {
                "id": "image_review",
                "title": "Image Review",
                "subtitle": "Raster metadata and preview",
            }
        ],
        artifacts={
            "qc": {
                "file_count": len(files),
                "readable_count": sum(1 for item in items if bool((item.get("preview") or {}).get("available", False))),
            },
            "metadata": {
                "items": items,
                "preview": lead.get("preview"),
            },
            "image_review": {
                "modality_hint": hints[0] if hints else "general-raster-medical-image",
                "next_tools": next_tools,
                "preview": lead.get("preview"),
                "metadata_items": items,
            },
        },
        sources=[],
    )


def _read_dicom_metadata(raw: bytes) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "patient_id": "not available",
        "study_instance_uid": "not available",
        "series_instance_uid": "not available",
        "study_description": "not available",
        "series_description": "not available",
        "modality": "not available",
        "rows": "not available",
        "columns": "not available",
        "instance_number": "not available",
        "preview": {"available": False, "image_data_url": None, "message": "Preview not available"},
        "preview_presets": {},
    }
    if pydicom is None:
        return meta
    try:
        dataset = pydicom.dcmread(io.BytesIO(raw), stop_before_pixels=True, force=True)
        meta = {
            "patient_id": str(getattr(dataset, "PatientID", "not available")),
            "study_instance_uid": str(getattr(dataset, "StudyInstanceUID", "not available")),
            "series_instance_uid": str(getattr(dataset, "SeriesInstanceUID", "not available")),
            "study_description": str(getattr(dataset, "StudyDescription", "not available")),
            "series_description": str(getattr(dataset, "SeriesDescription", "not available")),
            "modality": str(getattr(dataset, "Modality", "not available")),
            "rows": str(getattr(dataset, "Rows", "not available")),
            "columns": str(getattr(dataset, "Columns", "not available")),
            "instance_number": str(getattr(dataset, "InstanceNumber", "not available")),
            "preview": _build_dicom_preview(raw),
            "preview_presets": _build_dicom_preview_presets(raw),
        }
    except Exception:
        pass
    return meta


def _normalize_dicom_array(raw: bytes, window_width: Optional[float] = None, window_center: Optional[float] = None) -> tuple[Optional[Any], str]:
    if pydicom is None or np is None or Image is None:
        return None, "Install numpy and Pillow to render DICOM previews."

    try:
        dataset = pydicom.dcmread(io.BytesIO(raw), force=True)
        pixel_array = dataset.pixel_array
        array = np.asarray(pixel_array)

        if array.ndim == 4:
            array = array[0, 0]
        elif array.ndim == 3:
            if array.shape[-1] in (3, 4):
                pass
            else:
                array = array[0]

        if array.ndim == 2:
            array = array.astype(np.float32)
            slope = float(getattr(dataset, "RescaleSlope", 1) or 1)
            intercept = float(getattr(dataset, "RescaleIntercept", 0) or 0)
            array = array * slope + intercept

            ww = window_width
            wc = window_center
            if ww is None or wc is None:
                ww = getattr(dataset, "WindowWidth", None)
                wc = getattr(dataset, "WindowCenter", None)
                if isinstance(ww, (list, tuple)):
                    ww = ww[0]
                if isinstance(wc, (list, tuple)):
                    wc = wc[0]

            if ww is not None and wc is not None:
                ww = float(ww)
                wc = float(wc)
                lower = wc - ww / 2.0
                upper = wc + ww / 2.0
                normalized = ((array - lower) / max(upper - lower, 1e-6) * 255.0).clip(0, 255).astype(np.uint8)
            else:
                min_value = float(array.min())
                max_value = float(array.max())
                if max_value == min_value:
                    normalized = np.zeros_like(array, dtype=np.uint8)
                else:
                    normalized = ((array - min_value) / (max_value - min_value) * 255.0).clip(0, 255).astype(np.uint8)
            if str(getattr(dataset, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
                normalized = 255 - normalized
            return normalized, "Preview generated"

        if array.ndim == 3 and array.shape[-1] in (3, 4):
            normalized = array.astype(np.float32)
            min_value = float(normalized.min())
            max_value = float(normalized.max())
            if max_value != min_value:
                normalized = ((normalized - min_value) / (max_value - min_value) * 255.0).clip(0, 255)
            return normalized.astype(np.uint8), "Preview generated"

        return None, f"Unsupported DICOM pixel shape: {tuple(array.shape)}"
    except Exception as exc:
        return None, f"Preview generation failed: {exc}"


def _build_dicom_preview(raw: bytes, window_width: Optional[float] = None, window_center: Optional[float] = None) -> dict[str, Any]:
    normalized, message = _normalize_dicom_array(raw, window_width=window_width, window_center=window_center)
    if normalized is None:
        return {
            "available": False,
            "image_data_url": None,
            "message": message,
        }

    if getattr(normalized, "ndim", 0) == 2:
        image = Image.fromarray(normalized, mode="L")
    else:
        image = Image.fromarray(normalized, mode="RGB" if normalized.shape[-1] == 3 else "RGBA")

    preview_buffer = io.BytesIO()
    image.save(preview_buffer, format="PNG")
    encoded = base64.b64encode(preview_buffer.getvalue()).decode("ascii")
    return {
        "available": True,
        "image_data_url": f"data:image/png;base64,{encoded}",
        "message": message,
    }


def _build_dicom_preview_presets(raw: bytes) -> dict[str, Any]:
    previews: dict[str, Any] = {}
    for preset in WINDOW_PRESETS:
        previews[preset["id"]] = {
            "label": preset["label"],
            **_build_dicom_preview(raw, window_width=preset["width"], window_center=preset["center"]),
        }
    return previews


def _summarize_dicom_series(files: list[tuple[str, bytes, str, str]]) -> IntakeSummaryResponse:
    items = []
    for file_name, raw, suffix, source_path in files:
        meta = _read_dicom_metadata(raw)
        meta["file_name"] = file_name
        meta["size_bytes"] = len(raw)
        meta["file_type"] = suffix
        meta["source_file_path"] = source_path
        items.append(meta)

    by_series: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        key = item["series_instance_uid"]
        by_series.setdefault(key, []).append(item)

    series_rows = []
    for series_uid, series_items in by_series.items():
        first = series_items[0]
        series_rows.append(
            {
                "series_instance_uid": series_uid,
                "study_instance_uid": first["study_instance_uid"],
                "modality": first["modality"],
                "study_description": first["study_description"],
                "series_description": first["series_description"],
                "instance_count": len(series_items),
                "example_files": [item["file_name"] for item in series_items[:3]],
                "all_files": [item["file_name"] for item in series_items],
                "all_file_paths": [item["source_file_path"] for item in series_items],
                "preview": first.get("preview"),
                "preview_presets": first.get("preview_presets", {}),
            }
        )

    study_count = len({item["study_instance_uid"] for item in items})
    summary = (
        f"This imaging upload contains {len(items)} DICOM file(s), grouped into {len(series_rows)} series and {study_count} study/studies. "
        f"Detected modalities include {', '.join(sorted({item['modality'] for item in items}))}. "
        "This deterministic summary should be followed by series-level QC, orientation checks, and image viewing."
    )

    studio_cards = [
        {"id": "metadata", "title": "DICOM Review", "subtitle": "Metadata and image previews"},
    ]
    artifacts = {
        "qc": {
            "file_count": len(items),
            "series_count": len(series_rows),
            "study_count": study_count,
            "modalities": sorted({item["modality"] for item in items}),
        },
        "metadata": {
            "items": items[:12],
            "source_file_paths": [item["source_file_path"] for item in items],
        },
        "series": {
            "series": series_rows,
            "source_file_paths": [item["source_file_path"] for item in items],
        },
    }
    return IntakeSummaryResponse(
        source=UploadedSourceSummary(
            file_name=f"{len(items)} DICOM files",
            file_type="dicom",
            modality="medical-image",
            size_bytes=sum(int(item["size_bytes"]) for item in items),
            status="parsed",
        ),
        grounded_summary=summary,
        studio_cards=studio_cards,
        artifacts=artifacts,
        sources=[],
    )


def _summarize_raster_image(file_name: str, raw: bytes, suffix: str, source_path: Optional[str] = None) -> IntakeSummaryResponse:
    meta: dict[str, Any] = {
        "width": "not available",
        "height": "not available",
        "mode": "not available",
        "format": suffix.upper(),
        "file_size_bytes": len(raw),
    }
    preview: dict[str, Any] = {"available": False, "image_data_url": None}
    try:
        from PIL import Image as PILImage
        import io as _io
        img = PILImage.open(_io.BytesIO(raw))
        meta["width"] = img.width
        meta["height"] = img.height
        meta["mode"] = img.mode
        meta["format"] = img.format or suffix.upper()
        img.thumbnail((512, 512))
        buf = _io.BytesIO()
        img.save(buf, format="PNG")
        preview = {
            "available": True,
            "image_data_url": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii"),
        }
    except Exception:
        pass

    summary = (
        f"Raster image '{file_name}' received. "
        f"Format: {meta['format']}, size: {meta.get('width', '?')} x {meta.get('height', '?')} px, "
        f"color mode: {meta['mode']}. File size: {len(raw):,} bytes."
    )
    return IntakeSummaryResponse(
        source=UploadedSourceSummary(
            file_name=file_name,
            file_type=suffix,
            modality="medical-image",
            size_bytes=len(raw),
            status="parsed",
        ),
        grounded_summary=summary,
        studio_cards=[{"id": "metadata", "title": "Image Review", "subtitle": "Metadata and preview"}],
        artifacts={
            "metadata": {**meta, "source_file_path": source_path, "preview": preview},
            "qc": {"file_size_bytes": len(raw)},
        },
        sources=[],
    )



def _prefixed_response(response: IntakeSummaryResponse, source_index: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prefixed_cards: list[dict[str, Any]] = []
    prefixed_artifacts: dict[str, Any] = {}
    for card in response.studio_cards:
        prefixed_id = f"source{source_index}::{card['id']}"
        prefixed_cards.append(
            {
                **card,
                "id": prefixed_id,
                "base_id": card.get("base_id") or card["id"],
                "source_index": source_index,
                "source_name": response.source.file_name,
                "source_modality": response.source.modality,
            }
        )
        if card["id"] in response.artifacts:
            prefixed_artifacts[prefixed_id] = response.artifacts[card["id"]]
    for artifact_id, artifact_value in response.artifacts.items():
        prefixed_artifacts.setdefault(f"source{source_index}::{artifact_id}", artifact_value)
    return prefixed_cards, prefixed_artifacts


def _merge_responses(responses: list[IntakeSummaryResponse]) -> IntakeSummaryResponse:
    if not responses:
        summary = "No supported sources were parsed."
        return IntakeSummaryResponse(
            source=UploadedSourceSummary(
                file_name="No supported sources",
                file_type="unknown",
                modality="unknown",
                size_bytes=0,
                status="unsupported",
            ),
            grounded_summary=summary,
            studio_cards=[],
            artifacts={},
            sources=[],
        )

    if len(responses) == 1:
        cards, artifacts = _prefixed_response(responses[0], 0)
        summary = responses[0].grounded_summary
        return IntakeSummaryResponse(
            source=responses[0].source,
            grounded_summary=summary,
            studio_cards=cards,
            artifacts=artifacts,
            sources=[responses[0].source],
            used_tools=list(responses[0].used_tools or []),
        )

    studio_cards: list[dict[str, Any]] = []
    artifacts: dict[str, Any] = {}
    used_tools: list[str] = []
    summary_lines = ["This upload includes multiple patient-linked sources. Deterministic first-pass findings by source:"]
    for index, response in enumerate(responses):
        cards, card_artifacts = _prefixed_response(response, index)
        studio_cards.extend(cards)
        artifacts.update(card_artifacts)
        for tool_name in response.used_tools or []:
            if tool_name not in used_tools:
                used_tools.append(tool_name)
        summary_lines.append(
            f"- {response.source.file_name} ({response.source.modality}): {response.grounded_summary}"
        )

    combined_source = UploadedSourceSummary(
        file_name=f"{len(responses)} sources",
        file_type="mixed",
        modality="multi-source",
        size_bytes=sum(item.source.size_bytes for item in responses),
        status="parsed",
    )
    return IntakeSummaryResponse(
        source=combined_source,
        grounded_summary="\n".join(summary_lines),
        studio_cards=studio_cards,
        artifacts=artifacts,
        sources=[item.source for item in responses],
        used_tools=used_tools,
    )


def _source_artifact_views(analysis: IntakeSummaryResponse) -> list[dict[str, Any]]:
    views: list[dict[str, Any]] = []
    sources = analysis.sources or []
    if not sources:
        return views
    for index, source in enumerate(sources):
        prefix = f"source{index}::"
        source_artifacts = {
            key.removeprefix(prefix): value
            for key, value in analysis.artifacts.items()
            if key.startswith(prefix)
        }
        views.append(
            {
                "source_index": index,
                "source_name": source.file_name,
                "source_modality": source.modality,
                "artifacts": source_artifacts,
            }
        )
    return views


def _merged_source_artifacts(analysis: IntakeSummaryResponse, active_source_index: int | None = None) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for view in _source_artifact_views(analysis):
        merged.update(view["artifacts"])
    if isinstance(active_source_index, int):
        prefix = f"source{active_source_index}::"
        active_scoped = {
            key.removeprefix(prefix): value
            for key, value in analysis.artifacts.items()
            if key.startswith(prefix)
        }
        merged.update(active_scoped)
    return {**analysis.artifacts, **merged}


def _artifact_guided_answer(payload: ArtifactChatRequest) -> str:
    original_question = payload.question.strip()
    question = original_question.lower()
    wants_korean = _wants_korean(payload.question)
    source = payload.analysis.source
    artifacts = payload.analysis.artifacts
    active_view = payload.active_view or ""
    active_card = payload.active_card or {}
    active_artifact = payload.active_artifact or {}
    base_view = str(active_card.get("base_id") or active_view.split("::")[-1] if active_view else "")
    active_source_index = active_card.get("source_index")
    source_views = _source_artifact_views(payload.analysis)

    artifacts = _merged_source_artifacts(payload.analysis, active_source_index if isinstance(active_source_index, int) else None)
    used_tools = list(payload.analysis.used_tools or [])
    tool_result_items: list[dict[str, Any]] = []
    for artifact_key, artifact_value in payload.analysis.artifacts.items():
        if not artifact_key.startswith("tool_result::"):
            continue
        if isinstance(artifact_value, dict):
            tool_result_items.append(artifact_value)

    if active_artifact and base_view:
        artifacts = {**artifacts, base_view: active_artifact}

    cohort_browser = artifacts.get("cohort_browser") or {}
    all_cohort_browsers: list[tuple[str, dict[str, Any]]] = []
    for artifact_key, artifact_value in payload.analysis.artifacts.items():
        if not isinstance(artifact_value, dict):
            continue
        if artifact_key == "cohort_browser" or artifact_key.endswith("::cohort_browser"):
            all_cohort_browsers.append((artifact_key, artifact_value))
    if not all_cohort_browsers and isinstance(cohort_browser, dict) and cohort_browser:
        all_cohort_browsers.append(("cohort_browser", cohort_browser))
    subject_tokens = {
        token.lower()
        for token in re.findall(r"[A-Za-z0-9_.-]+", original_question)
        if len(token) >= 3
    }

    cross_cohort_matches: list[dict[str, Any]] = []
    if subject_tokens and _contains_any(
        question,
        [
            "patient",
            "patients",
            "subject",
            "subjects",
            "participant",
            "participants",
            "환자",
            "피험자",
            "대상자",
            "설명",
            "describe",
            "tell me about",
        ],
    ):
        seen_sheet_names: set[str] = set()
        for artifact_key, scoped_browser in all_cohort_browsers:
            if not isinstance(scoped_browser, dict):
                continue
            scoped_roles = scoped_browser.get("roles") or {}
            scoped_subject_columns = [str(item) for item in (scoped_roles.get("subject_id_columns") or []) if item]
            scoped_subjects = scoped_browser.get("subjects") or []
            scoped_grid = scoped_browser.get("grid") or {}
            scoped_rows = list(scoped_grid.get("rows") or []) if isinstance(scoped_grid, dict) else []
            scoped_overview = scoped_browser.get("overview") or {}
            scoped_sheet_name = str(
                scoped_overview.get("selected_sheet")
                or scoped_browser.get("sheet_name")
                or artifact_key.replace("::cohort_browser", "")
                or "current sheet"
            )
            if scoped_sheet_name in seen_sheet_names:
                continue
            seen_sheet_names.add(scoped_sheet_name)
            matched_subject_id: str | None = None
            matched_subject_preview: dict[str, Any] | None = None

            for item in scoped_subjects:
                if not isinstance(item, dict):
                    continue
                subject_id = str(item.get("subject_id") or "").strip()
                if subject_id and subject_id.lower() in subject_tokens:
                    matched_subject_id = subject_id
                    matched_subject_preview = item
                    break

            if not matched_subject_id and scoped_subject_columns:
                for row in scoped_rows:
                    if not isinstance(row, dict):
                        continue
                    for column in scoped_subject_columns:
                        value = str(row.get(column) or "").strip()
                        if value and value.lower() in subject_tokens:
                            matched_subject_id = value
                            break
                    if matched_subject_id:
                        break

            if not matched_subject_id:
                continue

            matched_rows: list[dict[str, Any]] = []
            if scoped_subject_columns:
                for row in scoped_rows:
                    if not isinstance(row, dict):
                        continue
                    row_subject = next(
                        (str(row.get(column) or "").strip() for column in scoped_subject_columns if str(row.get(column) or "").strip()),
                        "",
                    )
                    if row_subject.lower() == matched_subject_id.lower():
                        matched_rows.append(row)

            cross_cohort_matches.append(
                {
                    "sheet_name": scoped_sheet_name,
                    "subject_id": matched_subject_id,
                    "roles": scoped_roles,
                    "subject_preview": matched_subject_preview or {},
                    "rows": matched_rows,
                }
            )

    if cross_cohort_matches:
        sections: list[str] = []
        for match in cross_cohort_matches:
            matched_subject_id = str(match.get("subject_id") or "n/a")
            scoped_roles = match.get("roles") or {}
            scoped_subject_columns = [str(item) for item in (scoped_roles.get("subject_id_columns") or []) if item]
            matched_subject_preview = match.get("subject_preview") or {}
            matched_rows = match.get("rows") or []
            representative_row = matched_rows[0] if matched_rows else {}
            preferred_columns = [
                *scoped_subject_columns[:2],
                *(scoped_roles.get("visit_columns") or [])[:2],
                *(scoped_roles.get("site_columns") or [])[:2],
                *(scoped_roles.get("arm_columns") or [])[:2],
                *(scoped_roles.get("outcome_columns") or [])[:4],
            ]
            seen_columns: set[str] = set()
            key_pairs: list[str] = []
            for column in preferred_columns:
                column = str(column)
                if not column or column in seen_columns:
                    continue
                seen_columns.add(column)
                value = representative_row.get(column) if isinstance(representative_row, dict) else None
                if value not in (None, "", "nan"):
                    key_pairs.append(f"{column}={value}")
            if isinstance(representative_row, dict):
                for column, value in representative_row.items():
                    if column in seen_columns or value in (None, "", "nan"):
                        continue
                    key_pairs.append(f"{column}={value}")
                    if len(key_pairs) >= 8:
                        break

            record_count = len(matched_rows) or matched_subject_preview.get("record_count") or "n/a"
            visits = ", ".join(matched_subject_preview.get("visits") or []) or "n/a"
            site = matched_subject_preview.get("site") or "n/a"
            arm = matched_subject_preview.get("arm") or "n/a"
            latest_outcome = matched_subject_preview.get("latest_outcome") or "n/a"
            sheet_name = str(match.get("sheet_name") or "current sheet")

            if wants_korean:
                lines = [
                    f"`{sheet_name}` sheet에서 찾은 환자/subject `{matched_subject_id}` 요약입니다.",
                    f"- record 수: {record_count}",
                    f"- visits: {visits}",
                    f"- site: {site}",
                    f"- arm/group: {arm}",
                    f"- latest outcome: {latest_outcome}",
                ]
                if key_pairs:
                    lines.append("- 대표 row 정보: " + "; ".join(key_pairs[:8]))
            else:
                lines = [
                    f"Here is the summary for subject `{matched_subject_id}` in sheet `{sheet_name}`.",
                    f"- record count: {record_count}",
                    f"- visits: {visits}",
                    f"- site: {site}",
                    f"- arm/group: {arm}",
                    f"- latest outcome: {latest_outcome}",
                ]
                if key_pairs:
                    lines.append("- representative row values: " + "; ".join(key_pairs[:8]))
            sections.append("\n".join(lines))

        if len(sections) == 1:
            return sections[0]
        if wants_korean:
            return "여러 cohort sheet에서 같은 subject를 찾았습니다.\n\n" + "\n\n".join(sections)
        return "I found the same subject across multiple cohort sheets.\n\n" + "\n\n".join(sections)

    if not base_view and isinstance(cohort_browser, dict) and _contains_any(
        question,
        [
            "patient",
            "patients",
            "subject",
            "subjects",
            "participant",
            "participants",
            "cohort",
            "sheet",
            "grid",
            "환자",
            "피험자",
            "대상자",
            "코호트",
        ],
    ):
        base_view = "cohort_browser"

    if _contains_any(question, ["what is your name", "who are you", "hello", "hi"]):
        if wants_korean:
            source_hint = ""
            if payload.analysis.sources:
                source_hint = f" 현재 {len(payload.analysis.sources)}개의 업로드 source를 바탕으로 답변 중입니다."
            elif payload.analysis.source.file_name:
                source_hint = f" 현재 source는 `{payload.analysis.source.file_name}` 입니다."
            return f"제 이름은 ChatClinic 입니다.{source_hint}"
        source_hint = ""
        if payload.analysis.sources:
            source_hint = f" I am currently grounded on {len(payload.analysis.sources)} uploaded source(s)."
        elif payload.analysis.source.file_name:
            source_hint = f" The current source is `{payload.analysis.source.file_name}`."
        return f"My name is ChatClinic.{source_hint}"

    if _contains_any(
        question,
        [
            "tool status",
            "tool usage",
            "used tools",
            "which tool",
            "which tools",
            "tool log",
            "current tool",
            "현재 tool",
            "현재 툴",
            "툴 사용",
            "도구 사용",
            "사용중인 tool",
            "사용 중인 tool",
            "사용중인 툴",
            "사용 중인 툴",
        ],
    ):
        if wants_korean:
            lines = ["현재 tool 사용 현황입니다.", ""]
            if used_tools:
                lines.append("- intake/analysis 과정에서 사용된 tool:")
                lines.extend(f"  - `{tool_name}`" for tool_name in used_tools)
            if tool_result_items:
                lines.append("- 사용자가 실행한 tool 결과:")
                for item in tool_result_items:
                    tool_meta = item.get("tool") or {}
                    summary = str(item.get("summary") or "").strip() or "요약 없음"
                    lines.append(
                        f"  - `{tool_meta.get('name', 'n/a')}`"
                        f" | team: {tool_meta.get('team', 'n/a')}"
                        f" | task: {tool_meta.get('task_type', 'n/a')}"
                    )
                    lines.append(f"    - summary: {summary}")
            if not used_tools and not tool_result_items:
                lines.append("- 아직 실행되거나 기록된 tool이 없습니다.")
            return "\n".join(lines)

        lines = ["Here is the current tool usage status.", ""]
        if used_tools:
            lines.append("- Tools used during intake/analysis:")
            lines.extend(f"  - `{tool_name}`" for tool_name in used_tools)
        if tool_result_items:
            lines.append("- User-triggered tool runs:")
            for item in tool_result_items:
                tool_meta = item.get("tool") or {}
                summary = str(item.get("summary") or "").strip() or "No summary"
                lines.append(
                    f"  - `{tool_meta.get('name', 'n/a')}`"
                    f" | team: {tool_meta.get('team', 'n/a')}"
                    f" | task: {tool_meta.get('task_type', 'n/a')}"
                )
                lines.append(f"    - summary: {summary}")
        if not used_tools and not tool_result_items:
            lines.append("- No tools have been executed or recorded yet.")
        return "\n".join(lines)

    if isinstance(cohort_browser, dict):
        if "intake" in cohort_browser and "intake" not in artifacts:
            artifacts["intake"] = cohort_browser.get("intake")
        if "overview" in cohort_browser and "cohort" not in artifacts:
            overview = cohort_browser.get("overview") or {}
            composition = cohort_browser.get("composition") or {}
            artifacts["cohort"] = {
                "record_count": overview.get("row_count"),
                "field_count": overview.get("column_count"),
                "subject_count": overview.get("subject_count"),
                "visit_count": overview.get("visit_count"),
                "site_count": overview.get("site_count"),
                "arm_count": overview.get("arm_count"),
                "completeness_rate": overview.get("completeness_rate"),
                "categorical_breakdowns": composition.get("site_distribution", []) + composition.get("arm_distribution", []) + composition.get("outcome_distribution", []),
                "numeric_breakdowns": composition.get("numeric_breakdowns", []),
                "age_histogram": composition.get("age_histogram", []),
            }
        if "schema_highlights" in cohort_browser and "schema" not in artifacts:
            artifacts["schema"] = {"profiles": cohort_browser.get("schema_highlights") or []}
        if "roles" in cohort_browser and "roles" not in artifacts:
            artifacts["roles"] = cohort_browser.get("roles")
        if "missingness" in cohort_browser and "missingness" not in artifacts:
            artifacts["missingness"] = cohort_browser.get("missingness")
        if "subjects" in cohort_browser and "subjects" not in artifacts:
            artifacts["subjects"] = cohort_browser.get("subjects")
        if "grid" in cohort_browser and "grid" not in artifacts:
            artifacts["grid"] = cohort_browser.get("grid")
        if "domains" in cohort_browser and "domains" not in artifacts:
            artifacts["domains"] = cohort_browser.get("domains")

    if base_view == "cohort_browser" and wants_korean and question.strip() in {"한국어로", "한글로", "korean", "in korean"}:
        overview = (cohort_browser if isinstance(cohort_browser, dict) else {}).get("overview") or {}
        intake = (cohort_browser if isinstance(cohort_browser, dict) else {}).get("intake") or {}
        composition = (cohort_browser if isinstance(cohort_browser, dict) else {}).get("composition") or {}
        roles = (cohort_browser if isinstance(cohort_browser, dict) else {}).get("roles") or {}
        missingness = (cohort_browser if isinstance(cohort_browser, dict) else {}).get("missingness") or {}
        domains = (cohort_browser if isinstance(cohort_browser, dict) else {}).get("domains") or []
        subjects = (cohort_browser if isinstance(cohort_browser, dict) else {}).get("subjects") or []

        lines = [
            "현재 Cohort Browser 요약입니다.",
            "",
            f"- 분석 모드: {intake.get('analysis_mode', 'cohort')}",
            f"- 레코드 수: {overview.get('row_count', intake.get('row_count', 'n/a'))}",
            f"- 변수 수: {overview.get('column_count', 'n/a')}",
            f"- subject 수: {overview.get('subject_count', intake.get('subject_unique_count', 'n/a'))}",
            f"- visit 수: {overview.get('visit_count', 'n/a')}",
            f"- site 수: {overview.get('site_count', 'n/a')}",
            f"- arm 수: {overview.get('arm_count', 'n/a')}",
            f"- completeness: {overview.get('completeness_rate', 'n/a')}%",
        ]
        subject_columns = roles.get("subject_id_columns") or []
        if subject_columns:
            lines.append(f"- subject ID 컬럼: {', '.join(subject_columns[:4])}")
        domain_names = [str(item.get("sheet_name", "n/a")) for item in domains[:8] if isinstance(item, dict)]
        if domain_names:
            lines.append(f"- workbook sheet/cohort: {', '.join(domain_names)}")
        site_distribution = composition.get("site_distribution") or []
        if site_distribution:
            top_sites = ", ".join(
                f"{item.get('label', 'n/a')} ({item.get('count', 'n/a')})" for item in site_distribution[:4]
            )
            lines.append(f"- site 분포: {top_sites}")
        arm_distribution = composition.get("arm_distribution") or []
        if arm_distribution:
            top_arms = ", ".join(
                f"{item.get('label', 'n/a')} ({item.get('count', 'n/a')})" for item in arm_distribution[:4]
            )
            lines.append(f"- arm 분포: {top_arms}")
        missing_columns = missingness.get("top_missing_columns") or []
        if missing_columns:
            top_missing = ", ".join(
                f"{item.get('column', 'n/a')} ({round(float(item.get('missing_rate', 0)) * 100, 1)}%)"
                for item in missing_columns[:4]
            )
            lines.append(f"- 결측 상위 컬럼: {top_missing}")
        if subjects:
            preview = ", ".join(str(item.get("subject_id", "n/a")) for item in subjects[:6] if isinstance(item, dict))
            if preview:
                lines.append(f"- subject preview: {preview}")
        lines.extend(
            [
                "",
                "원하시면 다음도 이어서 설명할 수 있습니다.",
                "- schema/변수 타입",
                "- 각 sheet cohort 비교",
                "- 결측 패턴",
                "- subject grid 탐색",
            ]
        )
        return "\n".join(lines)

    if base_view == "cohort_browser":
        overview = (cohort_browser if isinstance(cohort_browser, dict) else {}).get("overview") or {}
        intake = (cohort_browser if isinstance(cohort_browser, dict) else {}).get("intake") or {}
        composition = (cohort_browser if isinstance(cohort_browser, dict) else {}).get("composition") or {}
        roles = (cohort_browser if isinstance(cohort_browser, dict) else {}).get("roles") or {}
        missingness = (cohort_browser if isinstance(cohort_browser, dict) else {}).get("missingness") or {}
        domains = (cohort_browser if isinstance(cohort_browser, dict) else {}).get("domains") or []
        subjects = (cohort_browser if isinstance(cohort_browser, dict) else {}).get("subjects") or []
        grid = (cohort_browser if isinstance(cohort_browser, dict) else {}).get("grid") or {}
        sheet_name = overview.get("selected_sheet") or active_card.get("title") or "current sheet"
        subject_columns = [str(item) for item in (roles.get("subject_id_columns") or []) if item]
        grid_rows = list(grid.get("rows") or []) if isinstance(grid, dict) else []

        matched_subject_id: str | None = None
        matched_subject_preview: dict[str, Any] | None = None
        for item in subjects:
            if not isinstance(item, dict):
                continue
            subject_id = str(item.get("subject_id") or "").strip()
            if subject_id and subject_id.lower() in subject_tokens:
                matched_subject_id = subject_id
                matched_subject_preview = item
                break

        matched_rows: list[dict[str, Any]] = []
        if subject_columns and not matched_subject_id and subject_tokens:
            for row in grid_rows:
                if not isinstance(row, dict):
                    continue
                for column in subject_columns:
                    value = str(row.get(column) or "").strip()
                    if value and value.lower() in subject_tokens:
                        matched_subject_id = value
                        break
                if matched_subject_id:
                    break

        if matched_subject_id and subject_columns:
            for row in grid_rows:
                if not isinstance(row, dict):
                    continue
                row_subject = next(
                    (str(row.get(column) or "").strip() for column in subject_columns if str(row.get(column) or "").strip()),
                    "",
                )
                if row_subject.lower() == matched_subject_id.lower():
                    matched_rows.append(row)

        if matched_subject_id:
            representative_row = matched_rows[0] if matched_rows else {}
            preferred_columns = [
                *subject_columns[:2],
                *(roles.get("visit_columns") or [])[:2],
                *(roles.get("site_columns") or [])[:2],
                *(roles.get("arm_columns") or [])[:2],
                *(roles.get("outcome_columns") or [])[:4],
            ]
            seen_columns: set[str] = set()
            key_pairs: list[str] = []
            for column in preferred_columns:
                column = str(column)
                if not column or column in seen_columns:
                    continue
                seen_columns.add(column)
                value = representative_row.get(column) if isinstance(representative_row, dict) else None
                if value not in (None, "", "nan"):
                    key_pairs.append(f"{column}={value}")
            if isinstance(representative_row, dict):
                for column, value in representative_row.items():
                    if column in seen_columns or value in (None, "", "nan"):
                        continue
                    key_pairs.append(f"{column}={value}")
                    if len(key_pairs) >= 8:
                        break

            record_count = len(matched_rows) or (matched_subject_preview or {}).get("record_count") or "n/a"
            visits = ", ".join((matched_subject_preview or {}).get("visits") or []) or "n/a"
            latest_outcome = (matched_subject_preview or {}).get("latest_outcome") or "n/a"
            site = (matched_subject_preview or {}).get("site") or "n/a"
            arm = (matched_subject_preview or {}).get("arm") or "n/a"

            if wants_korean:
                lines = [
                    f"`{sheet_name}` sheet에서 찾은 환자/subject `{matched_subject_id}` 요약입니다.",
                    "",
                    f"- record 수: {record_count}",
                    f"- visits: {visits}",
                    f"- site: {site}",
                    f"- arm/group: {arm}",
                    f"- latest outcome: {latest_outcome}",
                ]
                if key_pairs:
                    lines.append("- 대표 row 정보: " + "; ".join(key_pairs[:8]))
                return "\n".join(lines)

            lines = [
                f"Here is the summary for subject `{matched_subject_id}` in sheet `{sheet_name}`.",
                "",
                f"- record count: {record_count}",
                f"- visits: {visits}",
                f"- site: {site}",
                f"- arm/group: {arm}",
                f"- latest outcome: {latest_outcome}",
            ]
            if key_pairs:
                lines.append("- representative row values: " + "; ".join(key_pairs[:8]))
            return "\n".join(lines)

        if _contains_any(
            question,
            [
                "patient",
                "patients",
                "subject",
                "subjects",
                "participant",
                "participants",
                "cohort browser",
                "subject explorer",
                "grid",
                "sheet",
                "환자",
                "피험자",
                "대상자",
                "코호트",
            ],
        ):
            lines: list[str] = []
            if wants_korean:
                lines.append(f"현재 선택된 cohort sheet는 `{sheet_name}` 입니다.")
                lines.append("")
                lines.append(f"- 레코드 수: {overview.get('row_count', intake.get('row_count', 'n/a'))}")
                lines.append(f"- subject 수: {overview.get('subject_count', intake.get('subject_unique_count', 'n/a'))}")
                lines.append(f"- subject ID 컬럼: {', '.join(subject_columns[:4]) or 'n/a'}")
                if subjects:
                    preview = "; ".join(
                        f"{item.get('subject_id', 'n/a')} | records {item.get('record_count', 'n/a')} | visits {', '.join(item.get('visits', [])[:4]) or 'n/a'}"
                        for item in subjects[:8]
                        if isinstance(item, dict)
                    )
                    lines.append(f"- subject preview: {preview}")
                if isinstance(grid, dict):
                    lines.append(
                        f"- grid: {len(grid.get('columns') or [])} columns, {grid.get('row_count', len(grid.get('rows') or []))} rows"
                    )
                if domains:
                    domain_text = ", ".join(
                        str(item.get("sheet_name", "n/a")) for item in domains[:8] if isinstance(item, dict)
                    )
                    lines.append(f"- workbook 내 다른 cohort sheet: {domain_text}")
                return "\n".join(lines)

            lines.append(f"The currently selected cohort sheet is `{sheet_name}`.")
            lines.append("")
            lines.append(f"- records: {overview.get('row_count', intake.get('row_count', 'n/a'))}")
            lines.append(f"- subjects: {overview.get('subject_count', intake.get('subject_unique_count', 'n/a'))}")
            lines.append(f"- subject ID columns: {', '.join(subject_columns[:4]) or 'n/a'}")
            if subjects:
                preview = "; ".join(
                    f"{item.get('subject_id', 'n/a')} | records {item.get('record_count', 'n/a')} | visits {', '.join(item.get('visits', [])[:4]) or 'n/a'}"
                    for item in subjects[:8]
                    if isinstance(item, dict)
                )
                lines.append(f"- subject preview: {preview}")
            if isinstance(grid, dict):
                lines.append(
                    f"- grid: {len(grid.get('columns') or [])} columns, {grid.get('row_count', len(grid.get('rows') or []))} rows"
                )
            if domains:
                domain_text = ", ".join(
                    str(item.get("sheet_name", "n/a")) for item in domains[:8] if isinstance(item, dict)
                )
                lines.append(f"- other cohort sheets in workbook: {domain_text}")
            return "\n".join(lines)

        if _contains_any(
            question,
            [
                "site",
                "arm",
                "outcome",
                "distribution",
                "visit",
                "missing",
                "schema",
                "column",
                "변수",
                "분포",
                "결측",
                "방문",
            ],
        ):
            lines: list[str] = []
            if wants_korean:
                lines.append(f"`{sheet_name}` cohort 분석 요약:")
                lines.append("")
                lines.append(f"- completeness: {overview.get('completeness_rate', 'n/a')}%")
                site_distribution = composition.get("site_distribution") or []
                arm_distribution = composition.get("arm_distribution") or []
                outcome_distribution = composition.get("outcome_distribution") or []
                if site_distribution:
                    lines.append(
                        "- site 분포: "
                        + ", ".join(f"{item.get('label', 'n/a')} ({item.get('count', 'n/a')})" for item in site_distribution[:4])
                    )
                if arm_distribution:
                    lines.append(
                        "- arm 분포: "
                        + ", ".join(f"{item.get('label', 'n/a')} ({item.get('count', 'n/a')})" for item in arm_distribution[:4])
                    )
                if outcome_distribution:
                    lines.append(
                        "- outcome 분포: "
                        + ", ".join(f"{item.get('label', 'n/a')} ({item.get('count', 'n/a')})" for item in outcome_distribution[:4])
                    )
                missing_columns = missingness.get("top_missing_columns") or []
                if missing_columns:
                    lines.append(
                        "- 결측 상위 컬럼: "
                        + ", ".join(
                            f"{item.get('column', 'n/a')} ({round(float(item.get('missing_rate', 0)) * 100, 1)}%)"
                            for item in missing_columns[:4]
                        )
                    )
                schema_highlights = (cohort_browser.get("schema_highlights") if isinstance(cohort_browser, dict) else []) or []
                if schema_highlights:
                    lines.append(
                        "- schema preview: "
                        + ", ".join(f"{item.get('name', 'n/a')}:{item.get('inferred_type', 'n/a')}" for item in schema_highlights[:6])
                    )
                return "\n".join(lines)

            lines.append(f"Cohort analysis summary for `{sheet_name}`:")
            lines.append("")
            lines.append(f"- completeness: {overview.get('completeness_rate', 'n/a')}%")
            site_distribution = composition.get("site_distribution") or []
            arm_distribution = composition.get("arm_distribution") or []
            outcome_distribution = composition.get("outcome_distribution") or []
            if site_distribution:
                lines.append(
                    "- site distribution: "
                    + ", ".join(f"{item.get('label', 'n/a')} ({item.get('count', 'n/a')})" for item in site_distribution[:4])
                )
            if arm_distribution:
                lines.append(
                    "- arm distribution: "
                    + ", ".join(f"{item.get('label', 'n/a')} ({item.get('count', 'n/a')})" for item in arm_distribution[:4])
                )
            if outcome_distribution:
                lines.append(
                    "- outcome distribution: "
                    + ", ".join(f"{item.get('label', 'n/a')} ({item.get('count', 'n/a')})" for item in outcome_distribution[:4])
                )
            missing_columns = missingness.get("top_missing_columns") or []
            if missing_columns:
                lines.append(
                    "- top missing columns: "
                    + ", ".join(
                        f"{item.get('column', 'n/a')} ({round(float(item.get('missing_rate', 0)) * 100, 1)}%)"
                        for item in missing_columns[:4]
                    )
                )
            schema_highlights = (cohort_browser.get("schema_highlights") if isinstance(cohort_browser, dict) else []) or []
            if schema_highlights:
                lines.append(
                    "- schema preview: "
                    + ", ".join(f"{item.get('name', 'n/a')}:{item.get('inferred_type', 'n/a')}" for item in schema_highlights[:6])
                )
            return "\n".join(lines)

    if base_view == "schema" and _is_generic_explanation_request(question, original_question):
        question = "schema " + question
    if base_view == "intake" and _is_generic_explanation_request(question, original_question):
        question = "intake " + question
    if base_view == "cohort_browser" and _is_generic_explanation_request(question, original_question):
        question = "intake schema cohort roles missingness " + question
    if base_view == "cohort" and _is_generic_explanation_request(question, original_question):
        question = "cohort " + question
    if base_view == "roles" and _is_generic_explanation_request(question, original_question):
        question = "roles " + question
    if base_view == "missingness" and _is_generic_explanation_request(question, original_question):
        question = "missingness " + question
    if base_view == "metadata" and _is_generic_explanation_request(question, original_question):
        question = "metadata " + question
    if base_view == "qc" and _is_generic_explanation_request(question, original_question):
        question = "qc " + question
    if base_view == "series" and _is_generic_explanation_request(question, original_question):
        question = "series " + question
    if base_view == "message" and _is_generic_explanation_request(question, original_question):
        question = "message " + question
    if base_view == "patient" and _is_generic_explanation_request(question, original_question):
        question = "patient " + question
    if base_view == "observations" and _is_generic_explanation_request(question, original_question):
        question = "observation " + question
    if base_view == "medications" and _is_generic_explanation_request(question, original_question):
        question = "medication " + question
    if base_view == "fhir_browser" and _is_generic_explanation_request(question, original_question):
        question = "patient observation medication " + question
    if base_view == "resources" and _is_generic_explanation_request(question, original_question):
        question = "resource " + question
    if base_view == "note" and _is_generic_explanation_request(question, original_question):
        question = "note " + question
    if wants_korean and question.strip() in {"한국어로", "한글로", "korean", "in korean"} and base_view:
        question = f"{base_view} " + question
    if not base_view and wants_korean and question.strip() in {"한국어로", "한글로", "korean", "in korean"}:
        return _korean_analysis_summary(payload.analysis)

    if base_view and not _contains_any(
        question,
        [
            "schema",
            "intake",
            "column",
            "cohort",
            "distribution",
            "role",
            "subject",
            "visit",
            "missing",
            "series",
            "study",
            "dicom",
            "metadata",
            "modality",
            "fhir",
            "hl7",
            "message",
            "fhir_browser",
            "patient",
            "observation",
            "medication",
            "demographic",
            "identifier",
            "contact",
            "resource",
            "note",
            "clinical note",
            "qc",
            "quality",
        ],
    ) and _is_generic_explanation_request(question, original_question):
        question = f"{base_view} {question}"

    if "intake" in question or "scope" in question or "cohort or patient" in question or "single patient" in question:
        intake = artifacts.get("intake") or {}
        rationale = (intake.get("rationale") or [])[:5]
        lines = [
            f"- mode: {intake.get('analysis_mode', 'n/a')}",
            f"- cohort_score: {intake.get('cohort_score', 'n/a')}",
            f"- single_patient_score: {intake.get('single_patient_score', 'n/a')}",
            f"- subject_column: {intake.get('subject_column', 'n/a')}",
            f"- unique_subjects: {intake.get('subject_unique_count', 'n/a')}",
        ]
        lines.extend(f"- rationale: {item}" for item in rationale)
        return ("현재 table intake classification:\n\n" if wants_korean else "Table intake classification:\n\n") + "\n".join(lines)

    if "schema" in question or "column" in question or "컬럼" in payload.question or "변수" in payload.question:
        profiles = (artifacts.get("schema") or {}).get("profiles", [])
        if not profiles:
            return "현재 source에 대해 사용할 수 있는 schema profile이 없습니다." if wants_korean else "No schema profile is available for the current source."
        lines = [
            f"- {item['name']}: {item['inferred_type']} | missing {item['missing_count']} | unique {item['unique_count']} | sample {', '.join(item['sample_values']) if item['sample_values'] else 'n/a'}"
            for item in profiles[:8]
        ]
        return ("현재 source의 schema review:\n\n" if wants_korean else "Schema review of the current source:\n\n") + "\n".join(lines)

    if "cohort" in question or "distribution" in question or "요약" in payload.question or "분포" in payload.question:
        cohort = artifacts.get("cohort") or {}
        lines = [
            f"- records: {cohort.get('record_count', 'n/a')}",
            f"- fields: {cohort.get('field_count', 'n/a')}",
        ]
        for item in (cohort.get("categorical_breakdowns") or [])[:3]:
            values = ", ".join(f"{entry['label']} ({entry['count']})" for entry in item.get("top_values", [])[:4])
            lines.append(f"- {item.get('column')}: {values}")
        for item in (cohort.get("numeric_breakdowns") or [])[:2]:
            summary = item.get("summary") or {}
            lines.append(
                f"- {item.get('column')}: min {summary.get('min', 'n/a')}, max {summary.get('max', 'n/a')}, mean {summary.get('mean', 'n/a')}"
            )
        return ("현재 source의 cohort summary:\n\n" if wants_korean else "Cohort summary of the current source:\n\n") + "\n".join(lines)

    if "roles" in question or "subject" in question or "visit" in question or "arm" in question or "site" in question:
        roles = artifacts.get("roles") or {}
        lines = [
            f"- subject_id_columns: {', '.join(roles.get('subject_id_columns', [])) or 'n/a'}",
            f"- visit_columns: {', '.join(roles.get('visit_columns', [])) or 'n/a'}",
            f"- site_columns: {', '.join(roles.get('site_columns', [])) or 'n/a'}",
            f"- arm_columns: {', '.join(roles.get('arm_columns', [])) or 'n/a'}",
            f"- date_columns: {', '.join(roles.get('date_columns', [])) or 'n/a'}",
            f"- outcome_columns: {', '.join(roles.get('outcome_columns', [])) or 'n/a'}",
        ]
        return ("현재 source의 variable roles:\n\n" if wants_korean else "Variable roles of the current source:\n\n") + "\n".join(lines)

    if "subject" in question or "patient list" in question or "grid" in question or "cohort browser" in question or "subject explorer" in question:
        subjects = artifacts.get("subjects") or []
        grid = artifacts.get("grid") or {}
        domains = artifacts.get("domains") or []
        lines = []
        if subjects:
            preview = "; ".join(
                f"{item.get('subject_id', 'n/a')} | records {item.get('record_count', 'n/a')} | visits {', '.join(item.get('visits', [])[:4]) or 'n/a'}"
                for item in subjects[:6]
            )
            lines.append(f"- subject preview: {preview}")
        if isinstance(grid, dict):
            lines.append(
                f"- sheet grid: {len(grid.get('columns') or [])} columns, {grid.get('row_count', len(grid.get('rows') or []))} rows"
            )
        if domains:
            domain_text = "; ".join(
                f"{item.get('sheet_name', 'n/a')} ({item.get('row_count', 'n/a')} rows, subject column {item.get('subject_column', 'n/a')})"
                for item in domains[:8]
            )
            lines.append(f"- workbook domains: {domain_text}")
        if not lines:
            return (
                "현재 source에 대해 사용할 수 있는 cohort subject explorer가 없습니다."
                if wants_korean
                else "No cohort subject explorer is available for the current source."
            )
        return ("현재 source의 cohort subject explorer:\n\n" if wants_korean else "Cohort subject explorer of the current source:\n\n") + "\n".join(lines)

    if "missingness" in question or "missing" in question or "결측" in payload.question:
        missingness = artifacts.get("missingness") or {}
        rows = missingness.get("top_missing_columns") or []
        if not rows:
            return "현재 source에 대해 결측 요약이 없습니다." if wants_korean else "No missingness summary is available for the current source."
        lines = [
            f"- {item.get('column')}: missing {round(float(item.get('missing_rate', 0)) * 100, 1)}% ({item.get('missing_count', 'n/a')}) | non-empty {item.get('non_empty_count', 'n/a')}"
            for item in rows[:8]
        ]
        return ("현재 source의 missingness summary:\n\n" if wants_korean else "Missingness summary of the current source:\n\n") + "\n".join(lines)

    if "series" in question or "study" in question or "시리즈" in payload.question:
        series_payload = artifacts.get("series") or {}
        series_rows = series_payload.get("series") or []
        if not series_rows:
            return "현재 source에 대해 그룹화된 imaging series가 없습니다." if wants_korean else "No grouped imaging series are available for the current source."
        lines = [
            f"- {item.get('series_description') or 'unnamed series'} | modality {item.get('modality', 'n/a')} | instances {item.get('instance_count', 'n/a')} | study {item.get('study_description', 'n/a')}"
            for item in series_rows[:6]
        ]
        if wants_korean:
            lines = [
                f"- {(item.get('series_description') or '이름 없는 시리즈')} | modality {item.get('modality', 'n/a')} | 인스턴스 {item.get('instance_count', 'n/a')}개 | study {item.get('study_description', 'n/a')}"
                for item in series_rows[:6]
            ]
            return "현재 imaging upload의 series review:\n\n" + "\n".join(lines)
        return "Series review of the current imaging upload:\n\n" + "\n".join(lines)

    if "dicom" in question or "metadata" in question or "modality" in question or "영상" in payload.question:
        metadata = artifacts.get("metadata") or {}
        if not metadata:
            return "현재 source에 대해 사용할 수 있는 imaging metadata가 없습니다." if wants_korean else "No imaging metadata is available for the current source."
        if "items" in metadata:
            first = (metadata.get("items") or [{}])[0]
            if wants_korean:
                return (
                    "영상 메타데이터 요약:\n\n"
                    f"- sample file: {first.get('file_name', 'n/a')}\n"
                    f"- modality: {first.get('modality', 'n/a')}\n"
                    f"- patient_id: {first.get('patient_id', 'n/a')}\n"
                    f"- study_description: {first.get('study_description', 'n/a')}\n"
                    f"- series_description: {first.get('series_description', 'n/a')}\n"
                    f"- matrix: {first.get('rows', 'n/a')} x {first.get('columns', 'n/a')}"
                )
            return (
                "Imaging metadata summary:\n\n"
                f"- sample file: {first.get('file_name', 'n/a')}\n"
                f"- modality: {first.get('modality', 'n/a')}\n"
                f"- patient_id: {first.get('patient_id', 'n/a')}\n"
                f"- study_description: {first.get('study_description', 'n/a')}\n"
                f"- series_description: {first.get('series_description', 'n/a')}\n"
                f"- matrix: {first.get('rows', 'n/a')} x {first.get('columns', 'n/a')}"
            )
        if wants_korean:
            return (
                "영상 메타데이터 요약:\n\n"
                f"- modality: {metadata.get('modality', 'n/a')}\n"
                f"- patient_id: {metadata.get('patient_id', 'n/a')}\n"
                f"- study_description: {metadata.get('study_description', 'n/a')}\n"
                f"- rows: {metadata.get('rows', 'n/a')}\n"
                f"- columns: {metadata.get('columns', 'n/a')}"
            )
        return (
            "Imaging metadata summary:\n\n"
            f"- modality: {metadata.get('modality', 'n/a')}\n"
            f"- patient_id: {metadata.get('patient_id', 'n/a')}\n"
            f"- study_description: {metadata.get('study_description', 'n/a')}\n"
            f"- rows: {metadata.get('rows', 'n/a')}\n"
            f"- columns: {metadata.get('columns', 'n/a')}"
        )

    if "fhir" in question or "hl7" in question or "message" in question or "resource" in question or "세그먼트" in payload.question:
        message = artifacts.get("message") or {}
        resources = artifacts.get("resources") or {}
        if not message:
            return "현재 source에 대해 사용할 수 있는 HL7/FHIR 메타데이터가 없습니다." if wants_korean else "No HL7/FHIR message metadata is available for the current source."
        lines = [f"- format: {message.get('format', 'n/a')}"]
        for key in ("resource_type", "id", "message_type", "control_id", "version", "patient_id", "patient_name", "visit_class"):
            if key in message:
                lines.append(f"- {key}: {message.get(key, 'n/a')}")
        if "segment_counts" in resources:
            counts = resources.get("segment_counts") or {}
            lines.append(f"- segment inventory: {', '.join(f'{k}={v}' for k, v in counts.items())}")
        if "top_level_keys" in resources:
            lines.append(f"- top-level keys: {', '.join((resources.get('top_level_keys') or [])[:10])}")
        if wants_korean:
            translated_lines = []
            for line in lines:
                translated_lines.append(
                    line.replace("format", "형식")
                    .replace("resource_type", "resource_type")
                    .replace("message_type", "message_type")
                    .replace("control_id", "control_id")
                    .replace("version", "version")
                    .replace("patient_id", "patient_id")
                    .replace("patient_name", "patient_name")
                    .replace("visit_class", "visit_class")
                    .replace("segment inventory", "segment inventory")
                    .replace("top-level keys", "top-level keys")
                )
            return "현재 source의 clinical message review:\n\n" + "\n".join(translated_lines)
        return "Clinical message review of the current source:\n\n" + "\n".join(lines)

    if "patient" in question or "demographic" in question or "identifier" in question or "contact" in question or "환자" in payload.question:
        patient = artifacts.get("patient") or {}
        if not patient:
            return "현재 source에 대해 사용할 수 있는 patient browser artifact가 없습니다." if wants_korean else "No FHIR patient browser artifact is available for the current source."
        lines = [
            f"- resource_type: {patient.get('resource_type', 'n/a')}",
            f"- id: {patient.get('id', 'n/a')}",
            f"- full_name: {patient.get('full_name', 'n/a')}",
            f"- gender: {patient.get('gender', 'n/a')}",
            f"- birth_date: {patient.get('birth_date', 'n/a')}",
            f"- active: {patient.get('active', 'n/a')}",
            f"- managing_organization: {patient.get('managing_organization', 'n/a')}",
        ]
        identifiers = patient.get("identifiers") or []
        telecom = patient.get("telecom") or []
        addresses = patient.get("addresses") or []
        allergies = (artifacts.get("allergies") or {}).get("items") or []
        vitals = (artifacts.get("vitals") or {}).get("items") or []
        if identifiers:
            lines.append("- identifiers: " + "; ".join(f"{item.get('system', 'n/a')}={item.get('value', 'n/a')}" for item in identifiers[:4]))
        if telecom:
            lines.append("- telecom: " + "; ".join(f"{item.get('system', 'n/a')}={item.get('value', 'n/a')}" for item in telecom[:4]))
        if addresses:
            lines.append("- addresses: " + "; ".join(f"{item.get('line', 'n/a')} / {item.get('city', 'n/a')}" for item in addresses[:3]))
        if allergies:
            lines.append("- allergies: " + "; ".join(f"{item.get('substance', 'n/a')} ({item.get('criticality', 'n/a')})" for item in allergies[:4]))
        if vitals:
            lines.append("- latest vitals: " + "; ".join(f"{item.get('label', 'n/a')}={item.get('value', 'n/a')}" for item in vitals[:4]))
        if wants_korean:
            translated = []
            for line in lines:
                translated.append(
                    line.replace("resource_type", "resource_type")
                    .replace("full_name", "full_name")
                    .replace("birth_date", "birth_date")
                    .replace("managing_organization", "managing_organization")
                    .replace("identifiers", "identifiers")
                    .replace("telecom", "telecom")
                    .replace("addresses", "addresses")
                )
            return "현재 source의 FHIR patient browser:\n\n" + "\n".join(translated)
        return "FHIR patient browser of the current source:\n\n" + "\n".join(lines)

    if "observation" in question or "vital" in question or "status change" in question or "검사" in payload.question or "관찰" in payload.question:
        observations = artifacts.get("observations") or {}
        items = observations.get("items") or []
        if not items:
            return "현재 source에 대해 사용할 수 있는 Observation artifact가 없습니다." if wants_korean else "No Observation artifact is available for the current source."
        lines = [
            f"- {item.get('code', 'n/a')} | value {item.get('value', 'n/a')} | status {item.get('status', 'n/a')} | effective {item.get('effective', 'n/a')}"
            for item in items[:8]
        ]
        if wants_korean:
            return "현재 source의 Observation Viewer:\n\n" + "\n".join(lines)
        return "Observation Viewer of the current source:\n\n" + "\n".join(lines)

    if "medication" in question or "prescription" in question or "drug" in question or "처방" in payload.question or "약" in payload.question:
        medications = artifacts.get("medications") or {}
        items = medications.get("items") or []
        if not items:
            return "현재 source에 대해 사용할 수 있는 medication artifact가 없습니다." if wants_korean else "No medication artifact is available for the current source."
        lines = [
            f"- {item.get('medication', 'n/a')} | status {item.get('status', 'n/a')} | intent {item.get('intent', 'n/a')} | date {item.get('date', 'n/a')}"
            for item in items[:8]
        ]
        if wants_korean:
            return "현재 source의 Medication Timeline:\n\n" + "\n".join(lines)
        return "Medication Timeline of the current source:\n\n" + "\n".join(lines)

    if "note" in question or "clinical note" in question or "노트" in payload.question:
        note = artifacts.get("note") or {}
        if not note:
            return "현재 source에 대해 사용할 수 있는 clinical note artifact가 없습니다." if wants_korean else "No clinical note artifact is available for the current source."
        if wants_korean:
            return (
                "현재 source의 clinical note review:\n\n"
                + f"- headline: {note.get('headline', 'n/a')}\n"
                + f"- line_count: {note.get('line_count', 'n/a')}\n"
                + f"- word_count: {note.get('word_count', 'n/a')}\n"
                + f"- preview:\n{note.get('preview', 'n/a')}"
            )
        return (
            "Clinical note review of the current source:\n\n"
            + f"- headline: {note.get('headline', 'n/a')}\n"
            + f"- line_count: {note.get('line_count', 'n/a')}\n"
            + f"- word_count: {note.get('word_count', 'n/a')}\n"
            + f"- preview:\n{note.get('preview', 'n/a')}"
        )

    if "qc" in question or "quality" in question or "결측" in payload.question:
        qc = artifacts.get("qc") or {}
        lines = [f"- {key}: {value}" for key, value in qc.items()]
        return ("현재 source의 QC summary:\n\n" if wants_korean else "QC summary of the current source:\n\n") + ("\n".join(lines) if lines else ("- QC 지표가 없습니다." if wants_korean else "- No QC metrics are available."))

    if wants_korean:
        return (
            f"현재 source는 `{source.file_name}` ({source.modality}) 입니다.\n\n"
            f"{_korean_analysis_summary(payload.analysis)}\n\n"
            "schema, cohort, QC, metadata, message structure, note content, 또는 현재 선택한 Studio card에 대해 구체적으로 질문하실 수 있습니다."
        )

    return (
        f"The current source is `{source.file_name}` ({source.modality}). "
        f"The current grounded summary is:\n\n{payload.analysis.grounded_summary}\n\n"
        "You can ask specifically about schema, cohort, QC, metadata, message structure, note content, or the active Studio card."
    )


def _compact_analysis_context(payload: ArtifactChatRequest) -> dict[str, Any]:
    analysis = payload.analysis
    active_card = payload.active_card or {}
    active_source_index = active_card.get("source_index")
    artifacts = _merged_source_artifacts(analysis, active_source_index if isinstance(active_source_index, int) else None)
    context: dict[str, Any] = {
        "source": analysis.source.model_dump(),
        "sources": [item.model_dump() for item in analysis.sources[:8]],
        "grounded_summary": analysis.grounded_summary,
        "studio_cards": analysis.studio_cards[:16],
        "artifacts_by_source": _source_artifact_views(analysis)[:8],
        "active_view": payload.active_view,
        "active_card": payload.active_card,
        "active_artifact": payload.active_artifact,
        "artifacts": {key: artifacts[key] for key in list(artifacts.keys())[:24]},
    }
    if payload.active_view and payload.active_view in artifacts:
        context["selected_artifact"] = artifacts[payload.active_view]
    return context


def _extract_output_text(result: dict[str, Any]) -> str:
    output_text = result.get("output_text")
    if output_text:
        return str(output_text).strip()

    texts: list[str] = []
    for item in result.get("output", []):
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"}:
                text = content.get("text")
                if isinstance(text, str) and text.strip():
                    texts.append(text.strip())
    return "\n\n".join(texts).strip()


def _call_openai_answer(payload: ArtifactChatRequest) -> str | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    model = os.getenv("OPENAI_MODEL", "gpt-5-mini")
    system_prompt = (
        "You are ChatClinic, a clinical data and medical imaging copilot. "
        "Answer from the provided intake analysis and Studio artifacts. "
        "Be grounded, concise, and structured. "
        "If the user asks about the active Studio card, explain that artifact directly. "
        "If the question is general and not clinically grounded, answer normally but briefly, then steer back to the uploaded sources. "
        "Use the same language as the user. Format in Markdown with short paragraphs or bullets when helpful."
    )
    compact_context = _compact_analysis_context(payload)
    history_lines = [{"role": turn.role, "content": turn.content} for turn in payload.history[-8:]]
    user_content = (
        f"User question:\n{payload.question}\n\n"
        "Grounded analysis context JSON:\n"
        f"{json.dumps(compact_context, ensure_ascii=False)}"
    )
    body = {
        "model": model,
        "input": [
            {"role": "system", "content": system_prompt},
            *history_lines,
            {"role": "user", "content": user_content},
        ],
    }
    request = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        data=json.dumps(body).encode("utf-8"),
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=float(os.getenv("OPENAI_TIMEOUT_SECONDS", "20"))) as response:
        result = json.loads(response.read().decode("utf-8"))
    return _extract_output_text(result) or None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/v1/intake/upload", response_model=IntakeSummaryResponse)
async def upload_source(files: list[UploadFile] = File(...)) -> IntakeSummaryResponse:
    if not files:
        raise ValueError("No files were uploaded")
    parsed_responses: list[IntakeSummaryResponse] = []
    image_files: list[tuple[str, bytes, str, str]] = []
    dicom_files: list[tuple[str, bytes, str, str]] = []
    raster_image_files: list[tuple[str, bytes, str, str]] = []
    ndjson_files: list[tuple[str, bytes, str]] = []

    for upload in files:
        raw = await upload.read()
        file_name = upload.filename or "uploaded-file"
        modality, suffix = _guess_modality(file_name)
        if modality == "clinical-table":
            try:
                parsed_responses.append(_summarize_table(file_name, raw, suffix))
            except Exception as exc:
                parsed_responses.append(
                    IntakeSummaryResponse(
                        source=UploadedSourceSummary(
                            file_name=file_name,
                            file_type=suffix,
                            modality="clinical-table",
                            size_bytes=len(raw),
                            status="unsupported",
                        ),
                        grounded_summary=f"This table-like source was received but could not be parsed deterministically: {exc}",
                        studio_cards=[],
                        artifacts={"error": {"message": str(exc)}},
                        sources=[],
                    )
                )
        elif modality == "clinical-message":
            if suffix == "ndjson":
                ndjson_files.append((file_name, raw, suffix))
            else:
                decoded = raw.decode("utf-8", errors="replace")
                if _looks_like_fhir_json(decoded) or _looks_like_fhir_xml(decoded):
                    try:
                        tool_result = run_tool(
                            "fhir_browser_tool",
                            {
                                "files": [
                                    {
                                        "file_name": file_name,
                                        "suffix": suffix,
                                        "raw_base64": base64.b64encode(raw).decode("ascii"),
                                    }
                                ]
                            },
                        )
                        tool_payload = tool_result.get("result", {})
                        parsed_responses.append(IntakeSummaryResponse.model_validate(tool_payload))
                    except Exception:
                        parsed_responses.append(_summarize_clinical_message(file_name, raw, suffix))
                else:
                    parsed_responses.append(_summarize_clinical_message(file_name, raw, suffix))
        elif modality == "clinical-note":
            parsed_responses.append(_summarize_clinical_note(file_name, raw, suffix))
        elif modality == "medical-image":
            source_path = _persist_uploaded_file(file_name, raw)
            if _is_dicom_suffix(suffix):
                dicom_files.append((file_name, raw, suffix, source_path))
            elif _is_raster_image_suffix(suffix):
                raster_image_files.append((file_name, raw, suffix, source_path))

    if ndjson_files:
        try:
            tool_result = run_tool(
                "fhir_browser_tool",
                {
                    "files": [
                        {
                            "file_name": file_name,
                            "suffix": suffix,
                            "raw_base64": base64.b64encode(raw).decode("ascii"),
                        }
                        for file_name, raw, suffix in ndjson_files
                    ]
                },
            )
            tool_payload = tool_result.get("result", {})
            parsed_responses.append(IntakeSummaryResponse.model_validate(tool_payload))
        except Exception:
            parsed_responses.append(_summarize_fhir_ndjson_group(ndjson_files))

    if dicom_files:
        try:
            tool_result = run_tool(
                "dicom_review_tool",
                {
                    "files": [
                        {
                            "file_name": file_name,
                            "suffix": suffix,
                            "raw_base64": base64.b64encode(raw).decode("ascii"),
                            "source_path": source_path,
                        }
                        for file_name, raw, suffix, source_path in dicom_files
                    ]
                },
            )
            tool_payload = tool_result.get("result", {})
            parsed_responses.append(IntakeSummaryResponse.model_validate(tool_payload))
        except Exception:
            if len(dicom_files) == 1:
                file_name, raw, suffix, source_path = dicom_files[0]
                parsed_responses.append(_summarize_dicom(file_name, raw, suffix, source_path=source_path))
            else:
                parsed_responses.append(_summarize_dicom_series(dicom_files))

    if raster_image_files:
        try:
            tool_result = run_tool(
                "image_review_tool",
                {
                    "files": [
                        {
                            "file_name": file_name,
                            "suffix": suffix,
                            "raw_base64": base64.b64encode(raw).decode("ascii"),
                            "source_path": source_path,
                        }
                        for file_name, raw, suffix, source_path in raster_image_files
                    ]
                },
            )
            tool_payload = tool_result.get("result", {})
            parsed_responses.append(IntakeSummaryResponse.model_validate(tool_payload))
        except Exception:
            if len(raster_image_files) == 1:
                file_name, raw, suffix, source_path = raster_image_files[0]
                parsed_responses.append(_summarize_raster_image(file_name, raw, suffix, source_path=source_path))
            else:
                parsed_responses.append(_summarize_raster_image_group(raster_image_files))

    if parsed_responses:
        return _merge_responses(parsed_responses)

    first_file = files[0]
    raw = await first_file.read()
    modality, suffix = _guess_modality(first_file.filename or "")
    summary = (
        "The uploaded source was received, but this scaffold currently supports clinical CSV/TSV files, "
        "FHIR JSON/XML/NDJSON, HL7 messages, plain-text clinical notes, DICOM uploads, and PNG/JPG/TIFF raster images only."
    )
    return IntakeSummaryResponse(
        source=UploadedSourceSummary(
            file_name=first_file.filename or "uploaded-file",
            file_type=suffix,
            modality=modality,
            size_bytes=len(raw),
            status="unsupported",
        ),
        grounded_summary=summary,
        studio_cards=[],
        artifacts={},
        sources=[],
    )


@app.post("/api/v1/chat/artifacts", response_model=ArtifactChatResponse)
def chat_about_artifacts(request: ArtifactChatRequest) -> ArtifactChatResponse:
    deterministic_answer = _artifact_guided_answer(request)

    generic_patterns = [
        "what is your name",
        "who are you",
        "hello",
        "hi",
        "한국어로",
        "한글로",
        "in korean",
        "이게 무슨 뜻이야",
        "설명해줘",
        "자세히",
    ]
    is_generic_followup = any(pattern in request.question.lower() for pattern in generic_patterns)

    studio_specific = any(
        token in request.question.lower()
        for token in [
            "schema",
            "column",
            "cohort",
            "patient",
            "patients",
            "subject",
            "subjects",
            "sheet",
            "grid",
            "distribution",
            "series",
            "study",
            "dicom",
            "metadata",
            "modality",
            "fhir",
            "hl7",
            "message",
            "resource",
            "note",
            "clinical note",
            "qc",
            "quality",
        ]
    ) or bool(request.active_card or request.active_artifact) or bool(
        re.search(r"[가-힣]", request.question)
        and any(token in request.question for token in ["환자", "피험자", "대상자", "코호트", "시트", "그리드", "결측", "분포"])
    )

    if not studio_specific or is_generic_followup:
        try:
            openai_answer = _call_openai_answer(request)
            if openai_answer:
                return ArtifactChatResponse(answer=openai_answer)
        except Exception:
            pass

    return ArtifactChatResponse(answer=deterministic_answer)
