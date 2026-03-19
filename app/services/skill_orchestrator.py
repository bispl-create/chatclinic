from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from app.services.tool_runner import discover_tools

ROOT = Path(__file__).resolve().parents[2]
SKILL_PATH = ROOT / "skills" / "chatclinic-orchestrator" / "SKILL.md"


def _skill_text() -> str:
    try:
        return SKILL_PATH.read_text(encoding="utf-8")
    except Exception:
        return ""


def _normalize(text: str) -> str:
    return " ".join((text or "").lower().split())


def _section_body(markdown: str, heading: str) -> str:
    if not markdown:
        return ""
    pattern = re.compile(
        rf"^##\s+{re.escape(heading)}\s*$\n(.*?)(?=^##\s+|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    match = pattern.search(markdown)
    if not match:
        return ""
    return match.group(1).strip()


def initial_chat_prompt() -> str:
    prompt = _section_body(_skill_text(), "Initial chat prompt")
    if prompt:
        lines: list[str] = []
        for line in prompt.splitlines():
            if not line.strip():
                if lines:
                    break
                continue
            lines.append(line.strip())
        if lines:
            return " ".join(lines)
    return (
        "Upload one clinical CSV/TSV file, FHIR JSON/XML/NDJSON, HL7 message files, "
        "plain-text clinical notes, or DICOM files. ChatClinic will generate a "
        "deterministic first-pass summary and open the matching Studio cards."
    )


def _analysis_modalities(analysis: dict[str, Any]) -> set[str]:
    modalities: set[str] = set()
    source = analysis.get("source") or {}
    modality = source.get("modality")
    if modality:
        modalities.add(str(modality))
    for item in analysis.get("sources") or []:
        modality = (item or {}).get("modality")
        if modality:
            modalities.add(str(modality))
    return modalities


def _score_tool(tool: dict[str, Any], question: str, analysis: dict[str, Any], active_view: str | None) -> tuple[int, str]:
    lowered = _normalize(question)
    score = 0
    rationale: list[str] = []
    modality = str(tool.get("modality", "") or "")
    keywords = [str(item).lower() for item in tool.get("keywords", []) or []]
    recommended_stage = str(tool.get("recommended_stage", "") or "")
    tool_name = str(tool.get("name", "") or "")

    if any(keyword and keyword in lowered for keyword in keywords):
        score += 6
        rationale.append("question keywords match the tool manifest")

    modalities = _analysis_modalities(analysis)
    if modality and modality in modalities:
        score += 3
        rationale.append("tool modality matches the uploaded source")

    if active_view and recommended_stage and recommended_stage in active_view:
        score += 2
        rationale.append("current Studio context aligns with the recommended stage")

    if tool_name == "cohort_analysis_tool":
        cohort_tokens = [
            "cohort",
            "sheet",
            "subject",
            "patient",
            "participants",
            "grid",
            "schema",
            "missing",
            "visit",
            "site",
            "arm",
            "outcome",
            "코호트",
            "시트",
            "환자",
            "피험자",
            "대상자",
            "그리드",
            "결측",
            "방문",
        ]
        if any(token in lowered for token in cohort_tokens):
            score += 8
            rationale.append("the orchestrator skill prefers cohort_analysis_tool for cohort/table questions")
        if "clinical-table" in modalities:
            score += 4
            rationale.append("the current upload contains a clinical table or workbook")

    return score, "; ".join(rationale)


def suggest_tool(question: str, analysis: dict[str, Any], active_view: str | None = None) -> dict[str, Any] | None:
    lowered = _normalize(question)
    if not lowered:
        return None

    tools = [tool for tool in discover_tools() if tool.get("name")]
    if not tools:
        return None

    skill_text = _skill_text()
    ranked: list[tuple[int, dict[str, Any], str]] = []
    for tool in tools:
        score, rationale = _score_tool(tool, lowered, analysis, active_view)
        if score > 0:
            ranked.append((score, tool, rationale))

    if not ranked:
        return None

    ranked.sort(key=lambda item: (item[0], int(item[1].get("priority", 0))), reverse=True)
    score, tool, rationale = ranked[0]
    if score < 5:
        return None

    extra = ""
    if tool.get("name") == "cohort_analysis_tool" and "cohort_analysis_tool" in skill_text:
        extra = " The orchestrator skill explicitly recommends this tool for table/eCRF cohort analysis."

    return {
        "tool": {
            "name": tool.get("name"),
            "team": tool.get("team"),
            "task_type": tool.get("task_type"),
            "modality": tool.get("modality"),
            "approval_required": bool(tool.get("approval_required", True)),
            "description": tool.get("description"),
            "runtime": dict(tool.get("runtime", {}) or {}),
        },
        "rationale": (rationale or "the orchestrator selected the most relevant registered tool") + extra,
    }
