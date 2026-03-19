from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any


def _normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


def _name_matches(name: str, patterns: tuple[str, ...]) -> bool:
    normalized = _normalize_name(name)
    return any(pattern in normalized for pattern in patterns)


def _is_float_like(value: str) -> bool:
    try:
        float(value)
        return True
    except Exception:
        return False


def _infer_type(values: list[str]) -> str:
    non_empty = [value for value in values if str(value or "").strip()]
    if not non_empty:
        return "categorical"
    float_like = sum(1 for value in non_empty if _is_float_like(str(value)))
    date_like = sum(1 for value in non_empty if re.match(r"^\d{4}-\d{1,2}-\d{1,2}", str(value)))
    if float_like == len(non_empty):
        if all(float(str(value)).is_integer() for value in non_empty):
            return "integer"
        return "float"
    if date_like >= max(1, math.ceil(len(non_empty) * 0.7)):
        return "date-like"
    unique_count = len(set(str(value) for value in non_empty))
    if unique_count > max(20, len(non_empty) * 0.6):
        return "free-text"
    return "categorical"


def _build_profiles(columns: list[str], rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    profiles: list[dict[str, Any]] = []
    for column in columns:
        values = [str(row.get(column, "") or "") for row in rows]
        non_empty = [value for value in values if value.strip()]
        missing_count = len(values) - len(non_empty)
        inferred = _infer_type(values)
        numeric_summary = None
        if inferred in {"integer", "float"} and non_empty:
            numeric_values = [float(value) for value in non_empty if _is_float_like(value)]
            if numeric_values:
                numeric_summary = {
                    "min": min(numeric_values),
                    "max": max(numeric_values),
                    "mean": round(sum(numeric_values) / len(numeric_values), 3),
                }
        profiles.append(
            {
                "name": column,
                "inferred_type": inferred,
                "non_empty_count": len(non_empty),
                "missing_count": missing_count,
                "missing_rate": missing_count / max(len(values), 1),
                "unique_count": len(set(non_empty)),
                "numeric_summary": numeric_summary,
            }
        )
    return profiles


def _infer_roles(columns: list[str], profiles: list[dict[str, Any]]) -> dict[str, list[str]]:
    role_map = {
        "subject_id_columns": ("subject", "patient_id", "participant", "screening", "mrn", "person_id", "subjectno"),
        "visit_columns": ("visit", "timepoint", "epoch", "cycle"),
        "site_columns": ("site", "center", "hospital"),
        "arm_columns": ("arm", "group", "cohort", "treatment"),
        "date_columns": ("date", "time", "datetime"),
        "outcome_columns": ("outcome", "response", "status", "grade", "severity"),
    }
    by_name = {profile["name"]: profile for profile in profiles}
    roles: dict[str, list[str]] = {key: [] for key in role_map}
    for column in columns:
        inferred = str((by_name.get(column) or {}).get("inferred_type", ""))
        for role, patterns in role_map.items():
            if _name_matches(column, patterns):
                if role == "date_columns" and inferred not in {"date-like", "integer", "float", "categorical"}:
                    continue
                roles[role].append(column)
    return roles


def _pick_subject_column(rows: list[dict[str, str]], roles: dict[str, list[str]], columns: list[str]) -> str | None:
    if roles["subject_id_columns"]:
        return roles["subject_id_columns"][0]
    preferred = ("subject", "patient", "participant", "person", "mrn", "screen", "id", "subjectno")
    for column in columns:
        normalized = _normalize_name(column)
        if any(token in normalized for token in preferred):
            return column
    return None


def _classify(file_name: str, rows: list[dict[str, str]], columns: list[str], profiles: list[dict[str, Any]], roles: dict[str, list[str]], suffix: str) -> dict[str, Any]:
    row_count = len(rows)
    cohort_score = 0
    single_score = 0
    rationale: list[str] = []
    subject_column = _pick_subject_column(rows, roles, columns)
    subject_unique = 0
    if subject_column:
        subject_values = {
            str(row.get(subject_column, "") or "").strip()
            for row in rows
            if str(row.get(subject_column, "") or "").strip()
        }
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
        "row_count": len(rows),
        "column_count": len(columns),
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


def _value_counts(rows: list[dict[str, str]], column: str | None, limit: int = 10) -> list[dict[str, Any]]:
    if not column:
        return []
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
    return [
        {
            "label": f"{(low + index * width):.1f}-{(low + (index + 1) * width):.1f}",
            "count": count,
        }
        for index, count in enumerate(counts)
    ]


def _cohort_summary(rows: list[dict[str, str]], profiles: list[dict[str, Any]]) -> dict[str, Any]:
    categorical = [item for item in profiles if item["inferred_type"] == "categorical" and item["unique_count"] > 0]
    numeric = [item for item in profiles if item["inferred_type"] in {"integer", "float"}]
    categorical_breakdowns = []
    for profile in sorted(categorical, key=lambda item: item["unique_count"])[:3]:
        counts = _value_counts(rows, profile["name"], limit=5)
        categorical_breakdowns.append({"column": profile["name"], "top_values": counts})
    numeric_breakdowns = [{"column": item["name"], "summary": item["numeric_summary"]} for item in numeric[:3]]
    return {
        "record_count": len(rows),
        "field_count": len(profiles),
        "categorical_breakdowns": categorical_breakdowns,
        "numeric_breakdowns": numeric_breakdowns,
    }


def _build_subject_preview(rows: list[dict[str, str]], roles: dict[str, list[str]], columns: list[str]) -> list[dict[str, Any]]:
    subject_column = _pick_subject_column(rows, roles, columns)
    if not subject_column:
        return []
    visit_column = roles["visit_columns"][0] if roles["visit_columns"] else None
    site_column = roles["site_columns"][0] if roles["site_columns"] else None
    arm_column = roles["arm_columns"][0] if roles["arm_columns"] else None
    outcome_column = roles["outcome_columns"][0] if roles["outcome_columns"] else None
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        subject = str(row.get(subject_column, "") or "").strip() or "(missing)"
        grouped.setdefault(subject, []).append(row)
    preview = []
    for subject, subject_rows in list(grouped.items())[:12]:
        latest_row = subject_rows[-1]
        visits = []
        if visit_column:
            visits = sorted(
                {
                    str(item.get(visit_column, "") or "").strip()
                    for item in subject_rows
                    if str(item.get(visit_column, "") or "").strip()
                }
            )
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


def _sheet_domain_name(sheet_name: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", sheet_name.lower()).strip("_")
    return normalized or "sheet"


def _build_artifact(
    rows: list[dict[str, str]],
    columns: list[str],
    profiles: list[dict[str, Any]],
    roles: dict[str, list[str]],
    intake: dict[str, Any],
    cohort: dict[str, Any],
    missingness: dict[str, Any],
    sheet_details: list[dict[str, Any]],
) -> dict[str, Any]:
    subject_column = intake.get("subject_column")
    visit_column = roles["visit_columns"][0] if roles["visit_columns"] else None
    site_column = roles["site_columns"][0] if roles["site_columns"] else None
    arm_column = roles["arm_columns"][0] if roles["arm_columns"] else None
    outcome_column = roles["outcome_columns"][0] if roles["outcome_columns"] else None
    subject_values = {
        str(row.get(subject_column, "") or "").strip()
        for row in rows
        if subject_column and str(row.get(subject_column, "") or "").strip()
    }
    visit_values = {
        str(row.get(visit_column, "") or "").strip()
        for row in rows
        if visit_column and str(row.get(visit_column, "") or "").strip()
    }
    age_profile = next((profile for profile in profiles if _name_matches(profile["name"], ("age",)) and profile["inferred_type"] in {"integer", "float"}), None)
    age_values = []
    if age_profile:
        for row in rows:
            value = str(row.get(age_profile["name"], "") or "").strip()
            if _is_float_like(value):
                age_values.append(float(value))
    composition = {
        "site_distribution": _value_counts(rows, site_column, limit=10),
        "arm_distribution": _value_counts(rows, arm_column, limit=10),
        "outcome_distribution": _value_counts(rows, outcome_column, limit=10),
        "age_histogram": _histogram(age_values, bins=6),
    }
    overview = {
        "row_count": len(rows),
        "column_count": len(columns),
        "subject_count": len(subject_values) if subject_values else cohort.get("record_count", len(rows)),
        "visit_count": len(visit_values),
        "site_count": len({item["label"] for item in composition["site_distribution"] if item["label"] != "(missing)"}),
        "arm_count": len({item["label"] for item in composition["arm_distribution"] if item["label"] != "(missing)"}),
        "completeness_rate": round(100 * (1 - sum(item["missing_rate"] for item in profiles) / max(len(profiles), 1)), 1) if profiles else 0.0,
        "analysis_mode": intake.get("analysis_mode", "n/a"),
        "selected_sheet": ((intake.get("table_meta") or {}).get("selected_sheet") or "n/a"),
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
        for item in sheet_details
    ]
    return {
        "overview": overview,
        "intake": intake,
        "composition": composition,
        "domains": domains,
        "subjects": _build_subject_preview(rows, roles, columns),
        "grid": {"columns": columns, "rows": rows, "row_count": len(rows)},
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    execution_context = payload.get("execution_context") or {}
    file_name = str(payload.get("file_name") or "workbook.xlsx")
    suffix = str(payload.get("suffix") or "xlsx")
    sheet_tables = list(payload.get("sheet_tables") or [])
    table_meta = dict(payload.get("table_meta") or {})

    sheet_details: list[dict[str, Any]] = []
    studio_cards: list[dict[str, Any]] = []
    artifacts: dict[str, Any] = {}
    summary_lines = [
        f"This workbook contains **{len(sheet_tables)}** non-empty sheet(s), each analyzed as an individual cohort view.",
        "",
        "## Sheet overview",
        "",
    ]

    for sheet_table in sheet_tables:
        sheet_rows = list(sheet_table.get("rows") or [])
        sheet_columns = list(sheet_table.get("columns") or [])
        sheet_name = str(sheet_table.get("sheet_name") or "Sheet")
        profiles = _build_profiles(sheet_columns, sheet_rows)
        roles = _infer_roles(sheet_columns, profiles)
        intake = _classify(file_name, sheet_rows, sheet_columns, profiles, roles, suffix)
        cohort = _cohort_summary(sheet_rows, profiles)
        missingness = _missingness_summary(profiles)
        detail = {
            "sheet_name": sheet_name,
            "domain": _sheet_domain_name(sheet_name),
            "row_count": len(sheet_rows),
            "subject_count": intake.get("subject_unique_count", 0),
            "subject_column": intake.get("subject_column", "n/a"),
            "visit_columns": roles.get("visit_columns", []),
            "date_columns": roles.get("date_columns", []),
        }
        sheet_details.append(detail)

        card_id = f"sheet::{sheet_name}::cohort_browser"
        sheet_table_meta = {
            "workbook_format": suffix,
            "sheet_names": table_meta.get("sheet_names", []),
            "selected_sheet": sheet_name,
            "sheet_name": sheet_name,
            "sheet_details": sheet_details,
        }
        studio_cards.append(
            {
                "id": card_id,
                "title": sheet_name,
                "subtitle": "Cohort Browser",
                "base_id": "cohort_browser",
            }
        )
        artifacts[card_id] = _build_artifact(
            sheet_rows,
            sheet_columns,
            profiles,
            roles,
            {**intake, "table_meta": sheet_table_meta},
            cohort,
            missingness,
            sheet_details,
        )
        summary_lines.append(
            f"- **{sheet_name}**: {len(sheet_rows)} row(s), {len(sheet_columns)} column(s), mode `{intake['analysis_mode']}`, subjects {intake.get('subject_unique_count', 'n/a')}."
        )

    selected_sheet_name = str(table_meta.get("selected_sheet") or (sheet_tables[0]["sheet_name"] if sheet_tables else "Sheet1"))
    result = {
        "summary": "\n".join(
            summary_lines
            + [
                "",
                f"**Selected sheet:** `{selected_sheet_name}`",
                "",
                "Ask about a specific sheet by name to inspect its cohort profile in detail.",
            ]
        ),
        "studio_cards": studio_cards,
        "artifacts": artifacts,
        "provenance": {
            "tool": "cohort_analysis_tool",
            "version": "0.1.0",
            "sheet_count": len(sheet_tables),
            "execution_context": execution_context,
        },
    }
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
