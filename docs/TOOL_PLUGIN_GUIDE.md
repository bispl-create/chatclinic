# Tool Plugin Guide

This guide explains how collaborators and student teams can add a new tool to `ChatClinic`.

<div style="padding:12px 16px; border-radius:12px; background:#fff1f2; border:2px solid #fb7185; color:#9f1239; margin:16px 0;">
  <strong>Revision history</strong><br/>
  <strong>March 2026 update:</strong> Plugins should now describe runtime compatibility for CPU/GPU hosts and may target raster image intake in addition to clinical tables, FHIR, DICOM, and notes.
</div>

## What to submit

Each team submits one plugin package and one Skill patch proposal.

Recommended final submission structure:

```text
team_name_submission.zip
  plugin/
    tool.json
    run.py
    README.md
    requirements.txt   # optional
  skill_update/
    skill_patch.md
    skill_rationale.md
```

The default class model is:

- `ChatClinic` provides the shared UI, orchestration Skill, and runner
- collaborators or student teams provide a `tool plugin`
- collaborators or student teams provide a `Skill patch proposal`
- teams do not need to operate a separate server

## Minimal manifest

Create `tool.json`.

```json
{
  "name": "team_name_tool",
  "team": "team_name",
  "task_type": "analysis-task",
  "modality": "clinical-table",
  "approval_required": true,
  "entrypoint": "run.py",
  "description": "Short human-readable description of the tool."
}
```

Recommended metadata:

```json
{
  "keywords": ["cohort", "sheet", "subject"],
  "recommended_stage": "post-intake",
  "priority": 80,
  "produces": ["cohort_browser"],
  "consumes": ["table_rows"]
}
```

These fields help the orchestration Skill choose tools with less manual hard-coding.

Recommended runtime metadata:

```json
{
  "runtime": {
    "host_compatible": ["cpu", "gpu"],
    "supported_accelerators": ["cpu"],
    "preferred_accelerator": "cpu",
    "requires_gpu": false,
    "allow_cpu_fallback": true,
    "estimated_runtime_sec": 10,
    "notes": "Explain whether the tool only needs CPU, prefers GPU, or requires GPU."
  }
}
```

## Execution contract

`ChatClinic` runs the plugin like this:

```bash
python3 run.py --input input.json --output output.json
```

Your script must:

1. read `--input`
2. write `--output`
3. exit with code `0` on success

## Input payload

The runner passes JSON with the current context. A typical payload looks like:

```json
{
  "question": "Explain this cohort",
  "analysis_source": {
    "file_name": "study.xlsx",
    "modality": "clinical-table"
  },
  "analysis_sources": [],
  "analysis_artifacts": {},
  "grounded_summary": "",
  "active_view": null,
  "active_card": {},
  "active_artifact": {}
}
```

Not every field is present for every workflow, so plugins should be defensive.

## Output payload

Write structured JSON like this:

```json
{
  "summary": "Cohort analysis completed.",
  "artifacts": {
    "cohort_browser": {}
  },
  "provenance": {
    "tool_version": "0.1.0"
  }
}
```

Recommended fields:

- `summary`
- `artifacts`
- `provenance`

Useful optional fields:

- `warnings`
- `measurements`
- `preview`
- `report_draft`

## Python template

```python
import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))

    result = {
        "summary": "Tool executed successfully.",
        "artifacts": {},
        "provenance": {"tool_version": "0.1.0"},
    }

    Path(args.output).write_text(json.dumps(result), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

## Best practices

- Keep outputs deterministic when possible.
- Return structured artifacts, not only free-form prose.
- Do not assume internet access.
- Do not assume a GPU unless your tool explicitly documents it.
- Do not write outside the plugin working area unless truly needed.
- Emit helpful stderr messages when a run fails.

## Supported source families in ChatClinic

Plugins can currently be designed around source families such as:

- clinical tables (`csv`, `tsv`, `xlsx`, `xlsm`, `xls`)
- FHIR / HL7 clinical messages
- DICOM medical images
- raster medical images (`png`, `jpg`, `jpeg`, `tif`, `tiff`)
- plain-text clinical notes

## How tools are discovered

`ChatClinic` automatically scans `plugins/*/tool.json`.

That means adding a new plugin is usually:

1. create the folder
2. add `tool.json`
3. add `run.py`
4. restart the backend

## How the Skill uses your tool

The orchestration Skill uses:

- `name`
- `task_type`
- `modality`
- `keywords`
- `recommended_stage`
- `priority`

to decide when a tool should be suggested or run.

If your new tool introduces a new analysis path, propose an update to:

- `/Users/jongcye/Documents/Codex/workspace/clinical_multimodal_workspace/skills/chatclinic-orchestrator/SKILL.md`

## When to update the Skill

You do **not** need to update the Skill for every tiny plugin change.

Usually update the Skill when:

- a new tool introduces a new analysis category
- a new tool should be preferred over an older tool
- a new tool requires approval before execution
- a new tool must run before or after another tool
- the initial chat guidance or orchestration policy should change

Usually you do **not** need to update the Skill when:

- you only improve the internal implementation of an existing tool
- you fix bugs without changing when the tool should be chosen
- you only change output formatting while keeping the same role in the workflow

## How to update the Skill

For class submissions, do not submit a full replacement Skill.

Instead, submit:

- `skill_update/skill_patch.md`
- `skill_update/skill_rationale.md`

The instructor will merge accepted proposals into the master Skill.

## How to update the Skill

Edit or propose changes for:

- `/Users/jongcye/Documents/Codex/workspace/clinical_multimodal_workspace/skills/chatclinic-orchestrator/SKILL.md`

Recommended steps:

1. Add the tool name to the relevant workflow or tool-order section.
2. Add a short rule explaining when the tool should be used.
3. Add approval guidance if the tool should not run automatically.
4. Mention important dependencies or ordering constraints.
5. Keep the Skill high-level; do not duplicate implementation details from `run.py`.

Example update pattern:

```md
- Use `my_new_tool` for pathology review when the uploaded source is a pathology image or when the user explicitly asks for pathology interpretation.
- Ask for approval before running `my_new_tool` if it creates a diagnostic draft.
```

## Recommended Skill fields to think about

When adding a tool, check these questions:

- What user questions should trigger it?
- What modality should it match?
- Does it depend on an earlier artifact?
- Should it be preferred over another tool?
- Does it need explicit approval?
- What Studio artifact or card should it produce?

If the answer to any of these changes platform behavior, update the Skill.

## Related documents

- [Course tool contract](COURSE_TOOLS.md)
- [Skill patch template](SKILL_PATCH_TEMPLATE.md)
- [Submission site specification](SUBMISSION_SITE_SPEC.md)
- [Master Skill integration guide](MASTER_SKILL_INTEGRATION.md)

## Recommended collaborator checklist

- `tool.json` is valid JSON
- `name` is unique
- `run.py` works with `--input` and `--output`
- output JSON is valid
- the tool can run in the shared classroom environment
- the result contains useful structured artifacts
- README describes dependencies and expected inputs
- the orchestration Skill was reviewed if the new tool changes routing or policy
