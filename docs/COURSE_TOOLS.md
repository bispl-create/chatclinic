# Course Tool Contract

`ChatClinic` can be extended by student teams without running separate servers.

<div style="padding:12px 16px; border-radius:12px; background:#fff1f2; border:2px solid #fb7185; color:#9f1239; margin:16px 0;">
  <strong>Revision history</strong><br/>
  <strong>March 2026 update:</strong> Tool manifests should now describe CPU/GPU runtime behavior explicitly, and image tools may target DICOM or raster inputs such as PNG/JPG/TIFF.
</div>

## Submission model

For the standard class project:

- the instructor provides `ChatClinic`
- the instructor maintains the orchestration Skill and shared runner
- each student team submits a **tool plugin**
- each student team also submits a **Skill patch proposal** for orchestration updates

Student teams normally do **not** need to submit:

- a complete replacement Skill
- a separate MCP server
- a separate web service
- a separate frontend

In other words:

- `Master Skill` = centrally managed behavior/policy
- `Tool plugin` = primary student deliverable
- `Skill patch` = student proposal for how their tool should be orchestrated

## When the Skill should be updated

The Skill is maintained centrally, but it should be reviewed whenever a new tool changes orchestration.

Update the Skill when:

- a new tool adds a new workflow
- a new tool should be selected for new keywords or modalities
- a new tool requires approval
- a new tool depends on an earlier tool or artifact
- a new tool should replace an older preferred tool

In practice:

- student teams submit the plugin
- student teams submit a small `skill_patch.md` proposal
- the instructor or maintainer merges multiple proposals into one master Skill if routing policy changes

## Goal

Each team submits a **plugin tool** that can be discovered and executed by a shared runner.

Examples:

- segmentation
- detection
- diagnosis draft
- image restoration
- FHIR summarization
- cohort analytics

## Directory contract

Each team submits one plugin package and one Skill proposal.

Recommended final submission package:

```text
team01_submission.zip
  plugin/
    tool.json
    run.py
    requirements.txt
    README.md
  skill_update/
    skill_patch.md
    skill_rationale.md
```

Optional extras are allowed:

- helper scripts
- model weights
- config files
- reference outputs for validation
- background paper list
- presentation slides describing tool integration

## Manifest contract

Use `tool.json`.

```json
{
  "name": "team_cohort_analysis_tool",
  "team": "team_cohort",
  "task_type": "cohort-browser",
  "modality": "clinical-table",
  "approval_required": false,
  "entrypoint": "run.py",
  "description": "Analyzes cohort tables and returns browser-ready artifacts."
}
```

## Execution contract

`ChatClinic` executes:

```bash
python3 run.py --input input.json --output output.json
```

The plugin must:

1. Read `--input`
2. Produce `--output`
3. Exit with code `0` on success

The plugin should:

4. Fail clearly with a useful stderr message on invalid input
5. Avoid writing outside its own working area unless explicitly designed to do so
6. Finish within a reasonable classroom demo time

## Input payload

The runner passes JSON like:

```json
{
  "question": "Segment the lungs from this chest x-ray",
  "analysis_source": {
    "file_name": "CT_small.dcm",
    "modality": "medical-image"
  },
  "artifacts": {
    "metadata": {},
    "series": {}
  }
}
```

## Output payload

The plugin returns structured JSON like:

```json
{
  "summary": "Cohort analysis completed.",
  "artifacts": {
    "cohort_browser": {}
  },
  "provenance": {
    "version": "0.1.0"
  }
}
```

Recommended output fields:

- `summary`
- `artifacts`
- `provenance`

Useful optional fields:

- `warnings`
- `measurements`
- `preview`
- `report_draft`

## Runtime metadata contract

If a tool should run correctly on CPU-only machines, GPU servers, or both, the manifest should describe that explicitly.

Recommended `runtime` block:

```json
{
  "runtime": {
    "host_compatible": ["cpu", "gpu"],
    "supported_accelerators": ["cpu"],
    "preferred_accelerator": "cpu",
    "requires_gpu": false,
    "allow_cpu_fallback": true,
    "estimated_runtime_sec": 10,
    "notes": "Runs on both CPU and GPU hosts without requiring GPU acceleration."
  }
}
```

For GPU-heavy tools, set:

- `supported_accelerators` to include `"gpu"`
- `requires_gpu` to `true` if there is no CPU fallback
- `allow_cpu_fallback` appropriately

## Recommended grading criteria

- input/output contract compliance
- deterministic execution
- artifact quality
- explanation quality
- integration with ChatClinic Studio

## Student submission checklist

- `tool.json` is valid JSON
- `name` is unique and stable
- `team` is clearly set
- `task_type` matches the actual function of the tool
- `run.py` works with `--input` and `--output`
- output JSON is valid
- the tool can run on the provided classroom environment
- the result can be rendered in `ChatClinic` as an artifact
- `skill_patch.md` clearly explains when and why the tool should be called
- `skill_patch.md` does not replace the entire master Skill
- `skill_rationale.md` explains the orchestration reason in plain language

## What to avoid

- do not require your own long-running server unless explicitly approved
- do not hard-code absolute paths to a personal machine
- do not assume internet access during execution
- do not rely on manual GUI interaction to finish the task
- do not return only free-form text when structured artifacts are available

## Suggested classroom workflow

1. Instructor provides a shared ChatClinic runner.
2. Each team implements one plugin.
3. Instructor reviews whether the orchestration Skill also needs an update.
4. ChatClinic proposes a tool call in chat.
5. User approves execution.
6. The tool runs and returns artifacts.
7. Studio visualizes the result.
8. The instructor merges accepted Skill patches into one master `SKILL.md`.

## Related documents

- [Tool plugin guide](TOOL_PLUGIN_GUIDE.md)
- [Skill patch template](SKILL_PATCH_TEMPLATE.md)
- [Submission site specification](SUBMISSION_SITE_SPEC.md)
- [Master Skill integration guide](MASTER_SKILL_INTEGRATION.md)
