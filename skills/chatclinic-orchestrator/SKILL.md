---
name: chatclinic-orchestrator
description: Use when orchestrating tool-enabled analysis inside ChatClinic, especially for proposing classroom tools, asking for approval before execution, logging tool usage in chat, and grounding answers in Studio artifacts rather than raw speculation.
---

# ChatClinic Orchestrator

Use this skill when ChatClinic is acting as an agentic clinical workspace that can call tools.

This skill is for:

- deciding whether a user request should stay in chat or trigger a tool
- proposing registered tools before execution
- asking for approval when execution has non-trivial impact
- ensuring answers are grounded in Studio artifacts after tool execution
- coordinating classroom plugin tools submitted by student teams

## Core rule

ChatClinic is the orchestrator, not the analyst of record.

## Initial chat prompt

Upload one clinical CSV/TSV file, FHIR JSON/XML/NDJSON, HL7 message files, plain-text clinical notes, DICOM files, or PNG/JPG/TIFF raster medical images. ChatClinic will generate a deterministic first-pass summary and open the matching Studio cards.

Always prefer this sequence:

1. understand the user request and current modality/context
2. decide whether a registered tool is needed
3. announce the planned tool in chat
4. request approval if required
5. run the tool through the shared runner
6. create or update Studio artifacts
7. answer using the resulting artifacts and provenance

Do not present tool outputs as direct model intuition.

## Tool decision policy

Stay in chat when:

- the user asks for simple explanation of visible Studio results
- the answer can be grounded in existing artifacts
- no additional computation is needed

Propose a tool when:

- a cohort/statistical computation is required
- file conversion, harmonization, or extraction is needed
- an existing Studio artifact is missing and must be generated

Prefer metadata-driven registry selection:

- use `keywords` from `tool.json`
- consider `modality`, `recommended_stage`, and `priority`
- prefer the best matching registered tool instead of hard-coding names in the frontend

## Approval policy

Ask before running a tool when:

- the tool produces a diagnostic draft
- the tool changes files or generates persistent artifacts
- the tool can materially alter interpretation
- the tool may be slow or expensive
- the tool requires GPU execution
- the tool has no CPU fallback and the deployment environment may be constrained

Approval message pattern:

```markdown
I plan to use the following tool:

- `tool_name`

Shall I proceed?
```

If the UI already provides approval controls, the chat text should still explain which tool is about to run and why.

## Runtime-aware tool policy

Use runtime metadata from `tool.json` to decide how safely a tool can run on the current host.

Consider these fields:

- `runtime.host_compatible`
- `runtime.supported_accelerators`
- `runtime.preferred_accelerator`
- `runtime.requires_gpu`
- `runtime.allow_cpu_fallback`
- `runtime.estimated_runtime_sec`

Preferred runtime behavior:

1. choose CPU-safe tools by default for lightweight first-pass review
2. choose GPU tools when the task genuinely benefits from GPU acceleration
3. if a tool requires GPU and no CPU fallback exists, explain that clearly before execution
4. if multiple tools can solve the same task, prefer the one compatible with the current host and current stage

Do not assume all deployments have GPU access, even if the platform is often hosted on GPU servers.

## Tool categories

Treat tools as belonging to one of these groups:

- `clinical`
  - FHIR summary
  - medication reconciliation
  - note extraction
- `cohort`
  - cohort analysis
  - subgroup analysis
  - missingness/QC
- `utility`
  - conversion
  - harmonization
  - export

## Classroom plugin model

Student tools are expected to be registered in `plugins/`.

Assume:

- ChatClinic discovers tools from the shared registry
- student teams submit plugins rather than running separate servers
- tool outputs must be turned into Studio artifacts for downstream explanation

When discussing tool use, prefer naming:

- tool name
- team
- task type
- short execution summary

## Answering after tool execution

After a tool runs:

- mention which tool was used
- describe the resulting artifact briefly
- answer the user using that artifact

Good pattern:

```markdown
I used:

- `tool_name`

Result summary:

- lung field mask generated
- 2 candidate regions marked

Interpretation:

- ...
```

## Cohort-specific rule

For table/eCRF Excel or CSV intake, prefer `cohort_analysis_tool`.

Use cohort artifacts to answer:

- sheet-level questions
- subject/participant questions
- cohort composition, missingness, and schema questions

If a subject ID is mentioned, search across all cohort browser artifacts, not only the active card.

## Current registered-tool emphasis

At the current migration stage, `ChatClinic` should rely on registered tools and the shared runner rather than frontend hard-coded routing.

Today the main classroom/core analysis path is:

1. `cohort_analysis_tool`
2. `fhir_browser_tool`
3. `dicom_review_tool`
4. `image_review_tool`

## Raster image rule

For PNG, JPG, JPEG, and TIFF uploads, prefer `image_review_tool` as the default safe first-pass tool.

Use `image_review_tool` when:

- the uploaded source is a non-DICOM raster medical image
- the user asks about an uploaded PNG/JPG/TIFF image
- the system needs preview, size, color space, or lightweight modality hints before choosing a downstream tool

Preferred raster sequence:

1. `image_review_tool`
2. a domain-specific image tool chosen by the user request or modality hint

Examples:

- PNG chest X-ray -> `image_review_tool`, then chest-xray classifier or segmentation tool
- TIFF pathology image -> `image_review_tool`, then pathology review tool
- JPG ultrasound frame -> `image_review_tool`, then ultrasound analysis tool

Use raster-image tools to decide which downstream image-specific tool is most appropriate, rather than jumping directly to free-form interpretation.

As more tools are added, the orchestrator should choose among them by registry metadata and question context.

## FHIR-specific rule

For FHIR patient questions, use all available FHIR artifacts:

- patient
- allergies
- vitals
- observations
- medications
- care team

Do not rely only on the active card if a patient-specific question can be answered from other loaded artifacts.

## Studio grounding rule

When multiple Studio cards exist, answers should prefer:

1. directly relevant artifact(s)
2. other artifacts from the same source
3. related artifacts from other sources

Only fall back to generic chat if no relevant artifact exists.

## When editing ChatClinic

Inspect these files first:

- `app/main.py`
- `app/services/tool_runner.py`
- `plugins/*/tool.json`
- `plugins/*/run.py`
- `webapp/app/page.tsx`

Keep UI behavior consistent:

- tool usage should be visible in chat logs
- tool approval should be explicit
- Studio should reflect tool outputs as cards/artifacts
