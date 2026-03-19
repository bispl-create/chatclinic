# ChatClinic

Clinical data and medical imaging analysis workspace for teaching and building **Agentic AI**.

> [!IMPORTANT]
> **Revision history**
> - **March 2026 update:** ChatClinic now supports CPU/GPU-aware tool runtime metadata and first-pass raster medical image intake for `PNG`, `JPG/JPEG`, and `TIFF` via `image_review_tool`.

![ChatClinic UI preview](docs/chatclinic-ui-preview.svg)

## Why this project exists

`ChatClinic` is designed for a class in which students learn how to build an AI system that can accept **clinical data** and **medical images**, then perform:

- integrated analysis
- diagnosis support
- multimodal reasoning
- structured review
- prediction-oriented downstream workflows

The central educational goal is not merely to build a chatbot. It is to show students how to build an **Agentic AI system** that can coordinate multiple specialized tools under the guidance of an orchestrating LLM.

In this project, the AI may receive:

- clinical tables such as CSV, TSV, and Excel eCRF files
- FHIR JSON/XML/NDJSON
- HL7 message files
- plain-text clinical notes
- DICOM medical imaging files
- raster medical images such as PNG, JPG/JPEG, and TIFF

and combine them into a single, grounded workflow.

## Educational perspective: what students should learn

This repository is written for **beginner students** who are learning what Agentic AI actually means in practice.

The key lesson is:

- an LLM should not do every calculation itself
- the LLM should decide **what to do next**
- deterministic tools should perform the actual analysis
- the system should keep results as structured artifacts
- the UI should make the reasoning trace visible

This lets students see that a modern AI system is not just:

- `user -> chatbot -> answer`

but rather:

- `user -> orchestrator -> tools -> artifacts -> grounded answer`

That is the practical teaching value of Agentic AI in this course.

## The main idea: orchestrator LLM + Skill + tools

`ChatClinic` separates the system into three educational layers:

1. **Orchestrator LLM**
   - reads the current request and context
   - decides whether a tool is needed
   - explains results after tools finish

2. **Skill**
   - stores the orchestration prompt and policy in `SKILL.md`
   - describes when tools should be used
   - controls tool ordering, approval rules, and answer style

3. **Tool Registry**
   - contains the available analysis tools
   - lets the orchestrator choose among them
   - makes the system easy to extend with new student-built tools

This design is intentionally educational: students can change the behavior of the agent by editing a **Skill** rather than hunting through backend logic only.

## Why Skill-based orchestration is important

One of the most important concepts in this class is that the logic for calling tools is not hidden entirely inside backend code.

Instead, the orchestration rule can live in a readable prompt file:

- which tool should be preferred
- when approval should be requested
- what order tools should run in
- how the LLM should explain results

This has a major teaching advantage:

- instructors can easily change the workflow
- students can understand the current orchestration policy
- the behavior of the system can be discussed as part of prompt design
- new tools can be added without rewriting the whole application

In short:

- `Skill` = the instructions that guide the orchestrator LLM
- `Tool` = the deterministic function that actually performs the analysis

## High-level diagram

```mermaid
flowchart LR
    U["User"] --> UI["ChatClinic UI<br/>Sources / Chat / Studio"]
    UI --> LLM["Orchestrator LLM"]
    LLM --> SK["Skill<br/>prompt-based workflow policy"]
    LLM --> REG["Tool Registry"]
    REG --> RUN["Shared Runner"]
    RUN --> T1["Cohort Analysis Tool"]
    RUN --> T2["FHIR Browser Tool"]
    RUN --> T3["DICOM Review Tool"]
    RUN --> T4["Image Review Tool"]
    T1 --> ART["Structured artifacts"]
    T2 --> ART
    T3 --> ART
    T4 --> ART
    ART --> UI
    UI --> ANS["Grounded explanation,<br/>reasoning, and next-step response"]
```

## Why this is a good classroom architecture

For teaching, this architecture is simpler and more powerful than asking each student team to build a separate full application.

The recommended model is:

- the instructor operates one shared `ChatClinic`
- the instructor maintains the orchestration `Skill`
- each student team submits one or more tool plugins
- the shared runner executes those tools
- the UI shows the resulting artifacts in `Studio`

This helps students focus on what matters:

- building a useful analysis tool
- defining its inputs and outputs clearly
- understanding how an orchestrator LLM chooses and uses tools

## What makes this Agentic AI

This project is an Agentic AI system because the LLM is used as an **orchestrator**, not merely a text generator.

It can:

1. inspect uploaded data and user intent
2. decide whether a tool is needed
3. follow the policy written in a `SKILL.md` file
4. choose the appropriate tool from the registry
5. execute the tool through the shared runner
6. receive structured outputs
7. produce a grounded answer based on the resulting artifacts

This is exactly the kind of system architecture students should understand if they want to build reliable multimodal AI systems for healthcare.

## What students can build in this class

Student teams can contribute tools such as:

- cohort analysis
- FHIR patient browsing
- DICOM review
- QC and harmonization tools
- segmentation
- detection
- structured reporting
- prediction and triage tools

As long as a tool follows the plugin contract, it can be registered and orchestrated by `ChatClinic`.

## Classroom materials

The main `ChatClinic` repository should stay focused on the platform itself.

For class signup, plugin submission, and Skill patch workflow, use the separate classroom materials repository:

- `chatclinic-class` (recommended separate repo/workspace for students and instructors)

That classroom repo should contain:

- tool submission rules
- Skill patch templates
- signup and final submission forms/specifications
- instructor merge workflow for the master Skill

## Core project pieces

- `skills/chatclinic-orchestrator/SKILL.md`
  - orchestration prompt and policy for the LLM
- `plugins/`
  - registered tools
- `app/services/tool_runner.py`
  - shared runner that executes tools
- `app/services/skill_orchestrator.py`
  - Skill-guided routing helpers
- `webapp/app/page.tsx`
  - UI for Sources, Chat, Studio, and review panels

## Example tools in this repository

Current example tools include:

- `plugins/cohort_sheet_browser/`
- `plugins/fhir_browser_tool/`
- `plugins/dicom_review_tool/`
- `plugins/image_review_tool/`

Together they demonstrate how one agentic system can support:

- cohort-oriented structured data analysis
- patient-centered clinical browsing
- medical image review
- raster image preview and lightweight modality triage

under one orchestration layer.

## Runtime model for current tools

Current tools are registered with runtime metadata so that the same platform can run on:

- CPU-only hosts
- GPU-equipped servers

The current built-in tools do not require GPU acceleration, but they now declare runtime compatibility explicitly in `tool.json`. This keeps the current classroom tools simple while allowing future student tools to prefer or require GPU execution without changing the overall orchestration model.

## Example data

The `examples/` folder includes demo files such as:

- cohort Excel sheets
- FHIR JSON/XML and bulk NDJSON
- HL7 messages
- chest X-ray and DICOM examples

These are intended for both classroom demonstration and student tool development.

## Local companion repo

If you are organizing a course, keep a separate companion repository for course operations, for example:

- `/Users/jongcye/Documents/Codex/workspace/chatclinic-class`

That repo can store:

- team signup instructions
- plugin submission templates
- Skill patch proposals
- instructor review workflow

## Quick start

```bash
cd /Users/jongcye/Documents/Codex/workspace/clinical_multimodal_workspace
cp .env.example .env
```

Then set:

```bash
OPENAI_API_KEY=sk-...
OPENAI_WORKFLOW_MODEL=gpt-5-nano
OPENAI_MODEL=gpt-5-mini
```

## Summary

`ChatClinic` is a teaching-oriented multimodal clinical Agentic AI workspace.

Its educational message is simple:

- the LLM should orchestrate
- the Skill should describe the orchestration policy
- the tools should perform the actual analysis
- the UI should expose the artifacts clearly

That separation makes the system easier to teach, easier to extend, and much better suited for project-based learning in medical AI.
