"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import DicomInteractiveViewer from "./components/DicomInteractiveViewer";

type IntakeResponse = {
  source: {
    file_name: string;
    file_type: string;
    modality: string;
    size_bytes: number;
    status: string;
  };
  grounded_summary: string;
  studio_cards: Array<{
    id: string;
    title: string;
    subtitle: string;
    base_id?: string;
    source_index?: number;
    source_name?: string;
    source_modality?: string;
  }>;
  artifacts: Record<string, unknown>;
  sources?: Array<{
    file_name: string;
    file_type: string;
    modality: string;
    size_bytes: number;
    status: string;
  }>;
  used_tools?: string[];
};

type ToolInfo = {
  name: string;
  team?: string | null;
  task_type?: string | null;
  modality?: string | null;
  approval_required?: boolean;
  description?: string | null;
};

type ToolRunResponse = {
  tool: ToolInfo;
  summary: string;
  artifacts: Record<string, unknown>;
  provenance: Record<string, unknown>;
  stdout?: string;
  stderr?: string;
};

type ToolSuggestion = {
  tool: ToolInfo;
  rationale: string;
};

type PendingToolRequest = {
  tool: ToolInfo;
  question: string;
  rationale?: string;
};

type ToolSuggestionResponse = {
  tool?: ToolInfo | null;
  rationale?: string | null;
};

type UiBootstrapResponse = {
  initial_chat_prompt?: string;
};

type SchemaProfile = {
  name: string;
  inferred_type: string;
  non_empty_count: number;
  missing_count: number;
  missing_rate: number;
  unique_count: number;
  sample_values: string[];
  numeric_summary?: {
    min: number;
    max: number;
    mean: number;
  } | null;
};

type CohortArtifact = {
  record_count: number;
  field_count: number;
  categorical_breakdowns: Array<{
    column: string;
    top_values: Array<{ label: string; count: number }>;
  }>;
  numeric_breakdowns: Array<{
    column: string;
    summary: { min: number; max: number; mean: number } | null;
  }>;
};

type TableIntakeArtifact = {
  analysis_mode?: string;
  cohort_score?: number;
  single_patient_score?: number;
  subject_column?: string | null;
  subject_unique_count?: number;
  visit_columns?: string[];
  site_columns?: string[];
  arm_columns?: string[];
  rationale?: string[];
  row_count?: number;
  column_count?: number;
  table_meta?: {
    workbook_format?: string;
    sheet_names?: string[];
    selected_sheet?: string;
    merged_from_sheets?: string[];
  };
};

type TableRolesArtifact = {
  subject_id_columns?: string[];
  visit_columns?: string[];
  site_columns?: string[];
  arm_columns?: string[];
  date_columns?: string[];
  outcome_columns?: string[];
};

type MissingnessArtifact = {
  top_missing_columns?: Array<{
    column: string;
    missing_rate: number;
    missing_count: number;
    non_empty_count: number;
  }>;
};

type CohortBrowserArtifact = {
  overview?: {
    row_count?: number;
    column_count?: number;
    subject_count?: number;
    visit_count?: number;
    site_count?: number;
    arm_count?: number;
    completeness_rate?: number;
    analysis_mode?: string;
    selected_sheet?: string;
  };
  composition?: {
    site_distribution?: Array<{ label: string; count: number }>;
    arm_distribution?: Array<{ label: string; count: number }>;
    outcome_distribution?: Array<{ label: string; count: number }>;
    age_histogram?: Array<{ label: string; count: number }>;
  };
  domains?: Array<{
    sheet_name: string;
    domain: string;
    row_count: number;
    subject_count: number;
    subject_column: string;
    visit_columns?: string[];
    date_columns?: string[];
  }>;
  subjects?: Array<{
    subject_id: string;
    record_count: number;
    site: string;
    arm: string;
    latest_outcome: string;
    visits: string[];
  }>;
  grid?: {
    columns?: string[];
    rows?: Array<Record<string, string>>;
    row_count?: number;
  };
  schema_highlights?: Array<{
    name: string;
    inferred_type: string;
    missing_count: number;
    unique_count: number;
  }>;
  roles?: TableRolesArtifact;
  missingness?: MissingnessArtifact;
  intake?: TableIntakeArtifact;
};

type CohortBrowserDomain = {
  sheet_name: string;
  domain: string;
  row_count: number;
  subject_count: number;
  subject_column: string;
  visit_columns?: string[];
  date_columns?: string[];
};

type CohortBrowserSubject = {
  subject_id: string;
  record_count: number;
  site: string;
  arm: string;
  latest_outcome: string;
  visits: string[];
};

type DicomMetadataItem = {
  file_name: string;
  patient_id: string;
  study_instance_uid: string;
  series_instance_uid: string;
  study_description: string;
  series_description: string;
  modality: string;
  rows: string;
  columns: string;
  instance_number: string;
  preview?: {
    available: boolean;
    image_data_url: string | null;
    message: string;
  };
  preview_presets?: Record<
    string,
    {
      available: boolean;
      image_data_url: string | null;
      message: string;
      label?: string;
    }
  >;
};

function renderMetadataRows(rows: Array<{ label: string; value: string | undefined | null }>) {
  return (
    <div className="metadataTable">
      {rows.map((row) => (
        <div key={row.label} className="metadataRow">
          <span className="metadataLabel">{row.label}</span>
          <span className="metadataValue">{row.value && String(row.value).trim() ? row.value : "n/a"}</span>
        </div>
      ))}
    </div>
  );
}

function formatAge(birthDate?: string) {
  if (!birthDate || birthDate === "n/a") {
    return "n/a";
  }
  const birth = new Date(birthDate);
  if (Number.isNaN(birth.getTime())) {
    return "n/a";
  }
  const now = new Date();
  let age = now.getFullYear() - birth.getFullYear();
  const monthDiff = now.getMonth() - birth.getMonth();
  if (monthDiff < 0 || (monthDiff === 0 && now.getDate() < birth.getDate())) {
    age -= 1;
  }
  return String(age);
}

function dateToNumber(value?: string) {
  if (!value || value === "n/a") {
    return null;
  }
  const parsed = new Date(value).getTime();
  return Number.isFinite(parsed) ? parsed : null;
}

function clampPercent(value: number) {
  return Math.max(0, Math.min(100, value));
}

function renderBarList(items: Array<{ label: string; count: number }>, emptyLabel: string) {
  const max = Math.max(...items.map((item) => item.count), 1);
  if (!items.length) {
    return <p className="mutedText">{emptyLabel}</p>;
  }
  return (
    <div className="cohortBarList">
      {items.map((item) => (
        <div key={item.label} className="cohortBarRow">
          <div className="cohortBarMeta">
            <span className="cohortBarLabel">{item.label}</span>
            <span className="cohortBarValue">{item.count}</span>
          </div>
          <div className="cohortBarTrack">
            <div className="cohortBarFill" style={{ width: `${(item.count / max) * 100}%` }} />
          </div>
        </div>
      ))}
    </div>
  );
}

function asArray<T>(value: unknown): T[] {
  return Array.isArray(value) ? (value as T[]) : [];
}

type DicomSeriesArtifact = {
  series: Array<{
    series_instance_uid: string;
    study_instance_uid: string;
    modality: string;
    study_description: string;
    series_description: string;
    instance_count: number;
    example_files: string[];
    all_files?: string[];
    preview?: {
      available: boolean;
      image_data_url: string | null;
      message: string;
    };
    preview_presets?: Record<
      string,
      {
        available: boolean;
        image_data_url: string | null;
        message: string;
        label?: string;
      }
    >;
  }>;
};

type ViewerSeriesGroup = {
  id: string;
  label: string;
  files: Array<{
    fileName: string;
    preview?: {
      available: boolean;
      image_data_url: string | null;
      message: string;
    };
    preview_presets?: Record<
      string,
      {
        available: boolean;
        image_data_url: string | null;
        message: string;
        label?: string;
      }
    >;
  }>;
};

type FhirPatientArtifact = {
  resource_type?: string;
  id?: string;
  full_name?: string;
  gender?: string;
  birth_date?: string;
  active?: string;
  managing_organization?: string;
  identifiers?: Array<{ system?: string; value?: string; use?: string }>;
  telecom?: Array<{ system?: string; value?: string; use?: string }>;
  addresses?: Array<{ line?: string; city?: string; state?: string; postalCode?: string; country?: string }>;
};

type FhirObservationArtifact = {
  count?: number;
  items?: Array<{
    code?: string;
    value?: string;
    status?: string;
    effective?: string;
    category?: string;
    numeric_value?: number | null;
    unit?: string;
    reference_low?: number | null;
    reference_high?: number | null;
  }>;
};

type FhirMedicationArtifact = {
  count?: number;
  items?: Array<{
    medication?: string;
    status?: string;
    intent?: string;
    date?: string;
    dosage?: string;
    start?: string;
    end?: string;
    duration_days?: number | null;
    current?: boolean;
  }>;
};

type FhirAllergyArtifact = {
  count?: number;
  items?: Array<{
    substance?: string;
    criticality?: string;
    clinical_status?: string;
    verification_status?: string;
  }>;
};

type FhirVitalArtifact = {
  items?: Array<{ label?: string; value?: string; effective?: string; status?: string }>;
};

type FhirTimelineArtifact = {
  events?: Array<{ type?: string; label?: string; start?: string; end?: string; status?: string }>;
};

type FhirLabArtifact = {
  series?: Array<{
    label?: string;
    points?: Array<{ date?: string; value?: number; unit?: string; low?: number | null; high?: number | null }>;
  }>;
  latest?: Array<{ label?: string; value?: number | string; unit?: string; low?: number | null; high?: number | null }>;
};

type FhirCareTeamArtifact = {
  practitioners?: Array<{ name?: string; role?: string; contact?: string; organization?: string }>;
  organizations?: Array<{ name?: string; contact?: string }>;
};

export default function Page() {
  const defaultInitialPrompt =
    "Upload one clinical CSV/TSV file, FHIR JSON/XML/NDJSON, HL7 message files, plain-text clinical notes, or DICOM files. ChatClinic will generate a deterministic first-pass summary and open the matching Studio cards.";
  const chatStreamRef = useRef<HTMLDivElement | null>(null);
  const [apiBase, setApiBase] = useState("http://127.0.0.1:8010");
  const [attachedFiles, setAttachedFiles] = useState<File[]>([]);
  const [status, setStatus] = useState("Waiting for a clinical source");
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<IntakeResponse | null>(null);
  const [chatTurns, setChatTurns] = useState<Array<{ role: "assistant" | "user"; content: string }>>([
    {
      role: "assistant",
      content: defaultInitialPrompt,
    },
  ]);
  const [composerText, setComposerText] = useState("");
  const [activeStudioView, setActiveStudioView] = useState<string | null>(null);
  const [cohortGridFilter, setCohortGridFilter] = useState("");
  const [cohortGridPage, setCohortGridPage] = useState(0);
  const [availableTools, setAvailableTools] = useState<ToolInfo[]>([]);
  const [pendingTool, setPendingTool] = useState<PendingToolRequest | null>(null);
  const [toolRegistryOpen, setToolRegistryOpen] = useState(false);

  const usedTools = useMemo(() => result?.used_tools ?? [], [result]);

  useEffect(() => {
    if (!chatStreamRef.current) {
      return;
    }
    chatStreamRef.current.scrollTo({
      top: chatStreamRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [chatTurns, status]);

  useEffect(() => {
    setCohortGridPage(0);
  }, [activeStudioView, cohortGridFilter]);

  useEffect(() => {
    let cancelled = false;
    async function loadTools() {
      try {
        const response = await fetch(`${apiBase.replace(/\/$/, "")}/api/v1/tools`);
        if (!response.ok) {
          return;
        }
        const payload = (await response.json()) as { tools?: ToolInfo[] };
        if (!cancelled) {
          setAvailableTools(payload.tools ?? []);
        }
      } catch {
        if (!cancelled) {
          setAvailableTools([]);
        }
      }
    }
    void loadTools();
    return () => {
      cancelled = true;
    };
  }, [apiBase]);

  useEffect(() => {
    let cancelled = false;
    async function loadBootstrap() {
      try {
        const response = await fetch(`${apiBase.replace(/\/$/, "")}/api/v1/ui/bootstrap`);
        if (!response.ok) {
          return;
        }
        const payload = (await response.json()) as UiBootstrapResponse;
        const prompt = payload.initial_chat_prompt?.trim();
        if (!cancelled && prompt) {
          setChatTurns((current) => {
            if (!current.length || current[0]?.role !== "assistant") {
              return current;
            }
            const next = [...current];
            next[0] = { ...next[0], content: prompt };
            return next;
          });
        }
      } catch {
        return;
      }
    }
    void loadBootstrap();
    return () => {
      cancelled = true;
    };
  }, [apiBase]);

  function mergeFiles(currentFiles: File[], incomingFiles: File[]) {
    const seen = new Set<string>();
    const merged: File[] = [];
    for (const file of [...currentFiles, ...incomingFiles]) {
      const key = `${file.name}::${file.size}::${file.lastModified}`;
      if (seen.has(key)) {
        continue;
      }
      seen.add(key);
      merged.push(file);
    }
    return merged;
  }

  function getSourceArtifact(baseId: string, sourceIndex?: number) {
    if (!result) {
      return null;
    }
    const artifacts = result.artifacts ?? {};
    if (typeof sourceIndex === "number") {
      return artifacts[`source${sourceIndex}::${baseId}`] ?? null;
    }
    return artifacts[baseId] ?? null;
  }

  async function handleFileChange(event: React.ChangeEvent<HTMLInputElement>) {
    const incomingFiles = Array.from(event.target.files ?? []);
    if (!incomingFiles.length) {
      return;
    }
    const mergedFiles = mergeFiles(attachedFiles, incomingFiles);
    const isWorkbookUpload = mergedFiles.some((file) => {
      const lowered = file.name.toLowerCase();
      return lowered.endsWith(".xlsx") || lowered.endsWith(".xlsm") || lowered.endsWith(".xls");
    });
    setAttachedFiles(mergedFiles);
    setActiveStudioView(null);
    setError(null);
    setStatus(
      isWorkbookUpload
        ? attachedFiles.length
          ? "Uploading and merging..."
          : "Uploading and parsing..."
        : attachedFiles.length
          ? "Uploading and merging..."
          : "Uploading and parsing...",
    );

    const formData = new FormData();
    for (const file of mergedFiles) {
      formData.append("files", file);
    }

    try {
      const response = await fetch(`${apiBase.replace(/\/$/, "")}/api/v1/intake/upload`, {
        method: "POST",
        body: formData,
      });
      if (!response.ok) {
        throw new Error(await response.text());
      }
      const payload: IntakeResponse = await response.json();
      setResult(payload);
      setStatus("Summary ready");
      setChatTurns((current) => [...current, { role: "assistant", content: payload.grounded_summary }]);
    } catch (caught) {
      const message = caught instanceof Error ? caught.message : String(caught);
      setError(message);
      setStatus("Upload failed");
    } finally {
      event.target.value = "";
    }
  }

  async function handleChatSubmit() {
    const text = composerText.trim();
    if (!text) {
      return;
    }

    if (!result) {
      setChatTurns((current) => [
        ...current,
        { role: "user", content: text },
        {
          role: "assistant",
          content: "Upload a clinical source first so ChatClinic has deterministic artifacts to explain.",
        },
      ]);
      setComposerText("");
      return;
    }

    let suggestion: ToolSuggestion | null = null;
    try {
      const suggestionResponse = await fetch(`${apiBase.replace(/\/$/, "")}/api/v1/tools/suggest`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: text,
          analysis: result,
          active_view: activeStudioView,
          active_card: activeStudioCard,
          active_artifact: studioArtifacts,
        }),
      });
      if (suggestionResponse.ok) {
        const suggestionPayload = (await suggestionResponse.json()) as ToolSuggestionResponse;
        if (suggestionPayload.tool?.name) {
          suggestion = {
            tool: suggestionPayload.tool,
            rationale: suggestionPayload.rationale ?? "The orchestrator selected a registered tool for this request.",
          };
        }
      }
    } catch {
      suggestion = null;
    }

    if (suggestion) {
      setChatTurns((current) => [
        ...current,
        { role: "user", content: text },
        {
          role: "assistant",
          content:
            "I plan to use the following tool:\n\n" +
            `- \`${suggestion.tool.name}\`\n\n` +
            `${suggestion.rationale}\n\nShall I proceed?`,
        },
      ]);
      setComposerText("");
      setPendingTool({ tool: suggestion.tool, question: text, rationale: suggestion.rationale });
      setStatus("Awaiting approval");
      return;
    }

    setStatus("Generating answer...");
    setChatTurns((current) => [...current, { role: "user", content: text }]);
    setComposerText("");
    await new Promise((resolve) => window.requestAnimationFrame(() => resolve(undefined)));
    try {
      const response = await fetch(`${apiBase.replace(/\/$/, "")}/api/v1/chat/artifacts`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: text,
          analysis: result,
          history: chatTurns,
          active_view: activeStudioView,
          active_card: activeStudioCard,
          active_artifact: studioArtifacts,
        }),
      });
      if (!response.ok) {
        throw new Error(await response.text());
      }
      const payload = await response.json();
      setChatTurns((current) => [...current, { role: "assistant", content: payload.answer }]);
      setStatus("Answer ready");
    } catch (caught) {
      const message = caught instanceof Error ? caught.message : String(caught);
      setChatTurns((current) => [
        ...current,
        { role: "assistant", content: `Chat request failed: ${message}` },
      ]);
      setStatus("Answer failed");
    }
  }

  const studioArtifacts = useMemo(() => {
    if (!result || !activeStudioView) {
      return null;
    }
    return (result.artifacts ?? {})[activeStudioView] ?? null;
  }, [activeStudioView, result]);

  const displayStudioCards = useMemo(() => {
    const cards = Array.isArray(result?.studio_cards) ? result.studio_cards : [];

    const normalized: typeof cards = [];
    const seenIds = new Set<string>();
    let cohortSourceIndex: number | undefined;
    let cohortSourceName: string | undefined;
    let cohortSourceModality: string | undefined;
    let hasLegacyCohortCards = false;

    for (const card of cards) {
      const baseId = card.base_id ?? card.id;
      if (["intake", "schema", "cohort", "roles", "missingness"].includes(baseId)) {
        hasLegacyCohortCards = true;
        if (typeof card.source_index === "number") {
          cohortSourceIndex = card.source_index;
        }
        cohortSourceName = card.source_name ?? cohortSourceName;
        cohortSourceModality = card.source_modality ?? cohortSourceModality;
        continue;
      }
      if (seenIds.has(card.id)) {
        continue;
      }
      seenIds.add(card.id);
      normalized.push(card);
    }

    if (hasLegacyCohortCards && !normalized.some((card) => (card.base_id ?? card.id) === "cohort_browser")) {
      normalized.unshift({
        id: typeof cohortSourceIndex === "number" ? `source${cohortSourceIndex}::cohort_browser` : "cohort_browser",
        title: "Cohort Browser",
        subtitle: "Intake, schema, and cohort review in one card",
        base_id: "cohort_browser",
        source_index: cohortSourceIndex,
        source_name: cohortSourceName,
        source_modality: cohortSourceModality,
      });
    }

    return normalized;
  }, [result]);
  const hasStudioCards = displayStudioCards.length > 0;

  const activeStudioCard = useMemo(() => {
    if (!activeStudioView) {
      return null;
    }
    return displayStudioCards.find((card) => card.id === activeStudioView) ?? null;
  }, [activeStudioView, displayStudioCards]);

  const activeBaseView = activeStudioCard?.base_id ? activeStudioCard.base_id : activeStudioView ? activeStudioView : null;
  const activeSourceIndex = activeStudioCard?.source_index;

  const visibleSources = useMemo(() => {
    if (Array.isArray(result?.sources) && result.sources.length) {
      return result.sources;
    }
    return attachedFiles.map((file) => ({
      file_name: file.name,
      file_type: file.name.split(".").pop() ?? "unknown",
      modality: "pending",
      size_bytes: file.size,
      status: "pending",
    }));
  }, [attachedFiles, result]);

  const dicomSeriesGroups = useMemo<ViewerSeriesGroup[]>(() => {
    const metadataArtifact = getSourceArtifact("metadata", activeSourceIndex) as
      | ({
          items?: DicomMetadataItem[];
          patient_id?: string;
          study_description?: string;
          modality?: string;
          rows?: string;
          columns?: string;
          preview?: {
            available: boolean;
            image_data_url: string | null;
            message: string;
          };
          preview_presets?: Record<
            string,
            {
              available: boolean;
              image_data_url: string | null;
              message: string;
              label?: string;
            }
          >;
        } & Record<string, unknown>)
      | undefined;
    const metadataItems = (metadataArtifact?.items ?? []).filter(Boolean);
    const seriesArtifact = (getSourceArtifact("series", activeSourceIndex) as DicomSeriesArtifact | undefined)?.series ?? [];
    if (!metadataItems.length && metadataArtifact?.preview) {
      return [
        {
          id: "single-dicom",
          label: activeStudioCard?.source_name ?? result?.source.file_name ?? "DICOM file",
          files: [
            {
              fileName: activeStudioCard?.source_name ?? result?.source.file_name ?? "uploaded.dcm",
              preview: metadataArtifact.preview,
              preview_presets: metadataArtifact.preview_presets,
            },
          ],
        },
      ];
    }

    if (!metadataItems.length) {
      return [];
    }

    if (!seriesArtifact.length) {
      return [
        {
          id: "single-series",
          label: metadataItems.length === 1 ? metadataItems[0].file_name : `All DICOM files (${metadataItems.length})`,
          files: metadataItems.map((item) => ({
            fileName: item.file_name,
            preview: item.preview,
            preview_presets: item.preview_presets,
          })),
        },
      ];
    }

    const itemByName = new Map(metadataItems.map((item) => [item.file_name, item]));
    const groups: ViewerSeriesGroup[] = [];
    for (const series of seriesArtifact) {
        const files = (series.all_files ?? series.example_files)
          .map((fileName) => itemByName.get(fileName))
          .filter((item): item is DicomMetadataItem => Boolean(item))
          .map((item) => ({
            fileName: item.file_name,
            preview: item.preview,
            preview_presets: item.preview_presets,
          }));
        if (!files.length) {
          continue;
        }
        groups.push({
          id: series.series_instance_uid,
          label: `${series.series_description || "Unnamed series"} (${series.instance_count})`,
          files,
        });
      }

    return groups.length
      ? groups
      : [
          {
            id: "all-dicom",
            label: `All DICOM files (${metadataItems.length})`,
            files: metadataItems.map((item) => ({
              fileName: item.file_name,
              preview: item.preview,
              preview_presets: item.preview_presets,
            })),
          },
        ];
  }, [result, activeSourceIndex, activeStudioCard?.source_name]);

  function renderStudioCanvas() {
    if (!result || !activeBaseView) {
      return <p className="mutedText">Upload a source, then open a Studio card to inspect deterministic artifacts.</p>;
    }

    if (activeBaseView === "cohort_browser") {
      const browser =
        ((activeStudioView ? result.artifacts[activeStudioView] : null) as CohortBrowserArtifact | undefined) ??
        (getSourceArtifact("cohort_browser", activeSourceIndex) as CohortBrowserArtifact | undefined);
      const intake = browser?.intake ?? ((getSourceArtifact("intake", activeSourceIndex) as TableIntakeArtifact | undefined) ?? undefined);
      const schema = asArray<SchemaProfile>(
        browser?.schema_highlights ?? ((getSourceArtifact("schema", activeSourceIndex) as { profiles?: SchemaProfile[] } | undefined)?.profiles ?? []),
      );
      const cohort = browser?.overview;
      const composition = browser?.composition;
      const domains = asArray<CohortBrowserDomain>(browser?.domains ?? []);
      const roles = browser?.roles ?? ((getSourceArtifact("roles", activeSourceIndex) as TableRolesArtifact | undefined) ?? undefined);
      const missingness = browser?.missingness ?? ((getSourceArtifact("missingness", activeSourceIndex) as MissingnessArtifact | undefined) ?? undefined);
      const subjects = asArray<CohortBrowserSubject>(browser?.subjects ?? []);
      const gridColumns = asArray<string>(browser?.grid?.columns ?? []);
      const gridRows = asArray<Record<string, unknown>>(browser?.grid?.rows ?? []);
      const siteDistribution = asArray<{ label: string; count: number }>(composition?.site_distribution ?? []);
      const armDistribution = asArray<{ label: string; count: number }>(composition?.arm_distribution ?? []);
      const outcomeDistribution = asArray<{ label: string; count: number }>(composition?.outcome_distribution ?? []);
      const ageHistogram = asArray<{ label: string; count: number }>(composition?.age_histogram ?? []);
      const normalizedGridFilter = cohortGridFilter.trim().toLowerCase();
      const filteredGridRows = normalizedGridFilter
        ? gridRows.filter((row) =>
            gridColumns.some((column) => String(row?.[column] ?? "").toLowerCase().includes(normalizedGridFilter)),
          )
        : gridRows;
      const cohortGridPageSize = 100;
      const cohortGridPageCount = Math.max(1, Math.ceil(filteredGridRows.length / cohortGridPageSize));
      const safeCohortGridPage = Math.min(cohortGridPage, cohortGridPageCount - 1);
      const pagedGridRows = filteredGridRows.slice(
        safeCohortGridPage * cohortGridPageSize,
        safeCohortGridPage * cohortGridPageSize + cohortGridPageSize,
      );
      return (
        <div className="artifactStack">
          <section className="cohortBrowserSection">
            <div className="cohortSectionHeader">
              <strong>Overview</strong>
              <span className="mutedText">Cohort snapshot, schema, and intake reasoning</span>
            </div>
            <div className="cohortKpiGrid">
              <article className="artifactCard cohortKpiCard">
                <span className="cohortKpiLabel">Subjects</span>
                <strong>{cohort?.subject_count ?? intake?.subject_unique_count ?? "n/a"}</strong>
              </article>
              <article className="artifactCard cohortKpiCard">
                <span className="cohortKpiLabel">Records</span>
                <strong>{cohort?.row_count ?? intake?.row_count ?? "n/a"}</strong>
              </article>
              <article className="artifactCard cohortKpiCard">
                <span className="cohortKpiLabel">Visits</span>
                <strong>{cohort?.visit_count ?? "n/a"}</strong>
              </article>
              <article className="artifactCard cohortKpiCard">
                <span className="cohortKpiLabel">Completeness</span>
                <strong>{cohort?.completeness_rate ?? "n/a"}%</strong>
              </article>
            </div>
            <div className="cohortOverviewGrid">
              <article className="artifactCard">
                <strong>Table intake</strong>
                <p>Mode: {intake?.analysis_mode ?? "n/a"}</p>
                <p>
                  Cohort score {intake?.cohort_score ?? "n/a"} | Single-patient score {intake?.single_patient_score ?? "n/a"}
                </p>
                <p>
                  Subject column {intake?.subject_column ?? "n/a"} | Unique subjects {intake?.subject_unique_count ?? "n/a"}
                </p>
                {intake?.table_meta?.selected_sheet ? <p>Selected sheet: {intake.table_meta.selected_sheet}</p> : null}
                {intake?.table_meta?.merged_from_sheets?.length ? (
                  <p>Merged sheets: {intake.table_meta.merged_from_sheets.join(", ")}</p>
                ) : null}
              </article>
              <article className="artifactCard">
                <strong>Schema highlights</strong>
                <ul className="artifactList">
                  {schema.slice(0, 8).map((profile) => (
                    <li key={profile.name}>
                      {profile.name}: {profile.inferred_type} | missing {profile.missing_count} | unique {profile.unique_count}
                    </li>
                  ))}
                </ul>
              </article>
            </div>
            {domains.length ? (
              <article className="artifactCard">
                <strong>Workbook domains</strong>
                <div className="cohortDomainList">
                  {domains.map((domain) => (
                    <div key={domain.sheet_name} className="cohortDomainItem">
                      <div className="cohortDomainTitle">
                        <span>{domain.sheet_name}</span>
                        <span className="cohortSubjectBadge">{domain.row_count} rows</span>
                      </div>
                      <p>
                        Subject column {domain.subject_column || "n/a"} | Subjects {domain.subject_count}
                      </p>
                      <p>
                        Visits {(domain.visit_columns?.join(", ") || "n/a")} | Dates {(domain.date_columns?.join(", ") || "n/a")}
                      </p>
                    </div>
                  ))}
                </div>
              </article>
            ) : null}
            {(intake?.rationale ?? []).length ? (
              <article className="artifactCard">
                <strong>Classifier rationale</strong>
                <ul className="artifactList">
                  {(intake?.rationale ?? []).map((item, index) => (
                    <li key={`cohort-browser-rationale-${index}`}>{item}</li>
                  ))}
                </ul>
              </article>
            ) : null}
          </section>

          <section className="cohortBrowserSection">
            <div className="cohortSectionHeader">
              <strong>Composition</strong>
              <span className="mutedText">Sites, arms, outcomes, age distribution, roles, and missingness</span>
            </div>
            <div className="cohortCompositionGrid">
              <article className="artifactCard">
                <strong>Sites</strong>
                {renderBarList(siteDistribution, "No site distribution available.")}
              </article>
              <article className="artifactCard">
                <strong>Arms</strong>
                {renderBarList(armDistribution, "No arm distribution available.")}
              </article>
              <article className="artifactCard">
                <strong>Outcomes</strong>
                {renderBarList(outcomeDistribution, "No outcome distribution available.")}
              </article>
              <article className="artifactCard">
                <strong>Age histogram</strong>
                {renderBarList(ageHistogram, "No age histogram available.")}
              </article>
            </div>
            <div className="cohortOverviewGrid">
              <article className="artifactCard">
                <strong>Variable roles</strong>
                <p>Subject identifiers: {roles?.subject_id_columns?.join(", ") || "None detected"}</p>
                <p>Visit columns: {roles?.visit_columns?.join(", ") || "None detected"}</p>
                <p>Site columns: {roles?.site_columns?.join(", ") || "None detected"}</p>
                <p>Arm / group columns: {roles?.arm_columns?.join(", ") || "None detected"}</p>
                <p>Date / time columns: {roles?.date_columns?.join(", ") || "None detected"}</p>
                <p>Outcome / status columns: {roles?.outcome_columns?.join(", ") || "None detected"}</p>
              </article>
              <article className="artifactCard">
                <strong>Missingness</strong>
                <ul className="artifactList">
                  {(missingness?.top_missing_columns ?? []).slice(0, 8).map((item) => (
                    <li key={`cohort-browser-missing-${item.column}`}>
                      {item.column}: missing {(item.missing_rate * 100).toFixed(1)}% ({item.missing_count}) | non-empty {item.non_empty_count}
                    </li>
                  ))}
                </ul>
              </article>
            </div>
          </section>

          <section className="cohortBrowserSection">
            <div className="cohortSectionHeader">
              <strong>Subjects</strong>
              <span className="mutedText">Scrollable cohort CRF grid for patient-level review</span>
            </div>
            <article className="artifactCard">
              <div className="cohortSectionHeader">
                <strong>Cohort sheet grid</strong>
                <span className="mutedText">
                  {filteredGridRows.length} / {browser?.grid?.row_count ?? gridRows.length} rows
                </span>
              </div>
              <div className="cohortGridToolbar">
                <input
                  className="cohortGridSearch"
                  type="text"
                  value={cohortGridFilter}
                  onChange={(event) => setCohortGridFilter(event.target.value)}
                  placeholder="Filter the cohort sheet by any value"
                />
                <div className="cohortGridPager">
                  <button
                    type="button"
                    className="cohortPagerButton"
                    onClick={() => setCohortGridPage((current) => Math.max(0, current - 1))}
                    disabled={safeCohortGridPage === 0}
                  >
                    Prev
                  </button>
                  <span className="mutedText">
                    Page {safeCohortGridPage + 1} / {cohortGridPageCount}
                  </span>
                  <button
                    type="button"
                    className="cohortPagerButton"
                    onClick={() => setCohortGridPage((current) => Math.min(cohortGridPageCount - 1, current + 1))}
                    disabled={safeCohortGridPage >= cohortGridPageCount - 1}
                  >
                    Next
                  </button>
                </div>
              </div>
              {subjects.length ? (
                <div className="cohortGridSummary">
                  <span className="cohortSubjectBadge">{subjects.length} subjects in preview</span>
                  <span className="mutedText">
                    {subjects.slice(0, 5).map((item) => item.subject_id).join(", ")}
                      {subjects.length > 5 ? " ..." : ""}
                    </span>
                  </div>
                ) : null}
              {gridColumns.length > 0 && filteredGridRows.length > 0 ? (
                <div className="cohortGridWrap">
                  <table className="cohortGridTable">
                    <thead>
                      <tr>
                        {gridColumns.map((column) => (
                          <th key={`cohort-grid-head-${column}`}>{column}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {pagedGridRows.map((row, rowIndex) => (
                        <tr key={`cohort-grid-row-${safeCohortGridPage * cohortGridPageSize + rowIndex}`}>
                          {gridColumns.map((column) => {
                            const cellValue = row && typeof row === "object" ? row[column] : "";
                            return (
                              <td key={`cohort-grid-cell-${safeCohortGridPage * cohortGridPageSize + rowIndex}-${column}`}>
                                {String(cellValue ?? "").trim() || "—"}
                              </td>
                            );
                          })}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <p className="mutedText">No cohort grid rows are available for this sheet.</p>
              )}
            </article>
          </section>
        </div>
      );
    }

    if (activeBaseView === "intake") {
      const intake = getSourceArtifact("intake", activeSourceIndex) as TableIntakeArtifact | undefined;
      if (!intake) {
        return <p className="mutedText">No table intake classification is available.</p>;
      }
      return (
        <div className="artifactStack">
          <article className="artifactCard">
            <strong>Table mode classification</strong>
            <p>Mode: {intake.analysis_mode ?? "n/a"}</p>
            <p>
              Cohort score {intake.cohort_score ?? "n/a"} | Single-patient score {intake.single_patient_score ?? "n/a"}
            </p>
            <p>
              Rows {intake.row_count ?? "n/a"} | Columns {intake.column_count ?? "n/a"} | Subject column {intake.subject_column ?? "n/a"}
            </p>
            <p>Unique subject IDs: {intake.subject_unique_count ?? "n/a"}</p>
            {intake.table_meta?.selected_sheet ? <p>Selected sheet: {intake.table_meta.selected_sheet}</p> : null}
            {intake.table_meta?.sheet_names?.length ? <p>Workbook sheets: {intake.table_meta.sheet_names.join(", ")}</p> : null}
          </article>
          {(intake.rationale ?? []).map((item, index) => (
            <article key={`rationale-${index}`} className="artifactCard">
              <strong>Classifier rationale {index + 1}</strong>
              <p>{item}</p>
            </article>
          ))}
        </div>
      );
    }

    if (activeBaseView === "schema") {
      const schema = (getSourceArtifact("schema", activeSourceIndex) as { profiles?: SchemaProfile[] } | undefined)?.profiles ?? [];
      return (
        <div className="artifactStack">
          {schema.map((profile) => (
            <article key={profile.name} className="artifactCard">
              <strong>
                {profile.name} <span className="artifactType">{profile.inferred_type}</span>
              </strong>
              <p>
                Non-empty {profile.non_empty_count} | Missing {profile.missing_count} | Unique {profile.unique_count}
              </p>
              <p>Sample values: {profile.sample_values.length ? profile.sample_values.join(", ") : "n/a"}</p>
              {profile.numeric_summary ? (
                <p>
                  Range {profile.numeric_summary.min} to {profile.numeric_summary.max} | Mean {profile.numeric_summary.mean}
                </p>
              ) : null}
            </article>
          ))}
        </div>
      );
    }

    if (activeBaseView === "cohort") {
      const cohort = getSourceArtifact("cohort", activeSourceIndex) as CohortArtifact | undefined;
      if (!cohort) {
        return <p className="mutedText">No cohort summary is available.</p>;
      }
      return (
        <div className="artifactStack">
          <article className="artifactCard">
            <strong>Cohort overview</strong>
            <p>Records: {cohort.record_count}</p>
            <p>Fields: {cohort.field_count}</p>
          </article>
          {cohort.categorical_breakdowns.map((item) => (
            <article key={item.column} className="artifactCard">
              <strong>{item.column}</strong>
              <ul className="artifactList">
                {item.top_values.map((entry) => (
                  <li key={`${item.column}-${entry.label}`}>
                    {entry.label}: {entry.count}
                  </li>
                ))}
              </ul>
            </article>
          ))}
          {cohort.numeric_breakdowns.map((item) => (
            <article key={item.column} className="artifactCard">
              <strong>{item.column}</strong>
              <p>
                Min {item.summary?.min ?? "n/a"} | Max {item.summary?.max ?? "n/a"} | Mean {item.summary?.mean ?? "n/a"}
              </p>
            </article>
          ))}
        </div>
      );
    }

    if (activeBaseView === "roles") {
      const roles = getSourceArtifact("roles", activeSourceIndex) as TableRolesArtifact | undefined;
      if (!roles) {
        return <p className="mutedText">No variable roles are available.</p>;
      }
      const sections: Array<[string, string[] | undefined]> = [
        ["Subject identifiers", roles.subject_id_columns],
        ["Visit columns", roles.visit_columns],
        ["Site columns", roles.site_columns],
        ["Arm / group columns", roles.arm_columns],
        ["Date / time columns", roles.date_columns],
        ["Outcome / status columns", roles.outcome_columns],
      ];
      return (
        <div className="artifactStack">
          {sections.map(([label, values]) => (
            <article key={label} className="artifactCard">
              <strong>{label}</strong>
              <p>{values && values.length ? values.join(", ") : "None detected"}</p>
            </article>
          ))}
        </div>
      );
    }

    if (activeBaseView === "missingness") {
      const missingness = getSourceArtifact("missingness", activeSourceIndex) as MissingnessArtifact | undefined;
      const columns = missingness?.top_missing_columns ?? [];
      if (!columns.length) {
        return <p className="mutedText">No missingness summary is available.</p>;
      }
      return (
        <div className="artifactStack">
          {columns.map((item) => (
            <article key={item.column} className="artifactCard">
              <strong>{item.column}</strong>
              <p>
                Missing {(item.missing_rate * 100).toFixed(1)}% ({item.missing_count}) | Non-empty {item.non_empty_count}
              </p>
            </article>
          ))}
        </div>
      );
    }

    if (activeBaseView === "metadata") {
      const metadata = getSourceArtifact("metadata", activeSourceIndex) as
        | {
            items?: DicomMetadataItem[];
            patient_id?: string;
            study_description?: string;
            modality?: string;
            rows?: string;
            columns?: string;
            preview?: {
              available: boolean;
              image_data_url: string | null;
              message: string;
            };
            preview_presets?: Record<
              string,
              {
                available: boolean;
                image_data_url: string | null;
                message: string;
                label?: string;
              }
            >;
          }
        | undefined;
      if (!metadata) {
        return <p className="mutedText">No imaging metadata is available.</p>;
      }
      if (metadata.items?.length) {
        return (
          <div className="artifactStack">
            {metadata.items.map((item) => (
              <article key={`${item.file_name}-${item.series_instance_uid}`} className="artifactCard">
                <strong>{item.file_name}</strong>
                {renderMetadataRows([
                  { label: "Modality", value: item.modality },
                  { label: "Patient ID", value: item.patient_id },
                  { label: "Study Description", value: item.study_description },
                  { label: "Series Description", value: item.series_description },
                  { label: "Instance Number", value: item.instance_number },
                  { label: "Study UID", value: item.study_instance_uid },
                  { label: "Series UID", value: item.series_instance_uid },
                  { label: "Matrix", value: `${item.rows} x ${item.columns}` },
                ])}
              </article>
            ))}
          </div>
        );
      }
      return (
        <div className="artifactStack">
          <article className="artifactCard">
            <strong>Metadata overview</strong>
            {renderMetadataRows([
              { label: "Modality", value: metadata.modality },
              { label: "Patient ID", value: metadata.patient_id },
              { label: "Study Description", value: metadata.study_description },
              { label: "Rows", value: metadata.rows },
              { label: "Columns", value: metadata.columns },
              { label: "Matrix", value: `${metadata.rows ?? "n/a"} x ${metadata.columns ?? "n/a"}` },
            ])}
          </article>
        </div>
      );
    }

    if (activeBaseView === "series") {
      const series = (getSourceArtifact("series", activeSourceIndex) as DicomSeriesArtifact | undefined)?.series ?? [];
      if (!series.length) {
        return <p className="mutedText">No grouped imaging series are available.</p>;
      }
      return (
        <div className="artifactStack">
          {series.map((item) => (
            <article key={item.series_instance_uid} className="artifactCard">
              {item.preview?.available && item.preview.image_data_url ? (
                <img src={item.preview.image_data_url} alt={item.series_description || "DICOM series preview"} className="dicomPreview" />
              ) : (
                <div className="dicomPreviewPlaceholder">{item.preview?.message ?? "Preview not available"}</div>
              )}
              <strong>{item.series_description || "Unnamed series"}</strong>
              <p>
                {item.modality} | instances {item.instance_count}
              </p>
              <p>Study: {item.study_description || "n/a"}</p>
              <p>Examples: {item.example_files.join(", ")}</p>
            </article>
          ))}
        </div>
      );
    }

    if (activeBaseView === "message") {
      const message = (getSourceArtifact("message", activeSourceIndex) as Record<string, unknown> | undefined) ?? {};
      return (
        <div className="artifactStack">
          <article className="artifactCard">
            <strong>Clinical message overview</strong>
            {renderMetadataRows(
              Object.entries(message).map(([label, value]) => ({
                label,
                value: value == null ? "n/a" : typeof value === "string" ? value : JSON.stringify(value),
              })),
            )}
          </article>
        </div>
      );
    }

    if (activeBaseView === "fhir_browser" || activeBaseView === "patient") {
      const patient = (getSourceArtifact("patient", activeSourceIndex) as FhirPatientArtifact | undefined) ?? {};
      const allergies = (getSourceArtifact("allergies", activeSourceIndex) as FhirAllergyArtifact | undefined) ?? {};
      const vitals = (getSourceArtifact("vitals", activeSourceIndex) as FhirVitalArtifact | undefined) ?? {};
      const observations = (getSourceArtifact("observations", activeSourceIndex) as FhirObservationArtifact | undefined) ?? {};
      const medications = (getSourceArtifact("medications", activeSourceIndex) as FhirMedicationArtifact | undefined) ?? {};
      const timeline = (getSourceArtifact("timeline", activeSourceIndex) as FhirTimelineArtifact | undefined) ?? {};
      const labs = (getSourceArtifact("labs", activeSourceIndex) as FhirLabArtifact | undefined) ?? {};
      const careTeam = (getSourceArtifact("care_team", activeSourceIndex) as FhirCareTeamArtifact | undefined) ?? {};
      const patientInitial = (patient.full_name || "P").charAt(0).toUpperCase();
      const age = formatAge(patient.birth_date);
      const vitalItems = vitals.items ?? [];
      const allergyItems = allergies.items ?? [];
      const observationItems = observations.items ?? [];
      const medicationItems = medications.items ?? [];
      const eventItems = timeline.events ?? [];
      const labSeries = labs.series ?? [];
      const carePractitioners = careTeam.practitioners ?? [];
      const careOrganizations = careTeam.organizations ?? [];
      const datedEvents = [...eventItems, ...medicationItems].map((item) => dateToNumber("start" in item ? item.start : undefined) ?? dateToNumber("date" in item ? item.date : undefined)).filter((value): value is number => value !== null);
      const timelineMin = datedEvents.length ? Math.min(...datedEvents) : null;
      const timelineMax = datedEvents.length ? Math.max(...datedEvents) : null;
      const timelineSpan = timelineMin !== null && timelineMax !== null && timelineMax > timelineMin ? timelineMax - timelineMin : 1;
      const labPoints = labSeries.flatMap((series) => series.points ?? []);
      const labDates = labPoints.map((point) => dateToNumber(point.date)).filter((value): value is number => value !== null);
      const labValues = labPoints.map((point) => point.value).filter((value): value is number => typeof value === "number");
      const labMinDate = labDates.length ? Math.min(...labDates) : null;
      const labMaxDate = labDates.length ? Math.max(...labDates) : null;
      const labMinValue = labValues.length ? Math.min(...labValues) : null;
      const labMaxValue = labValues.length ? Math.max(...labValues) : null;
      const labDateSpan = labMinDate !== null && labMaxDate !== null && labMaxDate > labMinDate ? labMaxDate - labMinDate : 1;
      const labValueSpan = labMinValue !== null && labMaxValue !== null && labMaxValue > labMinValue ? labMaxValue - labMinValue : 1;
      const lineColors = ["#2f80ed", "#f97331", "#16a34a", "#8b5cf6", "#ef4444", "#0f766e"];
      return (
        <div className="artifactStack">
          <section className="fhirSummaryHeader">
            <article className="artifactCard">
              <strong>FHIR patient overview</strong>
              <div className="patientHero">
                <div className="patientAvatar">{patientInitial}</div>
                <div>
                  <div className="patientName">{patient.full_name || "Unknown patient"}</div>
                  <div className="mutedText">
                    {patient.gender || "n/a"} / age {age} / Patient ID {patient.id || "n/a"}
                  </div>
                </div>
              </div>
              {renderMetadataRows([
                { label: "Birth Date", value: patient.birth_date },
                { label: "Active", value: patient.active },
                { label: "Managing Org", value: patient.managing_organization },
              ])}
            </article>

            <article className="artifactCard">
              <strong>AllergyIntolerance alerts</strong>
              {allergyItems.length ? (
                <div className="alertStack">
                  {allergyItems.map((item, index) => (
                    <div key={`allergy-${index}`} className="allergyBadge">
                      <span className="allergyDot" />
                      <div>
                        <strong>{item.substance || "Unknown allergen"}</strong>
                        <p className="mutedText">
                          {item.criticality || "n/a"} / {item.clinical_status || "n/a"}
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="mutedText">No AllergyIntolerance resources are available.</p>
              )}
            </article>

            <article className="artifactCard">
              <strong>Latest vital signs</strong>
              {vitalItems.length ? (
                <div className="vitalGrid">
                  {vitalItems.map((item, index) => (
                    <div key={`vital-${index}`} className="vitalTile">
                      <span className="vitalLabel">{item.label || "Vital"}</span>
                      <strong>{item.value || "n/a"}</strong>
                      <small>{item.effective || item.status || "n/a"}</small>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="mutedText">No latest vital observations are available.</p>
              )}
            </article>
          </section>

          <section className="fhirMainGrid">
            <article className="artifactCard">
              <strong>Medical &amp; medication timeline</strong>
              <div className="timelineSection">
                <div className="timelineSectionTitle">Events</div>
                {eventItems.length ? (
                  <div className="timelineTrackList">
                    {eventItems.map((item, index) => {
                      const start = dateToNumber(item.start);
                      const end = dateToNumber(item.end) ?? start;
                      const left = start !== null && timelineMin !== null ? clampPercent(((start - timelineMin) / timelineSpan) * 100) : 0;
                      const width = start !== null && end !== null ? Math.max(8, clampPercent((((end - start) || 1) / timelineSpan) * 100)) : 12;
                      return (
                        <div key={`event-${index}`} className="timelineRow">
                          <div className="timelineLabel">
                            <strong>{item.label || item.type || "Event"}</strong>
                            <small>{item.type || "Event"} / {item.status || "n/a"}</small>
                          </div>
                          <div className="timelineBarArea">
                            <div className="timelineBar eventBar" style={{ left: `${left}%`, width: `${width}%` }} />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                ) : (
                  <p className="mutedText">No Encounter or Procedure events are available.</p>
                )}
              </div>

              <div className="timelineSection">
                <div className="timelineSectionTitle">Medications</div>
                {medicationItems.length ? (
                  <div className="fhirAccordionList">
                    {medicationItems.map((item, index) => {
                      const start = dateToNumber(item.start || item.date);
                      const end = dateToNumber(item.end) ?? start;
                      const left = start !== null && timelineMin !== null ? clampPercent(((start - timelineMin) / timelineSpan) * 100) : 0;
                      const width = start !== null && end !== null ? Math.max(8, clampPercent((((end - start) || 1) / timelineSpan) * 100)) : 12;
                      return (
                        <details key={`med-${index}`} className="fhirAccordionItem">
                          <summary>
                            <div className="timelineSummary">
                              <span>{item.medication || "Unknown medication"}</span>
                              <span className={`timelineStatus ${item.current ? "timelineStatusCurrent" : "timelineStatusPast"}`}>
                                {item.current ? "Current" : item.status || "Ended"}
                              </span>
                            </div>
                          </summary>
                          <div className="timelineBarArea timelineBarAreaExpanded">
                            <div className={`timelineBar ${item.current ? "medBarActive" : "medBarPast"}`} style={{ left: `${left}%`, width: `${width}%` }} />
                          </div>
                          {renderMetadataRows([
                            { label: "Status", value: item.status },
                            { label: "Intent", value: item.intent },
                            { label: "Start", value: item.start || item.date },
                            { label: "End", value: item.end },
                            { label: "Duration (days)", value: item.duration_days != null ? String(item.duration_days) : "n/a" },
                            { label: "Dosage", value: item.dosage },
                          ])}
                        </details>
                      );
                    })}
                  </div>
                ) : (
                  <p className="mutedText">No medication history is available.</p>
                )}
              </div>
            </article>

            <article className="artifactCard">
              <strong>Lab graph &amp; insights</strong>
              {labSeries.length ? (
                <>
                  <svg className="labChart" viewBox="0 0 560 280" preserveAspectRatio="none">
                    {labSeries.slice(0, 4).map((series, seriesIndex) => {
                      const points = (series.points ?? [])
                        .map((point) => {
                          const xDate = dateToNumber(point.date);
                          if (xDate === null || labMinDate === null || labMaxDate === null || labMinValue === null || labMaxValue === null || typeof point.value !== "number") {
                            return null;
                          }
                          const x = 30 + ((xDate - labMinDate) / labDateSpan) * 500;
                          const y = 240 - ((point.value - labMinValue) / labValueSpan) * 190;
                          return `${x},${y}`;
                        })
                        .filter(Boolean)
                        .join(" ");
                      if (!points) {
                        return null;
                      }
                      return <polyline key={`series-${seriesIndex}`} fill="none" stroke={lineColors[seriesIndex % lineColors.length]} strokeWidth="3" points={points} />;
                    })}
                    {(labs.latest ?? []).map((item, index) =>
                      typeof item.low === "number" && typeof item.high === "number" && labMinValue !== null && labMaxValue !== null ? (
                        <rect
                          key={`range-${index}`}
                          x="30"
                          width="500"
                          y={240 - ((item.high - labMinValue) / labValueSpan) * 190}
                          height={Math.max(6, ((item.high - item.low) / labValueSpan) * 190)}
                          fill="rgba(47,128,237,0.08)"
                        />
                      ) : null,
                    )}
                  </svg>
                  <div className="insightGrid">
                    {(labs.latest ?? []).slice(0, 4).map((item, index) => (
                      <div key={`insight-${index}`} className="insightTile">
                        <strong>{item.label || "Lab"}</strong>
                        <span>{item.value} {item.unit || ""}</span>
                        <small>
                          Normal {item.low ?? "n/a"} - {item.high ?? "n/a"}
                        </small>
                      </div>
                    ))}
                  </div>
                </>
              ) : (
                <p className="mutedText">No numeric Observation trends are available for graph review.</p>
              )}
            </article>
          </section>

          <article className="artifactCard">
            <strong>Care team map</strong>
            <div className="careTeamGrid">
              <div>
                <div className="timelineSectionTitle">Practitioners</div>
                {carePractitioners.length ? (
                  <div className="careCardStack">
                    {carePractitioners.map((item, index) => (
                      <div key={`prac-${index}`} className="careCard">
                        <strong>{item.name || "Unknown practitioner"}</strong>
                        <small>{item.role || "Practitioner"}</small>
                        <small>{item.contact || "n/a"}</small>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="mutedText">No Practitioner resources are available.</p>
                )}
              </div>
              <div>
                <div className="timelineSectionTitle">Organizations</div>
                {careOrganizations.length ? (
                  <div className="careCardStack">
                    {careOrganizations.map((item, index) => (
                      <div key={`org-${index}`} className="careCard">
                        <strong>{item.name || "Unknown organization"}</strong>
                        <small>{item.contact || "n/a"}</small>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="mutedText">No Organization resources are available.</p>
                )}
              </div>
            </div>
          </article>
        </div>
      );
    }

    if (activeBaseView === "resources") {
      const resources = (getSourceArtifact("resources", activeSourceIndex) as Record<string, unknown> | undefined) ?? {};
      return (
        <div className="artifactStack">
          {"top_level_keys" in resources ? (
            <article className="artifactCard">
              <strong>FHIR structure</strong>
              <p>Top-level keys: {((resources.top_level_keys as string[] | undefined) ?? []).join(", ") || "n/a"}</p>
              <pre>{JSON.stringify(resources.sample ?? {}, null, 2)}</pre>
            </article>
          ) : null}
          {"segment_counts" in resources ? (
            <article className="artifactCard">
              <strong>HL7 segment inventory</strong>
              <ul className="artifactList">
                {Object.entries((resources.segment_counts as Record<string, number> | undefined) ?? {}).map(([name, count]) => (
                  <li key={name}>
                    {name}: {count}
                  </li>
                ))}
              </ul>
              <pre>{(((resources.segments as string[] | undefined) ?? []).slice(0, 12)).join("\n")}</pre>
            </article>
          ) : null}
        </div>
      );
    }

    if (activeBaseView === "note") {
      const note = (getSourceArtifact("note", activeSourceIndex) as Record<string, unknown> | undefined) ?? {};
      return (
        <div className="artifactStack">
          <article className="artifactCard">
            <strong>Clinical note</strong>
            {renderMetadataRows([
              { label: "Headline", value: String(note.headline ?? "n/a") },
              { label: "Line count", value: String(note.line_count ?? "n/a") },
              { label: "Word count", value: String(note.word_count ?? "n/a") },
            ])}
            <pre>{String(note.preview ?? "n/a")}</pre>
          </article>
        </div>
      );
    }

    if (activeBaseView === "tool_result") {
      const toolResult = (studioArtifacts as ToolRunResponse | null) ?? null;
      if (!toolResult) {
        return <p className="mutedText">No tool output is available.</p>;
      }
      const toolArtifacts = (toolResult.artifacts ?? {}) as Record<string, unknown>;
      const reportSections = (toolArtifacts.report_sections as Record<string, unknown> | undefined) ?? null;
      const regions = (toolArtifacts.regions as Array<Record<string, unknown>> | undefined) ?? [];
      const detections = (toolArtifacts.detections as Array<Record<string, unknown>> | undefined) ?? [];
      const qualityAssessment = (toolArtifacts.quality_assessment as Record<string, unknown> | undefined) ?? null;
      const restorationPlan = (toolArtifacts.restoration_plan as string[] | undefined) ?? [];
      const tissueSummary = (toolArtifacts.tissue_summary as Record<string, unknown> | undefined) ?? null;
      const producedFiles = (toolArtifacts.produced_files as string[] | undefined) ?? [];
      const installSteps = (toolArtifacts.install_steps as string[] | undefined) ?? [];
      const summaryPoints = (toolArtifacts.summary_points as string[] | undefined) ?? [];
      const reviewTasks = (toolArtifacts.review_tasks as string[] | undefined) ?? [];
      const searchPlan = (toolArtifacts.search_plan as string[] | undefined) ?? [];
      const suggestedTerms = (toolArtifacts.suggested_terms as string[] | undefined) ?? [];
      const fieldTags = (toolArtifacts.field_tags as string[] | undefined) ?? [];
      const meshStrategy = (toolArtifacts.mesh_strategy as string[] | undefined) ?? [];
      const apiWorkflow = (toolArtifacts.api_workflow as string[] | undefined) ?? [];
      const bestPractices = (toolArtifacts.best_practices as string[] | undefined) ?? [];
      const articles = (toolArtifacts.articles as Array<Record<string, unknown>> | undefined) ?? [];
      const searchAxes = (toolArtifacts.search_axes as string[] | undefined) ?? [];
      const suggestedQueries = (toolArtifacts.suggested_queries as string[] | undefined) ?? [];
      const recommendedSources = (toolArtifacts.recommended_sources as string[] | undefined) ?? [];
      const briefPoints = (toolArtifacts.brief_points as string[] | undefined) ?? [];
      const triageQuestions = (toolArtifacts.triage_questions as string[] | undefined) ?? [];
      const entities = (toolArtifacts.entities as Record<string, unknown> | undefined) ?? null;
      const plainLanguageSummary = (toolArtifacts.plain_language_summary as string[] | undefined) ?? [];
      const patientQuestions = (toolArtifacts.patient_questions as string[] | undefined) ?? [];
      const implementationSteps = (toolArtifacts.implementation_steps as string[] | undefined) ?? [];
      const likelyResources = (toolArtifacts.likely_resources as string[] | undefined) ?? [];
      const developerNotes = (toolArtifacts.developer_notes as string[] | undefined) ?? [];
      const protocolSections = (toolArtifacts.protocol_sections as string[] | undefined) ?? [];
      const designQuestions = (toolArtifacts.design_questions as string[] | undefined) ?? [];
      const requiredEvidence = (toolArtifacts.required_evidence as string[] | undefined) ?? [];
      const reviewSteps = (toolArtifacts.review_steps as string[] | undefined) ?? [];
      const studyPlan = (toolArtifacts.study_plan as string[] | undefined) ?? [];
      const reviewChecklist = (toolArtifacts.review_checklist as string[] | undefined) ?? [];
      const findingCategories = (toolArtifacts.finding_categories as string[] | undefined) ?? [];
      const followUpQuestions = (toolArtifacts.follow_up_questions as string[] | undefined) ?? [];
      return (
        <div className="artifactStack">
          <article className="artifactCard">
            <strong>{toolResult.tool.name}</strong>
            <p>{toolResult.summary}</p>
            {renderMetadataRows([
              { label: "Team", value: toolResult.tool.team ?? "n/a" },
              { label: "Task type", value: toolResult.tool.task_type ?? "n/a" },
              { label: "Modality", value: toolResult.tool.modality ?? "n/a" },
            ])}
          </article>
          {(() => {
            const imageArtifacts = Object.entries(toolArtifacts).filter(
              ([, v]) => v && typeof v === "object" && (v as Record<string, unknown>).type === "image" && (v as Record<string, unknown>).image_data_url,
            );
            if (!imageArtifacts.length) return null;
            return imageArtifacts.map(([key, v]) => {
              const art = v as { image_data_url: string; description?: string };
              return (
                <article key={`tool-img-${key}`} className="artifactCard">
                  <strong>{art.description ?? key}</strong>
                  <img
                    src={art.image_data_url}
                    alt={art.description ?? key}
                    style={{ width: "100%", borderRadius: 6, marginTop: 8 }}
                  />
                </article>
              );
            });
          })()}
          {reportSections ? (
            <article className="artifactCard">
              <strong>Structured report</strong>
              {renderMetadataRows([{ label: "Exam", value: String(reportSections.exam ?? "n/a") }])}
              <div className="artifactStack">
                <div>
                  <strong>Findings</strong>
                  <ul className="artifactList">
                    {((reportSections.findings as string[] | undefined) ?? []).map((item, index) => (
                      <li key={`finding-${index}`}>{item}</li>
                    ))}
                  </ul>
                </div>
                <div>
                  <strong>Impression</strong>
                  <ul className="artifactList">
                    {((reportSections.impression as string[] | undefined) ?? []).map((item, index) => (
                      <li key={`impression-${index}`}>{item}</li>
                    ))}
                  </ul>
                </div>
                {Array.isArray(toolArtifacts.recommendations) ? (
                  <div>
                    <strong>Recommendations</strong>
                    <ul className="artifactList">
                      {((toolArtifacts.recommendations as string[]) ?? []).map((item, index) => (
                        <li key={`recommendation-${index}`}>{item}</li>
                      ))}
                    </ul>
                  </div>
                ) : null}
              </div>
            </article>
          ) : null}
          {regions.length ? (
            <article className="artifactCard">
              <strong>Segmentation regions</strong>
              <div className="careCardStack">
                {regions.map((item, index) => (
                  <div key={`region-${index}`} className="careCard">
                    <strong>{String(item.name ?? `Region ${index + 1}`)}</strong>
                    <small>Area: {String(item.area_px ?? "n/a")} px</small>
                    <small>Confidence: {String(item.confidence ?? "n/a")}</small>
                  </div>
                ))}
              </div>
            </article>
          ) : null}
          {detections.length ? (
            <article className="artifactCard">
              <strong>Detection candidates</strong>
              <div className="careCardStack">
                {detections.map((item, index) => (
                  <div key={`detection-${index}`} className="careCard">
                    <strong>{String(item.label ?? `Candidate ${index + 1}`)}</strong>
                    <small>Confidence: {String(item.confidence ?? "n/a")}</small>
                    <small>BBox: {Array.isArray(item.bbox) ? item.bbox.join(", ") : "n/a"}</small>
                  </div>
                ))}
              </div>
            </article>
          ) : null}
          {qualityAssessment ? (
            <article className="artifactCard">
              <strong>Restoration review</strong>
              {renderMetadataRows(
                Object.entries(qualityAssessment).map(([label, value]) => ({
                  label,
                  value: String(value ?? "n/a"),
                })),
              )}
              {restorationPlan.length ? (
                <div>
                  <strong>Restoration plan</strong>
                  <ul className="artifactList">
                    {restorationPlan.map((item, index) => (
                      <li key={`restoration-${index}`}>{item}</li>
                    ))}
                  </ul>
                </div>
              ) : null}
            </article>
          ) : null}
          {tissueSummary ? (
            <article className="artifactCard">
              <strong>Pathology review</strong>
              {renderMetadataRows(
                Object.entries(tissueSummary).map(([label, value]) => ({
                  label,
                  value: String(value ?? "n/a"),
                })),
              )}
              {reviewTasks.length ? (
                <div>
                  <strong>Review tasks</strong>
                  <ul className="artifactList">
                    {reviewTasks.map((item, index) => (
                      <li key={`review-task-${index}`}>{item}</li>
                    ))}
                  </ul>
                </div>
              ) : null}
            </article>
          ) : null}
          {reviewChecklist.length || findingCategories.length || followUpQuestions.length ? (
            <article className="artifactCard">
              <strong>Medical imaging review</strong>
              {reviewChecklist.length ? (
                <div>
                  <strong>Review checklist</strong>
                  <ul className="artifactList">
                    {reviewChecklist.map((item, index) => (
                      <li key={`review-check-${index}`}>{item}</li>
                    ))}
                  </ul>
                </div>
              ) : null}
              {findingCategories.length ? (
                <div>
                  <strong>Finding categories</strong>
                  <ul className="artifactList">
                    {findingCategories.map((item, index) => (
                      <li key={`finding-category-${index}`}>{item}</li>
                    ))}
                  </ul>
                </div>
              ) : null}
              {followUpQuestions.length ? (
                <div>
                  <strong>Follow-up questions</strong>
                  <ul className="artifactList">
                    {followUpQuestions.map((item, index) => (
                      <li key={`follow-up-question-${index}`}>{item}</li>
                    ))}
                  </ul>
                </div>
              ) : null}
            </article>
          ) : null}
          {searchAxes.length || suggestedQueries.length || recommendedSources.length ? (
            <article className="artifactCard">
              <strong>Biomedical search</strong>
              {searchAxes.length ? (
                <div>
                  <strong>Search axes</strong>
                  <ul className="artifactList">
                    {searchAxes.map((item, index) => (
                      <li key={`search-axis-${index}`}>{item}</li>
                    ))}
                  </ul>
                </div>
              ) : null}
              {suggestedQueries.length ? (
                <div>
                  <strong>Suggested queries</strong>
                  <ul className="artifactList">
                    {suggestedQueries.map((item, index) => (
                      <li key={`search-query-${index}`}>{item}</li>
                    ))}
                  </ul>
                </div>
              ) : null}
              {recommendedSources.length ? (
                <div>
                  <strong>Recommended sources</strong>
                  <ul className="artifactList">
                    {recommendedSources.map((item, index) => (
                      <li key={`recommended-source-${index}`}>{item}</li>
                    ))}
                  </ul>
                </div>
              ) : null}
            </article>
          ) : null}
          {briefPoints.length || triageQuestions.length ? (
            <article className="artifactCard">
              <strong>Specialty brief</strong>
              {briefPoints.length ? (
                <div>
                  <strong>Brief points</strong>
                  <ul className="artifactList">
                    {briefPoints.map((item, index) => (
                      <li key={`brief-point-${index}`}>{item}</li>
                    ))}
                  </ul>
                </div>
              ) : null}
              {triageQuestions.length ? (
                <div>
                  <strong>Triage questions</strong>
                  <ul className="artifactList">
                    {triageQuestions.map((item, index) => (
                      <li key={`triage-question-${index}`}>{item}</li>
                    ))}
                  </ul>
                </div>
              ) : null}
            </article>
          ) : null}
          {entities ? (
            <article className="artifactCard">
              <strong>Extracted entities</strong>
              {renderMetadataRows(
                Object.entries(entities).map(([label, value]) => ({
                  label,
                  value: Array.isArray(value) ? value.join(", ") || "n/a" : String(value ?? "n/a"),
                })),
              )}
            </article>
          ) : null}
          {plainLanguageSummary.length || patientQuestions.length ? (
            <article className="artifactCard">
              <strong>Patient-friendly explanation</strong>
              {plainLanguageSummary.length ? (
                <div>
                  <strong>Plain language summary</strong>
                  <ul className="artifactList">
                    {plainLanguageSummary.map((item, index) => (
                      <li key={`plain-language-${index}`}>{item}</li>
                    ))}
                  </ul>
                </div>
              ) : null}
              {patientQuestions.length ? (
                <div>
                  <strong>Patient questions</strong>
                  <ul className="artifactList">
                    {patientQuestions.map((item, index) => (
                      <li key={`patient-question-${index}`}>{item}</li>
                    ))}
                  </ul>
                </div>
              ) : null}
            </article>
          ) : null}
          {implementationSteps.length || likelyResources.length || developerNotes.length ? (
            <article className="artifactCard">
              <strong>FHIR developer guide</strong>
              {implementationSteps.length ? (
                <div>
                  <strong>Implementation steps</strong>
                  <ul className="artifactList">
                    {implementationSteps.map((item, index) => (
                      <li key={`implementation-step-${index}`}>{item}</li>
                    ))}
                  </ul>
                </div>
              ) : null}
              {likelyResources.length ? (
                <div>
                  <strong>Likely resources</strong>
                  <ul className="artifactList">
                    {likelyResources.map((item, index) => (
                      <li key={`likely-resource-${index}`}>{item}</li>
                    ))}
                  </ul>
                </div>
              ) : null}
              {developerNotes.length ? (
                <div>
                  <strong>Developer notes</strong>
                  <ul className="artifactList">
                    {developerNotes.map((item, index) => (
                      <li key={`developer-note-${index}`}>{item}</li>
                    ))}
                  </ul>
                </div>
              ) : null}
            </article>
          ) : null}
          {protocolSections.length || designQuestions.length ? (
            <article className="artifactCard">
              <strong>Clinical trial protocol</strong>
              {protocolSections.length ? (
                <div>
                  <strong>Protocol sections</strong>
                  <ul className="artifactList">
                    {protocolSections.map((item, index) => (
                      <li key={`protocol-section-${index}`}>{item}</li>
                    ))}
                  </ul>
                </div>
              ) : null}
              {designQuestions.length ? (
                <div>
                  <strong>Design questions</strong>
                  <ul className="artifactList">
                    {designQuestions.map((item, index) => (
                      <li key={`design-question-${index}`}>{item}</li>
                    ))}
                  </ul>
                </div>
              ) : null}
            </article>
          ) : null}
          {requiredEvidence.length || reviewSteps.length ? (
            <article className="artifactCard">
              <strong>Prior auth review</strong>
              {requiredEvidence.length ? (
                <div>
                  <strong>Required evidence</strong>
                  <ul className="artifactList">
                    {requiredEvidence.map((item, index) => (
                      <li key={`required-evidence-${index}`}>{item}</li>
                    ))}
                  </ul>
                </div>
              ) : null}
              {reviewSteps.length ? (
                <div>
                  <strong>Review steps</strong>
                  <ul className="artifactList">
                    {reviewSteps.map((item, index) => (
                      <li key={`prior-auth-step-${index}`}>{item}</li>
                    ))}
                  </ul>
                </div>
              ) : null}
            </article>
          ) : null}
          {studyPlan.length ? (
            <article className="artifactCard">
              <strong>USMLE helper</strong>
              <ul className="artifactList">
                {studyPlan.map((item, index) => (
                  <li key={`study-plan-${index}`}>{item}</li>
                ))}
              </ul>
            </article>
          ) : null}
          <article className="artifactCard">
            <strong>Artifacts</strong>
            <pre>{JSON.stringify(toolResult.artifacts ?? {}, null, 2)}</pre>
          </article>
          <article className="artifactCard">
            <strong>Provenance</strong>
            <pre>{JSON.stringify(toolResult.provenance ?? {}, null, 2)}</pre>
          </article>
        </div>
      );
    }

    return studioArtifacts ? <pre>{JSON.stringify(studioArtifacts, null, 2)}</pre> : null;
  }

  function proposeToolRun(tool: ToolInfo) {
    setPendingTool({
      tool,
      question: composerText.trim() || tool.description || tool.name,
      rationale: tool.description ?? undefined,
    });
    setStatus("Awaiting approval");
    setChatTurns((current) => [
      ...current,
      {
        role: "assistant",
        content: `Tool proposal: run \`${tool.name}\` (${tool.task_type ?? "tool"})${tool.description ? ` to ${tool.description}` : ""}?`,
      },
    ]);
  }

  async function approveToolRun() {
    if (!pendingTool || !result) {
      return;
    }
    const toolName = pendingTool.tool.name;
    setStatus("Running tool");
    try {
      const response = await fetch(`${apiBase.replace(/\/$/, "")}/api/v1/tools/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          tool_name: toolName,
          analysis: result,
          active_view: activeStudioView,
          active_card: activeStudioCard,
          active_artifact: studioArtifacts,
          question: pendingTool.question,
        }),
      });
      if (!response.ok) {
        throw new Error(await response.text());
      }
      const payload = (await response.json()) as ToolRunResponse;
      const toolCardId = `tool_result::${payload.tool.name}`;
      const nextResult: IntakeResponse = {
        ...result,
        studio_cards: [
          ...result.studio_cards.filter((card) => card.id !== toolCardId),
          {
            id: toolCardId,
            title: `${payload.tool.name}`,
            subtitle: payload.summary,
            base_id: "tool_result",
          },
        ],
        artifacts: {
          ...result.artifacts,
          [toolCardId]: payload,
        },
      };
      setResult(nextResult);
      setActiveStudioView(toolCardId);
      setChatTurns((current) => [...current, { role: "assistant", content: payload.summary }]);
      setPendingTool(null);
      setStatus("Tool result ready");
    } catch (caught) {
      const message = caught instanceof Error ? caught.message : String(caught);
      setChatTurns((current) => [
        ...current,
        {
          role: "assistant",
          content: `Tool run failed: ${message}`,
        },
      ]);
      setPendingTool(null);
      setStatus("Tool failed");
    }
  }

  function cancelToolRun() {
    setPendingTool(null);
    setStatus(result ? "Summary ready" : "Waiting for a clinical source");
  }

  function renderStudioReviewPanel() {
    if (!result || !activeStudioView || !activeStudioCard) {
      return null;
    }

    if (activeBaseView === "metadata" && activeStudioCard.source_modality === "medical-image") {
      const metadata = getSourceArtifact("metadata", activeSourceIndex) as
        | {
            items?: DicomMetadataItem[];
            patient_id?: string;
            study_description?: string;
            modality?: string;
            rows?: string;
            columns?: string;
            preview?: {
              available: boolean;
              image_data_url: string | null;
              message: string;
            };
          }
        | undefined;

      return (
        <section className="reviewPanel">
          <div className="reviewPanelHeader">
            <div>
              <h2>{activeStudioCard.title}</h2>
              <p className="mutedText">{activeStudioCard.subtitle}</p>
            </div>
          </div>
          <div className="reviewPanelGrid">
            <div className="reviewPanelViewer">
              {dicomSeriesGroups.length ? (
                <DicomInteractiveViewer seriesGroups={dicomSeriesGroups} />
              ) : (
                <p className="mutedText">No DICOM files are attached for interactive review.</p>
              )}
            </div>
            <div className="reviewPanelMetadata">
              {metadata?.items?.length ? (
                <div className="artifactStack">
                  {metadata.items.map((item) => (
                    <article key={`${item.file_name}-${item.series_instance_uid}`} className="artifactCard">
                      <strong>{item.file_name}</strong>
                      {renderMetadataRows([
                        { label: "Modality", value: item.modality },
                        { label: "Patient ID", value: item.patient_id },
                        { label: "Study Description", value: item.study_description },
                        { label: "Series Description", value: item.series_description },
                        { label: "Instance Number", value: item.instance_number },
                        { label: "Study UID", value: item.study_instance_uid },
                        { label: "Series UID", value: item.series_instance_uid },
                        { label: "Matrix", value: `${item.rows} x ${item.columns}` },
                      ])}
                    </article>
                  ))}
                </div>
              ) : metadata ? (
                <article className="artifactCard">
                  <strong>Metadata overview</strong>
                  {renderMetadataRows([
                    { label: "Modality", value: metadata.modality },
                    { label: "Patient ID", value: metadata.patient_id },
                    { label: "Study Description", value: metadata.study_description },
                    { label: "Rows", value: metadata.rows },
                    { label: "Columns", value: metadata.columns },
                    { label: "Matrix", value: `${metadata.rows ?? "n/a"} x ${metadata.columns ?? "n/a"}` },
                  ])}
                </article>
              ) : (
                <p className="mutedText">No imaging metadata is available.</p>
              )}
            </div>
          </div>
        </section>
      );
    }

    return (
      <section className="reviewPanel">
        <div className="reviewPanelHeader">
          <div>
            <h2>{activeStudioCard.title}</h2>
            <p className="mutedText">{activeStudioCard.subtitle}</p>
          </div>
        </div>
        <div className="reviewPanelBody">{renderStudioCanvas()}</div>
      </section>
    );
  }

  return (
    <main className="shell">
      <header className="topbar">
        <div className="brandWrap">
          <svg
            className="brandIcon"
            viewBox="0 0 64 64"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
            aria-hidden="true"
          >
            <rect x="14" y="8" width="36" height="24" rx="8" stroke="currentColor" strokeWidth="3.5" />
            <path d="M32 4V8" stroke="currentColor" strokeWidth="3.5" strokeLinecap="round" />
            <path d="M14 18H10C8.89543 18 8 18.8954 8 20V24C8 25.1046 8.89543 26 10 26H14" stroke="currentColor" strokeWidth="3.5" />
            <path d="M50 18H54C55.1046 18 56 18.8954 56 20V24C56 25.1046 55.1046 26 54 26H50" stroke="currentColor" strokeWidth="3.5" />
            <circle cx="24" cy="20" r="2.5" fill="currentColor" />
            <circle cx="40" cy="20" r="2.5" fill="currentColor" />
            <path d="M27 38H37" stroke="currentColor" strokeWidth="3.5" strokeLinecap="round" />
            <path d="M20 32V38" stroke="currentColor" strokeWidth="3.5" strokeLinecap="round" />
            <path d="M44 32V38" stroke="currentColor" strokeWidth="3.5" strokeLinecap="round" />
            <path d="M20 38C16.6863 38 14 40.6863 14 44V58H50V44C50 40.6863 47.3137 38 44 38H20Z" stroke="currentColor" strokeWidth="3.5" />
            <path d="M22 58V50C22 47.7909 23.7909 46 26 46H28V52" stroke="currentColor" strokeWidth="3.5" strokeLinecap="round" strokeLinejoin="round" />
            <path d="M42 58V50C42 47.7909 40.2091 46 38 46H36V52" stroke="currentColor" strokeWidth="3.5" strokeLinecap="round" strokeLinejoin="round" />
            <circle cx="40" cy="51" r="3.5" stroke="currentColor" strokeWidth="3.5" />
          </svg>
          <div className="brand">ChatClinic</div>
        </div>
        <div className="copyright">Copyright 2026. BISPL@KAIST AI, All rights reserved.</div>
      </header>
      <div className="grid">
        <section className="panel leftPanel">
          <section className="leftPanelSection">
            <div className="leftPanelHeader">
              <h2>Sources</h2>
            </div>
            <div className="leftPanelBody">
              <label className="uploadButton">
                + Add clinical source
                <input type="file" accept=".csv,.tsv,.xlsx,.xls,.xlsm,.json,.xml,.ndjson,.hl7,.txt,.dcm,.dicom,.png,.jpg,.jpeg,.tif,.tiff,.bmp" multiple onChange={handleFileChange} />
              </label>
              {attachedFiles.length ? (
                <div className="sourceStack">
                  {visibleSources.map((item) => (
                    <article key={item.file_name} className="sourceCard">
                      <strong>{item.file_name}</strong>
                      <span>{item.modality}</span>
                    </article>
                  ))}
                </div>
              ) : (
                <p className="mutedText">Attach one or more clinical sources: CSV/TSV or Excel eCRF tables, FHIR JSON/XML/NDJSON, HL7 messages, plain-text notes, or DICOM files from the same patient/study.</p>
              )}
              <div className="statusBlock">
                <span>Status</span>
                <strong>{status}</strong>
              </div>
              {error ? <p className="errorText">{error}</p> : null}
            </div>
          </section>

          <section className="leftPanelSection">
            <div className="leftPanelHeader">
              <h2>Tool Registry</h2>
            </div>
            <div className="leftPanelBody">
              {availableTools.length ? (
                <div className="toolRegistryStack">
                  <div className="toolRegistryAccordion">
                    <button
                      type="button"
                      className="toolRegistrySummary"
                      aria-expanded={toolRegistryOpen}
                      onClick={() => setToolRegistryOpen((current) => !current)}
                    >
                      <span>Available tools</span>
                      <span className="toolRegistryCount">{availableTools.length}</span>
                    </button>
                    {toolRegistryOpen ? (
                      <div className="toolRegistryPanel">
                        <div className="toolRegistryList">
                          {availableTools.map((tool) => (
                            <button key={tool.name} type="button" className="toolRegistryChip" onClick={() => proposeToolRun(tool)}>
                              {tool.name}
                            </button>
                          ))}
                        </div>
                      </div>
                    ) : null}
                  </div>
                  <div className="toolUsageBlock">
                    <div className="toolUsageHeader">
                      <span>Used tools</span>
                      <strong>{usedTools.length}</strong>
                    </div>
                    {usedTools.length ? (
                      <div className="toolUsageLog">
                        {usedTools.map((toolName, index) => (
                          <div key={`${toolName}-${index}`} className="toolUsageLine">
                            - using tool: `{toolName}`
                          </div>
                        ))}
                      </div>
                    ) : (
                      <p className="mutedText">Used tool logs will appear here after analysis starts.</p>
                    )}
                  </div>
                </div>
              ) : (
                <p className="mutedText">No classroom tools are currently registered.</p>
              )}
            </div>
          </section>
        </section>

        <section className="panel">
          <div className="chatHeader">
            <h2>Chat</h2>
            <span className="statusPill">{status}</span>
          </div>
          <div ref={chatStreamRef} className="chatStream">
            {chatTurns.map((turn, index) => (
              <article key={`turn-${index}`} className={turn.role === "user" ? "userBubble" : "assistantBlock"}>
                {turn.role === "user" ? (
                  <p>{turn.content}</p>
                ) : (
                  <div className="markdownBody">
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>{turn.content}</ReactMarkdown>
                  </div>
                )}
              </article>
            ))}
          </div>
          <div className="chatComposer">
            {pendingTool ? (
              <div className="approvalBar">
                <span>
                  Run <strong>{pendingTool.tool.name}</strong>?
                </span>
                <div className="approvalActions">
                  <button type="button" className="miniActionButton" onClick={approveToolRun}>
                    Approve
                  </button>
                  <button type="button" className="miniActionButton secondary" onClick={cancelToolRun}>
                    Cancel
                  </button>
                </div>
              </div>
            ) : null}
            <input
              value={composerText}
              onChange={(event) => setComposerText(event.target.value)}
              onKeyDown={(event) => {
                if (event.key !== "Enter") {
                  return;
                }
                const native = event.nativeEvent as KeyboardEvent & { isComposing?: boolean };
                if (native.isComposing) {
                  return;
                }
                event.preventDefault();
                void handleChatSubmit();
              }}
              placeholder="Ask a follow-up question about the current scaffold..."
              className="chatInput"
            />
            <button type="button" className="sendButton" onClick={handleChatSubmit}>
              →
            </button>
          </div>
        </section>

        <section className="panel">
          <h2>Studio</h2>
          {hasStudioCards ? (
            <>
              <div className="studioGrid">
                {displayStudioCards.map((card) => (
                  <button
                    key={card.id}
                    type="button"
                    className={`studioCard ${activeStudioView === card.id ? "studioCardActive" : ""}`}
                    onClick={() => setActiveStudioView(card.id)}
                    disabled={!result}
                  >
                    <strong>{card.title}</strong>
                    <span>{card.subtitle}</span>
                  </button>
                ))}
              </div>
              <div className="studioCanvas">
                {activeStudioView ? (
                  <p className="mutedText">
                    {activeStudioCard ? `${activeStudioCard.title} is open in the panel below.` : "The selected Studio card is open below."}
                  </p>
                ) : (
                  <p className="mutedText">Select a Studio card to open its detailed result below.</p>
                )}
              </div>
            </>
          ) : (
            <div className="studioCanvas">
              <p className="mutedText">Studio cards will appear after tool-driven analysis results are ready.</p>
            </div>
          )}
        </section>
      </div>
      {renderStudioReviewPanel()}
    </main>
  );
}
