from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PLUGINS_DIR = ROOT / "plugins"


def discover_tools() -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    if not PLUGINS_DIR.exists():
        return tools

    for manifest_path in sorted(PLUGINS_DIR.glob("*/tool.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        manifest["plugin_dir"] = str(manifest_path.parent)
        tools.append(manifest)
    return tools


def get_tool(tool_name: str) -> dict[str, Any] | None:
    for tool in discover_tools():
        if tool.get("name") == tool_name:
            return tool
    return None


def _detect_gpu_available() -> bool:
    override = str(os.environ.get("CHATCLINIC_GPU_AVAILABLE", "")).strip().lower()
    if override in {"1", "true", "yes", "y", "on"}:
        return True
    if override in {"0", "false", "no", "n", "off"}:
        return False

    cuda_visible = str(os.environ.get("CUDA_VISIBLE_DEVICES", "")).strip()
    if cuda_visible and cuda_visible != "-1":
        return True

    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return False
    try:
        completed = subprocess.run(
            [nvidia_smi, "-L"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception:
        return False
    return completed.returncode == 0 and bool((completed.stdout or "").strip())


def _normalize_runtime(tool: dict[str, Any]) -> dict[str, Any]:
    runtime = dict(tool.get("runtime") or {})
    supported_accelerators = runtime.get("supported_accelerators")
    if not isinstance(supported_accelerators, list) or not supported_accelerators:
        supported_accelerators = ["cpu"]
    normalized_accelerators = []
    for item in supported_accelerators:
        value = str(item).strip().lower()
        if value in {"cpu", "gpu"} and value not in normalized_accelerators:
            normalized_accelerators.append(value)
    if not normalized_accelerators:
        normalized_accelerators = ["cpu"]

    preferred_accelerator = str(runtime.get("preferred_accelerator", "cpu")).strip().lower()
    if preferred_accelerator not in {"cpu", "gpu"}:
        preferred_accelerator = "cpu"

    requires_gpu = bool(runtime.get("requires_gpu", False))
    allow_cpu_fallback = bool(runtime.get("allow_cpu_fallback", not requires_gpu))

    host_compatible = runtime.get("host_compatible")
    if not isinstance(host_compatible, list) or not host_compatible:
        host_compatible = ["cpu", "gpu"]
    normalized_hosts = []
    for item in host_compatible:
        value = str(item).strip().lower()
        if value in {"cpu", "gpu"} and value not in normalized_hosts:
            normalized_hosts.append(value)
    if not normalized_hosts:
        normalized_hosts = ["cpu", "gpu"]

    return {
        "host_compatible": normalized_hosts,
        "supported_accelerators": normalized_accelerators,
        "preferred_accelerator": preferred_accelerator,
        "requires_gpu": requires_gpu,
        "allow_cpu_fallback": allow_cpu_fallback,
        "min_vram_gb": runtime.get("min_vram_gb"),
        "estimated_runtime_sec": runtime.get("estimated_runtime_sec"),
        "notes": runtime.get("notes"),
    }


def _resolve_execution(runtime: dict[str, Any]) -> dict[str, Any]:
    gpu_available = _detect_gpu_available()
    host_environment = "gpu" if gpu_available else "cpu"
    supported_accelerators = list(runtime.get("supported_accelerators") or ["cpu"])
    preferred_accelerator = str(runtime.get("preferred_accelerator") or "cpu")
    requires_gpu = bool(runtime.get("requires_gpu", False))
    allow_cpu_fallback = bool(runtime.get("allow_cpu_fallback", not requires_gpu))

    selected_accelerator = "cpu"
    if gpu_available and "gpu" in supported_accelerators and preferred_accelerator == "gpu":
        selected_accelerator = "gpu"
    elif requires_gpu and gpu_available and "gpu" in supported_accelerators:
        selected_accelerator = "gpu"
    elif requires_gpu and not gpu_available:
        if allow_cpu_fallback and "cpu" in supported_accelerators:
            selected_accelerator = "cpu"
        else:
            raise RuntimeError("Tool requires GPU but no GPU is available on the current host.")
    elif gpu_available and "gpu" in supported_accelerators and "cpu" not in supported_accelerators:
        selected_accelerator = "gpu"
    elif "cpu" in supported_accelerators:
        selected_accelerator = "cpu"
    elif "gpu" in supported_accelerators and gpu_available:
        selected_accelerator = "gpu"
    else:
        raise RuntimeError("Tool runtime requirements cannot be satisfied on the current host.")

    return {
        "host_environment": host_environment,
        "gpu_available": gpu_available,
        "selected_accelerator": selected_accelerator,
    }


def run_tool(tool_name: str, payload: dict[str, Any], timeout_seconds: int = 120) -> dict[str, Any]:
    tool = get_tool(tool_name)
    if not tool:
        raise ValueError(f"Unknown tool: {tool_name}")

    plugin_dir = Path(tool["plugin_dir"])
    entrypoint = plugin_dir / str(tool.get("entrypoint", "run.py"))
    if not entrypoint.exists():
        raise ValueError(f"Tool entrypoint not found: {entrypoint}")
    runtime = _normalize_runtime(tool)
    execution = _resolve_execution(runtime)

    with tempfile.TemporaryDirectory(prefix="chatclinic_tool_") as tmpdir:
        tmp_path = Path(tmpdir)
        input_path = tmp_path / "input.json"
        output_path = tmp_path / "output.json"
        payload_with_runtime = dict(payload)
        payload_with_runtime["execution_context"] = {
            **execution,
            "runtime": runtime,
        }
        input_path.write_text(json.dumps(payload_with_runtime, ensure_ascii=False, indent=2), encoding="utf-8")
        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH", "").strip()
        pythonpath_parts = [str(ROOT)]
        if existing_pythonpath:
            pythonpath_parts.append(existing_pythonpath)
        env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
        env["CHATCLINIC_HOST_ENVIRONMENT"] = str(execution["host_environment"])
        env["CHATCLINIC_GPU_AVAILABLE"] = "true" if execution["gpu_available"] else "false"
        env["CHATCLINIC_ACCELERATOR"] = str(execution["selected_accelerator"])
        env["CHATCLINIC_TOOL_RUNTIME"] = json.dumps(runtime, ensure_ascii=False)
        python_executable = str(Path(sys.executable))

        completed = subprocess.run(
            [python_executable, str(entrypoint), "--input", str(input_path), "--output", str(output_path)],
            cwd=str(plugin_dir),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
            env=env,
        )

        if completed.returncode != 0:
            raise RuntimeError(
                "Tool execution failed.\n"
                f"tool={tool_name}\n"
                f"returncode={completed.returncode}\n"
                f"stdout={completed.stdout}\n"
                f"stderr={completed.stderr}"
            )

        if not output_path.exists():
            raise RuntimeError(f"Tool completed without creating output file: {output_path}")

        result = json.loads(output_path.read_text(encoding="utf-8"))
        return {
            "tool": {
                "name": tool.get("name"),
                "team": tool.get("team"),
                "task_type": tool.get("task_type"),
                "modality": tool.get("modality"),
                "approval_required": tool.get("approval_required", True),
                "runtime": runtime,
                "execution": execution,
            },
            "result": result,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
