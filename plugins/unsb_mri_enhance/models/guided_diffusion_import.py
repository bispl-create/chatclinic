import os
import sys
from pathlib import Path


def _candidate_paths():
    repo_root = Path(__file__).resolve().parents[1]
    env_path = os.environ.get("GUIDED_DIFFUSION_PATH")

    candidates = []
    if env_path:
        candidates.append(Path(env_path).expanduser())

    candidates.extend([
        repo_root / "guided-diffusion",
        repo_root.parent / "guided-diffusion",
        repo_root / "third_party" / "guided-diffusion",
    ])
    return candidates


def _import_guided_diffusion_modules():
    from guided_diffusion.script_util import create_model
    from guided_diffusion.gaussian_diffusion import get_named_beta_schedule

    return create_model, get_named_beta_schedule


def load_guided_diffusion():
    try:
        return _import_guided_diffusion_modules()
    except ModuleNotFoundError as exc:
        if exc.name != "guided_diffusion":
            raise

    tried_paths = []
    for candidate in _candidate_paths():
        candidate_str = str(candidate)
        tried_paths.append(candidate_str)
        if candidate.exists() and candidate.is_dir() and candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)

    try:
        return _import_guided_diffusion_modules()
    except ModuleNotFoundError as exc:
        if exc.name != "guided_diffusion":
            raise
        searched = "\n".join(f"- {p}" for p in tried_paths)
        raise ModuleNotFoundError(
            "Could not import 'guided_diffusion'. "
            "Set GUIDED_DIFFUSION_PATH to the directory containing the "
            "'guided_diffusion' package, or place the guided-diffusion repo "
            "in one of these locations:\n"
            f"{searched}"
        ) from exc
