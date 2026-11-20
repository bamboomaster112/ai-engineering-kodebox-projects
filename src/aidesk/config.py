"""Settings for AI Desk, read once from the environment and config/models.toml."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
ROLES = ("fast", "main", "reasoning", "judge", "embed", "image")


def _load_env_files() -> None:
    # The KodeBox image keeps the KodeKey settings in ~/.env. A repo-level .env may add to it.
    # Values already exported in the shell always win.
    for path in (Path.home() / ".env", REPO_ROOT / ".env"):
        if path.is_file():
            load_dotenv(path, override=False)


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else default


@dataclass(frozen=True)
class Settings:
    provider: str
    base_url: str
    api_key: str = field(repr=False)
    models: dict[str, str]
    embed_dimensions: int
    embed_mode: str
    playground_models: list[str]
    probe_models: list[str]
    data_dir: Path
    cache_enabled: bool
    run_token_cap: int
    daily_token_cap: int
    search_backend: str
    kb_dir: Path

    @property
    def db_path(self) -> Path:
        return self.data_dir / "aidesk.db"

    @property
    def checkpoint_path(self) -> Path:
        return self.data_dir / "checkpoints.db"

    def model_for(self, role: str) -> str:
        if role not in self.models:
            raise KeyError(f"unknown model role {role!r}; known roles: {sorted(self.models)}")
        return self.models[role]


def _read_models_toml() -> dict:
    path = Path(os.environ.get("AIDESK_MODELS_FILE", REPO_ROOT / "config" / "models.toml"))
    with path.open("rb") as fh:
        return tomllib.load(fh)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    _load_env_files()
    raw = _read_models_toml()
    models = dict(raw.get("roles", {}))
    for role in ROLES:
        override = os.environ.get(f"AIDESK_MODEL_{role.upper()}", "").strip()
        if override:
            models[role] = override

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    provider = os.environ.get("AIDESK_PROVIDER", "").strip().lower() or ("kodekey" if api_key else "fake")
    if provider not in {"kodekey", "fake"}:
        raise ValueError(f"AIDESK_PROVIDER must be kodekey or fake, not {provider!r}")

    data_dir = Path(os.environ.get("AIDESK_DATA_DIR", REPO_ROOT / "var")).expanduser()
    data_dir.mkdir(parents=True, exist_ok=True)

    return Settings(
        provider=provider,
        base_url=os.environ.get("OPENAI_BASE_URL", "https://api.ai.kodekloud.com/v1").strip(),
        api_key=api_key,
        models=models,
        embed_dimensions=int(raw.get("embeddings", {}).get("dimensions", 512)),
        embed_mode=os.environ.get("AIDESK_EMBEDDINGS", raw.get("embeddings", {}).get("mode", "auto")).strip().lower(),
        playground_models=list(raw.get("playground", {}).get("models", [])),
        probe_models=list(raw.get("probe", {}).get("chat", [])),
        data_dir=data_dir,
        cache_enabled=os.environ.get("AIDESK_CACHE", "1").strip() != "0",
        run_token_cap=_int_env("AIDESK_RUN_TOKEN_CAP", 40_000),
        daily_token_cap=_int_env("AIDESK_DAILY_TOKEN_CAP", 150_000),
        search_backend=os.environ.get("AIDESK_SEARCH", "ddgs").strip().lower(),
        kb_dir=Path(os.environ.get("AIDESK_KB_DIR", REPO_ROOT / "data" / "kb")),
    )


def reset_settings() -> None:
    """Forget cached settings (tests change the environment between cases)."""
    get_settings.cache_clear()
