"""Configuration loading: secrets from the environment (.env), the rest from
config.yaml. Nothing secret is ever read from the YAML file."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

log = logging.getLogger(__name__)

# Repo root = parent of the parkingbot package.
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config.yaml"


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


@dataclass(frozen=True)
class Secrets:
    bot_token: str
    nvr_ip: str
    rtsp_user: str
    rtsp_pass: str
    allowed_chat_id: int


@dataclass(frozen=True)
class Spot:
    name: str
    channel: int
    polygon: list[tuple[int, int]]


@dataclass(frozen=True)
class Settings:
    secrets: Secrets
    cameras: dict[int, str]
    spots: list[Spot]
    raw: dict = field(repr=False)

    # ---- convenience accessors into the raw yaml (with sane defaults) ----
    @property
    def rtsp(self) -> dict:
        return self.raw.get("rtsp", {})

    @property
    def capture(self) -> dict:
        return self.raw.get("capture", {})

    @property
    def detection(self) -> dict:
        return self.raw.get("detection", {})

    @property
    def push(self) -> dict:
        return self.raw.get("push", {})

    @property
    def verify(self) -> dict:
        return self.raw.get("verify", {})

    @property
    def feedback(self) -> dict:
        return self.raw.get("feedback", {})

    @property
    def inference(self) -> dict:
        return self.raw.get("inference", {})


def _require_env(key: str) -> str:
    val = os.environ.get(key, "").strip()
    if not val:
        raise ConfigError(
            f"Missing required environment variable {key!r}. "
            f"Copy .env.example to .env and fill it in (see README)."
        )
    return val


def load_secrets() -> Secrets:
    # load_dotenv is a no-op if .env is absent (e.g. in Docker, vars come from
    # env_file). It never overrides already-set process env vars.
    load_dotenv(ROOT / ".env")
    try:
        chat_id = int(_require_env("ALLOWED_CHAT_ID"))
    except ValueError as exc:
        raise ConfigError("ALLOWED_CHAT_ID must be an integer.") from exc
    return Secrets(
        bot_token=_require_env("BOT_TOKEN"),
        nvr_ip=_require_env("NVR_IP"),
        rtsp_user=_require_env("RTSP_USER"),
        rtsp_pass=_require_env("RTSP_PASS"),
        allowed_chat_id=chat_id,
    )


def _parse_spots(raw: dict) -> list[Spot]:
    spots: list[Spot] = []
    for entry in raw.get("spots", []):
        poly = [(int(x), int(y)) for x, y in entry["polygon"]]
        if len(poly) < 3:
            raise ConfigError(
                f"Spot {entry.get('name')!r} polygon needs >=3 points."
            )
        spots.append(
            Spot(name=str(entry["name"]), channel=int(entry["channel"]), polygon=poly)
        )
    if not spots:
        raise ConfigError("config.yaml defines no parking spots.")
    return spots


def load_settings(config_path: Path | None = None) -> Settings:
    path = config_path or DEFAULT_CONFIG
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    raw = yaml.safe_load(path.read_text()) or {}

    cameras = {int(k): str(v) for k, v in (raw.get("cameras") or {}).items()}
    if not cameras:
        raise ConfigError("config.yaml defines no cameras.")

    settings = Settings(
        secrets=load_secrets(),
        cameras=cameras,
        spots=_parse_spots(raw),
        raw=raw,
    )
    log.info(
        "Loaded config: %d cameras, %d parking spots, push=%s",
        len(settings.cameras),
        len(settings.spots),
        settings.push.get("enabled", False),
    )
    return settings
