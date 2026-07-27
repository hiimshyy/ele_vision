"""
Smart Cabin Platform - Configuration Management

Loads configuration from YAML file with environment variable overrides.
Environment variables follow the pattern: SC_{SECTION}_{KEY} (uppercase)
Example: SC_CAMERA_URL overrides config.camera.url
"""

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, ValidationError


# --- Configuration Models ---


class CameraConfig(BaseModel):
    """Camera/RTSP stream configuration."""

    url: str = Field(description="RTSP stream URL")
    capture_fps: int = Field(default=25, ge=1, le=60, description="Native capture FPS")
    process_fps: int = Field(
        default=5, ge=1, le=30, description="FPS distributed to plugins"
    )
    reconnect_interval: float = Field(
        default=5.0, gt=0.0, description="Seconds between reconnection attempts"
    )
    max_reconnect_attempts: int = Field(
        default=0, ge=0, description="Max reconnect attempts (0 = infinite)"
    )


class MQTTConfig(BaseModel):
    """MQTT broker connection configuration."""

    broker_host: str = Field(default="localhost", description="MQTT broker hostname")
    broker_port: int = Field(default=1883, ge=1, le=65535, description="MQTT broker port")
    username: str = Field(default="", description="MQTT username")
    password: str = Field(default="", description="MQTT password")
    device_id: str = Field(default="cabin-001", description="Unique device identifier")
    keepalive: int = Field(default=60, ge=10, description="MQTT keepalive interval (s)")


class PluginEntry(BaseModel):
    """Configuration for a single plugin."""

    name: str = Field(description="Plugin module name")
    enabled: bool = Field(default=True, description="Whether plugin is active")
    config: dict[str, Any] = Field(
        default_factory=dict, description="Plugin-specific configuration"
    )


class PluginsConfig(BaseModel):
    """Plugins configuration section."""

    directory: str = Field(
        default="edge/plugins", description="Path to plugins directory"
    )
    modules: list[PluginEntry] = Field(
        default_factory=list, description="List of plugin configurations"
    )


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = Field(default="INFO", description="Log level")
    format: str = Field(
        default="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        description="Log format string",
    )
    file: str = Field(default="", description="Log file path (empty = stdout only)")


class SmartCabinConfig(BaseModel):
    """Root configuration model for Smart Cabin Platform."""

    camera: CameraConfig
    mqtt: MQTTConfig = Field(default_factory=MQTTConfig)
    plugins: PluginsConfig = Field(default_factory=PluginsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


# --- Configuration Loader ---

# Environment variable prefix
ENV_PREFIX = "SC_"


def _apply_env_overrides(config_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Apply environment variable overrides to config dictionary.

    Pattern: SC_{SECTION}_{KEY}=value
    Examples:
        SC_CAMERA_URL=rtsp://192.168.1.100/stream
        SC_MQTT_BROKER_HOST=mqtt.example.com
        SC_LOGGING_LEVEL=DEBUG
    """
    for env_key, env_value in os.environ.items():
        if not env_key.startswith(ENV_PREFIX):
            continue

        # Remove prefix and split into section + key
        parts = env_key[len(ENV_PREFIX) :].lower().split("_", 1)
        if len(parts) != 2:
            continue

        section, key = parts

        # Initialize section if not exists
        if section not in config_dict:
            config_dict[section] = {}

        # Only override scalar values (not nested dicts/lists)
        if isinstance(config_dict[section], dict):
            # Try to cast to appropriate type
            config_dict[section][key] = _cast_env_value(env_value)

    return config_dict


def _cast_env_value(value: str) -> Any:
    """Cast environment variable string to appropriate Python type."""
    # Boolean
    if value.lower() in ("true", "1", "yes"):
        return True
    if value.lower() in ("false", "0", "no"):
        return False

    # Integer
    try:
        return int(value)
    except ValueError:
        pass

    # Float
    try:
        return float(value)
    except ValueError:
        pass

    # String (default)
    return value


def load_config(
    config_path: str | Path | None = None,
    apply_env: bool = True,
) -> SmartCabinConfig:
    """
    Load Smart Cabin configuration from YAML file.

    Args:
        config_path: Path to YAML config file. If None, searches default locations.
        apply_env: Whether to apply environment variable overrides.

    Returns:
        Validated SmartCabinConfig instance.

    Raises:
        FileNotFoundError: If config file not found.
        ValidationError: If config values are invalid.
    """
    if config_path is None:
        config_path = _find_config_file()

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # Load YAML
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f) or {}

    # Apply environment variable overrides
    if apply_env:
        config_dict = _apply_env_overrides(config_dict)

    # Validate and return
    return SmartCabinConfig(**config_dict)


def _find_config_file() -> Path:
    """Search for config file in default locations."""
    search_paths = [
        Path("config.yaml"),
        Path("edge/config.yaml"),
        Path(os.environ.get("SC_CONFIG_PATH", "")),
    ]

    for path in search_paths:
        if path and path.exists():
            return path

    raise FileNotFoundError(
        "No config.yaml found. Searched: ./config.yaml, ./edge/config.yaml. "
        "Set SC_CONFIG_PATH environment variable or pass config_path explicitly."
    )


# --- CLI Entry Point ---

if __name__ == "__main__":
    import json
    import sys

    print("Smart Cabin Platform - Configuration Loader")
    print("=" * 50)

    try:
        # Allow passing config path as argument
        path = sys.argv[1] if len(sys.argv) > 1 else None
        config = load_config(config_path=path)

        print(f"\nConfiguration loaded successfully!\n")
        print(json.dumps(config.model_dump(), indent=2, default=str))
        print(f"\n{'=' * 50}")
        print("Tip: Override with env vars like SC_CAMERA_URL=rtsp://...")

    except FileNotFoundError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except ValidationError as e:
        print(f"\nVALIDATION ERROR:\n{e}", file=sys.stderr)
        sys.exit(1)
