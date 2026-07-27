"""
Tests for edge/core/config.py - Configuration Management System.

Covers:
- YAML loading from file
- Environment variable overrides
- Validation (missing required fields, invalid values)
- Default values when fields are absent
"""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from edge.core.config import (
    SmartCabinConfig,
    load_config,
    _apply_env_overrides,
    _cast_env_value,
)


# --- Fixtures ---


@pytest.fixture
def minimal_config_yaml(tmp_path):
    """Create a minimal valid config file (only required fields)."""
    config = {
        "camera": {
            "url": "rtsp://192.168.1.100:554/stream",
        }
    }
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump(config), encoding="utf-8")
    return config_file


@pytest.fixture
def full_config_yaml(tmp_path):
    """Create a fully specified config file."""
    config = {
        "camera": {
            "url": "rtsp://10.0.0.1:554/main",
            "capture_fps": 30,
            "process_fps": 10,
            "reconnect_interval": 3.0,
            "max_reconnect_attempts": 5,
        },
        "mqtt": {
            "broker_host": "mqtt.example.com",
            "broker_port": 8883,
            "username": "cabin_user",
            "password": "secret123",
            "device_id": "cabin-002",
            "keepalive": 120,
        },
        "plugins": {
            "directory": "/opt/plugins",
            "modules": [
                {
                    "name": "face_recognition",
                    "enabled": True,
                    "config": {"threshold": 0.7},
                },
                {
                    "name": "people_counter",
                    "enabled": False,
                    "config": {},
                },
            ],
        },
        "logging": {
            "level": "DEBUG",
            "format": "%(message)s",
            "file": "/var/log/cabin.log",
        },
    }
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump(config), encoding="utf-8")
    return config_file


@pytest.fixture
def invalid_config_yaml(tmp_path):
    """Create a config file with invalid values."""
    config = {
        "camera": {
            "url": "rtsp://host/stream",
            "capture_fps": -1,  # Invalid: must be >= 1
            "process_fps": 100,  # Invalid: must be <= 30
        }
    }
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump(config), encoding="utf-8")
    return config_file


@pytest.fixture
def missing_required_config_yaml(tmp_path):
    """Create a config file missing required 'camera.url'."""
    config = {
        "camera": {
            "capture_fps": 25,
            # 'url' is missing
        }
    }
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump(config), encoding="utf-8")
    return config_file


@pytest.fixture(autouse=True)
def clean_env():
    """Remove any SC_ environment variables before/after each test."""
    sc_vars = [key for key in os.environ if key.startswith("SC_")]
    for key in sc_vars:
        del os.environ[key]
    yield
    sc_vars = [key for key in os.environ if key.startswith("SC_")]
    for key in sc_vars:
        del os.environ[key]


# --- Test: Load from YAML ---


class TestLoadConfig:
    """Tests for loading configuration from YAML files."""

    def test_load_minimal_config(self, minimal_config_yaml):
        """Minimal config (only camera.url) should load with defaults."""
        config = load_config(config_path=minimal_config_yaml)

        assert config.camera.url == "rtsp://192.168.1.100:554/stream"
        assert config.camera.capture_fps == 25  # default
        assert config.camera.process_fps == 5  # default
        assert config.mqtt.broker_host == "localhost"  # default
        assert config.mqtt.broker_port == 1883  # default
        assert config.logging.level == "INFO"  # default

    def test_load_full_config(self, full_config_yaml):
        """Full config should load all values correctly."""
        config = load_config(config_path=full_config_yaml)

        # Camera
        assert config.camera.url == "rtsp://10.0.0.1:554/main"
        assert config.camera.capture_fps == 30
        assert config.camera.process_fps == 10
        assert config.camera.reconnect_interval == 3.0
        assert config.camera.max_reconnect_attempts == 5

        # MQTT
        assert config.mqtt.broker_host == "mqtt.example.com"
        assert config.mqtt.broker_port == 8883
        assert config.mqtt.username == "cabin_user"
        assert config.mqtt.password == "secret123"
        assert config.mqtt.device_id == "cabin-002"
        assert config.mqtt.keepalive == 120

        # Plugins
        assert config.plugins.directory == "/opt/plugins"
        assert len(config.plugins.modules) == 2
        assert config.plugins.modules[0].name == "face_recognition"
        assert config.plugins.modules[0].enabled is True
        assert config.plugins.modules[0].config == {"threshold": 0.7}
        assert config.plugins.modules[1].name == "people_counter"
        assert config.plugins.modules[1].enabled is False

        # Logging
        assert config.logging.level == "DEBUG"
        assert config.logging.file == "/var/log/cabin.log"

    def test_load_nonexistent_file(self):
        """Should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_config(config_path="/nonexistent/config.yaml")

    def test_load_empty_yaml(self, tmp_path):
        """Empty YAML (no camera section) should fail validation."""
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("", encoding="utf-8")

        with pytest.raises(Exception):  # ValidationError from pydantic
            load_config(config_path=config_file)


# --- Test: Validation ---


class TestConfigValidation:
    """Tests for config validation with Pydantic."""

    def test_invalid_capture_fps(self, invalid_config_yaml):
        """FPS out of range should raise ValidationError."""
        with pytest.raises(Exception):  # ValidationError
            load_config(config_path=invalid_config_yaml)

    def test_missing_required_field(self, missing_required_config_yaml):
        """Missing camera.url should raise ValidationError."""
        with pytest.raises(Exception):  # ValidationError
            load_config(config_path=missing_required_config_yaml)

    def test_invalid_mqtt_port(self, tmp_path):
        """MQTT port out of range should raise ValidationError."""
        config = {
            "camera": {"url": "rtsp://host/stream"},
            "mqtt": {"broker_port": 99999},  # Invalid: > 65535
        }
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(config), encoding="utf-8")

        with pytest.raises(Exception):
            load_config(config_path=config_file)

    def test_valid_edge_values(self, tmp_path):
        """Boundary values should pass validation."""
        config = {
            "camera": {
                "url": "rtsp://host/stream",
                "capture_fps": 1,  # minimum
                "process_fps": 30,  # maximum
            },
            "mqtt": {
                "broker_port": 1,  # minimum
                "keepalive": 10,  # minimum
            },
        }
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(config), encoding="utf-8")

        result = load_config(config_path=config_file)
        assert result.camera.capture_fps == 1
        assert result.camera.process_fps == 30
        assert result.mqtt.broker_port == 1


# --- Test: Default Values ---


class TestDefaultValues:
    """Tests for default values when fields are absent."""

    def test_mqtt_defaults(self, minimal_config_yaml):
        """MQTT section should use defaults when not specified."""
        config = load_config(config_path=minimal_config_yaml)

        assert config.mqtt.broker_host == "localhost"
        assert config.mqtt.broker_port == 1883
        assert config.mqtt.username == ""
        assert config.mqtt.password == ""
        assert config.mqtt.device_id == "cabin-001"
        assert config.mqtt.keepalive == 60

    def test_plugins_defaults(self, minimal_config_yaml):
        """Plugins section should use defaults when not specified."""
        config = load_config(config_path=minimal_config_yaml)

        assert config.plugins.directory == "edge/plugins"
        assert config.plugins.modules == []

    def test_logging_defaults(self, minimal_config_yaml):
        """Logging section should use defaults when not specified."""
        config = load_config(config_path=minimal_config_yaml)

        assert config.logging.level == "INFO"
        assert "%(asctime)s" in config.logging.format
        assert config.logging.file == ""

    def test_camera_defaults(self, tmp_path):
        """Camera should use defaults for optional fields."""
        config = {"camera": {"url": "rtsp://host/stream"}}
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(config), encoding="utf-8")

        result = load_config(config_path=config_file)
        assert result.camera.capture_fps == 25
        assert result.camera.process_fps == 5
        assert result.camera.reconnect_interval == 5.0
        assert result.camera.max_reconnect_attempts == 0


# --- Test: Environment Variable Overrides ---


class TestEnvOverrides:
    """Tests for environment variable override mechanism."""

    def test_override_camera_url(self, minimal_config_yaml):
        """SC_CAMERA_URL should override camera.url from YAML."""
        os.environ["SC_CAMERA_URL"] = "rtsp://override:554/stream"

        config = load_config(config_path=minimal_config_yaml)
        assert config.camera.url == "rtsp://override:554/stream"

    def test_override_mqtt_host(self, minimal_config_yaml):
        """SC_MQTT_BROKER_HOST should override mqtt.broker_host."""
        os.environ["SC_MQTT_BROKER_HOST"] = "mqtt.production.com"

        config = load_config(config_path=minimal_config_yaml)
        # Note: env key SC_MQTT_BROKER_HOST maps to mqtt.broker_host
        # The current implementation splits on first _ only: section=mqtt, key=broker_host
        # This maps correctly because the split is SC_ + MQTT + BROKER_HOST
        # Actually: parts = "MQTT_BROKER_HOST".split("_", 1) = ["mqtt", "broker_host"]
        assert config.mqtt.broker_host == "mqtt.production.com"

    def test_override_logging_level(self, minimal_config_yaml):
        """SC_LOGGING_LEVEL should override logging.level."""
        os.environ["SC_LOGGING_LEVEL"] = "DEBUG"

        config = load_config(config_path=minimal_config_yaml)
        assert config.logging.level == "DEBUG"

    def test_override_integer_value(self, minimal_config_yaml):
        """Integer env values should be cast correctly."""
        os.environ["SC_CAMERA_CAPTURE_FPS"] = "15"

        config = load_config(config_path=minimal_config_yaml)
        # SC_CAMERA_CAPTURE_FPS splits to section=camera, key=capture_fps
        # Wait - split("_", 1) on "CAMERA_CAPTURE_FPS" gives ["camera", "capture_fps"]
        assert config.camera.capture_fps == 15

    def test_override_disabled(self, minimal_config_yaml):
        """When apply_env=False, env vars should be ignored."""
        os.environ["SC_CAMERA_URL"] = "rtsp://should-be-ignored/stream"

        config = load_config(config_path=minimal_config_yaml, apply_env=False)
        assert config.camera.url == "rtsp://192.168.1.100:554/stream"

    def test_non_sc_env_vars_ignored(self, minimal_config_yaml):
        """Non SC_ prefixed variables should not affect config."""
        os.environ["OTHER_CAMERA_URL"] = "rtsp://other/stream"
        os.environ["CAMERA_URL"] = "rtsp://plain/stream"

        config = load_config(config_path=minimal_config_yaml)
        assert config.camera.url == "rtsp://192.168.1.100:554/stream"


# --- Test: Cast Env Value Helper ---


class TestCastEnvValue:
    """Tests for _cast_env_value helper function."""

    def test_cast_boolean_true(self):
        assert _cast_env_value("true") is True
        assert _cast_env_value("True") is True
        assert _cast_env_value("1") is True
        assert _cast_env_value("yes") is True

    def test_cast_boolean_false(self):
        assert _cast_env_value("false") is False
        assert _cast_env_value("False") is False
        assert _cast_env_value("0") is False
        assert _cast_env_value("no") is False

    def test_cast_integer(self):
        assert _cast_env_value("42") == 42
        assert _cast_env_value("1883") == 1883

    def test_cast_float(self):
        assert _cast_env_value("3.14") == 3.14
        assert _cast_env_value("5.0") == 5.0

    def test_cast_string(self):
        assert _cast_env_value("hello") == "hello"
        assert _cast_env_value("rtsp://host/stream") == "rtsp://host/stream"

    def test_cast_empty_string(self):
        assert _cast_env_value("") == ""
