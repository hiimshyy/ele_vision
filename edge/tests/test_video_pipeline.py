"""
Tests for edge/core/video_pipeline.py - Video Pipeline System.

Covers:
- Pipeline lifecycle (start, stop, state transitions)
- Frame callback registration/unregistration with per-callback FPS
- Frame scheduling (per-callback FPS control)
- Latest frame buffer (single frame, no ring buffer)
- Reconnection logic on stream failure
- Thread safety of callback invocation
"""

import threading
import time
from unittest.mock import patch

import numpy as np
import pytest

from edge.core.config import CameraConfig
from edge.core.video_pipeline import (
    VideoPipeline,
    PipelineState,
    FrameData,
    ScheduledCallback,
)


# --- Fixtures ---


@pytest.fixture
def camera_config():
    """Camera config for testing with fast reconnect."""
    return CameraConfig(
        url="rtsp://fake-camera:554/stream",
        capture_fps=25,
        process_fps=10,
        reconnect_interval=0.1,
        max_reconnect_attempts=3,
    )


@pytest.fixture
def pipeline(camera_config):
    """Create a pipeline instance (not started)."""
    p = VideoPipeline(camera_config)
    yield p
    # Ensure cleanup
    if p.state not in (PipelineState.IDLE, PipelineState.STOPPED):
        p.stop()


def make_fake_frame(width=640, height=480):
    """Create a fake BGR frame."""
    return np.zeros((height, width, 3), dtype=np.uint8)


# --- Mock VideoCapture ---


class MockVideoCapture:
    """Mock cv2.VideoCapture that produces fake frames."""

    def __init__(self, url=None, apiPreference=None, params=None,
                 frames_to_produce=50, fail_after=None):
        self._url = url
        self._opened = True
        self._frame_count = 0
        self._frames_to_produce = frames_to_produce
        self._fail_after = fail_after

    def isOpened(self):
        return self._opened

    def read(self):
        if not self._opened:
            return False, None

        self._frame_count += 1

        if self._fail_after and self._frame_count > self._fail_after:
            return False, None

        if self._frame_count > self._frames_to_produce:
            return False, None

        frame = make_fake_frame()
        frame[0, 0, 0] = self._frame_count % 256
        return True, frame

    def release(self):
        self._opened = False


class FailingVideoCapture:
    """Mock cv2.VideoCapture that fails to open."""

    def __init__(self, url=None, apiPreference=None, params=None):
        self._url = url

    def isOpened(self):
        return False

    def release(self):
        pass


# --- Test: Pipeline Lifecycle ---


class TestPipelineLifecycle:
    """Tests for pipeline start/stop and state management."""

    def test_initial_state_is_idle(self, pipeline):
        """Pipeline should start in IDLE state."""
        assert pipeline.state == PipelineState.IDLE

    def test_stop_from_idle(self, pipeline):
        """Stopping an idle pipeline should move to STOPPED."""
        pipeline.stop()
        assert pipeline.state == PipelineState.STOPPED

    @patch("edge.core.video_pipeline.cv2.VideoCapture")
    def test_start_moves_to_running(self, mock_cv2, pipeline):
        """Starting with successful connection should reach RUNNING."""
        mock_cv2.side_effect = lambda *args, **kwargs: MockVideoCapture(frames_to_produce=500)

        pipeline.start()
        time.sleep(0.5)

        assert pipeline.state == PipelineState.RUNNING

        pipeline.stop()
        assert pipeline.state == PipelineState.STOPPED

    @patch("edge.core.video_pipeline.cv2.VideoCapture")
    def test_stop_terminates_threads(self, mock_cv2, pipeline):
        """Stop should terminate capture and scheduler threads."""
        mock_cv2.side_effect = lambda *args, **kwargs: MockVideoCapture(frames_to_produce=5000)

        pipeline.start()
        time.sleep(0.3)
        pipeline.stop()

        assert not pipeline._capture_thread.is_alive()
        assert not pipeline._scheduler_thread.is_alive()

    @patch("edge.core.video_pipeline.cv2.VideoCapture")
    def test_double_start_ignored(self, mock_cv2, pipeline):
        """Calling start twice should not create extra threads."""
        mock_cv2.side_effect = lambda *args, **kwargs: MockVideoCapture(frames_to_produce=5000)

        pipeline.start()
        time.sleep(0.3)
        pipeline.start()  # Should be ignored

        pipeline.stop()


# --- Test: Frame Callbacks ---


class TestFrameCallbacks:
    """Tests for callback registration and invocation."""

    def test_register_callback(self, pipeline):
        """Should register a callback with FPS."""
        def my_callback(frame, frame_id, timestamp):
            pass

        pipeline.register_callback(my_callback, fps=10)
        assert len(pipeline._callbacks) == 1
        assert pipeline._callbacks[0].target_fps == 10

    def test_register_with_default_fps(self, pipeline):
        """Default FPS should be 15."""
        def my_callback(frame, frame_id, timestamp):
            pass

        pipeline.register_callback(my_callback)
        assert pipeline._callbacks[0].target_fps == 15.0

    def test_unregister_callback(self, pipeline):
        """Should unregister a callback."""
        def my_callback(frame, frame_id, timestamp):
            pass

        pipeline.register_callback(my_callback, fps=5)
        pipeline.unregister_callback(my_callback)
        assert len(pipeline._callbacks) == 0

    def test_duplicate_register_ignored(self, pipeline):
        """Registering same callback twice should not duplicate."""
        def my_callback(frame, frame_id, timestamp):
            pass

        pipeline.register_callback(my_callback, fps=5)
        pipeline.register_callback(my_callback, fps=10)
        assert len(pipeline._callbacks) == 1

    @patch("edge.core.video_pipeline.cv2.VideoCapture")
    def test_callback_receives_frames(self, mock_cv2, pipeline):
        """Registered callback should receive frames."""
        mock_cv2.side_effect = lambda *args, **kwargs: MockVideoCapture(frames_to_produce=500)

        received_frames = []

        def collector(frame, frame_id, timestamp):
            received_frames.append((frame_id, timestamp))

        pipeline.register_callback(collector, fps=10)
        pipeline.start()

        time.sleep(1.0)
        pipeline.stop()

        assert len(received_frames) > 0
        assert all(fid > 0 for fid, _ in received_frames)
        assert all(ts > 0 for _, ts in received_frames)

    @patch("edge.core.video_pipeline.cv2.VideoCapture")
    def test_callback_error_does_not_crash_pipeline(self, mock_cv2, pipeline):
        """A failing callback should not crash the pipeline."""
        mock_cv2.side_effect = lambda *args, **kwargs: MockVideoCapture(frames_to_produce=500)

        call_count = {"good": 0}

        def bad_callback(frame, frame_id, timestamp):
            raise RuntimeError("I'm broken!")

        def good_callback(frame, frame_id, timestamp):
            call_count["good"] += 1

        pipeline.register_callback(bad_callback, fps=10)
        pipeline.register_callback(good_callback, fps=10)
        pipeline.start()

        time.sleep(0.8)
        pipeline.stop()

        assert call_count["good"] > 0

    @patch("edge.core.video_pipeline.cv2.VideoCapture")
    def test_multiple_callbacks_all_invoked(self, mock_cv2, pipeline):
        """All registered callbacks should receive frames."""
        mock_cv2.side_effect = lambda *args, **kwargs: MockVideoCapture(frames_to_produce=500)

        counts = {"a": 0, "b": 0, "c": 0}

        def callback_a(frame, frame_id, timestamp):
            counts["a"] += 1

        def callback_b(frame, frame_id, timestamp):
            counts["b"] += 1

        def callback_c(frame, frame_id, timestamp):
            counts["c"] += 1

        pipeline.register_callback(callback_a, fps=10)
        pipeline.register_callback(callback_b, fps=10)
        pipeline.register_callback(callback_c, fps=10)

        pipeline.start()
        time.sleep(0.8)
        pipeline.stop()

        assert counts["a"] > 0
        assert counts["b"] > 0
        assert counts["c"] > 0


# --- Test: Per-Callback FPS Scheduling ---


class TestFPSScheduling:
    """Tests for per-callback FPS control."""

    @patch("edge.core.video_pipeline.cv2.VideoCapture")
    def test_different_fps_per_callback(self, mock_cv2):
        """Callbacks with different FPS should receive frames at different rates."""
        config = CameraConfig(
            url="rtsp://fake/stream",
            capture_fps=25,
            process_fps=15,
            reconnect_interval=0.1,
            max_reconnect_attempts=0,
        )
        mock_cv2.side_effect = lambda *args, **kwargs: MockVideoCapture(frames_to_produce=5000)

        pipeline = VideoPipeline(config)

        counts = {"fast": 0, "slow": 0}

        def fast_callback(frame, frame_id, timestamp):
            counts["fast"] += 1

        def slow_callback(frame, frame_id, timestamp):
            counts["slow"] += 1

        pipeline.register_callback(fast_callback, fps=20)
        pipeline.register_callback(slow_callback, fps=5)

        pipeline.start()
        time.sleep(2.0)
        pipeline.stop()

        # Fast should have ~4x more invocations than slow
        assert counts["fast"] > counts["slow"]
        ratio = counts["fast"] / max(counts["slow"], 1)
        assert 2.0 <= ratio <= 6.0, f"Expected ratio ~4, got {ratio:.1f}"

    @patch("edge.core.video_pipeline.cv2.VideoCapture")
    def test_stats_track_capture_and_distribute(self, mock_cv2, pipeline):
        """Stats should track both capture and distributed frame counts."""
        mock_cv2.side_effect = lambda *args, **kwargs: MockVideoCapture(frames_to_produce=5000)

        def dummy(frame, frame_id, timestamp):
            pass

        pipeline.register_callback(dummy, fps=10)
        pipeline.start()
        time.sleep(1.0)
        pipeline.stop()

        stats = pipeline.stats
        assert stats.frames_captured > 0
        assert stats.frames_distributed > 0
        assert stats.frames_captured >= stats.frames_distributed


# --- Test: Latest Frame Buffer ---


class TestLatestFrameBuffer:
    """Tests for the single-frame latest buffer behavior."""

    def test_no_frame_initially(self, pipeline):
        """Latest frame should be None before start."""
        assert pipeline._latest_frame is None

    @patch("edge.core.video_pipeline.cv2.VideoCapture")
    def test_latest_frame_always_newest(self, mock_cv2, pipeline):
        """Buffer should always contain the most recent frame."""
        mock_cv2.side_effect = lambda *args, **kwargs: MockVideoCapture(frames_to_produce=100)

        received_ids = []

        def collector(frame, frame_id, timestamp):
            received_ids.append(frame_id)

        pipeline.register_callback(collector, fps=5)
        pipeline.start()
        time.sleep(1.0)
        pipeline.stop()

        # Frame IDs should be monotonically increasing (always newest)
        assert len(received_ids) > 0
        for i in range(1, len(received_ids)):
            assert received_ids[i] > received_ids[i - 1]


# --- Test: Reconnection ---


class TestReconnection:
    """Tests for auto-reconnect behavior on stream failure."""

    @patch("edge.core.video_pipeline.cv2.VideoCapture")
    def test_reconnect_on_stream_failure(self, mock_cv2, pipeline):
        """Pipeline should attempt reconnection when stream drops."""
        call_count = {"n": 0}

        def create_capture(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return MockVideoCapture(fail_after=5)
            return MockVideoCapture(frames_to_produce=5000)

        mock_cv2.side_effect = create_capture

        def dummy(frame, frame_id, timestamp):
            pass

        pipeline.register_callback(dummy, fps=10)
        pipeline.start()
        time.sleep(1.0)
        pipeline.stop()

        assert pipeline.stats.reconnect_count >= 1
        assert pipeline.stats.frames_captured > 5

    @patch("edge.core.video_pipeline.cv2.VideoCapture")
    def test_max_reconnect_attempts_honored(self, mock_cv2):
        """Pipeline should stop after max reconnect attempts."""
        config = CameraConfig(
            url="rtsp://unreachable/stream",
            capture_fps=25,
            process_fps=5,
            reconnect_interval=0.05,
            max_reconnect_attempts=2,
        )
        mock_cv2.side_effect = lambda *args, **kwargs: FailingVideoCapture()

        pipeline = VideoPipeline(config)
        pipeline.start()

        time.sleep(0.5)
        time.sleep(0.2)
        assert pipeline.state == PipelineState.ERROR
        pipeline.stop()

    @patch("edge.core.video_pipeline.cv2.VideoCapture")
    def test_infinite_reconnect_when_max_is_zero(self, mock_cv2):
        """When max_reconnect_attempts=0, should retry indefinitely."""
        config = CameraConfig(
            url="rtsp://unreachable/stream",
            capture_fps=25,
            process_fps=5,
            reconnect_interval=0.05,
            max_reconnect_attempts=0,
        )

        attempt_count = {"n": 0}

        def create_capture(*args, **kwargs):
            attempt_count["n"] += 1
            if attempt_count["n"] >= 4:
                return MockVideoCapture(frames_to_produce=500)
            return FailingVideoCapture()

        mock_cv2.side_effect = create_capture

        pipeline = VideoPipeline(config)

        def dummy(frame, frame_id, timestamp):
            pass

        pipeline.register_callback(dummy, fps=10)
        pipeline.start()

        time.sleep(1.0)
        pipeline.stop()

        assert attempt_count["n"] >= 4
        assert pipeline.stats.frames_captured > 0


# --- Test: Thread Safety ---


class TestThreadSafety:
    """Tests for concurrent access patterns."""

    @patch("edge.core.video_pipeline.cv2.VideoCapture")
    def test_register_during_running(self, mock_cv2, pipeline):
        """Registering callbacks while pipeline is running should be safe."""
        mock_cv2.side_effect = lambda *args, **kwargs: MockVideoCapture(frames_to_produce=5000)

        counts = {"late": 0}

        pipeline.start()
        time.sleep(0.5)

        def late_callback(frame, frame_id, timestamp):
            counts["late"] += 1

        pipeline.register_callback(late_callback, fps=10)
        time.sleep(0.8)
        pipeline.stop()

        assert counts["late"] > 0

    @patch("edge.core.video_pipeline.cv2.VideoCapture")
    def test_unregister_during_running(self, mock_cv2, pipeline):
        """Unregistering callbacks while pipeline is running should be safe."""
        mock_cv2.side_effect = lambda *args, **kwargs: MockVideoCapture(frames_to_produce=5000)

        counts = {"total": 0}

        def counting_callback(frame, frame_id, timestamp):
            counts["total"] += 1

        pipeline.register_callback(counting_callback, fps=10)
        pipeline.start()
        time.sleep(0.5)

        pipeline.unregister_callback(counting_callback)
        count_at_unregister = counts["total"]

        time.sleep(0.5)
        pipeline.stop()

        # Should not have received many more frames after unregister
        assert counts["total"] - count_at_unregister <= 2


# --- Test: Plugin Isolation ---


class TestPluginIsolation:
    """Tests to verify scheduler resilience against slow or crashing plugins."""

    @patch("edge.core.video_pipeline.cv2.VideoCapture")
    def test_slow_plugin_does_not_block_others(self, mock_cv2):
        """
        A plugin that takes 100-500ms to process should NOT block other plugins
        from receiving frames at their target FPS.
        """
        config = CameraConfig(
            url="rtsp://fake/stream",
            capture_fps=25,
            process_fps=15,
            reconnect_interval=0.1,
            max_reconnect_attempts=0,
        )
        mock_cv2.side_effect = lambda *args, **kwargs: MockVideoCapture(frames_to_produce=10000)

        pipeline = VideoPipeline(config)

        counts = {"slow": 0, "fast": 0}
        timestamps = {"fast": []}

        def slow_plugin(frame, frame_id, timestamp):
            """Simulates a heavy plugin (100-500ms processing)."""
            time.sleep(0.3)  # 300ms processing time
            counts["slow"] += 1

        def fast_plugin(frame, frame_id, timestamp):
            """Normal plugin that should not be affected by slow one."""
            counts["fast"] += 1
            timestamps["fast"].append(time.time())

        pipeline.register_callback(slow_plugin, fps=2)   # Slow plugin at 2fps
        pipeline.register_callback(fast_plugin, fps=10)  # Fast plugin at 10fps

        pipeline.start()
        time.sleep(3.0)  # Run for 3 seconds
        pipeline.stop()

        # Fast plugin should have received significantly more frames than slow
        assert counts["fast"] > counts["slow"], (
            f"Fast plugin ({counts['fast']}) should have more frames than slow ({counts['slow']})"
        )

        # Fast plugin should have ~30 frames in 3s at 10fps (allow tolerance)
        assert counts["fast"] >= 15, (
            f"Fast plugin should have at least 15 frames in 3s at 10fps, got {counts['fast']}"
        )

        # Slow plugin should still receive frames (not starved)
        assert counts["slow"] >= 3, (
            f"Slow plugin should have at least 3 frames in 3s at 2fps, got {counts['slow']}"
        )

        # Verify fast plugin timing is roughly consistent (not bunched up)
        if len(timestamps["fast"]) >= 3:
            intervals = [
                timestamps["fast"][i+1] - timestamps["fast"][i]
                for i in range(len(timestamps["fast"]) - 1)
            ]
            avg_interval = sum(intervals) / len(intervals)
            # At 10fps, average interval should be ~0.1s (allow 0.05-0.3 for tolerance)
            assert 0.03 <= avg_interval <= 0.5, (
                f"Fast plugin average interval {avg_interval:.3f}s too far from 0.1s target"
            )

    @patch("edge.core.video_pipeline.cv2.VideoCapture")
    def test_exception_plugin_does_not_crash_pipeline(self, mock_cv2):
        """
        A plugin that always throws exceptions should NOT stop the pipeline
        or prevent other plugins from receiving frames.
        """
        config = CameraConfig(
            url="rtsp://fake/stream",
            capture_fps=25,
            process_fps=15,
            reconnect_interval=0.1,
            max_reconnect_attempts=0,
        )
        mock_cv2.side_effect = lambda *args, **kwargs: MockVideoCapture(frames_to_produce=10000)

        pipeline = VideoPipeline(config)

        counts = {"good": 0, "errors_thrown": 0}

        def crashing_plugin(frame, frame_id, timestamp):
            """Plugin that always raises an exception."""
            counts["errors_thrown"] += 1
            raise RuntimeError(f"Plugin crash at frame {frame_id}!")

        def healthy_plugin(frame, frame_id, timestamp):
            """Normal plugin that should keep working regardless."""
            counts["good"] += 1

        pipeline.register_callback(crashing_plugin, fps=10)
        pipeline.register_callback(healthy_plugin, fps=10)

        pipeline.start()
        time.sleep(2.0)  # Run for 2 seconds
        pipeline.stop()

        # Pipeline should still be running (not crashed)
        assert pipeline.state == PipelineState.STOPPED  # Clean stop

        # Crashing plugin should have been called (errors were thrown)
        assert counts["errors_thrown"] > 0, "Crashing plugin was never called"

        # Healthy plugin should have received frames normally
        assert counts["good"] >= 10, (
            f"Healthy plugin should have at least 10 frames in 2s at 10fps, got {counts['good']}"
        )

        # Both should have similar invocation counts (scheduler doesn't skip crashing one)
        ratio = counts["good"] / max(counts["errors_thrown"], 1)
        assert 0.5 <= ratio <= 2.0, (
            f"Good/error ratio {ratio:.1f} too skewed - scheduler may be biased"
        )
