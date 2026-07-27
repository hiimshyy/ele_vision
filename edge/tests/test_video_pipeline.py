"""
Tests for edge/core/video_pipeline.py - Video Pipeline System.

Covers:
- Pipeline lifecycle (start, stop, state transitions)
- Frame callback registration/unregistration
- FPS throttling (capture vs distribute rate)
- Ring buffer behavior (bounded, drops oldest)
- Reconnection logic on stream failure
- Thread safety of callback invocation
"""

import threading
import time
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

from edge.core.config import CameraConfig
from edge.core.video_pipeline import (
    VideoPipeline,
    PipelineState,
    FrameData,
)


# --- Fixtures ---


@pytest.fixture
def camera_config():
    """Camera config for testing with fast reconnect."""
    return CameraConfig(
        url="rtsp://fake-camera:554/stream",
        capture_fps=25,
        process_fps=10,
        reconnect_interval=0.1,  # Fast reconnect for tests
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

        # Simulate stream failure after N frames
        if self._fail_after and self._frame_count > self._fail_after:
            return False, None

        if self._frame_count > self._frames_to_produce:
            return False, None

        frame = make_fake_frame()
        # Add frame_count as pixel value for identification
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
        time.sleep(0.5)  # Give capture thread time to connect

        assert pipeline.state == PipelineState.RUNNING

        pipeline.stop()
        assert pipeline.state == PipelineState.STOPPED

    @patch("edge.core.video_pipeline.cv2.VideoCapture")
    def test_stop_terminates_threads(self, mock_cv2, pipeline):
        """Stop should terminate capture and distribute threads."""
        mock_cv2.side_effect = lambda *args, **kwargs: MockVideoCapture(frames_to_produce=5000)

        pipeline.start()
        time.sleep(0.3)
        pipeline.stop()

        # Threads should have terminated
        assert not pipeline._capture_thread.is_alive()
        assert not pipeline._distribute_thread.is_alive()

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
        """Should register a callback."""
        def my_callback(frame, frame_id, timestamp):
            pass

        pipeline.register_callback(my_callback)
        assert my_callback in pipeline._callbacks

    def test_unregister_callback(self, pipeline):
        """Should unregister a callback."""
        def my_callback(frame, frame_id, timestamp):
            pass

        pipeline.register_callback(my_callback)
        pipeline.unregister_callback(my_callback)
        assert my_callback not in pipeline._callbacks

    def test_duplicate_register_ignored(self, pipeline):
        """Registering same callback twice should not duplicate."""
        def my_callback(frame, frame_id, timestamp):
            pass

        pipeline.register_callback(my_callback)
        pipeline.register_callback(my_callback)
        assert pipeline._callbacks.count(my_callback) == 1

    @patch("edge.core.video_pipeline.cv2.VideoCapture")
    def test_callback_receives_frames(self, mock_cv2, pipeline):
        """Registered callback should receive frames."""
        mock_cv2.side_effect = lambda *args, **kwargs: MockVideoCapture(frames_to_produce=500)

        received_frames = []

        def collector(frame, frame_id, timestamp):
            received_frames.append((frame_id, timestamp))

        pipeline.register_callback(collector)
        pipeline.start()

        # Wait for some frames to be distributed
        time.sleep(0.8)
        pipeline.stop()

        assert len(received_frames) > 0
        # Frame IDs should be positive integers
        assert all(fid > 0 for fid, _ in received_frames)
        # Timestamps should be reasonable
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

        pipeline.register_callback(bad_callback)
        pipeline.register_callback(good_callback)
        pipeline.start()

        time.sleep(0.8)
        pipeline.stop()

        # Good callback should still receive frames despite bad one crashing
        assert call_count["good"] > 0

    @patch("edge.core.video_pipeline.cv2.VideoCapture")
    def test_multiple_callbacks_all_invoked(self, mock_cv2, pipeline):
        """All registered callbacks should receive each frame."""
        mock_cv2.side_effect = lambda *args, **kwargs: MockVideoCapture(frames_to_produce=500)

        counts = {"a": 0, "b": 0, "c": 0}

        def callback_a(frame, frame_id, timestamp):
            counts["a"] += 1

        def callback_b(frame, frame_id, timestamp):
            counts["b"] += 1

        def callback_c(frame, frame_id, timestamp):
            counts["c"] += 1

        pipeline.register_callback(callback_a)
        pipeline.register_callback(callback_b)
        pipeline.register_callback(callback_c)

        pipeline.start()
        time.sleep(0.8)
        pipeline.stop()

        # All callbacks should have received frames
        assert counts["a"] > 0
        assert counts["b"] > 0
        assert counts["c"] > 0
        # All should receive same number of frames
        assert counts["a"] == counts["b"] == counts["c"]


# --- Test: FPS Throttling ---


class TestFPSThrottling:
    """Tests for FPS limiting between capture and distribution."""

    @patch("edge.core.video_pipeline.cv2.VideoCapture")
    def test_distribute_fps_lower_than_capture(self, mock_cv2):
        """Distribution rate should be <= configured process_fps."""
        config = CameraConfig(
            url="rtsp://fake/stream",
            capture_fps=25,
            process_fps=5,  # Should distribute ~5 fps
            reconnect_interval=0.1,
            max_reconnect_attempts=0,
        )
        mock_cv2.side_effect = lambda *args, **kwargs: MockVideoCapture(frames_to_produce=5000)

        pipeline = VideoPipeline(config)
        received = []

        def collector(frame, frame_id, timestamp):
            received.append(time.time())

        pipeline.register_callback(collector)
        pipeline.start()

        time.sleep(2.0)  # Run for 2 seconds
        pipeline.stop()

        # At 5 fps for 2s, we expect ~10 frames (accounting for startup)
        # Allow reasonable tolerance
        assert 4 <= len(received) <= 15, f"Expected ~10 frames at 5fps, got {len(received)}"

    @patch("edge.core.video_pipeline.cv2.VideoCapture")
    def test_stats_track_capture_and_distribute(self, mock_cv2, pipeline):
        """Stats should track both capture and distributed frame counts."""
        mock_cv2.side_effect = lambda *args, **kwargs: MockVideoCapture(frames_to_produce=5000)

        pipeline.start()
        time.sleep(1.0)
        pipeline.stop()

        stats = pipeline.stats
        # Captured should be > distributed (due to throttling)
        assert stats.frames_captured > 0
        assert stats.frames_distributed > 0
        assert stats.frames_captured >= stats.frames_distributed


# --- Test: Ring Buffer ---


class TestRingBuffer:
    """Tests for the bounded ring buffer behavior."""

    def test_buffer_max_size(self, pipeline):
        """Buffer should be bounded at maxlen=30."""
        assert pipeline._buffer.maxlen == 30

    def test_buffer_drops_oldest_when_full(self, pipeline):
        """When buffer is full, oldest frames should be dropped."""
        # Manually fill buffer beyond capacity
        for i in range(40):
            frame_data = FrameData(
                frame=make_fake_frame(),
                frame_id=i + 1,
                timestamp=time.time(),
            )
            pipeline._buffer.append(frame_data)

        # Buffer should only contain last 30 frames
        assert len(pipeline._buffer) == 30
        # Oldest frame should be #11 (first 10 were dropped)
        assert pipeline._buffer[0].frame_id == 11


# --- Test: Reconnection ---


class TestReconnection:
    """Tests for auto-reconnect behavior on stream failure."""

    @patch("edge.core.video_pipeline.cv2.VideoCapture")
    def test_reconnect_on_stream_failure(self, mock_cv2, pipeline):
        """Pipeline should attempt reconnection when stream drops."""
        # First connection works for 5 frames, then fails
        # Second connection works indefinitely
        call_count = {"n": 0}

        def create_capture(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return MockVideoCapture(fail_after=5)
            return MockVideoCapture(frames_to_produce=5000)

        mock_cv2.side_effect = create_capture

        pipeline.start()
        time.sleep(1.0)  # Give time for failure + reconnect + new frames
        pipeline.stop()

        # Should have reconnected
        assert pipeline.stats.reconnect_count >= 1
        # Should have captured frames from both connections
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

        time.sleep(0.5)  # Wait for all attempts to fail

        # After max attempts, capture loop exits. State should be ERROR.
        # Give extra time for thread to finish
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
            max_reconnect_attempts=0,  # Infinite
        )

        attempt_count = {"n": 0}

        def create_capture(*args, **kwargs):
            attempt_count["n"] += 1
            # Succeed after 4 attempts
            if attempt_count["n"] >= 4:
                return MockVideoCapture(frames_to_produce=500)
            return FailingVideoCapture()

        mock_cv2.side_effect = create_capture

        pipeline = VideoPipeline(config)
        pipeline.start()

        time.sleep(1.0)
        pipeline.stop()

        # Should have eventually connected
        assert attempt_count["n"] >= 4
        assert pipeline.stats.frames_captured > 0


# --- Test: Thread Safety ---


class TestThreadSafety:
    """Tests for concurrent access patterns."""

    @patch("edge.core.video_pipeline.cv2.VideoCapture")
    def test_register_during_distribution(self, mock_cv2, pipeline):
        """Registering callbacks while pipeline is running should be safe."""
        mock_cv2.side_effect = lambda *args, **kwargs: MockVideoCapture(frames_to_produce=5000)

        counts = {"late": 0}

        pipeline.start()
        time.sleep(0.5)

        # Register callback while running
        def late_callback(frame, frame_id, timestamp):
            counts["late"] += 1

        pipeline.register_callback(late_callback)
        time.sleep(0.8)
        pipeline.stop()

        assert counts["late"] > 0

    @patch("edge.core.video_pipeline.cv2.VideoCapture")
    def test_unregister_during_distribution(self, mock_cv2, pipeline):
        """Unregistering callbacks while pipeline is running should be safe."""
        mock_cv2.side_effect = lambda *args, **kwargs: MockVideoCapture(frames_to_produce=5000)

        counts = {"total": 0}

        def counting_callback(frame, frame_id, timestamp):
            counts["total"] += 1

        pipeline.register_callback(counting_callback)
        pipeline.start()
        time.sleep(0.5)

        # Unregister while running
        pipeline.unregister_callback(counting_callback)
        count_at_unregister = counts["total"]

        time.sleep(0.5)
        pipeline.stop()

        # Should not have received many more frames after unregister
        # Allow a small margin for race condition
        assert counts["total"] - count_at_unregister <= 2
