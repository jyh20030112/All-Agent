from __future__ import annotations

import subprocess
import sys
import tempfile
import time
import unittest
from collections.abc import Callable
from pathlib import Path

from streamlit.testing.v1 import AppTest

from ejagent.contracts import AssistantMessage, ControlStatus, RunStatus
from ejagent.harness import HarnessStatus
from examples.streamlit_runtime import (
    RuntimeConfig,
    RuntimeSnapshot,
    StreamlitRuntimeController,
)


def _wait_for(
    controller: StreamlitRuntimeController,
    predicate: Callable[[RuntimeSnapshot], bool],
    *,
    timeout: float = 3.0,
) -> RuntimeSnapshot:
    deadline = time.monotonic() + timeout
    snapshot = controller.snapshot()
    while not predicate(snapshot):
        if time.monotonic() >= deadline:
            raise AssertionError(f"runtime condition timed out: {snapshot!r}")
        time.sleep(0.01)
        snapshot = controller.snapshot()
    return snapshot


class StreamlitRuntimeControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.store_root = Path(temporary.name)
        self.controllers: list[StreamlitRuntimeController] = []

    def tearDown(self) -> None:
        for controller in reversed(self.controllers):
            controller.close()

    def controller(self, *, delay: float = 0.05) -> StreamlitRuntimeController:
        controller = StreamlitRuntimeController(
            RuntimeConfig(
                agent_id="streamlit-test",
                store_root=self.store_root,
                probe_delay_seconds=delay,
            )
        )
        self.controllers.append(controller)
        return controller

    def test_parallel_run_is_committed_and_restored_from_jsonl(self) -> None:
        controller = self.controller()

        controller.start_run("validate parallel execution")
        snapshot = _wait_for(controller, lambda item: item.revision == 1)

        self.assertEqual(snapshot.status, HarnessStatus.READY)
        self.assertEqual(len(snapshot.audits), 1)
        self.assertEqual(snapshot.audits[0].result.status, RunStatus.COMPLETED)
        self.assertEqual(len(snapshot.probes), 2)
        first, second = snapshot.probes
        assert first.finished_at is not None
        assert second.finished_at is not None
        self.assertLess(
            max(first.started_at, second.started_at),
            min(first.finished_at, second.finished_at),
        )
        committed_messages = snapshot.messages

        controller.close()
        restored = self.controller()
        restored_snapshot = restored.snapshot()

        self.assertEqual(restored_snapshot.revision, 1)
        self.assertEqual(restored_snapshot.messages, committed_messages)
        self.assertEqual(len(restored_snapshot.audits), 1)

    def test_cancel_stops_tools_without_advancing_revision(self) -> None:
        controller = self.controller(delay=0.5)
        controller.start_run("cancel this validation")
        _wait_for(
            controller,
            lambda item: sum(probe.finished_at is None for probe in item.probes) == 2,
        )

        self.assertTrue(controller.cancel("test cancellation"))
        snapshot = _wait_for(
            controller,
            lambda item: item.status is HarnessStatus.READY and len(item.audits) == 1,
        )

        self.assertEqual(snapshot.revision, 0)
        self.assertIsNone(snapshot.last_result)
        self.assertIsNotNone(snapshot.latest_outcome)
        assert snapshot.latest_outcome is not None
        self.assertEqual(snapshot.latest_outcome.result.status, RunStatus.CANCELLED)
        self.assertTrue(all(probe.cancelled for probe in snapshot.probes))

    def test_steering_and_follow_up_are_admitted_during_active_run(self) -> None:
        controller = self.controller(delay=0.1)
        controller.start_run("initial validation")
        _wait_for(
            controller,
            lambda item: sum(probe.finished_at is None for probe in item.probes) == 2,
        )

        steering = controller.steer("mention the accepted steering")
        follow_up = controller.follow_up("run the queued validation")

        self.assertEqual(steering.status, ControlStatus.ACCEPTED)
        self.assertEqual(follow_up.status, ControlStatus.ACCEPTED)
        snapshot = _wait_for(
            controller,
            lambda item: (
                item.revision == 2
                and item.status is HarnessStatus.READY
                and item.pending_follow_ups == 0
            ),
        )
        self.assertEqual(len(snapshot.audits), 2)
        self.assertEqual(len(snapshot.probes), 4)
        self.assertTrue(
            any(
                isinstance(message, AssistantMessage)
                and message.content is not None
                and "Applied steering" in message.content
                for message in snapshot.messages
            )
        )


class StreamlitAppSmokeTests(unittest.TestCase):
    def test_app_entrypoint_imports_from_outside_repository(self) -> None:
        app_path = Path(__file__).parents[1] / "examples" / "streamlit_app.py"
        with tempfile.TemporaryDirectory() as working_directory:
            completed = subprocess.run(
                (sys.executable, str(app_path)),
                cwd=working_directory,
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_app_renders_before_runtime_is_started(self) -> None:
        app_path = Path(__file__).parents[1] / "examples" / "streamlit_app.py"
        app = AppTest.from_file(app_path).run()

        self.assertFalse(app.exception)
        self.assertEqual(app.title[0].value, "EJAgent Runtime Validation")
        self.assertTrue(any("Start the runtime" in item.value for item in app.info))

    def test_app_starts_and_stops_demo_runtime(self) -> None:
        app_path = Path(__file__).parents[1] / "examples" / "streamlit_app.py"
        with tempfile.TemporaryDirectory() as root:
            app = AppTest.from_file(app_path, default_timeout=5).run()
            app.text_input[1].set_value(root).run()
            app.button[0].click().run()
            controller = app.session_state["ejagent_runtime_controller"]
            try:
                self.assertFalse(app.exception)
                self.assertTrue(
                    any(
                        metric.label == "Harness" and metric.value == "ready"
                        for metric in app.metric
                    )
                )
                stop = next(button for button in app.button if button.label == "Stop")
                stop.click().run()
                self.assertFalse(app.exception)
            finally:
                controller.close()
