"""ROS graph smoke test; run in CI after building AIVA's ROS packages."""
import json
import os
import threading
import time
from unittest.mock import Mock

import pytest

if os.environ.get("AIVA_TEST_WITH_ROS") != "1":
    pytest.skip("requires the ROS-enabled CI job", allow_module_level=True)

import rclpy
from telepathy_receptor import node as chat_node
from telepathy_receptor.node import TelepathyReceptor
from cryptic_signal_core.msg import TelepathicSignal
from std_msgs.msg import String


class FakeStreamingResponse:
    status_code = 200

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def raise_for_status(self):
        return None

    def iter_lines(self):
        yield 'data: {"choices":[{"delta":{"content":"Grace replied"}}]}'
        yield "data: [DONE]"


def spin_until(node, predicate, timeout=4):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
        if predicate():
            return
    pytest.fail("ROS topic event did not arrive before timeout")


def test_chat_input_topics_and_response_topic_on_ros_graph(tmp_path, monkeypatch):
    monkeypatch.setenv("AGI_HOME", str(tmp_path))
    monkeypatch.setenv("AIVA_MEMORY_PATH", str(tmp_path / "memory.sqlite3"))
    # Keep local AIVA processes and parallel ROS tests out of this graph.
    monkeypatch.setenv("ROS_DOMAIN_ID", str(50 + os.getpid() % 180))
    monkeypatch.setattr(chat_node.httpx, "stream", Mock(return_value=FakeStreamingResponse()))
    rclpy.init()
    node = TelepathyReceptor()
    received = []
    response_events = []
    submitted = threading.Event()

    def capture_prompt(function, text, identifier):
        received.append((function, text, identifier))
        submitted.set()

    node._pool.submit = capture_prompt
    def capture_response(message):
        wire_event = json.loads(message.data)
        encoded = wire_event.pop("content_signal", None)
        if encoded is not None:
            wire_event["content"] = node._cryptic.decode(
                bytes(encoded["signal"]), int(encoded["pulse"]), encoded["scheme"]
            )
        response_events.append(wire_event)

    node.create_subscription(String, "/telepathic/response", capture_response, 10)
    signal_publisher = node.create_publisher(TelepathicSignal, "/telepathic/stimulus", 10)

    try:
        deadline = time.monotonic() + 4
        while time.monotonic() < deadline and node.count_subscribers("/telepathic/stimulus") == 0:
            rclpy.spin_once(node, timeout_sec=0.05)
        assert node.count_subscribers("/telepathic/stimulus") > 0
        assert node.count_subscribers("/aiva/chat_input") == 0
        signal = TelepathicSignal()
        signal.channel = "grace"
        signal.scheme = "ternary_utf8_v1"
        encoded, signal.pulse = node._cryptic.encode("radio hello")
        signal.signal = list(encoded)
        signal.identifier = "radio-2"
        signal_publisher.publish(signal)
        spin_until(node, submitted.is_set)
        assert received[-1][1:] == ("radio hello", "radio-2")

        node._pool.submit = Mock()
        monkeypatch.setattr(chat_node.httpx, "stream", Mock(return_value=FakeStreamingResponse()))
        node._answer("hello", "response-3")
        spin_until(node, lambda: any(event["type"] == "done" for event in response_events))
        assert [event["type"] for event in response_events] == ["start", "delta", "done"]
        assert response_events[-1]["radio_id"] == "response-3"
        assert node._memory.recent() == [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "Grace replied"},
        ]
    finally:
        node.destroy_node()
        rclpy.shutdown()
