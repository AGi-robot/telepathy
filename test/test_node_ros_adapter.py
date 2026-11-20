import json
from unittest.mock import Mock


from telepathy_receptor import node as chat_node
from telepathy_receptor.memory import GraceMemory
from aiva_cryptic_signal_core import CrypticSignalCore


class FakeStreamingResponse:
    status_code = 200

    def __init__(self, lines):
        self.lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def raise_for_status(self):
        return None

    def iter_lines(self):
        return iter(self.lines)


def make_node(tmp_path):
    node = chat_node.TelepathyReceptor.__new__(chat_node.TelepathyReceptor)
    node._system_prompt = "You are Grace."
    node._base_url = "http://llm.test/v1"
    node._model = "test-model"
    node._api_key = "test-token"
    node._memory = GraceMemory(str(tmp_path / "grace.sqlite3"))
    node._response_pub = Mock()
    node._publish = chat_node.TelepathyReceptor._publish.__get__(node)
    node.get_logger = Mock(return_value=Mock())
    return node


def sse(content):
    return "data: " + json.dumps({"choices": [{"delta": {"content": content}}]})


def test_streamed_llm_reply_is_published_and_persisted(tmp_path, monkeypatch):
    node = make_node(tmp_path)
    response = FakeStreamingResponse([sse("Hello "), sse("from Grace"), "data: [DONE]"])
    request = Mock(return_value=response)
    monkeypatch.setattr(chat_node.httpx, "stream", request)

    node._answer("Hello Grace", "request-42")

    request.assert_called_once()
    args, kwargs = request.call_args
    assert args[:2] == ("POST", "http://llm.test/v1/chat/completions")
    assert kwargs["headers"] == {"Authorization": "Bearer test-token"}
    assert kwargs["json"]["stream"] is True
    messages = kwargs["json"]["messages"]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"].startswith("You are Grace.\n\nCurrent local date and time:")
    assert messages[1:] == [{"role": "user", "content": "Hello Grace"}]
    assert [json.loads(call.args[0].data)["type"] for call in node._response_pub.publish.call_args_list] == [
        "start", "delta", "delta", "done",
    ]
    assert node._memory.recent() == [
        {"role": "user", "content": "Hello Grace"},
        {"role": "assistant", "content": "Hello from Grace"},
    ]
    assert json.loads(node._response_pub.publish.call_args_list[-1].args[0].data)["radio_id"] == "request-42"
    assert all(json.loads(call.args[0].data)["radio_id"] == "request-42"
               for call in node._response_pub.publish.call_args_list)


def test_empty_llm_stream_reports_error_and_does_not_save_exchange(tmp_path, monkeypatch):
    node = make_node(tmp_path)
    monkeypatch.setattr(chat_node.httpx, "stream", Mock(return_value=FakeStreamingResponse(["data: [DONE]"])))

    node._answer("Hello", "request-43")

    event = json.loads(node._response_pub.publish.call_args_list[-1].args[0].data)
    assert event["type"] == "error"
    assert event["radio_id"] == "request-43"
    assert node._memory.recent() == []


def test_webui_input_supports_json_plain_text_and_web_search(tmp_path):
    node = make_node(tmp_path)
    node._submit_prompt = Mock()

    node._on_webui_input(type("Message", (), {"data": '{"text":"hello","web_search":true,"identifier":"web-1"}'})())
    node._on_webui_input(type("Message", (), {"data": "plain text"})())
    node._on_webui_input(type("Message", (), {"data": "[]"})())

    assert node._submit_prompt.call_args_list[0].args == ("/web hello", "web-1")
    assert node._submit_prompt.call_args_list[1].args == ("plain text", "")
    assert node._submit_prompt.call_count == 2


def test_telepathic_signal_accepts_chat_channels_and_ignores_other_channels(tmp_path):
    node = make_node(tmp_path)
    node._cryptic = CrypticSignalCore()
    node._submit_prompt = Mock()
    encoded, pulse = node._cryptic.encode("hello")
    signal = type("Signal", (), {
        "channel": "grace", "scheme": "ternary_utf8_v1", "signal": list(encoded),
        "pulse": pulse, "identifier": "radio-7",
    })()
    node._on_signal(signal)
    assert node._submit_prompt.call_args.args == ("hello", "radio-7")

    signal.channel = "fleet-control"
    node._on_signal(signal)
    assert node._submit_prompt.call_count == 1

    signal.channel = "aiva"
    signal.pulse = -1
    node._on_signal(signal)
    assert node._submit_prompt.call_count == 1


def test_receptor_decodes_structured_signal_with_search_and_identifier(tmp_path):
    node = make_node(tmp_path)
    node._cryptic = CrypticSignalCore()
    node._submit_prompt = Mock()
    encoded, pulse = node._cryptic.encode(json.dumps({'text': 'hello — 你好', 'web_search': True}))
    signal = type('Signal', (), dict(signal=encoded, pulse=pulse, scheme='ternary_utf8_v1',
                                    channel='grace', identifier='encoded-1'))()
    node._on_signal(signal)
    node._submit_prompt.assert_called_once_with('/web hello — 你好', 'encoded-1')
