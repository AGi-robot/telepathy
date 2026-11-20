"""Exercise the real subprocess protocol without CUDA or downloaded weights."""
import json
import os
import sys
import threading
import time

import pytest

from telepathy_receptor.moss import MossClient, MossPaintError, PNG_SIGNATURE, paint_size


@pytest.fixture
def engine(tmp_path, monkeypatch):
    server = tmp_path / 'moss-server'
    server.write_text(f'#!{sys.executable}\n' + r'''
import json
import os
from pathlib import Path
import sys
import time

loaded = False
for line in sys.stdin:
    request = json.loads(line)
    method, identifier = request['method'], request['id']
    params = request.get('params', {})
    with open(os.environ['FAKE_MOSS_LOG'], 'a') as log:
        log.write(json.dumps({'pid': os.getpid(), **request}) + '\n')
    mode = os.getenv('FAKE_MOSS_MODE', '')
    def reply(**fields):
        print(json.dumps({'id': identifier, **fields}), flush=True)
    if method == 'hello':
        reply(result={'protocol_version': 2})
    elif method == 'load':
        loaded = True
        reply(result={'model': params['model']})
    elif method == 'generate':
        if not loaded:
            reply(error={'code': 'no_engine', 'message': 'not loaded'})
            continue
        reply(result={'accepted': True})
        if mode == 'timeout':
            time.sleep(30)
        if mode == 'error':
            reply(event='done', status='error', error={'code': 'cuda', 'message': 'out of memory'})
        elif mode == 'cancelled':
            reply(event='done', status='cancelled')
        else:
            content = b'not an image' if mode == 'invalid_png' else b'\x89PNG\r\n\x1a\nFAKE'
            Path(params['output']).write_bytes(content)
            # Multiple events may arrive in a single OS read.
            print(json.dumps({'id': identifier, 'event': 'progress', 'step': 1}) + '\n' +
                  json.dumps({'id': identifier, 'event': 'done', 'status': 'ok'}), flush=True)
    elif method == 'release':
        # Protocol 2 retains the engine; generate automatically restores GPU weights.
        if mode == 'release_error':
            reply(error={'code': 'cuda', 'message': 'cannot release'})
        else:
            reply(result={'vram_released_bytes': 100})
''')
    server.chmod(0o755)
    log = tmp_path / 'requests.jsonl'
    monkeypatch.setenv('MOSS_PAINT_SERVER_BIN', str(server))
    monkeypatch.setenv('MOSS_PAINT_WEIGHTS_ROOT', str(tmp_path / 'weights'))
    monkeypatch.setenv('FAKE_MOSS_LOG', str(log))
    monkeypatch.setenv('MOSS_PAINT_DEVICE', '0')
    client = MossClient()
    for path in (client.model_dir / 'model_index.json', client.dit_file,
                 client.text_encoder_file, client.vae_file,
                 client.distill_dir / 'lora.safetensors',
                 client.distill_dir / 'time_embedding.safetensors'):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    yield client, log
    client.close()


def requests(log):
    return [json.loads(line) for line in log.read_text().splitlines()]


def test_two_paintings_reuse_engine_after_soft_release(engine):
    client, log = engine
    assert client.available()
    assert client.generate('a red apple', 512, 512).startswith(PNG_SIGNATURE)
    assert client.generate('a blue flower', 512, 384).startswith(PNG_SIGNATURE)
    calls = requests(log)
    assert [call['method'] for call in calls] == [
        'hello', 'load', 'generate', 'release', 'generate', 'release']
    assert len({call['pid'] for call in calls}) == 1
    assert calls[1]['params']['vram_budget_bytes'] == 4096 * 1024 * 1024
    assert calls[1]['params']['device'] == 0
    assert not any(os.path.exists(call['params']['output'])
                   for call in calls if call['method'] == 'generate')


@pytest.mark.parametrize(('mode', 'message'), [
    ('error', 'out of memory'), ('cancelled', 'cancelled'), ('invalid_png', 'PNG'),
])
def test_generation_errors_stop_the_child(engine, monkeypatch, mode, message):
    client, log = engine
    monkeypatch.setenv('FAKE_MOSS_MODE', mode)
    with pytest.raises(MossPaintError, match=message):
        client.generate('a flower', 256, 256)
    assert client._process is None
    with pytest.raises(ProcessLookupError):
        os.kill(requests(log)[0]['pid'], 0)


def test_timeout_stops_gpu_work_and_restarts_cleanly(engine, monkeypatch):
    client, log = engine
    client.timeout = 1
    monkeypatch.setenv('FAKE_MOSS_MODE', 'timeout')
    started = time.monotonic()
    with pytest.raises(MossPaintError, match='did not answer'):
        client.generate('an apple', 256, 256)
    assert time.monotonic() - started < 5
    old_pid = requests(log)[0]['pid']
    with pytest.raises(ProcessLookupError):
        os.kill(old_pid, 0)
    monkeypatch.delenv('FAKE_MOSS_MODE')
    assert client.generate('a flower', 256, 256).startswith(PNG_SIGNATURE)
    assert requests(log)[-1]['pid'] != old_pid


def test_close_interrupts_a_running_painting(engine, monkeypatch):
    client, log = engine
    monkeypatch.setenv('FAKE_MOSS_MODE', 'timeout')
    errors = []
    def generate():
        try:
            client.generate('an apple', 256, 256)
        except MossPaintError as error:
            errors.append(error)
    worker = threading.Thread(target=generate)
    worker.start()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if log.exists() and any(call['method'] == 'generate' for call in requests(log)):
            break
        time.sleep(0.01)
    client.close()
    worker.join(timeout=5)
    assert not worker.is_alive()
    assert errors and client._process is None
    with pytest.raises(MossPaintError, match='shutting down'):
        client.generate('another apple', 256, 256)


def test_failed_release_cannot_leave_gpu_memory_allocated(engine, monkeypatch):
    client, log = engine
    monkeypatch.setenv('FAKE_MOSS_MODE', 'release_error')
    assert client.generate('a flower', 256, 256).startswith(PNG_SIGNATURE)
    assert client._process is None
    with pytest.raises(ProcessLookupError):
        os.kill(requests(log)[0]['pid'], 0)


def test_missing_or_non_executable_engine_is_unavailable(engine):
    client, _ = engine
    client.binary.chmod(0o644)
    assert not client.available()


@pytest.mark.parametrize('width,height', [(float('nan'), 512), (512, float('inf')), (-2, 1)])
def test_invalid_sizes_use_a_safe_default(width, height):
    assert paint_size(width, height) == (512, 512)
