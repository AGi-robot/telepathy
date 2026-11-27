# Telepathy — AIVA entity receptor

The ROS package `telepathy_receptor` receives `cryptic_signal_core/msg/TelepathicSignal`
on `/telepathic/stimulus`, decodes it with Cypher through AIVA's scheme registry,
and invokes the Interactive conversation engine. It publishes JSON stream events
on `/telepathic/response` as `std_msgs/String`. Text content is Cypher-encoded
inside each event; Console decodes it after rosbridge delivers the event.

The external WebUI lives in [Console](https://github.com/Project-AGi/console).
Console encodes the prompt before publishing; AIVA decodes it here. AIVA
encodes each response event's text content before publishing; Console restores
that content after rosbridge. Event types, IDs, and other structured fields
remain in the JSON envelope. The current `ternary_utf8_v1` codec is reversible
encoding, not encryption.

Decoded input accepts plain text or a JSON object containing `text`, `identifier`,
`web_search`, and optional Canvas metadata. The old `/aiva/chat_input` subscription
is removed. Canvas snapshots retain their separate `/aiva/canvas_snapshot` topic.

The receptor currently also hosts the migrated AIVA memory adapter and Canvas
orchestration. Interactive and Memory remain imported libraries, not ROS nodes.
Recent context defaults to eight exchanges; completed exchanges persist in SQLite.

## Runtime setup

Check out Telepathy beside Shell and link `shell/src/telepathy_receptor` to
`../../telepathy`. Shell provides `aiva_cryptic_signal_core`; Cypher provides the
message and codec. Install Interactive, Memory, Canvas and Imaging in the runtime
venv. Build with colcon and run `ros2 run telepathy_receptor telepathy_receptor`.

The migrated entity source retains its GPL-3.0-only license (`LICENSE.inherited`);
the repository's existing AGPL license is retained for its prior history.


Conversation storage defaults are read from [`config/storage.yaml`](config/storage.yaml), retaining the established database filename. `AIVA_MEMORY_PATH` and the ROS `memory_path` parameter can still override the full path. Technical API/class names use AIVA; the companion's personal identity stays in `persona/identity.yaml` and `persona/SOUL.md`.
