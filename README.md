# Telepathy — Grace entity receptor

The ROS package `telepathy_receptor` receives `cryptic_signal_core/msg/TelepathicSignal`
on `/telepathic/stimulus`, decodes it with Cypher through Grace's scheme registry,
and invokes the Interactive conversation engine. It publishes JSON stream events
as `std_msgs/String` on `/telepathic/response`.

The external WebUI lives in [Console](https://github.com/AGi-robot/console).
Console encodes the prompt before publishing; Grace decodes it here. The current
`ternary_utf8_v1` codec is reversible encoding, not encryption.

Decoded input accepts plain text or a JSON object containing `text`, `identifier`,
`web_search`, and optional Canvas metadata. The old `/aiva/chat_input` subscription
is removed. Canvas snapshots retain their separate `/aiva/canvas_snapshot` topic.

The receptor currently also hosts the migrated Grace memory adapter and Canvas
orchestration. Interactive and Memory remain imported libraries, not ROS nodes.
Recent context defaults to eight exchanges; completed exchanges persist in SQLite.

## Runtime setup

Check out Telepathy beside Shell and link `shell/src/telepathy_receptor` to
`../../telepathy`. Shell provides `aiva_cryptic_signal_core`; Cypher provides the
message and codec. Install Interactive, Memory, Canvas and Imaging in the runtime
venv. Build with colcon and run `ros2 run telepathy_receptor telepathy_receptor`.

The migrated entity source retains its GPL-3.0-only license (`LICENSE.inherited`);
the repository's existing AGPL license is retained for its prior history.
