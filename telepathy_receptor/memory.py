"""Wire Mind's conversation store into AIVA's chat and Canvas context."""
from __future__ import annotations

import json
import re

from memory.conversation import ConversationStore


class AivaMemory(ConversationStore):
    """Apply AIVA's Canvas policies to the shared SQLite conversation store."""

    def __init__(self, database_path: str, max_turns: int = 8) -> None:
        super().__init__(database_path, max_turns, default_title="AIVA chat")
        self._initialize_drawing_studies()

    def _initialize_drawing_studies(self) -> None:
        with self._connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS drawing_studies ("
                       "id TEXT PRIMARY KEY, prompt TEXT NOT NULL, mode TEXT NOT NULL, "
                       "summary TEXT NOT NULL, actions TEXT NOT NULL, rating TEXT, notes TEXT, "
                       "created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)")

    def recent(self, conversation_id: str | None = None) -> list[dict[str, str]]:
        history = super().recent(conversation_id)
        for turn in history:
            role, content = turn["role"], turn["content"]
            # Older Canvas versions stored the full drawing-action JSON as an
            # assistant turn. Keep that legacy history useful without sending
            # a potentially huge tool payload back to the model.
            if role == "assistant" and content.startswith("Prepared ") and " canvas actions:" in content:
                content = "I made a drawing on the shared Canvas."
            if role == "assistant" and " canvas actions (playback not confirmed): " in content:
                content = content.split(" canvas actions (playback not confirmed): ", 1)[0] + " canvas actions (plan stored)."
            turn["content"] = content
        return history

    def save_study(self, identifier, prompt, mode, summary, actions):
        """Store the proposed drawing recipe; a human review is required before reuse."""
        with self._connect() as db:
            db.execute("INSERT OR IGNORE INTO drawing_studies (id,prompt,mode,summary,actions) VALUES (?,?,?,?,?)",
                       (identifier, prompt, mode, summary, json.dumps(actions, ensure_ascii=False)))

    def review_study(self, identifier, rating, notes=""):
        if rating not in ('keep', 'improve') or not isinstance(notes, str) or len(notes) > 1000:
            raise ValueError('Choose keep or improve, with at most 1,000 characters of feedback')
        with self._connect() as db:
            result = db.execute("UPDATE drawing_studies SET rating=?, notes=? WHERE id=?",
                                (rating, notes, identifier))
            if not result.rowcount:
                raise ValueError('Drawing study not found; wait for AIVA to finish first')

    def drawing_lessons(self, query):
        """Retrieve a few relevant, user-reviewed studies within a small context budget."""
        words = set(re.findall(r'\w{4,}', query.lower())) - {'draw', 'paint', 'please', 'with', 'something', 'picture'}
        with self._connect() as db:
            rows = db.execute("SELECT prompt,summary,actions,rating,notes FROM drawing_studies "
                              "WHERE rating IS NOT NULL ORDER BY rowid DESC LIMIT 100").fetchall()
        ranked = sorted(enumerate(rows), key=lambda pair: (
            -len(words & set(re.findall(r'\w{4,}', pair[1][0].lower()))), pair[0]))
        lessons = []
        for _, (prompt, summary, actions, rating, notes) in ranked[:3]:
            lesson = {'subject': prompt[:200], 'review': rating, 'user_feedback': (notes or '')[:400],
                      'summary': summary[:600]}
            # A small recipe helps repeat successful contours without resending dense drawings.
            if rating == 'keep':
                recipe = []
                for action in json.loads(actions):
                    if action.get('kind') in ('curve', 'path', 'polygon', 'line', 'rectangle', 'ellipse', 'text', 'letter') and len(json.dumps(action)) < 400:
                        recipe.append({k: v for k, v in action.items() if k != 'id'})
                    if len(recipe) == 3:
                        break
                lesson['example_actions'] = recipe
            lessons.append(lesson)
        while lessons and len(json.dumps(lessons, ensure_ascii=False)) > 6000:
            lessons.pop()
        return json.dumps(lessons, ensure_ascii=False)
