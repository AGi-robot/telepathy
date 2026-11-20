import sqlite3
import json
import tempfile
import unittest
from pathlib import Path

from telepathy_receptor.memory import GraceMemory


class GraceMemoryTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.database = Path(self.tempdir.name) / "profile" / "grace.sqlite3"

    def test_exchange_history_survives_reopening(self):
        memory = GraceMemory(str(self.database), max_turns=2)
        memory.add_exchange("Hello", "Hello back")
        memory.add_exchange("How are you?", "I am well")
        reopened = GraceMemory(str(self.database), max_turns=2)
        self.assertEqual(reopened.recent(), [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hello back"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I am well"},
        ])

    def test_recent_context_is_limited_without_deleting_old_history(self):
        memory = GraceMemory(str(self.database), max_turns=1)
        memory.add_exchange("first", "first reply")
        memory.add_exchange("second", "second reply")
        self.assertEqual(memory.recent(), [
            {"role": "user", "content": "second"},
            {"role": "assistant", "content": "second reply"},
        ])
        with sqlite3.connect(self.database) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM conversation_turns").fetchone()[0], 4)

    def test_canvas_geometry_is_shortened_in_context_but_preserved_in_storage(self):
        memory = GraceMemory(str(self.database))
        original = 'Here is your garden.\n\nPrepared 1 canvas actions (playback not confirmed): ' + '[{"points":' + '[1,2],' * 2000 + '[]}]'
        memory.add_exchange('Paint a garden', original)
        context = memory.recent()[-1]['content']
        self.assertIn('Here is your garden.', context)
        self.assertNotIn('points', context)
        self.assertLess(len(context), 150)
        with sqlite3.connect(self.database) as db:
            self.assertEqual(db.execute("SELECT content FROM conversation_turns WHERE role='assistant'").fetchone()[0], original)

    def test_reviewed_drawing_lessons_survive_reopening_and_prefer_matching_subjects(self):
        memory = GraceMemory(str(self.database))
        memory.save_study('flower', 'A garden flower', 'reference', 'Curved petals', [{'kind': 'ellipse', 'x': 20}])
        self.assertEqual(memory.drawing_lessons('flower'), '[]')
        memory.review_study('flower', 'keep', 'Petals look balanced')
        memory.save_study('car', 'A fast car', 'draw', 'Wheels', [])
        memory.review_study('car', 'improve', 'Fix perspective')
        reopened = GraceMemory(str(self.database))
        lessons = json.loads(reopened.drawing_lessons('flower'))
        self.assertEqual(lessons[0]['subject'], 'A garden flower')
        self.assertEqual(lessons[0]['user_feedback'], 'Petals look balanced')
        self.assertEqual(lessons[0]['example_actions'][0]['kind'], 'ellipse')
        with self.assertRaises(ValueError):
            reopened.review_study('unknown', 'keep')
        with self.assertRaises(ValueError):
            reopened.review_study('flower', 'automatic-success')

    def test_conversations_are_isolated_and_exchanges_share_an_id(self):
        memory = GraceMemory(str(self.database))
        memory.add_exchange("AIVA message", "AIVA reply")
        memory.add_exchange("Other message", "Other reply", conversation_id="other")
        self.assertEqual(memory.recent(), [
            {"role": "user", "content": "AIVA message"},
            {"role": "assistant", "content": "AIVA reply"},
        ])
        self.assertEqual(memory.recent("other"), [
            {"role": "user", "content": "Other message"},
            {"role": "assistant", "content": "Other reply"},
        ])
        with sqlite3.connect(self.database) as db:
            ids = db.execute(
                "SELECT DISTINCT exchange_id FROM conversation_turns WHERE conversation_id='default'"
            ).fetchall()
            self.assertEqual(len(ids), 1)


if __name__ == "__main__":
    unittest.main()
