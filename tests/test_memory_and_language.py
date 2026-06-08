import unittest

from app.routers.ask import _get_answer_language
from app.services.user_profile_service import ExtractedProfile, UserFact, UserProfileService

from app.services.prompt_service import build_conversation_only_messages, build_off_topic_messages

class AnswerLanguageTests(unittest.TestCase):
    def test_unmarked_english_question_defaults_to_english(self):
        self.assertEqual(_get_answer_language("Tell me apple pie recipes."), "English")

    def test_crop_rotation_table_prompt_is_english(self):
        self.assertEqual(
            _get_answer_language("Give me some crop rotation techniques in a tabular format."),
            "English",
        )

    def test_confident_languages_are_preserved(self):
        self.assertEqual(_get_answer_language("Hoe kan ik mijn boerderij helpen?"), "Dutch")
        self.assertEqual(_get_answer_language("Como posso ajudar minha fazenda?"), "Portuguese")


    def test_explicit_language_is_used_in_non_retrieval_modes(self):
        for builder in (build_conversation_only_messages, build_off_topic_messages):
            messages = builder("Tell me a joke", answer_language="English")
            self.assertIn("entire answer in English", messages[0]["content"])


class PerTurnLanguageSwitchTests(unittest.IsolatedAsyncioTestCase):
    async def test_each_turn_uses_only_its_own_language(self):
        from app.routers.ask import _resolve_answer_language

        questions = [
            "How can I improve my farm?",
            "What crop should I plant?",
            "Please give me some irrigation advice.",
            "Hoe kan ik mijn boerderij verbeteren?",
            "Hoe help ik mijn gewassen?",
            "Comment améliorer ma ferme?",
        ]
        self.assertEqual(
            [await _resolve_answer_language(question) for question in questions],
            ["English", "English", "English", "Dutch", "Dutch", "French"],
        )

    async def test_ambiguous_turn_classifier_receives_no_history(self):
        from unittest.mock import AsyncMock, patch
        from app.routers.ask import _resolve_answer_language

        with patch(
            "app.routers.ask.generate_once",
            new=AsyncMock(return_value='{"language_code":"fr"}'),
        ) as generate:
            language = await _resolve_answer_language("Bonjour")

        self.assertEqual(language, "French")
        prompt = generate.await_args.args[0]
        self.assertIn("User message:\nBonjour", prompt)
        self.assertNotIn("Previous Conversation", prompt)
class MemoryCandidateTests(unittest.TestCase):
    def test_promotes_explicit_stable_facts_to_visible_memories(self):
        extracted = ExtractedProfile(
            facts=[
                UserFact(category="location", text="User farms in Crete", confidence=0.95),
                UserFact(category="crop", text="User grows olives", confidence=0.9),
                UserFact(category="issue", text="User has aphids today", confidence=0.99),
                UserFact(category="tool", text="User uses drip irrigation", confidence=0.7),
            ],
        )

        self.assertEqual(
            UserProfileService._memory_candidates(extracted),
            ["Farms in Crete", "Grows olives"],
        )

    def test_keeps_llm_notes_and_persistent_communication_style(self):
        extracted = ExtractedProfile(
            memory_notes=["Writing a thesis on soil carbon"],
            communication_style="concise",
        )

        self.assertEqual(
            UserProfileService._memory_candidates(extracted),
            ["Writing a thesis on soil carbon", "Prefers concise answers"],
        )

    def test_backfills_existing_stable_facts(self):
        extracted = ExtractedProfile()
        existing = [
            UserFact(category="farm_type", text="User operates a dairy farm", confidence=0.95),
            UserFact(category="topic", text="User asked about fertilizer", confidence=0.99),
        ]

        self.assertEqual(
            UserProfileService._memory_candidates(extracted, existing),
            ["Operates a dairy farm"],
        )

    def test_consolidates_existing_memory_log_into_durable_notes(self):
        notes = [
            {"id": 1, "note_text": "Prefers concise answers"},
            {"id": 2, "note_text": "Uses or is associated with EU-FarmBook"},
            {"id": 3, "note_text": "Is monitoring sheep in Norway"},
            {"id": 4, "note_text": "Is cultivating mushrooms"},
            {"id": 5, "note_text": "Is cultivating morels"},
            {"id": 6, "note_text": "Is cultivating mushtoom"},
            {"id": 7, "note_text": "Grows mushrooms"},
            {"id": 8, "note_text": "Raises sheep"},
            {"id": 9, "note_text": "Wants information on crop rotation techniques"},
        ]

        self.assertEqual(
            [note["note_text"] for note in UserProfileService.filter_memory_notes(notes)],
            ["Prefers concise answers", "Grows mushrooms", "Grows morels", "Raises sheep"],
        )


if __name__ == "__main__":
    unittest.main()
