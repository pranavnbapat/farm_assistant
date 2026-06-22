import json
import unittest
from unittest.mock import AsyncMock, patch

from app.routers.ask import _parse_follow_ups, follow_ups
from app.schemas import FollowUpsIn, SourceItem


class FollowUpPolicyTests(unittest.TestCase):
    def test_filters_invented_platform_capabilities(self):
        raw = json.dumps([
            "How do I register my farm on EU-FarmBook?",
            "What are practical ways to improve crop planning?",
            "How can I link my farm data to the EU agricultural database?",
        ])

        self.assertEqual(
            _parse_follow_ups(raw),
            ["What are practical ways to improve crop planning?"],
        )

    def test_filters_overlong_and_duplicate_suggestions(self):
        valid = "How can cover crops improve soil health?"
        raw = json.dumps([valid, valid.upper(), "A" * 81])

        self.assertEqual(_parse_follow_ups(raw), [valid])

    def test_removes_orphan_citation_legend_and_inline_marker(self):
        from app.routers.ask import _strip_orphan_citations

        answer = "A KML example was described [1].\n[1] Unsupported source legend."
        self.assertEqual(
            _strip_orphan_citations(answer, set()),
            "A KML example was described.",
        )

    def test_removes_empty_citation_placeholders(self):
        from app.routers.ask import _sanitize_generated_markdown, _strip_orphan_citations

        answer = "Sensors include camera systems [], milk sensors [ ], and rumen pH sensors []."
        expected = "Sensors include camera systems, milk sensors, and rumen pH sensors."

        self.assertEqual(_sanitize_generated_markdown(answer), expected)
        self.assertEqual(_strip_orphan_citations(answer, {1, 2, 3}), expected)

        nested = "Vaccines keep their efficacy [[]]-\n\n.\n\nLong-term savings [[][]]- are possible."
        self.assertEqual(
            _sanitize_generated_markdown(nested),
            "Vaccines keep their efficacy\n\nLong-term savings are possible.",
        )


class FollowUpEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def test_suppressed_mode_skips_generation(self):
        payload = FollowUpsIn(
            user_message="Hi, how are you?",
            assistant_message="Hello. How can I help?",
            grounding_mode="conversation_only",
        )

        with patch("app.routers.ask.generate_once", new_callable=AsyncMock) as generate:
            response = await follow_ups(payload)

        self.assertEqual(response.follow_ups, [])
        self.assertEqual(response.meta["reason"], "unsupported_grounding_mode")
        generate.assert_not_awaited()

    async def test_general_fallback_kml_turn_has_no_suggestions(self):
        payload = FollowUpsIn(
            user_message="What about KML files?",
            assistant_message="KML is a geographic data format used by mapping tools.",
            grounding_mode="general_fallback",
        )

        with patch("app.routers.ask.generate_once", new_callable=AsyncMock) as generate:
            response = await follow_ups(payload)

        self.assertEqual(response.follow_ups, [])
        self.assertEqual(response.meta["reason"], "unsupported_grounding_mode")
        generate.assert_not_awaited()

    async def test_grounded_mode_without_cited_sources_has_no_suggestions(self):
        payload = FollowUpsIn(
            user_message="What about KML files?",
            assistant_message="No relevant source was cited.",
            grounding_mode="euf_supported",
        )

        with patch("app.routers.ask.generate_once", new_callable=AsyncMock) as generate:
            response = await follow_ups(payload)

        self.assertEqual(response.follow_ups, [])
        self.assertEqual(response.meta["reason"], "no_cited_sources")
        generate.assert_not_awaited()

    async def test_source_labels_anchor_generated_prompt(self):
        payload = FollowUpsIn(
            user_message="How can I improve soil health?",
            assistant_message="Cover crops can help protect and enrich soil.",
            grounding_mode="euf_supported",
            sources=[
                SourceItem(title="Cover crops for healthy soils", project="SoilCare"),
            ],
        )
        generated = json.dumps(["Which cover crops work best in dry climates?"])

        with patch(
            "app.routers.ask.generate_once",
            new=AsyncMock(return_value=generated),
        ) as generate:
            response = await follow_ups(payload)

        self.assertEqual(
            response.follow_ups,
            ["Which cover crops work best in dry climates?"],
        )
        prompt = generate.await_args.args[0]
        self.assertIn("Grounding mode: euf_supported", prompt)
        self.assertIn("Cover crops for healthy soils - SoilCare", prompt)


if __name__ == "__main__":
    unittest.main()
