"""Tests for dual-axis (scientific capability / paper fidelity) scoring."""
from __future__ import annotations

from unittest import TestCase

from evaluation import score as score_module
from evaluation.score import aggregate_scores, _score_single_item, _clamp
from evaluation.config import SCIENTIFIC_WEIGHT, FIDELITY_WEIGHT


class FakeAgent:
    """Duck-typed stand-in for structai.LLMAgent: returns a fixed dict."""
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def __call__(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return self.payload


class TestClamp(TestCase):
    def test_clamp_bounds_and_types(self):
        self.assertEqual(_clamp(150), 100)
        self.assertEqual(_clamp(-5), 0)
        self.assertEqual(_clamp("abc"), 0)
        self.assertEqual(_clamp(73), 73)
        self.assertEqual(_clamp(None), 0)


class TestSingleItem(TestCase):
    def test_text_item_returns_both_axes(self):
        agent = FakeAgent({
            "scientific_score": 85, "scientific_reasoning": "valid method, real evidence",
            "fidelity_score": 30, "fidelity_reasoning": "diverged from paper",
        })
        res = _score_single_item(agent, "report", {"type": "text", "content": "x"},
                                 None, [], "instructions")
        self.assertEqual(res["scientific_score"], 85)
        self.assertEqual(res["fidelity_score"], 30)
        self.assertIn("valid method", res["scientific_reasoning"])
        self.assertIn("diverged", res["fidelity_reasoning"])

    def test_image_item_uses_dual_format(self):
        agent = FakeAgent({
            "scientific_score": 60, "scientific_reasoning": "ok figure",
            "fidelity_score": 90, "fidelity_reasoning": "matches target",
        })
        res = _score_single_item(agent, "report", {"type": "image", "content": "x", "path": "img.png"},
                                 None, [], "instructions")
        self.assertEqual(res["scientific_score"], 60)
        self.assertEqual(res["fidelity_score"], 90)

    def test_malformed_response_yields_zeros(self):
        res = _score_single_item(FakeAgent(None), "report", {"type": "text"}, None, [], "")
        self.assertEqual(res["scientific_score"], 0)
        self.assertEqual(res["fidelity_score"], 0)

    def test_out_of_range_scores_clamped(self):
        agent = FakeAgent({"scientific_score": 250, "fidelity_score": -10,
                           "scientific_reasoning": "", "fidelity_reasoning": ""})
        res = _score_single_item(agent, "r", {"type": "text"}, None, [], "")
        self.assertEqual(res["scientific_score"], 100)
        self.assertEqual(res["fidelity_score"], 0)


class TestAggregate(TestCase):
    def _checklist(self):
        return [
            {"type": "text", "content": "A", "weight": 0.25},
            {"type": "image", "content": "B", "weight": 0.75},
        ]

    def _raw(self):
        return [
            {"scientific_score": 80, "scientific_reasoning": "s0", "fidelity_score": 40, "fidelity_reasoning": "f0"},
            {"scientific_score": 60, "scientific_reasoning": "s1", "fidelity_score": 100, "fidelity_reasoning": "f1"},
        ]

    def test_weighted_aggregation(self):
        agg = aggregate_scores(self._checklist(), self._raw())
        exp_sci = (80 * 0.25 + 60 * 0.75) / 1.0      # 65.0
        exp_fid = (40 * 0.25 + 100 * 0.75) / 1.0     # 85.0
        exp_total = SCIENTIFIC_WEIGHT * exp_sci + FIDELITY_WEIGHT * exp_fid
        self.assertAlmostEqual(agg["scientific_capability_score"], round(exp_sci, 2))
        self.assertAlmostEqual(agg["paper_fidelity_score"], round(exp_fid, 2))
        self.assertAlmostEqual(agg["total_score"], round(exp_total, 2))
        self.assertEqual(agg["total_weight"], 1.0)

    def test_item_has_dual_and_legacy_fields(self):
        agg = aggregate_scores(self._checklist(), self._raw())
        it = agg["items"][0]
        for key in ("scientific_score", "fidelity_score", "scientific_reasoning",
                    "fidelity_reasoning", "score", "reasoning", "weight", "type", "index"):
            self.assertIn(key, it)
        # legacy combined score matches the configured blend
        self.assertEqual(it["score"], round(SCIENTIFIC_WEIGHT * 80 + FIDELITY_WEIGHT * 40))

    def test_total_score_is_default_blend(self):
        agg = aggregate_scores(self._checklist(), self._raw())
        blend = (SCIENTIFIC_WEIGHT * agg["scientific_capability_score"]
                 + FIDELITY_WEIGHT * agg["paper_fidelity_score"])
        self.assertAlmostEqual(agg["total_score"], round(blend, 2))

    def test_missing_keys_and_empty_results_are_robust(self):
        # raw_results with missing keys / None must not crash and default to 0
        agg = aggregate_scores([{"weight": 1.0, "type": "text", "content": "A"}], [{}])
        self.assertEqual(agg["scientific_capability_score"], 0)
        self.assertEqual(agg["paper_fidelity_score"], 0)
        self.assertEqual(agg["total_score"], 0)
        agg2 = aggregate_scores([], [])
        self.assertEqual(agg2["total_score"], 0)
        self.assertEqual(agg2["total_weight"], 0)


class TestBackCompatScoreFile(TestCase):
    def test_legacy_score_file_shape_still_readable(self):
        # A legacy _score.json had only total_score + items[].score/reasoning.
        # The new aggregate still produces total_score + items[].score so any
        # consumer reading those keys keeps working.
        agg = aggregate_scores([{"weight": 1.0, "type": "text", "content": "A"}],
                               [{"scientific_score": 70, "fidelity_score": 50,
                                 "scientific_reasoning": "", "fidelity_reasoning": ""}])
        self.assertIn("total_score", agg)
        self.assertIn("score", agg["items"][0])
        self.assertIn("reasoning", agg["items"][0])
