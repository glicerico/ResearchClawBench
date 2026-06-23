"""Tests for the two-axis scorer: per-item paper fidelity + holistic research."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest import TestCase

from evaluation.score import (
    aggregate_scores,
    _score_item_fidelity,
    _normalize_research_result,
    _distill_trajectory,
    _gather_code,
    _gather_outputs,
    _clamp,
    RESEARCH_DIMENSIONS,
)
from evaluation.config import SCIENTIFIC_WEIGHT, FIDELITY_WEIGHT


class FakeAgent:
    """Duck-typed stand-in for structai.LLMAgent: returns a fixed dict."""
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def __call__(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return self.payload


def _research_payload(score=60, reasoning="r", gap="g"):
    """A nested research response covering every dimension."""
    return {d["key"]: {"score": score, "reasoning": reasoning, "gap": gap}
            for d in RESEARCH_DIMENSIONS}


class TestClamp(TestCase):
    def test_clamp_bounds_and_types(self):
        self.assertEqual(_clamp(150), 100)
        self.assertEqual(_clamp(-5), 0)
        self.assertEqual(_clamp("abc"), 0)
        self.assertEqual(_clamp(73), 73)
        self.assertEqual(_clamp(None), 0)


class TestFidelityItem(TestCase):
    def test_text_item_returns_fidelity(self):
        agent = FakeAgent({"fidelity_score": 42, "fidelity_reasoning": "matches paper"})
        res = _score_item_fidelity(agent, "report", {"type": "text", "content": "x"},
                                   None, [], "instructions")
        self.assertEqual(res["fidelity_score"], 42)
        self.assertIn("matches", res["fidelity_reasoning"])
        # text items must not attach images
        self.assertNotIn("image_paths", agent.calls[0][1])

    def test_image_item_uses_image_path(self):
        agent = FakeAgent({"fidelity_score": 90, "fidelity_reasoning": "matches target"})
        res = _score_item_fidelity(agent, "report", {"type": "image", "content": "x", "path": "img.png"},
                                   None, [], "instructions")
        self.assertEqual(res["fidelity_score"], 90)
        self.assertIn("image_paths", agent.calls[0][1])

    def test_malformed_response_yields_zero(self):
        res = _score_item_fidelity(FakeAgent(None), "report", {"type": "text"}, None, [], "")
        self.assertEqual(res["fidelity_score"], 0)

    def test_out_of_range_clamped(self):
        agent = FakeAgent({"fidelity_score": 250, "fidelity_reasoning": ""})
        res = _score_item_fidelity(agent, "r", {"type": "text"}, None, [], "")
        self.assertEqual(res["fidelity_score"], 100)


class TestResearchNormalize(TestCase):
    def test_nested_shape(self):
        out = _normalize_research_result(_research_payload(70, "ok", "add ablation"))
        for d in RESEARCH_DIMENSIONS:
            self.assertEqual(out[d["key"]]["score"], 70)
            self.assertEqual(out[d["key"]]["reasoning"], "ok")
            self.assertEqual(out[d["key"]]["gap"], "add ablation")

    def test_flat_shape_tolerated(self):
        flat = {}
        for d in RESEARCH_DIMENSIONS:
            flat[f"{d['key']}_score"] = 55
            flat[f"{d['key']}_reasoning"] = "flat"
            flat[f"{d['key']}_gap"] = "flat gap"
        out = _normalize_research_result(flat)
        self.assertEqual(out[RESEARCH_DIMENSIONS[0]["key"]]["score"], 55)
        self.assertEqual(out[RESEARCH_DIMENSIONS[0]["key"]]["reasoning"], "flat")
        self.assertEqual(out[RESEARCH_DIMENSIONS[0]["key"]]["gap"], "flat gap")

    def test_missing_and_malformed_default_zero(self):
        out = _normalize_research_result(None)
        for d in RESEARCH_DIMENSIONS:
            self.assertEqual(out[d["key"]]["score"], 0)
        out2 = _normalize_research_result({RESEARCH_DIMENSIONS[0]["key"]: {"score": 300}})
        self.assertEqual(out2[RESEARCH_DIMENSIONS[0]["key"]]["score"], 100)


class TestAggregate(TestCase):
    def _checklist(self):
        return [
            {"type": "text", "content": "A", "weight": 0.25},
            {"type": "image", "content": "B", "weight": 0.75},
        ]

    def _fidelity(self):
        return [
            {"fidelity_score": 40, "fidelity_reasoning": "f0"},
            {"fidelity_score": 100, "fidelity_reasoning": "f1"},
        ]

    def test_fidelity_is_weighted_over_items(self):
        agg = aggregate_scores(self._checklist(), self._fidelity(), _research_payload(60))
        exp_fid = (40 * 0.25 + 100 * 0.75) / 1.0  # 85.0
        self.assertAlmostEqual(agg["paper_fidelity_score"], round(exp_fid, 2))

    def test_scientific_is_weighted_over_dimensions(self):
        # uniform 60 across all dimensions -> weighted avg is 60 regardless of weights
        agg = aggregate_scores(self._checklist(), self._fidelity(), _research_payload(60))
        self.assertAlmostEqual(agg["scientific_capability_score"], 60.0)

    def test_scientific_respects_dimension_weights(self):
        # give the first dimension 100, the rest 0; expect weight-proportional avg
        payload = {d["key"]: {"score": 0, "reasoning": ""} for d in RESEARCH_DIMENSIONS}
        first = RESEARCH_DIMENSIONS[0]
        payload[first["key"]] = {"score": 100, "reasoning": ""}
        total_w = sum(d["weight"] for d in RESEARCH_DIMENSIONS)
        expected = round(100 * first["weight"] / total_w, 2)
        agg = aggregate_scores(self._checklist(), self._fidelity(), payload)
        self.assertAlmostEqual(agg["scientific_capability_score"], expected)

    def test_total_is_default_blend(self):
        agg = aggregate_scores(self._checklist(), self._fidelity(), _research_payload(60))
        blend = (SCIENTIFIC_WEIGHT * agg["scientific_capability_score"]
                 + FIDELITY_WEIGHT * agg["paper_fidelity_score"])
        self.assertAlmostEqual(agg["total_score"], round(blend, 2))

    def test_items_have_fidelity_and_legacy_fields(self):
        agg = aggregate_scores(self._checklist(), self._fidelity(), _research_payload(60))
        it = agg["items"][0]
        for key in ("fidelity_score", "fidelity_reasoning", "score", "reasoning",
                    "weight", "type", "index"):
            self.assertIn(key, it)
        # legacy per-item score now mirrors fidelity (research is holistic)
        self.assertEqual(it["score"], 40)
        self.assertEqual(it["reasoning"], "f0")

    def test_research_dimensions_emitted(self):
        agg = aggregate_scores(self._checklist(), self._fidelity(), _research_payload(60))
        self.assertEqual(len(agg["research_dimensions"]), len(RESEARCH_DIMENSIONS))
        d0 = agg["research_dimensions"][0]
        for key in ("key", "name", "weight", "score", "reasoning", "gap"):
            self.assertIn(key, d0)
        self.assertEqual(d0["gap"], "g")

    def test_robust_to_empty(self):
        agg = aggregate_scores([{"weight": 1.0, "type": "text", "content": "A"}], [{}], {})
        self.assertEqual(agg["paper_fidelity_score"], 0)
        self.assertEqual(agg["scientific_capability_score"], 0)
        self.assertEqual(agg["total_score"], 0)
        agg2 = aggregate_scores([], [], {})
        self.assertEqual(agg2["total_score"], 0)
        self.assertEqual(agg2["total_weight"], 0)


class TestTrajectoryDistill(TestCase):
    def test_distill_extracts_actions_and_skips_noise(self):
        lines = [
            {"type": "system", "subtype": "init", "session_id": "s"},
            {"type": "assistant", "message": {"content": [
                {"type": "text", "text": "I'll explore the workspace."},
                {"type": "tool_use", "name": "Bash", "input": {"command": "ls -la", "description": "list"}},
            ]}},
            {"type": "user", "message": {"content": [
                {"type": "tool_result", "content": [{"type": "text", "text": "file1\nfile2"}]},
            ]}},
            {"type": "system", "subtype": "token_count", "estimated_tokens": 10},
            {"type": "result", "subtype": "success", "result": "done"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            (ws / "_agent_output.jsonl").write_text(
                "\n".join(json.dumps(l) for l in lines), encoding="utf-8")
            out = _distill_trajectory(ws)
        self.assertIn("THINK: I'll explore", out)
        self.assertIn("ACTION Bash: command=ls -la", out)
        self.assertIn("RESULT: file1", out)
        self.assertIn("FINAL:", out)
        self.assertNotIn("token_count", out)

    def test_missing_trajectory_is_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(_distill_trajectory(Path(tmp)), "")


class TestArtifactGather(TestCase):
    def test_code_and_outputs_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            (ws / "code").mkdir()
            (ws / "code" / "run.py").write_text("print('hello')", encoding="utf-8")
            (ws / "outputs").mkdir()
            (ws / "outputs" / "results.json").write_text('{"acc": 0.9}', encoding="utf-8")
            code = _gather_code(ws)
            outputs = _gather_outputs(ws)
        self.assertIn("run.py", code)
        self.assertIn("print('hello')", code)
        self.assertIn("results.json", outputs)
        self.assertIn("0.9", outputs)

    def test_missing_dirs_are_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(_gather_outputs(Path(tmp)), "")


class TestBackCompat(TestCase):
    def test_new_score_file_keeps_legacy_keys(self):
        agg = aggregate_scores([{"weight": 1.0, "type": "text", "content": "A"}],
                               [{"fidelity_score": 50, "fidelity_reasoning": "ok"}],
                               _research_payload(60))
        self.assertIn("total_score", agg)
        self.assertIn("score", agg["items"][0])
        self.assertIn("reasoning", agg["items"][0])
