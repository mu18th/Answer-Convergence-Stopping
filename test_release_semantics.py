"""Small offline regression suite for the released stopping semantics."""
from __future__ import annotations

import importlib.util
import copy
import sys
import unittest
from unittest import mock
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = ROOT / "ACS"
EVAL_ROOT = PACKAGE_ROOT / "eval"
for source_root in (PACKAGE_ROOT, EVAL_ROOT):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

import analyze as az
from benchmark_data.sniah_data import adapt_ruler_row
from browsecomp_plus_judge import parse_judge_response
from draft_cleaner import is_non_answer
from run_fold import (
    compatible_disjoint_ruler_cohort,
    execution_config,
    execution_fingerprint,
    parse_gate_response,
    validate_execution_resume,
)


CFG = {"rule": "windowed", "theta": 0.995, "eps": 0.05,
       "window": 3, "min_chunks": 2}


def load_script(name: str, relative_path: str):
    """Load a released experiment script without executing its CLI."""
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


class ReleaseSemantics(unittest.TestCase):
    def test_browsecomp_launcher_rejects_legacy_completed_rows(self):
        launcher = load_script(
            "release_browsecomp_launcher",
            "ACS/experiments/browsecomp_plus/run_browsecomp_all830.py",
        )
        expected = {
            "version": 1, "protocol_id": "test", "model": "test/model",
            "base_url": "https://example.invalid/v1",
            "openrouter_provider": "", "fold": {}, "online_stop": {},
        }
        fingerprint = execution_fingerprint(expected)
        class ExistingPath:
            @staticmethod
            def exists():
                return True

        with mock.patch.object(
            launcher, "read_jsonl",
            return_value=[{"id": "legacy", "steps": [{}]}],
        ):
            with self.assertRaisesRegex(RuntimeError, "fingerprint"):
                launcher.successful_ids(ExistingPath(), expected, fingerprint)

    def test_only_ruler_may_append_a_disjoint_input_cohort(self):
        current = [
            {"id": "ruler_hotpotqa_8192_0050", "benchmark": "ruler_hotpotqa"}
        ]
        self.assertTrue(compatible_disjoint_ruler_cohort(
            {"id": "ruler_hotpotqa_8192_0000", "benchmark": "ruler_hotpotqa"},
            current,
        ))
        self.assertFalse(compatible_disjoint_ruler_cohort(
            {"id": "foreign", "benchmark": "other"}, current,
        ))
        self.assertFalse(compatible_disjoint_ruler_cohort(
            {"id": "foreign", "benchmark": "ruler_hotpotqa"},
            [{"id": "bcp_1", "construction": "ACS_min10doc_all830_v1"}],
        ))

    def test_resume_fingerprint_covers_trajectory_settings(self):
        config = {
            "server": {
                "model": "test/model",
                "base_url": "https://example.invalid/v1",
            },
            "fold": {
                "chunk_chars": 24000,
                "notes_chars": 6000,
                "notes_tokens": 1500,
                "draft_tokens": 32,
            },
        }
        stop = {
            "rule": "none", "theta": 0.995, "eps": 0.05,
            "window": 3, "min_chunks": 2,
            "record_verbalized": True, "record_gate": True,
        }
        expected = execution_config(config, stop)
        fingerprint = execution_fingerprint(expected)
        record = {
            "id": "synthetic",
            "execution_config": expected,
            "execution_fingerprint": fingerprint,
        }
        validate_execution_resume(record, expected, fingerprint)

        variants = []
        for field, value in (
            ("notes_chars", 3000),
            ("notes_tokens", 750),
            ("draft_tokens", 64),
        ):
            changed = copy.deepcopy(config)
            changed["fold"][field] = value
            variants.append(execution_config(changed, stop))
        changed_stop = dict(stop, record_gate=False)
        variants.append(execution_config(config, changed_stop))
        for changed in variants:
            with self.subTest(changed=changed):
                with self.assertRaisesRegex(RuntimeError, "fingerprint"):
                    validate_execution_resume(
                        record, changed, execution_fingerprint(changed)
                    )

    def test_legacy_resume_without_fingerprint_fails_closed(self):
        expected = {
            "version": 1,
            "protocol_id": "test",
            "model": "test/model",
            "base_url": "https://example.invalid/v1",
            "openrouter_provider": "",
            "fold": {},
            "online_stop": {},
        }
        with self.assertRaisesRegex(RuntimeError, "fingerprint"):
            validate_execution_resume(
                {"id": "legacy"}, expected, execution_fingerprint(expected)
            )

    def test_open_non_answer_resets_window(self):
        drafts = ["alpha", "alpha", "I do not know", "alpha", "alpha", "alpha"]
        steps = [{"t": index, "draft": draft, "draft_norm": draft.lower(),
                  "draft_conf": 0.999, "confidence": 0.999}
                 for index, draft in enumerate(drafts, 1)]
        trajectory = {"task_shape": "open", "steps": steps}
        self.assertEqual(az.stop_measured(trajectory, .995, .05, w=3), 6)
        self.assertEqual(az.stop_online(trajectory, cfg=CFG), 6)

    def test_mcq_online_offline_equivalence(self):
        low = {"A": .4, "B": .3, "C": .2, "D": .1}
        high = {"A": .997, "B": .001, "C": .001, "D": .001}
        steps = [{"t": index, "posterior": posterior,
                  "confidence": max(posterior.values())}
                 for index, posterior in enumerate([low, high, high, high], 1)]
        trajectory = {"task_shape": "mcq", "steps": steps}
        self.assertEqual(az.stop_measured(trajectory, .995, .05, w=3), 4)
        self.assertEqual(az.stop_online(trajectory, cfg=CFG), 4)

    def test_verbalized_threshold_uses_0_to_100_scale(self):
        trajectory = {"steps": [{"t": 1, "verbalized": 99.0},
                                 {"t": 2, "verbalized": 99.5}]}
        self.assertEqual(az.stop_verbalized(trajectory, 99.5), 2)

    def test_invalid_comparators_conservatively_continue(self):
        trajectory = {"steps": [
            {"t": 1, "verbalized": None, "verbalized_stop": None},
            {"t": 2, "verbalized": 99.4, "verbalized_stop": False},
            {"t": 3, "verbalized": None, "verbalized_stop": None},
        ]}
        self.assertEqual(az.stop_verbalized(trajectory, 99.5), 3)
        self.assertEqual(az.stop_verbalized_gate(trajectory), 3)

    def test_premature_stop_receives_no_capture(self):
        self.assertEqual(az.captured_saving(2, 3, 10), 0)
        self.assertEqual(az.captured_saving(3, 3, 10), 7)
        self.assertEqual(az.captured_saving(7, 3, 10), 3)

    def test_non_answer_cleaner(self):
        self.assertTrue(is_non_answer("I don't know"))
        self.assertTrue(is_non_answer("The number is not explicitly stated."))
        self.assertFalse(is_non_answer("Marie Curie"))

    def test_gate_parser_accepts_only_exact_decisions(self):
        self.assertIs(parse_gate_response("END"), True)
        self.assertIs(parse_gate_response("<next>continue</next>"), False)
        self.assertIs(parse_gate_response('"END"'), True)
        self.assertIs(
            parse_gate_response("<next>continue</next>\nI will explain why."),
            False,
        )
        self.assertIsNone(parse_gate_response("I think we should continue"))

    def test_ruler_invalid_verbalized_value_conservatively_continues(self):
        ruler = load_script(
            "release_ruler_analysis",
            "ACS/experiments/ruler_hotpotqa/analyze_ruler_hotpotqa.py",
        )
        trajectory = {"steps": [
            {"verbalized": None},
            {"verbalized": 99.6},
        ]}
        self.assertEqual(ruler.stop_verbal(trajectory, 99.5), 2)

    def test_browsecomp_judge_parser(self):
        parsed = parse_judge_response(
            "**extracted_final_answer:** Paris\n"
            "**reasoning:** equivalent\n"
            "**correct:** yes\n"
            "**confidence:** 100"
        )
        self.assertFalse(parsed.parse_error)
        self.assertTrue(parsed.correct)
        self.assertEqual(parsed.extracted_final_answer, "Paris")
        self.assertTrue(parse_judge_response("unstructured").parse_error)

    def test_browsecomp_scoring_requires_semantic_judge(self):
        trajectory = {
            "id": "bcp_synthetic",
            "task_shape": "open",
            "answer_type": "string",
            "question": "Where?",
            "answer": "Paris",
        }
        original = az.JUDGE
        az.JUDGE = None
        try:
            with self.assertRaisesRegex(RuntimeError, "requires semantic judging"):
                az.score_answer(trajectory, "Paris")
        finally:
            az.JUDGE = original

    def test_unknown_answer_type_fails_closed(self):
        trajectory = {
            "id": "unsupported_synthetic",
            "task_shape": "open",
            "answer_type": "obsolete_benchmark_metric",
            "answer": "42",
        }
        with self.assertRaisesRegex(ValueError, "unsupported answer_type"):
            az.score_answer(trajectory, "42")

    def test_sniah_dataset_adapter_schema(self):
        source = {
            "task": "niah_single_1",
            "index": 17,
            "input": "needle context\nWhere is the needle?",
            "outputs": ["haystack"],
            "length": 8192,
            "length_w_model_temp": 8192,
            "answer_prefix": " Answer:",
            "token_position_answer": 123,
        }
        row = adapt_ruler_row(source, 8192, 4)
        self.assertEqual(row["task_shape"], "open")
        self.assertEqual(row["answer"], "haystack")
        self.assertEqual(row["context"], "needle context")
        self.assertEqual(row["question"], "Where is the needle? Answer:")

    def test_sniah_ablation_validation_uses_all_250_rows(self):
        preparer = load_script(
            "release_sniah_ablation_prepare",
            "ACS/experiments/sniah_ablations/prepare_sniah.py",
        )
        buckets = [8192, 16384, 32768, 65536, 131072]
        rows = [
            {"id": f"synthetic_{bucket}_{index}", "bucket": bucket}
            for bucket in buckets for index in range(50)
        ]
        self.assertEqual(len(preparer.validate_rows(rows)), 250)

    def test_parameter_selection_objective(self):
        selection = load_script(
            "release_parameter_selection",
            "ACS/experiments/sniah_five_model/analyze_parameter_selection.py",
        )
        original = selection.metrics
        selection.metrics = lambda _rows, theta, eps: {
            "accuracy": theta,
            "mean_tokens": theta * 1000,
            "premature": 0.0,
        }
        try:
            theta, eps, summary = selection.select([[{}]])
        finally:
            selection.metrics = original
        self.assertEqual((theta, eps), (0.98, 0.05))
        self.assertEqual(summary["best_dev_accuracy"], 0.995)


if __name__ == "__main__":
    unittest.main()
