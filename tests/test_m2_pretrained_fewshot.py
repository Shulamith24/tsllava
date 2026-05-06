from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import math

import torch

import scripts.train_ucr_classification_pretrained_fewshot as fewshot_script
from scripts.train_ucr_classification_pretrained_fewshot import (
    answer_token_nll_from_prompt,
    evaluate,
    parse_args,
    score_semantic_candidates,
)
from opentslm.time_series_datasets.classification_utils import get_label_cards, get_label_verbalizers
from opentslm.time_series_datasets.univariate_fewshot import load_univariate_fewshot_bundle


def _write_ucr_dataset(root: Path, dataset_name: str) -> Path:
    archive = root / "data" / "UCRArchive_2018"
    dataset_dir = archive / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    train_rows = [
        [0, 0.00, 0.05, 0.00, 0.05],
        [0, 0.10, 0.00, 0.10, 0.00],
        [1, 1.00, 0.95, 1.00, 0.95],
        [1, 0.90, 1.00, 0.90, 1.00],
    ]
    test_rows = [
        [0, 0.02, 0.05, 0.02, 0.05],
        [1, 0.98, 0.96, 0.98, 0.96],
    ]

    def _write_rows(path: Path, rows: list[list[float]]) -> None:
        lines = ["\t".join(str(value) for value in row) for row in rows]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    _write_rows(dataset_dir / f"{dataset_name}_TRAIN.tsv", train_rows)
    _write_rows(dataset_dir / f"{dataset_name}_TEST.tsv", test_rows)
    return archive


class _DummyTokenizer:
    eos_token_id = 99
    _token_to_id = {"<c0>": 7, "<c1>": 8, "<eos>": 99}
    _id_to_token = {7: "<c0>", 8: "<c1>", 99: "<eos>"}

    def batch_decode(self, _gen_ids, skip_special_tokens: bool = False):
        del skip_special_tokens
        return ["<c0>"]

    def convert_tokens_to_ids(self, token: str):
        return self._token_to_id[token]

    def convert_ids_to_tokens(self, ids):
        if isinstance(ids, int):
            return self._id_to_token[ids]
        return [self._id_to_token[int(item)] for item in ids]


class _DummyLLM:
    def __init__(self):
        self.calls: list[dict[str, object]] = []
        self.forward_calls: list[dict[str, object]] = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return torch.tensor([[1, 2]])

    def __call__(self, *, inputs_embeds, attention_mask, return_dict: bool = True):
        self.forward_calls.append(
            {
                "inputs_embeds_shape": tuple(inputs_embeds.shape),
                "attention_mask_shape": tuple(attention_mask.shape),
                "return_dict": return_dict,
            }
        )
        batch_size, seq_len, _hidden = inputs_embeds.shape
        logits = torch.full((batch_size, seq_len, 100), -10.0)
        for row_idx in range(batch_size):
            last_pos = int(attention_mask[row_idx].sum().item()) - 1
            if row_idx == 0:
                logits[row_idx, last_pos, 7] = 6.0
                logits[row_idx, last_pos, 8] = 1.0
            else:
                logits[row_idx, last_pos, 7] = 0.5
                logits[row_idx, last_pos, 8] = 5.0
        return SimpleNamespace(logits=logits)


class _DummyFewshotModel:
    def __init__(self):
        self.tokenizer = _DummyTokenizer()
        self.llm = _DummyLLM()
        self.native_generate_calls = 0
        self.compute_loss_calls = 0

    def eval(self):
        return self

    def compute_loss(self, _batch):
        self.compute_loss_calls += 1
        return torch.tensor(0.0)

    def pad_and_apply_batch(self, batch):
        return torch.zeros(len(batch), 2, 3), torch.ones(len(batch), 2, dtype=torch.long)

    def generate(self, batch, max_new_tokens: int, skip_special_tokens: bool = False):
        del max_new_tokens
        del skip_special_tokens
        self.native_generate_calls += 1
        return [sample["answer"].replace(self.get_eos_token(), "").strip() for sample in batch]

    def get_eos_token(self) -> str:
        return "<eos>"


class _SemanticTokenizer:
    vocab = {"<pad>": 0, "short": 1, "long": 2, "phrase": 3}

    def __call__(self, answers, return_tensors, padding, truncation, add_special_tokens):
        del return_tensors, padding, truncation, add_special_tokens
        encoded = []
        for answer in answers:
            encoded.append([self.vocab[token] for token in answer.split()])
        max_len = max(len(row) for row in encoded)
        input_ids = []
        attention_mask = []
        for row in encoded:
            pad_len = max_len - len(row)
            input_ids.append(row + [0] * pad_len)
            attention_mask.append([1] * len(row) + [0] * pad_len)
        return SimpleNamespace(
            input_ids=torch.tensor(input_ids, dtype=torch.long),
            attention_mask=torch.tensor(attention_mask, dtype=torch.long),
        )


class _UniformLLM:
    def __init__(self, vocab_size: int = 5, hidden_size: int = 3):
        self.vocab_size = vocab_size
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)

    def get_input_embeddings(self):
        return self.embedding

    def __call__(self, *, inputs_embeds, attention_mask, return_dict: bool = True):
        del attention_mask, return_dict
        batch_size, seq_len, _hidden = inputs_embeds.shape
        return SimpleNamespace(logits=torch.zeros(batch_size, seq_len, self.vocab_size))


class _SemanticScoringModel:
    device = "cpu"

    def __init__(self):
        self.tokenizer = _SemanticTokenizer()
        self.llm = _UniformLLM()


class M2PretrainedFewshotTest(unittest.TestCase):
    def test_parse_args_tracks_constrained_decoding_flag(self) -> None:
        default_args = parse_args([])
        self.assertTrue(default_args.constrained_decoding)
        self.assertFalse(default_args.disable_constrained_decoding)
        self.assertEqual(default_args.eval_decode_mode, "generate")
        self.assertEqual(default_args.dataset_family, "ucr")
        self.assertEqual(default_args.split_protocol, "default")

        disabled_args = parse_args(["--disable_constrained_decoding"])
        self.assertFalse(disabled_args.constrained_decoding)
        self.assertTrue(disabled_args.disable_constrained_decoding)

        logits_args = parse_args(["--eval_decode_mode", "logits"])
        self.assertEqual(logits_args.eval_decode_mode, "logits")

        semantic_args = parse_args(["--label_interface", "semantic", "--dataset_family", "mitbih"])
        self.assertEqual(semantic_args.eval_decode_mode, "logits")
        self.assertEqual(semantic_args.semantic_target_mode, "class_token")
        self.assertEqual(semantic_args.semantic_score_mode, "calibrated")

        phrase_args = parse_args(
            [
                "--label_interface",
                "semantic",
                "--dataset_family",
                "mitbih",
                "--semantic_target_mode",
                "phrase",
            ]
        )
        self.assertEqual(phrase_args.eval_decode_mode, "phrase_likelihood")

    def test_canonical_verbalizers_follow_dataset_label_order(self) -> None:
        self.assertEqual(
            get_label_verbalizers("mitbih", ["N", "S", "V", "F", "Q"]),
            {
                "N": ["normal beat"],
                "S": ["supraventricular ectopic beat"],
                "V": ["ventricular ectopic beat"],
                "F": ["fusion beat"],
                "Q": ["unknown beat type"],
            },
        )
        self.assertEqual(
            list(get_label_verbalizers("sleepedf", ["W", "N1", "N2", "N3", "REM"]).values()),
            [
                ["wake stage"],
                ["N1 stage"],
                ["N2 stage"],
                ["N3 stage"],
                ["REM stage"],
            ],
        )
        self.assertEqual(
            list(get_label_verbalizers("cinc2017af", ["N", "A", "O", "~"]).values()),
            [
                ["normal rhythm"],
                ["atrial fibrillation"],
                ["other rhythm"],
                ["noisy signal"],
            ],
        )

    def test_multi_verbalizer_label_cards_follow_dataset_label_order(self) -> None:
        cards = get_label_cards("mitbih", ["N", "S", "V", "F", "Q"], verbalizer_mode="multi")
        self.assertEqual(list(cards.keys()), ["N", "S", "V", "F", "Q"])
        self.assertEqual(cards["V"]["class_token"], "<c2>")
        self.assertEqual(cards["V"]["canonical_name"], "ventricular ectopic beat")
        self.assertIn("ECG heartbeat segment: ventricular ectopic beat", cards["V"]["verbalizers"])

    def test_evaluate_uses_constrained_logits_processor_by_default(self) -> None:
        model = _DummyFewshotModel()
        batch = [{"answer": "<c0><eos>"}]

        metrics = evaluate(
            model=model,
            data_loader=[batch],
            max_new_tokens=2,
            class_token_ids=[7],
            disable_constrained_decoding=False,
            rank=1,
        )

        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["macro_f1"], 1.0)
        self.assertEqual(len(model.llm.calls), 1)
        self.assertIn("logits_processor", model.llm.calls[0])
        self.assertEqual(model.native_generate_calls, 0)

    def test_evaluate_can_disable_constrained_decoding(self) -> None:
        model = _DummyFewshotModel()
        batch = [{"answer": "<c0><eos>"}]

        metrics = evaluate(
            model=model,
            data_loader=[batch],
            max_new_tokens=2,
            class_token_ids=[7],
            disable_constrained_decoding=True,
            rank=1,
        )

        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["macro_f1"], 1.0)
        self.assertEqual(len(model.llm.calls), 0)
        self.assertEqual(model.native_generate_calls, 1)

    def test_evaluate_logits_mode_scores_class_tokens_without_generate(self) -> None:
        model = _DummyFewshotModel()
        batch = [
            {"answer": "<c0><eos>"},
            {"answer": "<c1><eos>"},
        ]

        metrics = evaluate(
            model=model,
            data_loader=[batch],
            max_new_tokens=2,
            class_token_ids=[7, 8],
            disable_constrained_decoding=True,
            eval_decode_mode="logits",
            rank=1,
        )

        self.assertEqual(metrics["eval_decode_mode"], "logits")
        self.assertEqual(metrics["eval_loss_type"], "class_token_ce")
        self.assertEqual(metrics["predictions"], ["<c0>", "<c1>"])
        self.assertEqual(metrics["labels"], ["<c0>", "<c1>"])
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["macro_f1"], 1.0)
        self.assertEqual(len(model.llm.calls), 0)
        self.assertEqual(model.native_generate_calls, 0)
        self.assertEqual(len(model.llm.forward_calls), 1)
        self.assertEqual(model.compute_loss_calls, 0)

    def test_length_normalized_answer_nll_is_not_biased_by_phrase_length(self) -> None:
        model = _SemanticScoringModel()
        inputs_embeds = torch.zeros(2, 1, 3)
        attention_mask = torch.ones(2, 1, dtype=torch.long)

        nll = answer_token_nll_from_prompt(
            model,
            inputs_embeds,
            attention_mask,
            ["short", "long phrase"],
        )

        expected = math.log(model.llm.vocab_size)
        self.assertTrue(torch.allclose(nll, torch.full((2,), expected), atol=1e-6))

    def test_calibrated_semantic_scores_subtract_null_time_series_prior(self) -> None:
        calls = []

        def _fake_score(_model, batch, _candidate_verbalizers):
            calls.append(batch)
            is_null = bool(torch.equal(batch[0]["time_series"][0], torch.zeros(3)))
            return torch.tensor([[1.0, 0.5]]) if is_null else torch.tensor([[3.0, 1.5]])

        batch = [{"time_series": [torch.ones(3)], "answer": "normal heartbeat<eos>"}]
        with mock.patch.object(
            fewshot_script,
            "score_candidate_verbalizers",
            side_effect=_fake_score,
        ):
            scores = score_semantic_candidates(
                _DummyFewshotModel(),
                batch,
                [["normal heartbeat"], ["atrial fibrillation"]],
                semantic_score_mode="calibrated",
            )

        self.assertEqual(len(calls), 2)
        self.assertTrue(torch.equal(calls[1][0]["time_series"][0], torch.zeros(3)))
        self.assertTrue(torch.equal(scores, torch.tensor([[2.0, 1.0]])))

    def test_support_calibrated_semantic_scores_subtract_support_bias(self) -> None:
        def _fake_score(_model, _batch, _candidate_verbalizers):
            return torch.tensor([[3.0, 1.5]])

        with mock.patch.object(
            fewshot_script,
            "score_candidate_verbalizers",
            side_effect=_fake_score,
        ):
            scores = score_semantic_candidates(
                _DummyFewshotModel(),
                [{"time_series": [torch.ones(3)], "answer": "normal heartbeat<eos>"}],
                [["normal heartbeat"], ["atrial fibrillation"]],
                semantic_score_mode="support_cal",
                support_calibration_scores=torch.tensor([1.0, 0.25]),
            )

        self.assertTrue(torch.equal(scores, torch.tensor([[2.0, 1.25]])))

    def test_evaluate_phrase_likelihood_maps_phrases_to_class_ids(self) -> None:
        model = _DummyFewshotModel()
        batch = [
            {"answer": "normal heartbeat<eos>", "time_series": [torch.ones(3)]},
            {"answer": "atrial fibrillation<eos>", "time_series": [torch.ones(3)]},
        ]

        with mock.patch.object(
            fewshot_script,
            "score_semantic_candidates",
            return_value=torch.tensor([[4.0, 1.0], [0.5, 3.0]]),
        ):
            metrics = evaluate(
                model=model,
                data_loader=[batch],
                max_new_tokens=4,
                eval_decode_mode="phrase_likelihood",
                label_verbalizers={0: ["normal heartbeat"], 1: ["atrial fibrillation"]},
                selected_class_ids=[0, 1],
                semantic_score_mode="calibrated",
                label_to_class_id={
                    "normal heartbeat": 0,
                    "atrial fibrillation": 1,
                },
                rank=1,
            )

        self.assertEqual(metrics["eval_decode_mode"], "phrase_likelihood")
        self.assertEqual(metrics["eval_loss_type"], "semantic_phrase_calibrated_ce")
        self.assertEqual(metrics["semantic_score_mode"], "calibrated")
        self.assertEqual(metrics["predictions"], ["normal heartbeat", "atrial fibrillation"])
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["macro_f1"], 1.0)

    def test_ucr_bundle_loader_preserves_existing_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            archive = _write_ucr_dataset(Path(tmp), "ToyWave")
            args = SimpleNamespace(
                dataset_family="ucr",
                dataset="ToyWave",
                split_protocol="default",
                data_path=str(archive),
            )

            bundle = load_univariate_fewshot_bundle(args, eos_token="<eos>")

            self.assertEqual(bundle.dataset_family, "ucr")
            self.assertEqual(bundle.dataset_name, "ToyWave")
            self.assertEqual(bundle.split_protocol, "official")
            self.assertEqual(bundle.num_classes, 2)
            self.assertEqual(bundle.class_tokens, ["<c0>", "<c1>"])
            self.assertEqual(len(bundle.train_dataset), 4)
            self.assertEqual(len(bundle.test_dataset), 2)

    def test_cinc2017_bundle_loader_uses_stable_label_order(self) -> None:
        rows = {
            "train": [
                {"record_name": "A00/A00001", "label": "N", "time_series": [0.0, 1.0], "sample_rate": 300.0, "source_sample_rate": 300.0, "original_length": 2},
                {"record_name": "A00/A00002", "label": "A", "time_series": [1.0, 0.0], "sample_rate": 300.0, "source_sample_rate": 300.0, "original_length": 2},
            ],
            "validation": [
                {"record_name": "A00/A00003", "label": "O", "time_series": [0.0, 0.0], "sample_rate": 300.0, "source_sample_rate": 300.0, "original_length": 2},
            ],
            "test": [
                {"record_name": "A00/A00004", "label": "~", "time_series": [1.0, 1.0], "sample_rate": 300.0, "source_sample_rate": 300.0, "original_length": 2},
            ],
        }

        def _fake_loader(**_kwargs):
            return rows["train"], rows["validation"], rows["test"]

        args = SimpleNamespace(
            dataset_family="cinc2017af",
            dataset=None,
            split_protocol="default",
            data_path="./data",
        )
        with mock.patch(
            "opentslm.time_series_datasets.cinc2017af.CinC2017AFClassificationDataset.load_cinc2017af_splits",
            side_effect=_fake_loader,
        ):
            bundle = load_univariate_fewshot_bundle(args, eos_token="<eos>")

        self.assertEqual(bundle.dataset_family, "cinc2017af")
        self.assertEqual(bundle.dataset_name, "CinC2017AF")
        self.assertEqual(bundle.split_protocol, "stratified")
        self.assertEqual(bundle.num_classes, 4)
        self.assertEqual(bundle.label_mapping, {"N": "<c0>", "A": "<c1>", "O": "<c2>", "~": "<c3>"})

    def test_cinc2017_semantic_bundle_uses_canonical_verbalizers(self) -> None:
        rows = {
            "train": [
                {"record_name": "A00/A00001", "label": "N", "time_series": [0.0, 1.0], "sample_rate": 300.0, "source_sample_rate": 300.0, "original_length": 2},
                {"record_name": "A00/A00002", "label": "A", "time_series": [1.0, 0.0], "sample_rate": 300.0, "source_sample_rate": 300.0, "original_length": 2},
            ],
            "validation": [
                {"record_name": "A00/A00003", "label": "O", "time_series": [0.0, 0.0], "sample_rate": 300.0, "source_sample_rate": 300.0, "original_length": 2},
            ],
            "test": [
                {"record_name": "A00/A00004", "label": "~", "time_series": [1.0, 1.0], "sample_rate": 300.0, "source_sample_rate": 300.0, "original_length": 2},
            ],
        }

        def _fake_loader(**_kwargs):
            return rows["train"], rows["validation"], rows["test"]

        args = SimpleNamespace(
            dataset_family="cinc2017af",
            dataset=None,
            split_protocol="default",
            data_path="./data",
            label_interface="semantic",
            verbalizer_set="canonical",
            verbalizer_mode="canonical",
            semantic_target_mode="class_token",
        )
        with mock.patch(
            "opentslm.time_series_datasets.cinc2017af.CinC2017AFClassificationDataset.load_cinc2017af_splits",
            side_effect=_fake_loader,
        ):
            bundle = load_univariate_fewshot_bundle(args, eos_token="<eos>")

        self.assertEqual(bundle.class_tokens, ["<c0>", "<c1>", "<c2>", "<c3>"])
        self.assertEqual(
            bundle.label_mapping,
            {
                "N": "<c0>",
                "A": "<c1>",
                "O": "<c2>",
                "~": "<c3>",
            },
        )
        self.assertEqual(
            bundle.label_verbalizers,
            {
                0: ["normal rhythm"],
                1: ["atrial fibrillation"],
                2: ["other rhythm"],
                3: ["noisy signal"],
            },
        )
        self.assertEqual(bundle.label_cards[1]["short_code"], "AF")
        sample = bundle.train_dataset[0]
        self.assertIn("<c0> = normal rhythm", sample["pre_prompt"])
        self.assertIn("Output exactly one class token", sample["pre_prompt"])
        self.assertEqual(sample["post_prompt"], "Class:")
        self.assertEqual(sample["answer"], "<c0><eos>")

    def test_cinc2017_semantic_phrase_mode_preserves_diagnostic_path(self) -> None:
        rows = {
            "train": [
                {"record_name": "A00/A00001", "label": "N", "time_series": [0.0, 1.0], "sample_rate": 300.0, "source_sample_rate": 300.0, "original_length": 2},
            ],
            "validation": [
                {"record_name": "A00/A00002", "label": "A", "time_series": [1.0, 0.0], "sample_rate": 300.0, "source_sample_rate": 300.0, "original_length": 2},
            ],
            "test": [
                {"record_name": "A00/A00003", "label": "O", "time_series": [0.0, 0.0], "sample_rate": 300.0, "source_sample_rate": 300.0, "original_length": 2},
            ],
        }

        def _fake_loader(**_kwargs):
            return rows["train"], rows["validation"], rows["test"]

        args = SimpleNamespace(
            dataset_family="cinc2017af",
            dataset=None,
            split_protocol="default",
            data_path="./data",
            label_interface="semantic",
            verbalizer_set="canonical",
            verbalizer_mode="canonical",
            semantic_target_mode="phrase",
        )
        with mock.patch(
            "opentslm.time_series_datasets.cinc2017af.CinC2017AFClassificationDataset.load_cinc2017af_splits",
            side_effect=_fake_loader,
        ):
            bundle = load_univariate_fewshot_bundle(args, eos_token="<eos>")

        sample = bundle.train_dataset[0]
        self.assertIn("normal rhythm", sample["pre_prompt"])
        self.assertIn("Output exactly one label phrase", sample["pre_prompt"])
        self.assertEqual(sample["post_prompt"], "Label:")
        self.assertEqual(sample["answer"], "normal rhythm<eos>")
        self.assertEqual(sample["answer_loss_normalization"], "sample")

    def test_heart_sound_bundle_loader_uses_stable_label_order(self) -> None:
        rows = {
            "train": [
                {
                    "record_name": "a0001",
                    "label": "normal",
                    "source_database": "training-a",
                    "time_series": [0.0, 1.0],
                    "sample_rate": 500.0,
                    "source_sample_rate": 2000.0,
                    "original_length": 8,
                    "wav_path": "/tmp/a0001.wav",
                },
                {
                    "record_name": "a0002",
                    "label": "abnormal",
                    "source_database": "training-a",
                    "time_series": [1.0, 0.0],
                    "sample_rate": 500.0,
                    "source_sample_rate": 2000.0,
                    "original_length": 8,
                    "wav_path": "/tmp/a0002.wav",
                },
            ],
            "validation": [],
            "test": [],
        }

        def _fake_loader(**_kwargs):
            return rows["train"], rows["validation"], rows["test"]

        args = SimpleNamespace(
            dataset_family="cinc2016heart",
            dataset=None,
            split_protocol="default",
            data_path="./data",
        )
        with mock.patch(
            "opentslm.time_series_datasets.heart_sound.HeartSoundClassificationDataset.load_heart_sound_splits",
            side_effect=_fake_loader,
        ):
            bundle = load_univariate_fewshot_bundle(args, eos_token="<eos>")

        self.assertEqual(bundle.dataset_family, "cinc2016heart")
        self.assertEqual(bundle.dataset_name, "CinC2016HeartSound")
        self.assertEqual(bundle.split_protocol, "stratified")
        self.assertEqual(bundle.num_classes, 2)
        self.assertEqual(bundle.label_mapping, {"normal": "<c0>", "abnormal": "<c1>"})

if __name__ == "__main__":
    unittest.main()
