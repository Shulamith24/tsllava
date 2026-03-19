# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "train_ucr_patchtst_classification_fewshot.py"
sys.path.insert(0, str(REPO_ROOT / "src"))

from opentslm.model.PatchTSTClassifier import (
    PatchTSTClassifierAdapter,
    prepare_patchtst_classification_batch,
)
from transformers import PatchTSTConfig, PatchTSTForClassification


def load_patchtst_fewshot_script_module():
    spec = spec_from_file_location("train_ucr_patchtst_classification_fewshot", SCRIPT_PATH)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_prepare_patchtst_classification_batch_shapes_and_masks():
    batch = [
        {
            "time_series": [np.array([1.0, 2.0, 3.0], dtype=np.float32)],
            "int_label": 1,
        },
        {
            "time_series": [np.array([4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float32)],
            "int_label": 0,
        },
    ]

    converted = prepare_patchtst_classification_batch(
        batch,
        context_length=6,
        device="cpu",
        pad_mode="last",
    )

    assert converted["past_values"].shape == (2, 6, 1)
    assert converted["target_values"].shape == (2,)
    assert converted["past_observed_mask"].shape == (2, 6, 1)
    assert converted["past_observed_mask"].dtype == torch.bool

    assert converted["target_values"].tolist() == [1, 0]
    assert converted["past_observed_mask"][0, :, 0].tolist() == [True, True, True, False, False, False]
    assert converted["past_observed_mask"][1, :, 0].tolist() == [True, True, True, True, True, False]
    assert converted["past_values"][0, :, 0].tolist() == [1.0, 2.0, 3.0, 3.0, 3.0, 3.0]


def test_build_model_from_local_pretrained_resets_head(tmp_path):
    pretrained_dir = tmp_path / "patchtst_pretrained"
    base_model = PatchTSTForClassification(
        PatchTSTConfig(
            num_input_channels=1,
            num_targets=2,
            context_length=8,
            patch_length=4,
            stride=4,
            d_model=16,
            num_attention_heads=4,
            num_hidden_layers=1,
            ffn_dim=32,
            use_cls_token=True,
        )
    )
    base_model.save_pretrained(pretrained_dir)

    adapter = PatchTSTClassifierAdapter.build_model(
        num_classes=3,
        context_length=8,
        device="cpu",
        patchtst_model_id=str(pretrained_dir),
        reset_head=True,
        use_cls_token=True,
    )

    outputs = adapter(
        past_values=torch.randn(2, 8, 1),
        target_values=torch.tensor([0, 1]),
        past_observed_mask=torch.ones(2, 8, 1, dtype=torch.bool),
        return_dict=True,
    )

    assert adapter.head_was_reset is True
    assert adapter.config.num_targets == 3
    assert outputs.loss is not None
    assert outputs.prediction_logits.shape == (2, 3)


def test_patchtst_fewshot_way_sampling_and_epoch_enforcement():
    script_module = load_patchtst_fewshot_script_module()

    label_to_indices = {
        0: [0, 1, 2],
        1: [3, 4, 5],
        2: [6, 7, 8],
    }
    support_info = script_module.sample_support_info(
        label_to_indices=label_to_indices,
        shot=2,
        seed=123,
        way=2,
    )

    assert support_info["way"] == 2
    assert len(support_info["selected_class_ids"]) == 2
    assert len(support_info["selected_indices"]) == 4
    assert set(support_info["selected_by_class"].keys()) == {
        str(class_id) for class_id in support_info["selected_class_ids"]
    }

    args = script_module.parse_args(["--protocol", "fewshot", "--epochs", "7", "--way", "2"])
    args = script_module.enforce_strict_fewshot_protocol(args)

    assert args.epochs == script_module.STRICT_FEWSHOT_EPOCHS == 100
