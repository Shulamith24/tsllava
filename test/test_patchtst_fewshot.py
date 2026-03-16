# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

import numpy as np
import torch

from opentslm.model.PatchTSTClassifier import (
    PatchTSTClassifierAdapter,
    prepare_patchtst_classification_batch,
)
from transformers import PatchTSTConfig, PatchTSTForClassification


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
