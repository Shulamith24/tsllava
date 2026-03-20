# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

from importlib import import_module
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
FEWSHOT_SCRIPT_PATH = REPO_ROOT / "scripts" / "train_ucr_classification_pretrained_fewshot.py"
FULL_SCRIPT_PATH = REPO_ROOT / "scripts" / "train_ucr_classification_pretrained_full.py"
STAGE12_SCRIPT_PATH = REPO_ROOT / "scripts" / "train_curriculum_pretrain_stage12.py"

from opentslm.model.encoder.NewTSDualBranchEncoder import NewTSDualBranchEncoder
from opentslm.model.encoder.NewTSVisionEncoder import NewTSVisionEncoder
from opentslm.model.llm.OpenTSLMSP import OpenTSLMSP
from opentslm.model_config import PATCH_SIZE


def load_script_module(script_path: Path, module_name: str):
    spec = spec_from_file_location(module_name, script_path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_train_ucr_fewshot_script_module():
    return load_script_module(
        FEWSHOT_SCRIPT_PATH,
        "train_ucr_classification_pretrained_fewshot",
    )


def load_train_ucr_full_script_module():
    return load_script_module(
        FULL_SCRIPT_PATH,
        "train_ucr_classification_pretrained_full",
    )


def load_train_curriculum_stage12_script_module():
    return load_script_module(
        STAGE12_SCRIPT_PATH,
        "train_curriculum_pretrain_stage12",
    )


class DummyProcessor:
    def __call__(self, images, return_tensors="pt"):
        return {"pixel_values": torch.zeros(len(images), 3, 32, 32)}


class DummyVisionModel(nn.Module):
    def __init__(self, num_layers: int, hidden_dim: int = 6, num_patches: int = 4):
        super().__init__()
        self.encoder = SimpleNamespace(layer=nn.ModuleList([nn.Linear(1, 1) for _ in range(num_layers)]))
        self.hidden_dim = hidden_dim
        self.num_patches = num_patches

    def forward(self, pixel_values, output_hidden_states=False):
        batch_size = pixel_values.size(0)
        token_count = self.num_patches + 1
        base = torch.arange(token_count * self.hidden_dim, dtype=torch.float32).view(1, token_count, self.hidden_dim)
        hidden_states = tuple(base.repeat(batch_size, 1, 1) + idx for idx in range(len(self.encoder.layer) + 1))
        return SimpleNamespace(
            last_hidden_state=hidden_states[-1],
            hidden_states=hidden_states if output_hidden_states else None,
        )


class DummyTokenizer:
    def __init__(self):
        self.pad_token = None
        self.eos_token = "</s>"

    def __len__(self):
        return 32

    def __call__(
        self,
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        add_special_tokens=True,
    ):
        del return_tensors, padding, truncation, add_special_tokens
        if isinstance(texts, str):
            texts = [texts]

        encoded = []
        for idx, _ in enumerate(texts):
            encoded.append(torch.tensor([idx + 2], dtype=torch.long))

        input_ids = pad_sequence(encoded, batch_first=True, padding_value=0)
        attention_mask = (input_ids != 0).long()
        return SimpleNamespace(input_ids=input_ids, attention_mask=attention_mask)


class DummyLLM(nn.Module):
    def __init__(self, hidden_size: int = 16, vocab_size: int = 32):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def resize_token_embeddings(self, _):
        return self.embed

    def get_input_embeddings(self):
        return self.embed


def install_dummy_vision_loader(monkeypatch):
    vision_module = import_module("opentslm.model.encoder.NewTSVisionEncoder")
    load_calls = []

    def fake_load_vision_backbone(model_name, *, num_hidden_layers=None):
        del model_name
        resolved_layers = num_hidden_layers or 12
        load_calls.append(resolved_layers)
        return DummyProcessor(), DummyVisionModel(resolved_layers), 6, 4, 32, resolved_layers

    monkeypatch.setattr(vision_module, "load_vision_backbone", fake_load_vision_backbone)
    return load_calls


def install_dummy_llm(monkeypatch):
    sp_module = import_module("opentslm.model.llm.OpenTSLMSP")
    monkeypatch.setattr(sp_module.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: DummyTokenizer())
    monkeypatch.setattr(sp_module.AutoModelForCausalLM, "from_pretrained", lambda *args, **kwargs: DummyLLM())


def test_newts_dual_branch_encoder_output_shapes_and_freeze_behavior(monkeypatch):
    install_dummy_vision_loader(monkeypatch)
    x_short = torch.randn(2, 12)
    x_long = torch.randn(2, 20)

    both_encoder = NewTSDualBranchEncoder(
        output_dim=8,
        patch_length=4,
        stride=4,
        d_model=8,
        num_hidden_layers=1,
        ffn_dim=16,
        branch_mode="both",
        freeze_ts_backbone=True,
        freeze_vision_backbone=True,
        device="cpu",
    )
    ts_only_encoder = NewTSDualBranchEncoder(
        output_dim=8,
        patch_length=4,
        stride=4,
        d_model=8,
        num_hidden_layers=1,
        ffn_dim=16,
        branch_mode="ts_only",
        device="cpu",
    )
    vision_only_encoder = NewTSDualBranchEncoder(
        output_dim=8,
        patch_length=4,
        stride=4,
        d_model=8,
        num_hidden_layers=1,
        ffn_dim=16,
        branch_mode="vision_only",
        device="cpu",
    )

    assert both_encoder(x_short).shape == (2, 7, 8)
    assert both_encoder(x_long).shape == (2, 9, 8)
    assert ts_only_encoder(x_short).shape == (2, 3, 8)
    assert ts_only_encoder(x_long).shape == (2, 5, 8)
    assert vision_only_encoder(x_short).shape == (2, 4, 8)
    assert vision_only_encoder(x_long).shape == (2, 4, 8)
    assert all(not param.requires_grad for param in both_encoder.ts_backbone.parameters())
    assert all(not param.requires_grad for param in both_encoder.vision_encoder.vit.parameters())


def test_newts_dual_branch_encoder_with_pma_returns_slot_tokens(monkeypatch):
    install_dummy_vision_loader(monkeypatch)
    x = torch.randn(2, 12)

    both_encoder = NewTSDualBranchEncoder(
        output_dim=8,
        patch_length=4,
        stride=4,
        d_model=8,
        num_hidden_layers=1,
        ffn_dim=16,
        branch_mode="both",
        use_pma=True,
        aggregator_num_queries=3,
        device="cpu",
    )
    ts_only_encoder = NewTSDualBranchEncoder(
        output_dim=8,
        patch_length=4,
        stride=4,
        d_model=8,
        num_hidden_layers=1,
        ffn_dim=16,
        branch_mode="ts_only",
        use_pma=True,
        aggregator_num_queries=3,
        device="cpu",
    )
    vision_only_encoder = NewTSDualBranchEncoder(
        output_dim=8,
        patch_length=4,
        stride=4,
        d_model=8,
        num_hidden_layers=1,
        ffn_dim=16,
        branch_mode="vision_only",
        use_pma=True,
        aggregator_num_queries=3,
        device="cpu",
    )

    assert both_encoder(x).shape == (2, 3, 8)
    assert ts_only_encoder(x).shape == (2, 3, 8)
    assert vision_only_encoder(x).shape == (2, 3, 8)


def test_newts_dual_branch_encoder_with_pma_projects_back_to_output_dim(monkeypatch):
    install_dummy_vision_loader(monkeypatch)
    encoder = NewTSDualBranchEncoder(
        output_dim=8,
        patch_length=4,
        stride=4,
        d_model=8,
        num_hidden_layers=1,
        ffn_dim=16,
        branch_mode="both",
        use_pma=True,
        aggregator_hidden_size=16,
        aggregator_num_heads=4,
        aggregator_num_queries=2,
        device="cpu",
    )

    outputs = encoder(torch.randn(2, 12))

    assert outputs.shape == (2, 2, 8)


def test_newts_vision_encoder_truncates_to_default_layer_4(monkeypatch):
    load_calls = install_dummy_vision_loader(monkeypatch)

    encoder = NewTSVisionEncoder(device="cpu")

    assert load_calls == [4]
    assert encoder.num_hidden_layers == 4
    assert len(encoder.vit.encoder.layer) == 4


def test_newts_vision_encoder_scalar_mix_uses_max_layer(monkeypatch):
    load_calls = install_dummy_vision_loader(monkeypatch)

    encoder = NewTSVisionEncoder(
        feature_mode="scalar_mix",
        mix_layers=[2, 4],
        device="cpu",
    )

    assert load_calls == [4]
    assert encoder.mix_layers == (2, 4)


def test_newts_vision_encoder_rejects_invalid_depth():
    with pytest.raises(ValueError, match="num_hidden_layers must be greater than or equal"):
        NewTSVisionEncoder(
            layer_idx=4,
            num_hidden_layers=3,
            device="cpu",
        )


@pytest.mark.parametrize(
    "script_loader",
    [load_train_ucr_fewshot_script_module, load_train_ucr_full_script_module],
    ids=["fewshot", "full"],
)
def test_opentslmsp_newts_checkpoint_metadata_roundtrip(script_loader, monkeypatch, tmp_path):
    install_dummy_vision_loader(monkeypatch)
    install_dummy_llm(monkeypatch)
    script_module = script_loader()

    model = OpenTSLMSP(
        llm_id="dummy-llm",
        device="cpu",
        encoder_type="newts_dual_branch",
        newts_dual_branch_config={
            "patch_length": 4,
            "stride": 4,
            "d_model": 8,
            "num_attention_heads": 2,
            "num_hidden_layers": 1,
            "ffn_dim": 16,
            "dropout": 0.0,
            "vit_layer_idx": 4,
            "use_pma": True,
            "aggregator_layers": 2,
            "aggregator_hidden_size": 8,
            "aggregator_num_heads": 2,
            "aggregator_ffn_dim": 16,
            "aggregator_num_queries": 3,
            "aggregator_query_mode": "shared",
            "aggregator_fusion_mode": "gated_sum",
            "aggregator_gate_type": "dynamic",
            "aggregator_fuse_layers": 1,
        },
    )

    checkpoint_path = tmp_path / "newts_sp.pt"
    model.store_to_file(str(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    assert checkpoint["model_config"]["llm_id"] == "dummy-llm"
    assert checkpoint["model_config"]["encoder_type"] == "newts_dual_branch"
    assert checkpoint["model_config"]["encoder_config"]["dynamic_length"] is True
    assert checkpoint["model_config"]["encoder_config"]["ts_positional_encoding"] == "sinusoidal"
    assert "context_length" not in checkpoint["model_config"]["encoder_config"]
    assert checkpoint["model_config"]["encoder_config"]["vit_num_hidden_layers"] == 4
    assert checkpoint["model_config"]["encoder_config"]["use_pma"] is True
    assert checkpoint["model_config"]["encoder_config"]["aggregator_num_queries"] == 3

    args = script_module.parse_args(["--local_checkpoint", str(checkpoint_path)])
    args.use_lora = False
    args = script_module.hydrate_args_from_local_checkpoint_metadata(args)
    init_kwargs = script_module.resolve_model_init_kwargs_from_checkpoint(args, checkpoint)

    assert args.encoder_type == "newts_dual_branch"
    assert args.llm_id == "dummy-llm"
    assert init_kwargs["newts_dual_branch_config"]["vit_num_hidden_layers"] == 4
    assert args.use_pma is True
    assert init_kwargs["newts_dual_branch_config"]["aggregator_num_queries"] == 3


@pytest.mark.parametrize(
    "script_loader",
    [load_train_ucr_fewshot_script_module, load_train_ucr_full_script_module],
    ids=["fewshot", "full"],
)
def test_train_ucr_newts_defaults_and_validation(script_loader):
    script_module = script_loader()

    args = script_module.parse_args(["--encoder_type", "newts_dual_branch"])
    script_module.validate_args(args)
    config = script_module.build_newts_dual_branch_config(args)

    assert args.vit_feature_mode == "single"
    assert args.vit_layer_idx == 4
    assert args.vit_truncate_to_feature_layer is True
    assert args.use_pma is False
    assert config["dynamic_length"] is True
    assert config["ts_positional_encoding"] == "sinusoidal"
    assert "context_length" not in config
    assert script_module.resolve_effective_pad_mode(args) == "last"

    explicit_pad_args = script_module.parse_args(
        ["--encoder_type", "newts_dual_branch", "--pad_mode", "repeat"]
    )
    script_module.validate_args(explicit_pad_args)
    assert script_module.resolve_effective_pad_mode(explicit_pad_args) == "repeat"

    invalid_args = script_module.parse_args(
        [
            "--encoder_type",
            "newts_dual_branch",
            "--pretrained_model",
            "OpenTSLM/llama-3.2-1b-m4-sp",
        ]
    )
    with pytest.raises(ValueError, match="cannot be specified together"):
        script_module.validate_args(invalid_args)

    pretrained_only_args = script_module.parse_args(
        ["--pretrained_model", "OpenTSLM/llama-3.2-1b-m4-sp"]
    )
    script_module.validate_args(pretrained_only_args)
    assert pretrained_only_args.encoder_type_explicit is False


def test_train_ucr_fewshot_way_sampling_and_epochs_are_user_controlled():
    script_module = load_train_ucr_fewshot_script_module()

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
    script_module.validate_args(args)
    assert args.epochs == 7

    full_args = script_module.parse_args(["--protocol", "full", "--epochs", "9"])
    script_module.validate_args(full_args)
    assert full_args.epochs == 9


def test_train_ucr_full_resume_state_helper():
    script_module = load_train_ucr_full_script_module()

    resume_state = script_module.resolve_training_resume_state(
        {
            "epoch": 3,
            "best_val_acc": 0.82,
            "patience_counter": 2,
            "loss_history": [{"epoch": 1}, {"epoch": 2}, {"epoch": 3}],
        }
    )

    assert resume_state["start_epoch"] == 4
    assert resume_state["best_val_acc"] == pytest.approx(0.82)
    assert resume_state["patience_counter"] == 2
    assert len(resume_state["loss_history"]) == 3


def test_train_ucr_fewshot_phase_resume_state_helper():
    script_module = load_train_ucr_fewshot_script_module()

    resume_state = script_module.resolve_phase_resume_state(
        {
            "epoch": 2,
            "phase": "phase2_joint",
            "phase_epochs": 5,
            "train_loss": 0.123,
            "loss_history": [0.4, 0.3],
        },
        phase_name="phase2_joint",
        phase_epochs=5,
    )

    assert resume_state["completed_epoch"] == 2
    assert resume_state["start_epoch"] == 3
    assert resume_state["last_loss"] == pytest.approx(0.123)
    assert resume_state["loss_history"] == [0.4, 0.3]
    assert resume_state["is_complete"] is False

    completed_state = script_module.resolve_phase_resume_state(
        {"epoch": 5, "phase": "phase2_joint", "phase_epochs": 5},
        phase_name="phase2_joint",
        phase_epochs=5,
    )
    assert completed_state["is_complete"] is True


@pytest.mark.parametrize(
    "script_loader",
    [load_train_ucr_fewshot_script_module, load_train_ucr_full_script_module],
    ids=["fewshot", "full"],
)
def test_train_ucr_newts_pma_config_and_validation(script_loader):
    script_module = script_loader()

    args = script_module.parse_args(
        [
            "--encoder_type",
            "newts_dual_branch",
            "--use_pma",
            "--aggregator_hidden_size",
            "16",
            "--aggregator_num_heads",
            "4",
            "--aggregator_ffn_dim",
            "32",
            "--aggregator_num_queries",
            "3",
            "--aggregator_query_mode",
            "separate",
            "--aggregator_fusion_mode",
            "concat_linear",
            "--aggregator_gate_type",
            "slot",
            "--aggregator_fuse_layers",
            "2",
        ]
    )
    script_module.validate_args(args)
    config = script_module.build_newts_dual_branch_config(args)

    assert config["use_pma"] is True
    assert config["dynamic_length"] is True
    assert config["aggregator_hidden_size"] == 16
    assert config["aggregator_num_heads"] == 4
    assert config["aggregator_ffn_dim"] == 32
    assert config["aggregator_num_queries"] == 3
    assert config["aggregator_query_mode"] == "separate"
    assert config["aggregator_fusion_mode"] == "concat_linear"
    assert config["aggregator_gate_type"] == "slot"
    assert config["aggregator_fuse_layers"] == 2

    invalid_args = script_module.parse_args(
        [
            "--encoder_type",
            "newts_dual_branch",
            "--use_pma",
            "--aggregator_hidden_size",
            "10",
            "--aggregator_num_heads",
            "4",
        ]
    )
    with pytest.raises(ValueError, match="must evenly divide"):
        script_module.validate_args(invalid_args)


def test_train_ucr_full_resolve_collate_patch_size():
    script_module = load_train_ucr_full_script_module()

    transformer_args = script_module.parse_args(["--encoder_type", "transformer_cnn"])
    tslanet_args = script_module.parse_args(["--encoder_type", "tslanet", "--tslanet_patch_size", "16"])
    newts_args = script_module.parse_args(["--encoder_type", "newts_dual_branch"])

    assert script_module.resolve_collate_patch_size(transformer_args) == PATCH_SIZE
    assert script_module.resolve_collate_patch_size(tslanet_args) == 16
    assert script_module.resolve_collate_patch_size(newts_args) == 1


def test_opentslmsp_pad_and_apply_batch_with_pma_tokens(monkeypatch):
    install_dummy_vision_loader(monkeypatch)
    install_dummy_llm(monkeypatch)

    model = OpenTSLMSP(
        llm_id="dummy-llm",
        device="cpu",
        encoder_type="newts_dual_branch",
        newts_dual_branch_config={
            "patch_length": 4,
            "stride": 4,
            "d_model": 8,
            "num_attention_heads": 2,
            "num_hidden_layers": 1,
            "ffn_dim": 16,
            "dropout": 0.0,
            "vit_layer_idx": 4,
            "use_pma": True,
            "aggregator_hidden_size": 8,
            "aggregator_num_heads": 2,
            "aggregator_ffn_dim": 16,
            "aggregator_num_queries": 2,
        },
    )

    batch = [
        {
            "pre_prompt": "Classify",
            "time_series_text": ["Signal"],
            "time_series": [torch.randn(10)],
            "post_prompt": "Answer",
        },
        {
            "pre_prompt": "Classify",
            "time_series_text": ["Signal"],
            "time_series": [torch.randn(12)],
            "post_prompt": "Answer",
        },
    ]

    inputs_embeds, attention_mask = model.pad_and_apply_batch(batch)

    assert inputs_embeds.shape == (2, 5, model.llm.config.hidden_size)
    assert attention_mask.shape == (2, 5)
    assert torch.all(attention_mask == 1)


@pytest.mark.parametrize(
    "script_loader",
    [
        load_train_ucr_fewshot_script_module,
        load_train_ucr_full_script_module,
        load_train_curriculum_stage12_script_module,
    ],
    ids=["fewshot", "full", "stage12"],
)
def test_newts_context_length_flag_warns_but_is_ignored(script_loader, capsys):
    script_module = script_loader()

    args = script_module.parse_args(
        ["--encoder_type", "newts_dual_branch", "--context_length", "128"]
    )
    script_module.validate_args(args)
    script_module.warn_deprecated_newts_context_length(args, rank=0)
    captured = capsys.readouterr()

    assert "deprecated" in captured.out
    assert script_module.build_newts_dual_branch_config(args)["dynamic_length"] is True


def test_stage12_length_bucket_batch_sampler_groups_similar_lengths():
    script_module = load_train_curriculum_stage12_script_module()

    class DummyDataset:
        def __init__(self, lengths):
            self.lengths = list(lengths)

        def __len__(self):
            return len(self.lengths)

        def __getitem__(self, idx):
            return {"time_series": [torch.arange(self.lengths[idx])]}

    dataset = DummyDataset([64, 512, 128, 256, 68, 500, 132, 260])
    sampler = script_module.LengthBucketBatchSampler(
        dataset,
        batch_size=2,
        shuffle=False,
        num_replicas=1,
    )

    batches = list(iter(sampler))
    batch_lengths = [[dataset.lengths[idx] for idx in batch] for batch in batches]

    assert batch_lengths == [[64, 68], [128, 132], [256, 260], [500, 512]]


def test_stage12_newts_run_name_and_pad_mode_defaults():
    script_module = load_train_curriculum_stage12_script_module()

    args = script_module.parse_args(["--encoder_type", "newts_dual_branch", "--branch_mode", "both"])

    assert script_module.default_run_name(args) == "newts_dual_branch_both_dynamic"
    assert script_module.resolve_effective_pad_mode(args) == "last"
