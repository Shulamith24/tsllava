from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from scripts.train_curriculum_pretrain_stage0_tsqa_m4 import (
    DualViewSSLModel,
    LengthBucketBatchSampler,
    STAGE2_CANONICAL_NAME,
    STAGE_ORDER,
    configure_sp_trainable_parameters,
    export_stage_alias_checkpoint,
    load_stage0_encoder_into_sp,
    parse_args,
    save_sp_checkpoint,
    sanitize_checkpoint_metadata,
    stage_dependency_candidates,
)
from scripts.train_ucr_classification_pretrained_fewshot import (
    extract_sp_component_states_from_checkpoint,
    parse_args as parse_fewshot_args,
    resolve_model_init_kwargs_from_checkpoint,
)
from opentslm.time_series_datasets.curriculum_pretrain_aux import (
    AlignmentTargetDataset,
    DEFAULT_SYNTHETIC_SAMPLE_TYPES,
    MixedPretrainDataset,
    load_ucr_train_raw_records,
)


class _DummySPModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(3, 3)
        self.projector = nn.Linear(3, 3)
        self.lora_enabled = False

    def get_checkpoint_metadata(self):
        return {
            "llm_id": "meta-llama/Llama-3.2-1B",
            "encoder_type": "newts_dual_branch",
            "encoder_config": {
                "patch_length": 16,
                "stride": 8,
                "d_model": 128,
                "num_attention_heads": 8,
                "num_hidden_layers": 3,
                "ffn_dim": 512,
                "dropout": 0.1,
                "branch_mode": "both",
                "vit_model_name": "facebook/dinov2-base",
                "vit_feature_mode": "single",
                "vit_layer_idx": 4,
                "vit_mix_layers": None,
                "vit_patch_size": 16,
                "vit_stride": 0.5,
                "vision_2d_mode": "tivit_sqrt_overlap",
                "vit_truncate_to_feature_layer": True,
                "vit_num_hidden_layers": 4,
                "projector_type": "mlp",
                "projector_dropout": 0.1,
                "use_pma": False,
                "aggregator_layers": 2,
                "aggregator_hidden_size": 128,
                "aggregator_num_heads": 8,
                "aggregator_ffn_dim": 512,
                "aggregator_num_queries": 2,
                "aggregator_query_mode": "separate",
                "aggregator_fusion_mode": "concat_linear",
                "aggregator_gate_type": "dynamic",
                "aggregator_fuse_layers": 1,
                "branch_dropout": 0.15,
                "enable_modality_embeddings": False,
                "freeze_ts_backbone": False,
                "freeze_vision_backbone": True,
            },
            "alignment_losses_enabled": True,
            "loss_w_align": 0.2,
        }

    def save_lora_state_to_checkpoint(self, checkpoint):
        checkpoint["lora_enabled"] = False


class _DummyEncoder(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.proj = nn.Linear(3, 3)
        self._config = dict(config or {"patch_length": 16, "stride": 8, "branch_mode": "both"})

    def get_config(self):
        return dict(self._config)


class _DummyStage0Model(nn.Module):
    def __init__(self, encoder_config=None):
        super().__init__()
        self.encoder = _DummyEncoder(encoder_config)


class _DummyPromptDataset:
    def __init__(self, answer: str):
        self.answer = answer

    def __len__(self):
        return 1

    def get_sample_length(self, idx: int) -> int:
        return 4

    def __getitem__(self, idx: int):
        return {
            "pre_prompt": "pre",
            "post_prompt": "post",
            "answer": self.answer,
            "time_series": [torch.ones(4)],
            "time_series_text": ["ts"],
        }


class _DummySyntheticDataset:
    def __len__(self):
        return 1

    def get_sample_length(self, idx: int) -> int:
        return 4

    def __getitem__(self, idx: int):
        return {
            "pre_prompt": "pre",
            "post_prompt": "post",
            "answer": "caption<eos>",
            "time_series": [torch.ones(4)],
            "time_series_text": ["ts"],
            "alignment_target_text": "caption",
            "source_name": "synthetic_attribute",
        }


class _DummyPatchBackbone(nn.Module):
    def __init__(self, patch_length: int, stride: int):
        super().__init__()
        self.patch_length = int(patch_length)
        self.stride = int(stride)
        self.mix = nn.Linear(self.patch_length, self.patch_length)

    def _extract_patches(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        if x.size(1) < self.patch_length:
            x = F.pad(x, (0, self.patch_length - x.size(1)))
        if x.size(1) == self.patch_length:
            patches = x.unsqueeze(1)
        else:
            patches = x.unfold(dimension=1, size=self.patch_length, step=self.stride).contiguous()
        return self.mix(patches)


class _DummyVisionBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = nn.Linear(1, 1)


class _DummyStage0EncoderModule(nn.Module):
    def __init__(self, **config):
        super().__init__()
        self.output_dim = int(config.get("output_dim", 8))
        self.patch_length = int(config.get("patch_length", 4))
        self.stride = int(config.get("stride", 2))
        self.branch_mode = str(config.get("branch_mode", "both"))
        self.branch_dropout = float(config.get("branch_dropout", 0.0))
        self.freeze_ts_backbone_default = bool(config.get("freeze_ts_backbone", False))
        self.freeze_vision_backbone_default = bool(config.get("freeze_vision_backbone", True))
        self.ts_backbone = _DummyPatchBackbone(self.patch_length, self.stride)
        self.vision_encoder = _DummyVisionBranch()
        self.ts_projector = nn.Linear(self.patch_length, self.output_dim)
        self.vision_projector = nn.Linear(self.patch_length, self.output_dim)
        self._config = {
            "output_dim": self.output_dim,
            "patch_length": self.patch_length,
            "stride": self.stride,
            "branch_mode": self.branch_mode,
            "projector_type": config.get("projector_type", "mlp"),
            "projector_dropout": config.get("projector_dropout", 0.1),
            "use_pma": False,
            "aggregator_layers": 2,
            "aggregator_hidden_size": self.output_dim,
            "aggregator_num_heads": 2,
            "aggregator_ffn_dim": self.output_dim * 4,
            "aggregator_num_queries": 2,
            "aggregator_query_mode": "separate",
            "aggregator_fusion_mode": "concat_linear",
            "aggregator_gate_type": "dynamic",
            "aggregator_fuse_layers": 1,
            "enable_modality_embeddings": config.get("enable_modality_embeddings", False),
            "branch_dropout": self.branch_dropout,
            "vision_train_mode": "none",
            "vision_topk_blocks": 4,
            "freeze_ts_backbone": self.freeze_ts_backbone_default,
            "freeze_vision_backbone": self.freeze_vision_backbone_default,
            "num_attention_heads": 2,
            "num_hidden_layers": 1,
            "ffn_dim": self.output_dim * 4,
            "dropout": 0.1,
            "vit_model_name": "dummy-vit",
            "vit_feature_mode": "single",
            "vit_layer_idx": 1,
            "vit_mix_layers": [],
            "vit_patch_size": self.patch_length,
            "vit_stride": 0.1,
            "vision_2d_mode": "tivit_sqrt_overlap",
            "vit_truncate_to_feature_layer": True,
            "vit_num_hidden_layers": 1,
            "dynamic_length": True,
            "ts_positional_encoding": "sinusoidal",
            "d_model": self.output_dim,
        }

        if self.freeze_ts_backbone_default:
            for param in self.ts_backbone.parameters():
                param.requires_grad = False
        if self.freeze_vision_backbone_default:
            for param in self.vision_encoder.vit.parameters():
                param.requires_grad = False

    def _resolve_runtime_branch_mode(self, runtime_branch_mode: str) -> str:
        mode = str(runtime_branch_mode).lower()
        if mode != "both" or not self.training or self.branch_dropout <= 0.0:
            return mode
        draw = torch.rand(1).item()
        if draw < self.branch_dropout:
            return "ts_only"
        if draw < 2 * self.branch_dropout:
            return "vision_only"
        return "both"

    def forward(self, x: torch.Tensor, *, runtime_branch_mode: str = "both", return_intermediates: bool = False):
        patches = self.ts_backbone._extract_patches(x)
        ts_tokens_all = self.ts_projector(patches)
        vision_tokens_all = self.vision_projector(patches + 0.25)
        effective_mode = self._resolve_runtime_branch_mode(runtime_branch_mode)

        ts_tokens = ts_tokens_all
        vision_tokens = vision_tokens_all
        if effective_mode == "ts_only":
            vision_tokens = None
        elif effective_mode == "vision_only":
            ts_tokens = None

        if ts_tokens is None:
            fused_tokens = vision_tokens
        elif vision_tokens is None:
            fused_tokens = ts_tokens
        else:
            fused_tokens = torch.cat([ts_tokens, vision_tokens], dim=1)

        if not return_intermediates:
            return fused_tokens

        pooled = lambda tokens: tokens.mean(dim=1) if tokens is not None else None
        return {
            "ts_tokens": ts_tokens,
            "vision_tokens": vision_tokens,
            "fused_tokens": fused_tokens,
            "pooled_ts": pooled(ts_tokens),
            "pooled_vision": pooled(vision_tokens),
            "pooled_fused": pooled(fused_tokens),
            "effective_branch_mode": effective_mode,
        }

    def get_config(self):
        return dict(self._config)


class CurriculumPretrainV2Test(unittest.TestCase):
    def test_parser_understands_new_stages_and_dependencies(self) -> None:
        args = parse_args([])
        self.assertEqual(args.stages, STAGE_ORDER)
        self.assertEqual(stage_dependency_candidates("stage1_tsqa_transfer"), ["stage0_encoder_ssl"])
        self.assertEqual(stage_dependency_candidates("stage2_synthetic_semantics"), ["stage1_tsqa_transfer"])
        self.assertEqual(
            stage_dependency_candidates("stage3_m4_caption"),
            [STAGE2_CANONICAL_NAME, "stage1_tsqa_transfer"],
        )
        alias_args = parse_args(["--stages", "stage0_encoder_ssl,stage1_tsqa_transfer,stage2_synthetic_semantics"])
        self.assertEqual(alias_args.stages[-1], STAGE2_CANONICAL_NAME)

    def test_stage2_default_synthetic_sample_types_exclude_match_mismatch(self) -> None:
        args = parse_args([])
        self.assertEqual(args.stage2_synthetic_sample_types, DEFAULT_SYNTHETIC_SAMPLE_TYPES)
        self.assertNotIn("match_mismatch", args.stage2_synthetic_sample_types)
        self.assertEqual(args.stage0_mix_weights, (2, 1, 2))
        self.assertEqual(args.stage0_branch_dropout, 0.1)
        self.assertEqual(args.stage0_w_ts_recon, 1.0)
        self.assertEqual(args.stage0_w_ts_vicreg, 0.25)
        self.assertEqual(args.stage0_w_vi_vicreg, 0.25)
        self.assertEqual(args.stage0_w_fuse_vicreg, 0.5)
        self.assertEqual(args.stage2_mix_weights, (1, 1, 2))
        self.assertEqual(args.stage0_downstream_pool, "ucr_train_list")
        self.assertFalse(args.stage0_train_vision)
        self.assertEqual(args.stage0_epochs, 8)
        self.assertEqual(args.stage1_epochs, 6)
        self.assertEqual(args.stage1_lr_encoder, 5e-5)
        self.assertEqual(args.stage1_lr_projector, 1e-4)
        self.assertEqual(args.stage1_projector_only_epochs, 2)
        self.assertEqual(args.stage2_epochs, 6)
        self.assertEqual(args.stage3_epochs, 12)

    def test_stage1_checkpoint_metadata_is_sanitized(self) -> None:
        metadata = sanitize_checkpoint_metadata(_DummySPModel().get_checkpoint_metadata())
        self.assertNotIn("branch_dropout", metadata["encoder_config"])
        self.assertNotIn("enable_modality_embeddings", metadata["encoder_config"])
        self.assertNotIn("alignment_losses_enabled", metadata)
        self.assertNotIn("loss_w_align", metadata)

        with tempfile.TemporaryDirectory() as tmpdir:
            model = _DummySPModel()
            optimizer = AdamW(model.parameters(), lr=1e-3)
            scheduler = LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
            checkpoint_path = f"{tmpdir}/best_model.pt"
            save_sp_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=3,
                train_loss=0.7,
                metrics={"loss_total": 0.5, "accuracy": 0.8},
                save_path=checkpoint_path,
                args=parse_args([]),
                stage_name="stage1_tsqa_transfer",
                rank=0,
            )
            saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            self.assertNotIn("branch_dropout", saved["model_config"]["encoder_config"])
            self.assertNotIn("alignment_losses_enabled", saved["model_config"])

    def test_stage_alias_checkpoint_is_fewshot_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            alias_path = f"{tmpdir}/stage1_transfer_checkpoint.pt"
            model = _DummySPModel()
            export_stage_alias_checkpoint(model, alias_path, rank=0)
            checkpoint = torch.load(alias_path, map_location="cpu", weights_only=False)

            self.assertIn("model_config", checkpoint)
            self.assertIn("encoder_state", checkpoint)
            self.assertIn("projector_state", checkpoint)
            self.assertNotIn("optimizer_state", checkpoint)
            self.assertNotIn("branch_dropout", checkpoint["model_config"]["encoder_config"])

            fewshot_args = parse_fewshot_args([])
            init_kwargs = resolve_model_init_kwargs_from_checkpoint(fewshot_args, checkpoint)
            encoder_state, projector_state = extract_sp_component_states_from_checkpoint(checkpoint)

            self.assertEqual(init_kwargs["encoder_type"], "newts_dual_branch")
            self.assertEqual(init_kwargs["llm_id"], "meta-llama/Llama-3.2-1B")
            self.assertNotIn("branch_dropout", init_kwargs["newts_dual_branch_config"])
            self.assertTrue(encoder_state)
            self.assertTrue(projector_state)

    def test_stage0_encoder_loading_is_strict_for_matching_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model = _DummyStage0Model()
            checkpoint_path = f"{tmpdir}/best_encoder.pt"
            torch.save(
                {
                    "encoder_state": model.encoder.state_dict(),
                    "model_config": {
                        "encoder_config": {
                            **model.encoder.get_config(),
                            "branch_dropout": 0.1,
                            "enable_modality_embeddings": True,
                        }
                    },
                },
                checkpoint_path,
            )

            load_info = load_stage0_encoder_into_sp(model, checkpoint_path, device="cpu")
            self.assertEqual(load_info["missing_keys"], [])
            self.assertEqual(load_info["unexpected_keys"], [])
            self.assertEqual(load_info["dropped_stage0_only_keys"], [])

    def test_stage0_encoder_loading_ignores_stage0_only_fused_pool_proj_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model = _DummyStage0Model()
            checkpoint_path = f"{tmpdir}/best_encoder.pt"
            checkpoint_state = dict(model.encoder.state_dict())
            checkpoint_state["fused_pool_proj.0.weight"] = torch.ones(3)
            checkpoint_state["fused_pool_proj.0.bias"] = torch.zeros(3)
            checkpoint_state["fused_pool_proj.1.weight"] = torch.ones(3, 3)
            checkpoint_state["fused_pool_proj.1.bias"] = torch.zeros(3)
            torch.save(
                {
                    "encoder_state": checkpoint_state,
                    "model_config": {"encoder_config": model.encoder.get_config()},
                },
                checkpoint_path,
            )

            load_info = load_stage0_encoder_into_sp(model, checkpoint_path, device="cpu")
            self.assertEqual(load_info["missing_keys"], [])
            self.assertEqual(load_info["unexpected_keys"], [])
            self.assertEqual(
                load_info["dropped_stage0_only_keys"],
                [
                    "fused_pool_proj.0.weight",
                    "fused_pool_proj.0.bias",
                    "fused_pool_proj.1.weight",
                    "fused_pool_proj.1.bias",
                ],
            )

    def test_stage0_loss_structure_is_finite_and_freezes_only_vision_backbone(self) -> None:
        encoder_config = {
            "output_dim": 8,
            "patch_length": 4,
            "stride": 2,
            "branch_mode": "both",
            "projector_type": "mlp",
            "projector_dropout": 0.1,
            "freeze_ts_backbone": False,
            "freeze_vision_backbone": True,
            "branch_dropout": 0.1,
        }
        with patch("scripts.train_curriculum_pretrain_stage0_tsqa_m4.NewTSDualBranchEncoder", _DummyStage0EncoderModule):
            model = DualViewSSLModel(
                encoder_config=encoder_config,
                device="cpu",
                train_vision=False,
                mask_ratio=0.25,
                jitter_std=0.01,
                scaling_range=(0.9, 1.1),
                time_mask_ratio=0.1,
                loss_w_ts_recon=1.0,
                loss_w_ts_vicreg=0.25,
                loss_w_vi_vicreg=0.25,
                loss_w_fuse_vicreg=0.5,
            )
            model.train()
            batch = {"series": torch.randn(4, 12)}
            losses = model.compute_losses(batch)
            for key in [
                "loss_total",
                "loss_ts_recon",
                "loss_ts_vicreg",
                "loss_vi_vicreg",
                "loss_fuse_vicreg",
            ]:
                self.assertIn(key, losses)
                self.assertTrue(torch.isfinite(losses[key]).item(), key)

            losses["loss_total"].backward()
            vit_grads = [param.grad for param in model.encoder.vision_encoder.vit.parameters()]
            projector_grads = [param.grad for param in model.encoder.vision_projector.parameters()]
            self.assertTrue(all(grad is None for grad in vit_grads))
            self.assertTrue(any(grad is not None and torch.isfinite(grad).all() for grad in projector_grads))

    def test_stage1_projector_only_epochs_toggle_encoder_trainability(self) -> None:
        args = parse_args([])
        model = _DummySPModel()

        model._curriculum_stage_epoch = 1
        configure_sp_trainable_parameters(model, args, "stage1_tsqa_transfer", rank=0)
        self.assertTrue(all(not param.requires_grad for param in model.encoder.parameters()))
        self.assertTrue(all(param.requires_grad for param in model.projector.parameters()))

        model._curriculum_stage_epoch = args.stage1_projector_only_epochs + 1
        configure_sp_trainable_parameters(model, args, "stage1_tsqa_transfer", rank=0)
        self.assertTrue(all(param.requires_grad for param in model.encoder.parameters()))

    def test_stage2_alignment_datasets_mark_targets_by_source_role(self) -> None:
        tsqa_dataset = AlignmentTargetDataset(
            _DummyPromptDataset(answer="(a)<eos>"),
            eos_token="<eos>",
            source_name="tsqa",
            alignment_from_answer=False,
        )
        m4_dataset = AlignmentTargetDataset(
            _DummyPromptDataset(answer="caption<eos>"),
            eos_token="<eos>",
            source_name="m4",
            alignment_from_answer=True,
        )
        synthetic_dataset = _DummySyntheticDataset()
        mixed = MixedPretrainDataset(
            [tsqa_dataset, m4_dataset, synthetic_dataset],
            [1, 1, 1],
            seed=0,
            epoch_size=3,
        )

        samples = [mixed[idx] for idx in range(len(mixed))]
        samples_by_source = {sample["source_name"]: sample for sample in samples}

        self.assertIsNone(samples_by_source["tsqa"]["alignment_target_text"])
        self.assertEqual(samples_by_source["m4"]["alignment_target_text"], "caption")
        self.assertEqual(samples_by_source["synthetic_attribute"]["alignment_target_text"], "caption")

    def test_ucr_train_list_skips_comment_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            list_path = Path(tmpdir) / "datasets.txt"
            list_path.write_text(
                "# UCR训练数据集列表\n# 注释应被跳过\nACSF1\n\nAdiac\n",
                encoding="utf-8",
            )

            calls = []

            def fake_load_ucr_dataset(dataset_name, raw_data_path="./data"):
                calls.append(dataset_name)
                return (
                    __import__("pandas").DataFrame([{"label": 0, "t1": 1.0, "t2": 2.0}]),
                    __import__("pandas").DataFrame([{"label": 0, "t1": 1.0, "t2": 2.0}]),
                )

            with patch("opentslm.time_series_datasets.curriculum_pretrain_aux.load_ucr_dataset", side_effect=fake_load_ucr_dataset):
                records = load_ucr_train_raw_records(
                    raw_data_path="./data",
                    dataset_list_path=str(list_path),
                )

            self.assertEqual(calls, ["ACSF1", "Adiac"])
            self.assertEqual(len(records), 2)

    def test_length_bucket_sampler_prefers_fast_length_api(self) -> None:
        class _FastLengthDataset:
            def __len__(self) -> int:
                return 4

            def get_sample_length(self, idx: int) -> int:
                return [9, 3, 6, 12][idx]

            def __getitem__(self, idx: int):
                raise AssertionError("__getitem__ should not be used for length inference")

        sampler = LengthBucketBatchSampler(
            _FastLengthDataset(),
            batch_size=2,
            shuffle=False,
        )

        self.assertEqual(sampler.sample_lengths, [9, 3, 6, 12])
        self.assertEqual(list(iter(sampler)), [[1, 2], [0, 3]])


if __name__ == "__main__":
    unittest.main()
