from __future__ import annotations

import tempfile
import unittest

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from scripts.train_curriculum_pretrain_stage0_tsqa_m4 import (
    STAGE_ORDER,
    export_stage_alias_checkpoint,
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
from opentslm.time_series_datasets.curriculum_pretrain_aux import DEFAULT_SYNTHETIC_SAMPLE_TYPES


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


class CurriculumPretrainV2Test(unittest.TestCase):
    def test_parser_understands_new_stages_and_dependencies(self) -> None:
        args = parse_args([])
        self.assertEqual(args.stages, STAGE_ORDER)
        self.assertEqual(stage_dependency_candidates("stage1_tsqa_transfer"), ["stage0_encoder_ssl"])
        self.assertEqual(stage_dependency_candidates("stage2_synthetic_semantics"), ["stage1_tsqa_transfer"])
        self.assertEqual(
            stage_dependency_candidates("stage3_m4_caption"),
            ["stage2_synthetic_semantics", "stage1_tsqa_transfer"],
        )

    def test_stage2_default_synthetic_sample_types_exclude_match_mismatch(self) -> None:
        args = parse_args([])
        self.assertEqual(args.stage2_synthetic_sample_types, DEFAULT_SYNTHETIC_SAMPLE_TYPES)
        self.assertNotIn("match_mismatch", args.stage2_synthetic_sample_types)

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


if __name__ == "__main__":
    unittest.main()
