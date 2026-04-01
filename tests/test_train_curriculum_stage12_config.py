from __future__ import annotations

import contextlib
import io
import unittest

from scripts.train_curriculum_pretrain_stage12 import (
    effective_config_snapshot,
    parse_args,
    resolve_effective_newts_vision_config,
    warn_newts_vision_runtime_config,
)


class CurriculumStage12VisionConfigTest(unittest.TestCase):
    def test_stage12_defaults_to_tivit_effective_stride(self) -> None:
        args = parse_args(["--encoder_type", "newts_dual_branch"])

        resolved = resolve_effective_newts_vision_config(args)
        snapshot = effective_config_snapshot(args, world_size=2)

        self.assertEqual(args.vision_2d_mode, "tivit_sqrt_overlap")
        self.assertFalse(args.vit_stride_explicit)
        self.assertEqual(resolved["effective_vit_stride"], 0.1)
        self.assertEqual(snapshot["effective_vit_stride"], 0.1)
        self.assertEqual(snapshot["effective_vit_patch_policy"], "sqrt_time_length")

    def test_stage12_warns_when_tivit_patch_size_is_explicit(self) -> None:
        args = parse_args(
            [
                "--encoder_type",
                "newts_dual_branch",
                "--vit_patch_size",
                "32",
            ]
        )

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            warn_newts_vision_runtime_config(args, rank=0)
        output = buffer.getvalue()

        snapshot = effective_config_snapshot(args, world_size=1)
        self.assertIn("--vit_patch_size is ignored", output)
        self.assertEqual(snapshot["effective_vit_patch_policy"], "sqrt_time_length")
        self.assertEqual(snapshot["effective_vit_stride"], 0.1)


if __name__ == "__main__":
    unittest.main()
