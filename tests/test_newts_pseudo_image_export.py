from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from scripts.visualization.export_newts_pseudo_image import (
    NewTSPseudoImageTransform,
    main as export_main,
)


def _write_ucr_dataset(root: Path, dataset_name: str) -> Path:
    archive = root / "data" / "UCRArchive_2018"
    dataset_dir = archive / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    train_rows = [
        [0, 0.00, 0.10, 0.00, 0.10, 0.00, 0.10, 0.00, 0.10],
        [1, 1.00, 0.90, 1.00, 0.90, 1.00, 0.90, 1.00, 0.90],
    ]
    test_rows = [
        [0, 0.05, 0.15, 0.05, 0.15, 0.05, 0.15, 0.05, 0.15],
        [1, 0.95, 0.85, 0.95, 0.85, 0.95, 0.85, 0.95, 0.85],
    ]

    def _write_rows(path: Path, rows: list[list[float]]) -> None:
        lines = ["\t".join(str(value) for value in row) for row in rows]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    _write_rows(dataset_dir / f"{dataset_name}_TRAIN.tsv", train_rows)
    _write_rows(dataset_dir / f"{dataset_name}_TEST.tsv", test_rows)
    return archive


class NewTSPseudoImageExportTest(unittest.TestCase):
    def test_transform_exposes_grayscale_and_rgb_views(self) -> None:
        transform = NewTSPseudoImageTransform(
            ts_patch_size=4,
            ts_stride=0.5,
            vision_2d_mode="reshape_serpentine",
            image_size=8,
        )
        time_series = torch.arange(10, dtype=torch.float32).view(1, 10, 1)

        grid = transform.ts2grid(time_series)
        gray = transform.ts2grayscale_image(time_series)
        rgb = transform.ts2image(time_series)

        self.assertEqual(tuple(grid.shape), (1, 1, 3, 4))
        self.assertGreater(float(grid[0, 0, 1, 0]), float(grid[0, 0, 1, -1]))
        self.assertEqual(tuple(gray.shape), (1, 1, 8, 8))
        self.assertEqual(tuple(rgb.shape), (1, 3, 8, 8))
        self.assertTrue(torch.allclose(rgb[:, 0], gray[:, 0]))
        self.assertTrue(torch.allclose(rgb[:, 1], gray[:, 0]))
        self.assertTrue(torch.allclose(rgb[:, 2], gray[:, 0]))

    def test_script_exports_expected_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            archive = _write_ucr_dataset(tmp_path, "ToyWave")
            output_dir = tmp_path / "results"

            metadata = export_main(
                [
                    "--dataset",
                    "ToyWave",
                    "--data_path",
                    str(archive.parent),
                    "--split",
                    "train",
                    "--sample_index",
                    "0",
                    "--image_size",
                    "32",
                    "--output_dir",
                    str(output_dir),
                ]
            )

            grid_path = Path(metadata["artifacts"]["pseudo_image_grid_png"])
            resized_path = Path(metadata["artifacts"]["pseudo_image_resized_png"])
            metadata_path = Path(metadata["artifacts"]["metadata_json"])

            self.assertTrue(grid_path.exists())
            self.assertTrue(resized_path.exists())
            self.assertTrue(metadata_path.exists())
            self.assertEqual(metadata["transform_config"]["vision_2d_mode"], "reshape_serpentine")
            self.assertEqual(metadata["resized_grid_shape"], [32, 32])

            with Image.open(grid_path) as image:
                self.assertEqual(image.mode, "L")
                self.assertGreaterEqual(min(image.size), 256)

            with Image.open(resized_path) as image:
                self.assertEqual(image.mode, "L")
                self.assertEqual(image.size, (256, 256))

    def test_script_can_hydrate_transform_config_from_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            archive = _write_ucr_dataset(tmp_path, "ToyWave")
            checkpoint_path = tmp_path / "demo_checkpoint.pt"
            torch.save(
                {
                    "model_config": {
                        "llm_id": "demo-llm",
                        "encoder_type": "newts_dual_branch",
                        "encoder_config": {
                            "vit_patch_size": 6,
                            "vit_stride": 1.0,
                            "vision_2d_mode": "legacy_unfold",
                        },
                    }
                },
                checkpoint_path,
            )

            metadata = export_main(
                [
                    "--dataset",
                    "ToyWave",
                    "--data_path",
                    str(archive.parent),
                    "--split",
                    "test",
                    "--sample_index",
                    "1",
                    "--image_size",
                    "24",
                    "--local_checkpoint",
                    str(checkpoint_path),
                    "--output_dir",
                    str(tmp_path / "results"),
                ]
            )

            self.assertEqual(metadata["transform_config"]["vit_patch_size"], 6)
            self.assertEqual(metadata["transform_config"]["vit_stride"], 1.0)
            self.assertEqual(metadata["transform_config"]["vision_2d_mode"], "legacy_unfold")
            self.assertEqual(metadata["checkpoint_metadata"]["llm_id"], "demo-llm")


if __name__ == "__main__":
    unittest.main()
