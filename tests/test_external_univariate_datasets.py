from __future__ import annotations

import unittest

import numpy as np

from opentslm.time_series_datasets.mitbih.mitbih_loader import (
    DE_CHAZAL_DS1_RECORDS,
    DE_CHAZAL_DS2_RECORDS,
    extract_centered_window,
    map_mitbih_symbol_to_aami,
)
from opentslm.time_series_datasets.sleep.sleepedf_classification_loader import (
    _normalize_sleepedf_channel_label,
    expand_sleep_stage_annotations,
    extract_sleepedf_subject_id,
    normalize_sleep_stage,
    sleepedf_pair_key,
    split_sleepedf_subject_ids,
)


class ExternalUnivariateDatasetHelpersTest(unittest.TestCase):
    def test_mitbih_de_chazal_split_is_disjoint(self) -> None:
        self.assertEqual(len(DE_CHAZAL_DS1_RECORDS), 22)
        self.assertEqual(len(DE_CHAZAL_DS2_RECORDS), 22)
        self.assertTrue(set(DE_CHAZAL_DS1_RECORDS).isdisjoint(DE_CHAZAL_DS2_RECORDS))

    def test_mitbih_extract_centered_window_pads_boundaries(self) -> None:
        signal = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

        left_window = extract_centered_window(signal, 0, window_size=4)
        right_window = extract_centered_window(signal, 3, window_size=4)

        np.testing.assert_allclose(left_window, np.asarray([0.0, 0.0, 1.0, 2.0], dtype=np.float32))
        np.testing.assert_allclose(right_window, np.asarray([2.0, 3.0, 4.0, 0.0], dtype=np.float32))

    def test_mitbih_aami_mapping_is_stable(self) -> None:
        self.assertEqual(map_mitbih_symbol_to_aami("N"), "N")
        self.assertEqual(map_mitbih_symbol_to_aami("A"), "S")
        self.assertEqual(map_mitbih_symbol_to_aami("V"), "V")
        self.assertEqual(map_mitbih_symbol_to_aami("F"), "F")
        self.assertEqual(map_mitbih_symbol_to_aami("/"), "Q")
        self.assertIsNone(map_mitbih_symbol_to_aami("|"))

    def test_sleepedf_subject_key_groups_two_nights(self) -> None:
        self.assertEqual(extract_sleepedf_subject_id("SC4001E0-PSG"), "SC400")
        self.assertEqual(extract_sleepedf_subject_id("SC4002EC-Hypnogram"), "SC400")
        self.assertEqual(sleepedf_pair_key("SC4001E0-PSG.edf"), "SC4001E")
        self.assertEqual(sleepedf_pair_key("SC4001EC-Hypnogram.edf"), "SC4001E")

    def test_sleepedf_stage_normalization(self) -> None:
        self.assertEqual(normalize_sleep_stage("Sleep stage W"), "W")
        self.assertEqual(normalize_sleep_stage("Sleep stage 1"), "N1")
        self.assertEqual(normalize_sleep_stage("Sleep stage 2"), "N2")
        self.assertEqual(normalize_sleep_stage("Sleep stage 3"), "N3")
        self.assertEqual(normalize_sleep_stage("Sleep stage 4"), "N3")
        self.assertEqual(normalize_sleep_stage("Sleep stage R"), "REM")
        self.assertIsNone(normalize_sleep_stage("Movement time"))
        self.assertIsNone(normalize_sleep_stage("Sleep stage ?"))

    def test_sleepedf_annotations_expand_to_fixed_epochs(self) -> None:
        rows = expand_sleep_stage_annotations(
            record_name="SC4001E0-PSG",
            subject_id="SC400",
            signal_path="/tmp/SC4001E0.npy",
            sample_rate=100.0,
            signal_length=20_000,
            onsets=[0.0, 60.0, 120.0],
            durations=[60.0, 30.0, 30.0],
            descriptions=["Sleep stage 2", "Movement time", "Sleep stage R"],
            epoch_seconds=30,
        )

        self.assertEqual(len(rows), 3)
        self.assertEqual([row["label"] for row in rows], ["N2", "N2", "REM"])
        self.assertEqual([row["start_sample"] for row in rows], [0, 3000, 12000])
        self.assertTrue(all(row["num_samples"] == 3000 for row in rows))

    def test_sleepedf_channel_normalization_handles_eeg_prefix(self) -> None:
        self.assertEqual(_normalize_sleepedf_channel_label("Fpz-Cz"), "fpz-cz")
        self.assertEqual(_normalize_sleepedf_channel_label("EEG Fpz-Cz"), "fpz-cz")
        self.assertEqual(_normalize_sleepedf_channel_label(" EEG   Fpz - Cz "), "fpz-cz")

    def test_sleepedf_subject_split_keeps_groups_together(self) -> None:
        subject_ids = [
            "SC400",
            "SC400",
            "SC401",
            "SC401",
            "SC402",
            "SC402",
            "SC403",
            "SC403",
            "SC404",
            "SC404",
            "SC405",
            "SC405",
        ]
        splits = split_sleepedf_subject_ids(subject_ids, seed=42)

        train_subjects = set(splits["train"])
        val_subjects = set(splits["validation"])
        test_subjects = set(splits["test"])
        self.assertTrue(train_subjects)
        self.assertTrue(val_subjects)
        self.assertTrue(test_subjects)
        self.assertTrue(train_subjects.isdisjoint(val_subjects))
        self.assertTrue(train_subjects.isdisjoint(test_subjects))
        self.assertTrue(val_subjects.isdisjoint(test_subjects))
        self.assertEqual(
            train_subjects | val_subjects | test_subjects,
            set(subject_ids),
        )


if __name__ == "__main__":
    unittest.main()
