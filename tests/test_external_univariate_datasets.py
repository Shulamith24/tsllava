from __future__ import annotations

import unittest
import tempfile
import wave
from pathlib import Path

import numpy as np

from opentslm.time_series_datasets.classification_utils import split_rows_stratified
from opentslm.time_series_datasets.cinc2017af.cinc2017af_loader import (
    CINC2017AF_SAMPLE_RATE,
    CINC2017AF_SOURCE_SAMPLE_RATE,
    CINC2017AF_TARGET_LENGTH,
    center_crop_or_pad as center_crop_or_pad_cinc2017af,
    normalize_cinc2017af_label,
)
from opentslm.time_series_datasets.heart_sound.heart_sound_loader import (
    HEART_SOUND_TARGET_LENGTH,
    build_heart_sound_rows,
    normalize_heart_sound_label,
    read_wav_mono,
    resample_series as resample_heart_sound_series,
)
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

    def test_cinc2017_af_label_normalization_is_stable(self) -> None:
        self.assertEqual(CINC2017AF_SOURCE_SAMPLE_RATE, 300.0)
        self.assertEqual(CINC2017AF_SAMPLE_RATE, 100.0)
        self.assertEqual(CINC2017AF_TARGET_LENGTH, 3000)
        self.assertEqual(normalize_cinc2017af_label("N"), "N")
        self.assertEqual(normalize_cinc2017af_label("A"), "A")
        self.assertEqual(normalize_cinc2017af_label("O"), "O")
        self.assertEqual(normalize_cinc2017af_label("~"), "~")
        self.assertIsNone(normalize_cinc2017af_label("x"))

    def test_heart_sound_label_normalization_is_stable(self) -> None:
        self.assertEqual(normalize_heart_sound_label("-1"), "normal")
        self.assertEqual(normalize_heart_sound_label(1), "abnormal")
        self.assertEqual(normalize_heart_sound_label("normal"), "normal")
        self.assertEqual(normalize_heart_sound_label("abnormal"), "abnormal")
        self.assertIsNone(normalize_heart_sound_label("0"))

    def test_stratified_split_is_reproducible_and_preserves_classes(self) -> None:
        rows = [
            {"record_name": f"{label}_{index:02d}", "label": label}
            for label in ("normal", "abnormal")
            for index in range(10)
        ]
        first = split_rows_stratified(rows, seed=42)
        second = split_rows_stratified(rows, seed=42)

        self.assertEqual(
            [[row["record_name"] for row in split] for split in first],
            [[row["record_name"] for row in split] for split in second],
        )
        self.assertEqual([len(split) for split in first], [14, 2, 4])
        for split in first:
            self.assertEqual({row["label"] for row in split}, {"normal", "abnormal"})

    def test_cinc2017_fixed_length_helper_pads_crops_and_cleans(self) -> None:
        short = center_crop_or_pad_cinc2017af(np.asarray([1.0, np.nan], dtype=np.float32), target_length=4)
        long = center_crop_or_pad_cinc2017af(np.arange(6, dtype=np.float32), target_length=4)

        np.testing.assert_allclose(short, np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float32))
        np.testing.assert_allclose(long, np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32))

    def test_heart_sound_wav_reader_and_resampler(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            wav_path = Path(tmp) / "sample.wav"
            values = np.linspace(-0.5, 0.5, 2000, dtype=np.float32)
            pcm = np.clip(values * 32767.0, -32768, 32767).astype("<i2")
            with wave.open(str(wav_path), "wb") as writer:
                writer.setnchannels(1)
                writer.setsampwidth(2)
                writer.setframerate(2000)
                writer.writeframes(pcm.tobytes())

            signal, sample_rate = read_wav_mono(wav_path)
            resampled = resample_heart_sound_series(signal, source_rate=sample_rate, target_rate=500.0)

            self.assertEqual(sample_rate, 2000.0)
            self.assertEqual(signal.shape, (2000,))
            self.assertEqual(resampled.shape, (500,))

    def test_heart_sound_build_rows_from_training_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "cinc2016heart"
            training_dir = root / "training-a"
            training_dir.mkdir(parents=True)
            (training_dir / "REFERENCE.csv").write_text("a0001,-1\na0002,1\n", encoding="utf-8")
            values = np.linspace(-0.25, 0.25, 2000, dtype=np.float32)
            pcm = np.clip(values * 32767.0, -32768, 32767).astype("<i2")
            for record_name in ("a0001", "a0002"):
                with wave.open(str(training_dir / f"{record_name}.wav"), "wb") as writer:
                    writer.setnchannels(1)
                    writer.setsampwidth(2)
                    writer.setframerate(2000)
                    writer.writeframes(pcm.tobytes())

            rows = build_heart_sound_rows(raw_data_path=str(root))

            self.assertEqual([row["label"] for row in rows], ["normal", "abnormal"])
            self.assertTrue(all(row["time_series"].shape == (HEART_SOUND_TARGET_LENGTH,) for row in rows))

if __name__ == "__main__":
    unittest.main()
