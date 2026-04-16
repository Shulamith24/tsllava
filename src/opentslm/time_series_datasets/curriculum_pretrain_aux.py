import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch.utils.data import Dataset

from opentslm.time_series_datasets.ucr.ucr_loader import load_ucr_dataset


DEFAULT_UCR_TRAIN_LIST = (
    Path(__file__).resolve().parent / "ucr" / "ucr_train_98_datasets.txt"
)

DEFAULT_SYNTHETIC_SAMPLE_TYPES = (
    "caption",
    "qa_trend",
    "qa_periodic",
    "qa_spikes",
    "qa_abrupt",
)
FULL_SYNTHETIC_SAMPLE_TYPES = DEFAULT_SYNTHETIC_SAMPLE_TYPES + ("match_mismatch",)


def _read_dataset_name_list(list_path: Path) -> List[str]:
    dataset_names: List[str] = []
    for raw_line in list_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        dataset_names.append(line)
    return dataset_names


def _normalize_series(series: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(series, dtype=torch.float32).flatten()
    tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
    if tensor.numel() == 0:
        raise ValueError("Encountered empty time series.")
    mean = tensor.mean()
    std = tensor.std(unbiased=False)
    if std > 1e-6:
        tensor = (tensor - mean) / (std + 1e-6)
    else:
        tensor = tensor - mean
    return tensor


def _build_prompt_sample(
    *,
    pre_prompt: str,
    post_prompt: str,
    answer: str,
    time_series: torch.Tensor,
    time_series_text: Optional[str] = None,
    eos_token: str = "",
    alignment_target_text: Optional[str] = None,
    source_name: Optional[str] = None,
    sample_type: Optional[str] = None,
) -> Dict[str, Any]:
    answer_text = answer.strip()
    if eos_token and not answer_text.endswith(eos_token):
        answer_text = answer_text + eos_token

    prompt_text = time_series_text or f"This is a univariate time series with {time_series.numel()} data points."
    sample = {
        "pre_prompt": pre_prompt.strip(),
        "post_prompt": post_prompt.strip(),
        "answer": answer_text,
        "time_series": [time_series.clone()],
        "time_series_text": [prompt_text],
    }
    if alignment_target_text:
        sample["alignment_target_text"] = alignment_target_text.strip()
    if source_name:
        sample["source_name"] = source_name
    if sample_type:
        sample["sample_type"] = sample_type
    return sample


def _negate_statement(statement: str) -> str:
    replacements = [
        (" is periodic.", " is not periodic."),
        (" are present in the time series.", " are not present in the time series."),
        (" contains an abrupt level change.", " does not contain an abrupt level change."),
        (" is stationary.", " is not stationary."),
    ]
    for old, new in replacements:
        if old in statement:
            return statement.replace(old, new)
    return "It is not true that " + statement[0].lower() + statement[1:]


def _split_tsqa_raw():
    from datasets import load_dataset
    from opentslm.time_series_datasets.TSQADataset import TEST_FRAC as TSQA_TEST_FRAC
    from opentslm.time_series_datasets.TSQADataset import VAL_FRAC as TSQA_VAL_FRAC

    ds_full = load_dataset("ChengsenWang/TSQA", split="train")
    train_val, test = ds_full.train_test_split(
        test_size=TSQA_TEST_FRAC,
        seed=42,
    ).values()
    train, val = train_val.train_test_split(
        test_size=TSQA_VAL_FRAC / (1 - TSQA_TEST_FRAC),
        seed=42,
    ).values()
    return {"train": train, "validation": val, "test": test}


def load_tsqa_raw_records(split: str) -> List[Dict[str, Any]]:
    split_map = _split_tsqa_raw()
    dataset = split_map[split]
    records = []
    for idx, row in enumerate(dataset):
        series = torch.tensor(json.loads(row["Series"]), dtype=torch.float32).flatten()
        series = _normalize_series(series)
        records.append(
            {
                "series_id": f"tsqa_{split}_{idx}",
                "source_name": "tsqa_raw",
                "series": series,
                "question": row["Question"],
                "task": row["Task"],
                "answer": row["Answer"],
            }
        )
    return records


def _split_m4_raw():
    from opentslm.time_series_datasets.m4.m4_loader import create_combined_dataset, load_all_m4_data

    data_dict = load_all_m4_data()
    train, val, test = create_combined_dataset(data_dict, seed=42)
    return {"train": train, "validation": val, "test": test}


def load_m4_raw_records(split: str) -> List[Dict[str, Any]]:
    dataset = _split_m4_raw()[split]
    records = []
    for idx, row in enumerate(dataset):
        series = _normalize_series(row["series"])
        records.append(
            {
                "series_id": f"m4_{split}_{row.get('id', idx)}",
                "source_name": "m4_raw",
                "series": series,
                "caption": row["caption"],
                "frequency": row.get("frequency"),
            }
        )
    return records


def load_ucr_train_raw_records(
    *,
    raw_data_path: str = "./data",
    dataset_list_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    list_path = Path(dataset_list_path or DEFAULT_UCR_TRAIN_LIST)
    dataset_names = _read_dataset_name_list(list_path)
    records: List[Dict[str, Any]] = []
    for dataset_name in dataset_names:
        train_df, _ = load_ucr_dataset(dataset_name, raw_data_path=raw_data_path)
        feature_cols = [col for col in train_df.columns if col != "label"]
        for idx, row in enumerate(train_df.to_dict("records")):
            values = [row[col] for col in feature_cols]
            series = _normalize_series(values)
            records.append(
                {
                    "series_id": f"ucr_{dataset_name}_{idx}",
                    "source_name": f"ucr_train:{dataset_name}",
                    "series": series,
                }
            )
    return records


class RawSeriesDataset(Dataset):
    def __init__(self, records: Sequence[Dict[str, Any]]):
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def get_sample_length(self, idx: int) -> int:
        return int(self.records[idx]["series"].numel())

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.records[idx]
        return {
            "series": record["series"].clone(),
            "series_id": record["series_id"],
            "source_name": record["source_name"],
        }


class AlignmentTargetDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        *,
        eos_token: str,
        source_name: str,
        alignment_from_answer: bool,
    ):
        self.dataset = dataset
        self.eos_token = eos_token
        self.source_name = source_name
        self.alignment_from_answer = alignment_from_answer

    def __len__(self) -> int:
        return len(self.dataset)

    def get_sample_length(self, idx: int) -> int:
        get_sample_length = getattr(self.dataset, "get_sample_length", None)
        if callable(get_sample_length):
            return int(get_sample_length(idx))
        sample = self.dataset[idx]
        return max(int(torch.as_tensor(ts).numel()) for ts in sample.get("time_series", []))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = dict(self.dataset[idx])
        sample["source_name"] = self.source_name
        if self.alignment_from_answer:
            sample["alignment_target_text"] = sample["answer"].replace(self.eos_token, "").strip()
        else:
            sample["alignment_target_text"] = None
        return sample


@dataclass(frozen=True)
class SeriesAttributes:
    trend: str
    periodic: bool
    volatility: str
    spikes: bool
    abrupt_change: bool
    dominant_frequency: str
    amplitude_range: str
    extrema_density: str
    stationary: bool


TREND_PHRASES = {
    "increasing": ["an increasing trend", "an upward trend", "a rising overall level"],
    "decreasing": ["a decreasing trend", "a downward trend", "a falling overall level"],
    "flat": ["a mostly flat trend", "little overall trend", "a broadly stable level"],
}
VOLATILITY_PHRASES = {
    "low": ["low variability", "gentle fluctuations", "mild variation"],
    "medium": ["moderate variability", "noticeable fluctuations", "medium variation"],
    "high": ["high variability", "strong fluctuations", "pronounced variation"],
}
FREQUENCY_PHRASES = {
    "low": ["a low dominant frequency", "slow oscillations", "coarse periodic structure"],
    "medium": ["a medium dominant frequency", "mid-speed oscillations", "moderate periodic structure"],
    "high": ["a high dominant frequency", "rapid oscillations", "fine periodic structure"],
}
AMPLITUDE_PHRASES = {
    "narrow": ["a narrow amplitude range", "compact amplitude swings", "limited vertical range"],
    "medium": ["a medium amplitude range", "moderate amplitude swings", "mid-sized vertical range"],
    "wide": ["a wide amplitude range", "large amplitude swings", "broad vertical range"],
}
EXTREMA_PHRASES = {
    "sparse": ["sparse local extrema", "few turning points", "low extrema density"],
    "moderate": ["moderate local extrema density", "a moderate number of turning points", "mid-density extrema"],
    "dense": ["dense local extrema", "many turning points", "high extrema density"],
}


class SyntheticAttributeDataset(Dataset):
    def __init__(
        self,
        split: str,
        *,
        eos_token: str,
        seed: int = 42,
        sample_types: Optional[Sequence[str]] = None,
    ):
        self.split = split
        self.eos_token = eos_token
        self.seed = int(seed)
        self.sample_types = tuple(sample_types or DEFAULT_SYNTHETIC_SAMPLE_TYPES)
        unknown_sample_types = sorted(set(self.sample_types) - set(FULL_SYNTHETIC_SAMPLE_TYPES))
        if unknown_sample_types:
            raise ValueError(f"Unsupported synthetic sample types: {unknown_sample_types}")
        self.records = load_m4_raw_records(split) + load_tsqa_raw_records(split)

    def __len__(self) -> int:
        return len(self.records) * len(self.sample_types)

    def get_sample_length(self, idx: int) -> int:
        base_idx = idx // len(self.sample_types)
        return int(self.records[base_idx]["series"].numel())

    def _compute_attributes(self, series: torch.Tensor) -> SeriesAttributes:
        x = series.float()
        n = x.numel()
        t = torch.linspace(-1.0, 1.0, steps=n, device=x.device)
        slope = float((t * x).mean().item())
        trend = "flat"
        if slope > 0.12:
            trend = "increasing"
        elif slope < -0.12:
            trend = "decreasing"

        diffs = torch.diff(x)
        diff_std = float(diffs.std(unbiased=False).item()) if diffs.numel() > 0 else 0.0
        if diff_std < 0.35:
            volatility = "low"
        elif diff_std < 0.8:
            volatility = "medium"
        else:
            volatility = "high"

        z = (x - x.mean()) / (x.std(unbiased=False) + 1e-6)
        spikes = bool((z.abs() > 3.0).any().item())
        abrupt_change = False
        if diffs.numel() > 0:
            abrupt_change = bool((diffs.abs().max() > 3.0 * diffs.abs().median().clamp_min(1e-3)).item())

        x_fft = torch.fft.rfft(x)
        power = x_fft.abs()
        if power.numel() > 2:
            dominant_idx = int(power[1:].argmax().item()) + 1
            freq_ratio = dominant_idx / max(power.numel() - 1, 1)
            dominant_power_ratio = float(power[dominant_idx].item() / (power[1:].sum().item() + 1e-6))
            periodic = dominant_power_ratio > 0.2
        else:
            freq_ratio = 0.0
            periodic = False
        if freq_ratio < 0.1:
            dominant_frequency = "low"
        elif freq_ratio < 0.3:
            dominant_frequency = "medium"
        else:
            dominant_frequency = "high"

        amplitude = float((x.max() - x.min()).item())
        if amplitude < 2.5:
            amplitude_range = "narrow"
        elif amplitude < 5.5:
            amplitude_range = "medium"
        else:
            amplitude_range = "wide"

        if diffs.numel() > 1:
            sign_change = torch.diff(torch.sign(diffs))
            extrema_count = int((sign_change != 0).sum().item())
            extrema_ratio = extrema_count / max(n - 2, 1)
        else:
            extrema_ratio = 0.0
        if extrema_ratio < 0.08:
            extrema_density = "sparse"
        elif extrema_ratio < 0.18:
            extrema_density = "moderate"
        else:
            extrema_density = "dense"

        half = max(1, n // 2)
        stationary = abs(float(x[:half].mean() - x[-half:].mean())) < 0.45 and abs(slope) < 0.12

        return SeriesAttributes(
            trend=trend,
            periodic=periodic,
            volatility=volatility,
            spikes=spikes,
            abrupt_change=abrupt_change,
            dominant_frequency=dominant_frequency,
            amplitude_range=amplitude_range,
            extrema_density=extrema_density,
            stationary=stationary,
        )

    def _attribute_caption(self, attrs: SeriesAttributes, variant_seed: int) -> str:
        rng = random.Random(variant_seed)
        trend_text = rng.choice(TREND_PHRASES[attrs.trend])
        volatility_text = rng.choice(VOLATILITY_PHRASES[attrs.volatility])
        frequency_text = rng.choice(FREQUENCY_PHRASES[attrs.dominant_frequency])
        amplitude_text = rng.choice(AMPLITUDE_PHRASES[attrs.amplitude_range])
        extrema_text = rng.choice(EXTREMA_PHRASES[attrs.extrema_density])
        periodic_text = "clear periodic structure" if attrs.periodic else "no strong periodic structure"
        spikes_text = "sharp spikes are present" if attrs.spikes else "no obvious sharp spikes are present"
        abrupt_text = "an abrupt change is visible" if attrs.abrupt_change else "no abrupt level change is evident"
        stationary_text = "the series appears stationary" if attrs.stationary else "the series appears non-stationary"
        return (
            f"The time series shows {trend_text}, {periodic_text}, {volatility_text}, "
            f"{spikes_text}, {abrupt_text}, {frequency_text}, {amplitude_text}, "
            f"{extrema_text}, and {stationary_text}."
        )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        base_idx = idx // len(self.sample_types)
        sample_variant = self.sample_types[idx % len(self.sample_types)]
        record = self.records[base_idx]
        series = record["series"]
        attrs = self._compute_attributes(series)
        variant_seed = self.seed * 1_000_003 + idx
        caption = self._attribute_caption(attrs, variant_seed)
        ts_text = f"This is a univariate time series with {series.numel()} normalized data points."

        if sample_variant == "caption":
            return _build_prompt_sample(
                pre_prompt="You are an expert in time series analysis.",
                post_prompt="Please describe the key temporal attributes of this time series.",
                answer=caption,
                time_series=series,
                time_series_text=ts_text,
                eos_token=self.eos_token,
                alignment_target_text=caption,
                source_name="synthetic_attribute",
                sample_type=sample_variant,
            )

        if sample_variant == "qa_trend":
            answer = "yes" if attrs.trend == "increasing" else "no"
            return _build_prompt_sample(
                pre_prompt="Does this time series show a clear upward trend?",
                post_prompt="Answer with yes or no.",
                answer=answer,
                time_series=series,
                time_series_text=ts_text,
                eos_token=self.eos_token,
                alignment_target_text=caption,
                source_name="synthetic_attribute",
                sample_type=sample_variant,
            )

        if sample_variant == "qa_periodic":
            answer = "yes" if attrs.periodic else "no"
            return _build_prompt_sample(
                pre_prompt="Is the time series periodic?",
                post_prompt="Answer with yes or no.",
                answer=answer,
                time_series=series,
                time_series_text=ts_text,
                eos_token=self.eos_token,
                alignment_target_text=caption,
                source_name="synthetic_attribute",
                sample_type=sample_variant,
            )

        if sample_variant == "qa_spikes":
            answer = "yes" if attrs.spikes else "no"
            return _build_prompt_sample(
                pre_prompt="Are sharp spikes present in this time series?",
                post_prompt="Answer with yes or no.",
                answer=answer,
                time_series=series,
                time_series_text=ts_text,
                eos_token=self.eos_token,
                alignment_target_text=caption,
                source_name="synthetic_attribute",
                sample_type=sample_variant,
            )

        if sample_variant == "qa_abrupt":
            answer = "yes" if attrs.abrupt_change else "no"
            return _build_prompt_sample(
                pre_prompt="Is there an abrupt level change in this time series?",
                post_prompt="Answer with yes or no.",
                answer=answer,
                time_series=series,
                time_series_text=ts_text,
                eos_token=self.eos_token,
                alignment_target_text=caption,
                source_name="synthetic_attribute",
                sample_type=sample_variant,
            )

        rng = random.Random(variant_seed)
        statements = [
            (attrs.periodic, "The time series is periodic."),
            (attrs.spikes, "Sharp spikes are present in the time series."),
            (attrs.abrupt_change, "The time series contains an abrupt level change."),
            (attrs.stationary, "The time series is stationary."),
        ]
        truth_value, statement = statements[base_idx % len(statements)]
        use_match = (base_idx + self.seed) % 2 == 0
        if use_match:
            chosen_statement = statement if truth_value else _negate_statement(statement)
            answer = "match"
        else:
            chosen_statement = statement if not truth_value else _negate_statement(statement)
            answer = "mismatch"

        return _build_prompt_sample(
            pre_prompt="Determine whether the statement matches the time series.",
            post_prompt=f"Statement: {chosen_statement}\nAnswer with match or mismatch.",
            answer=answer,
            time_series=series,
            time_series_text=ts_text,
            eos_token=self.eos_token,
            alignment_target_text=caption,
            source_name="synthetic_attribute",
            sample_type=sample_variant,
        )


class MixedPretrainDataset(Dataset):
    def __init__(
        self,
        datasets: Sequence[Dataset],
        weights: Sequence[int],
        *,
        seed: int = 42,
        epoch_size: Optional[int] = None,
    ):
        if len(datasets) != len(weights):
            raise ValueError("datasets and weights must have the same length")
        if not datasets:
            raise ValueError("At least one dataset is required")
        if any(weight <= 0 for weight in weights):
            raise ValueError("weights must all be positive")

        self.datasets = list(datasets)
        self.weights = [int(weight) for weight in weights]
        self.seed = int(seed)
        if epoch_size is None:
            epoch_units = max(
                math.ceil(len(dataset) / weight)
                for dataset, weight in zip(self.datasets, self.weights)
            )
            epoch_size = epoch_units * sum(self.weights)
        self.epoch_size = int(epoch_size)
        self.schedule: List[tuple[int, int]] = []
        self.set_epoch(0)

    def set_epoch(self, epoch: int):
        rng = random.Random(self.seed + int(epoch))
        shuffled_indices = [list(range(len(dataset))) for dataset in self.datasets]
        for indices in shuffled_indices:
            rng.shuffle(indices)

        cursors = [0 for _ in self.datasets]
        schedule: List[tuple[int, int]] = []
        while len(schedule) < self.epoch_size:
            for dataset_idx, weight in enumerate(self.weights):
                indices = shuffled_indices[dataset_idx]
                if not indices:
                    continue
                for _ in range(weight):
                    if len(schedule) >= self.epoch_size:
                        break
                    if cursors[dataset_idx] >= len(indices):
                        rng.shuffle(indices)
                        cursors[dataset_idx] = 0
                    schedule.append((dataset_idx, indices[cursors[dataset_idx]]))
                    cursors[dataset_idx] += 1
        self.schedule = schedule

    def __len__(self) -> int:
        return len(self.schedule)

    def get_sample_length(self, idx: int) -> int:
        dataset_idx, sample_idx = self.schedule[idx]
        dataset = self.datasets[dataset_idx]
        get_sample_length = getattr(dataset, "get_sample_length", None)
        if callable(get_sample_length):
            return int(get_sample_length(sample_idx))
        sample = dataset[sample_idx]
        return max(int(torch.as_tensor(ts).numel()) for ts in sample.get("time_series", []))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        dataset_idx, sample_idx = self.schedule[idx]
        return self.datasets[dataset_idx][sample_idx]
