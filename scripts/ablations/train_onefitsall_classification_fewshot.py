#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
Few-shot supervised classification for the One-Fits-All classification model.

Protocol:
- official TRAIN split is the support pool
- official TEST split is used only for final evaluation
- support sets are sampled per shot/run
- training keeps the original One-Fits-All single-stage optimization semantics
"""

from __future__ import annotations

import argparse
import datetime
import importlib.util
import json
import pickle
import random
import sys
import types
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
ONEFITSALL_SRC = PROJECT_ROOT / "temp" / "NeurIPS2023-One-Fits-All" / "Classification" / "src"
DEFAULT_DATA_PATH = str(PROJECT_ROOT / "data")
DEFAULT_FEWSHOT_SAVE_DIR = "results/ablations/onefitsall_fewshot"
DEFAULT_FULL_SAVE_DIR = "results/ablations/onefitsall_full"
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def bootstrap_sktime_compat() -> None:
    try:
        import sktime.utils as sktime_utils
        from sktime.datasets import load_from_tsfile_to_dataframe
    except ImportError:
        return

    if not hasattr(sktime_utils, "load_data"):
        sktime_utils.load_data = types.SimpleNamespace(
            load_from_tsfile_to_dataframe=load_from_tsfile_to_dataframe
        )


def bootstrap_optional_debug_modules() -> None:
    if "ipdb" not in sys.modules:
        sys.modules["ipdb"] = types.SimpleNamespace(set_trace=lambda: None)

    def _missing_excel_dependency(*_args, **_kwargs):
        raise RuntimeError("Excel export dependencies are unavailable in this environment.")

    if "xlrd" not in sys.modules:
        sys.modules["xlrd"] = types.SimpleNamespace(open_workbook=_missing_excel_dependency)

    if "xlwt" not in sys.modules:
        sys.modules["xlwt"] = types.SimpleNamespace(Workbook=_missing_excel_dependency)

    if "xlutils" not in sys.modules:
        sys.modules["xlutils"] = types.ModuleType("xlutils")
    if "xlutils.copy" not in sys.modules:
        xlutils_copy = types.ModuleType("xlutils.copy")
        xlutils_copy.copy = _missing_excel_dependency
        sys.modules["xlutils.copy"] = xlutils_copy


def bootstrap_onefitsall_packages() -> None:
    sys.path.insert(0, str(ONEFITSALL_SRC))

    for package_name in ("datasets", "models", "utils"):
        package_dir = ONEFITSALL_SRC / package_name
        for module_name in list(sys.modules.keys()):
            if module_name == package_name or module_name.startswith(f"{package_name}."):
                del sys.modules[module_name]

        spec = importlib.util.spec_from_file_location(
            package_name,
            package_dir / "__init__.py",
            submodule_search_locations=[str(package_dir)],
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to bootstrap One-Fits-All package: {package_name}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[package_name] = module
        spec.loader.exec_module(module)

bootstrap_sktime_compat()
bootstrap_optional_debug_modules()
bootstrap_onefitsall_packages()

from datasets.data import Normalizer, data_factory  # type: ignore  # noqa: E402
from datasets.dataset import ClassiregressionDataset, collate_superv  # type: ignore  # noqa: E402
from models.gpt4ts import gpt4ts  # type: ignore  # noqa: E402
from models.loss import get_loss_module  # type: ignore  # noqa: E402
from optimizers import get_optimizer  # type: ignore  # noqa: E402
from running import SupervisedRunner  # type: ignore  # noqa: E402
from utils import utils as onefitsall_utils  # type: ignore  # noqa: E402
from opentslm.time_series_datasets.ucr.ucr_loader import load_ucr_dataset  # noqa: E402

from fewshot_utils import (  # noqa: E402
    ShotType,
    aggregate_shot_results,
    build_label_to_indices,
    filter_indices_by_class_ids,
    parse_shots,
    sample_support_info,
    save_shot_summary_csv,
    shot_to_name,
    write_json,
)


def cli_flag_was_provided(argv: Optional[List[str]], flag_name: str) -> bool:
    if argv is None:
        argv = sys.argv[1:]
    return any(token == flag_name or token.startswith(f"{flag_name}=") for token in argv)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    provided_argv = list(argv) if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(
        description="Few-shot supervised classification for One-Fits-All GPT4TS"
    )

    parser.add_argument("--protocol", type=str, default="fewshot", choices=["fewshot", "full"])
    parser.add_argument("--shots", type=str, default="1,2,5,10,full")
    parser.add_argument("--way", type=int, default=None)
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument("--fewshot_seed_base", type=int, default=3407)

    parser.add_argument("--dataset", type=str, default=None, help="UCR dataset name, e.g. AllGestureWiimoteX.")
    parser.add_argument("--data_dir", type=str, default=None, help="Directory containing official TRAIN/TEST files.")
    parser.add_argument(
        "--data_path",
        type=str,
        default=DEFAULT_DATA_PATH,
        help="Base data path for UCRArchive_2018 compatibility. Defaults to the repo's ./data directory.",
    )
    parser.add_argument(
        "--data_format",
        type=str,
        default="auto",
        choices=["auto", "onefitsall_ts", "ucr_tsv"],
        help="Input layout: One-Fits-All .ts files or UCRArchive_2018 .tsv files.",
    )
    parser.add_argument("--data_class", type=str, default="tsra", choices=sorted(data_factory.keys()))
    parser.add_argument("--train_pattern", type=str, default="TRAIN")
    parser.add_argument("--test_pattern", type=str, default="TEST")
    parser.add_argument("--limit_size", type=float, default=None)
    parser.add_argument("--n_proc", type=int, default=-1)
    parser.add_argument(
        "--normalization",
        type=str,
        default="standardization",
        choices=["none", "standardization", "minmax", "per_sample_std", "per_sample_minmax"],
    )
    parser.add_argument("--norm_from", type=str, default=None)
    parser.add_argument("--subsample_factor", type=int, default=None)

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--optimizer", type=str, default="RAdam", choices=["Adam", "RAdam"])
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--print_interval", type=int, default=1)
    parser.add_argument("--l2_reg", type=float, default=0.0)
    parser.add_argument("--global_reg", action="store_true")
    parser.add_argument("--freeze", action="store_true")

    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--gpu", type=str, default="0", help="GPU index, -1 for CPU.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--comment", type=str, default="")
    parser.add_argument("--console", action="store_true")
    parser.add_argument("--save_dir", type=str, default=DEFAULT_FEWSHOT_SAVE_DIR)
    parser.add_argument("--resume", action="store_true", help="Resume from existing run checkpoints when available.")
    parser.add_argument(
        "--cleanup_checkpoints",
        action="store_true",
        help="Remove per-run checkpoints after writing final results to save disk space.",
    )

    args = parser.parse_args(argv)
    args.save_dir_explicit = cli_flag_was_provided(provided_argv, "--save_dir")
    return args


def normalize_protocol_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.protocol == "full":
        if args.way is not None:
            raise ValueError("--way is not allowed when --protocol=full; full supervision must use all classes.")
        args.shots = "full"
        args.num_runs = 1
        if not getattr(args, "save_dir_explicit", False):
            args.save_dir = DEFAULT_FULL_SAVE_DIR
        return args

    if not getattr(args, "save_dir_explicit", False):
        args.save_dir = DEFAULT_FEWSHOT_SAVE_DIR
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cleanup_checkpoint_files(paths: List[Path]) -> None:
    """Remove no-longer-needed checkpoints without failing the run."""
    for path in paths:
        if not path.exists():
            continue
        try:
            path.unlink()
            print(f"Removed checkpoint: {path}")
        except OSError as exc:
            print(f"Failed to remove checkpoint {path}: {exc}")


def infer_dataset_name(args: argparse.Namespace) -> str:
    if args.dataset:
        return args.dataset
    if args.data_dir:
        data_dir = Path(args.data_dir).resolve()
        return data_dir.name
    raise ValueError("--dataset is required when running the ablation script without --data_dir.")


def resolve_device(gpu_arg: str) -> torch.device:
    if gpu_arg == "-1" or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(f"cuda:{gpu_arg}")


def build_config(args: argparse.Namespace) -> Dict[str, Any]:
    config = vars(args).copy()
    config["task"] = "classification"
    config["model"] = "gpt4ts"
    config["freeze"] = bool(args.freeze)
    config["normalization"] = None if args.normalization == "none" else args.normalization
    config["lr_step"] = [1000000]
    config["lr_factor"] = [0.1]
    return config


class SimpleClassificationData:
    def __init__(
        self,
        feature_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        class_names: List[Any],
    ) -> None:
        self.feature_df = feature_df
        self.labels_df = labels_df
        self.all_df = feature_df
        self.feature_names = feature_df.columns.tolist()
        self.all_IDs = feature_df.index.unique().tolist()
        self.max_seq_len = int(feature_df.groupby(level=0).size().max())
        self.class_names = class_names


def infer_data_format(args: argparse.Namespace) -> str:
    if args.data_format != "auto":
        return args.data_format

    if args.data_dir:
        data_dir = Path(args.data_dir)
        if list(data_dir.glob("*_TRAIN.tsv")):
            return "ucr_tsv"
        if list(data_dir.glob("*.ts")):
            return "onefitsall_ts"

    if args.dataset:
        return "ucr_tsv"

    raise ValueError(
        "Unable to infer data format. Provide either a One-Fits-All .ts data_dir or a UCR-compatible --dataset."
    )


def infer_ucr_dataset_dir(args: argparse.Namespace, dataset_name: str) -> Path:
    if args.data_dir:
        candidate = Path(args.data_dir).resolve()
        if candidate.is_dir() and list(candidate.glob("*_TRAIN.tsv")):
            return candidate

    if args.data_path:
        candidate = Path(args.data_path).resolve() / "UCRArchive_2018" / dataset_name
        if candidate.is_dir():
            return candidate

    raise FileNotFoundError(
        "Unable to locate the UCR dataset directory. "
        "Use either --data_dir /path/to/UCRArchive_2018/<dataset> or --dataset <name> --data_path /path/to/data."
    )


def load_ucr_dataframes(dataset_dir: Path, dataset_name: Optional[str]) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    inferred_name = dataset_name
    if inferred_name is None:
        train_matches = sorted(dataset_dir.glob("*_TRAIN.tsv"))
        if not train_matches:
            raise FileNotFoundError(f"No *_TRAIN.tsv file found under {dataset_dir}")
        inferred_name = train_matches[0].name.rsplit("_TRAIN.tsv", 1)[0]

    train_path = dataset_dir / f"{inferred_name}_TRAIN.tsv"
    test_path = dataset_dir / f"{inferred_name}_TEST.tsv"
    if not train_path.exists() or not test_path.exists():
        train_df, test_df = load_ucr_dataset(inferred_name, raw_data_path=str(dataset_dir.parent.parent))
        return train_df, test_df, inferred_name

    train_df = pd.read_csv(train_path, sep="\t", header=None)
    test_df = pd.read_csv(test_path, sep="\t", header=None)
    n_cols = train_df.shape[1] - 1
    col_names = ["label"] + [f"t{i}" for i in range(1, n_cols + 1)]
    train_df.columns = col_names
    test_df.columns = col_names
    return train_df, test_df, inferred_name


def build_simple_ucr_data(
    split_df: pd.DataFrame,
    *,
    class_names: List[Any],
    label_to_id: Dict[Any, int],
) -> SimpleClassificationData:
    feature_cols = [col for col in split_df.columns if col != "label"]
    values = split_df[feature_cols].astype(np.float32).to_numpy()
    sample_ids = np.arange(len(split_df), dtype=np.int64)
    repeated_ids = np.repeat(sample_ids, len(feature_cols))

    feature_df = pd.DataFrame(
        {"dim_0": values.reshape(-1)},
        index=repeated_ids,
    )
    labels_df = pd.DataFrame(
        {"label": [label_to_id[label] for label in split_df["label"].tolist()]},
        index=sample_ids,
        dtype=np.int64,
    )
    return SimpleClassificationData(
        feature_df=feature_df,
        labels_df=labels_df,
        class_names=class_names,
    )


def load_raw_splits(config: Dict[str, Any], args: argparse.Namespace, dataset_name: str) -> tuple[Any, Any]:
    data_format = infer_data_format(args)
    if data_format == "ucr_tsv":
        dataset_dir = infer_ucr_dataset_dir(args, dataset_name)
        train_df, test_df, resolved_name = load_ucr_dataframes(dataset_dir, dataset_name)
        class_names = sorted(train_df["label"].unique().tolist())
        label_to_id = {label: idx for idx, label in enumerate(class_names)}
        config["dataset"] = resolved_name
        config["data_dir"] = str(dataset_dir)
        return (
            build_simple_ucr_data(train_df, class_names=class_names, label_to_id=label_to_id),
            build_simple_ucr_data(test_df, class_names=class_names, label_to_id=label_to_id),
        )

    data_class = data_factory[config["data_class"]]
    train_data = data_class(
        config["data_dir"],
        pattern=config["train_pattern"],
        n_proc=config["n_proc"],
        limit_size=config["limit_size"],
        config=config,
    )
    test_data = data_class(
        config["data_dir"],
        pattern=config["test_pattern"],
        n_proc=-1,
        limit_size=None,
        config=config,
    )
    return train_data, test_data


def clone_data_object(data_obj: Any) -> Any:
    return deepcopy(data_obj)


def maybe_build_normalizer(config: Dict[str, Any], support_feature_df) -> Optional[Normalizer]:
    if config["norm_from"]:
        with open(config["norm_from"], "rb") as f:
            norm_dict = pickle.load(f)
        return Normalizer(**norm_dict)

    if config["normalization"] is None:
        return None

    normalizer = Normalizer(config["normalization"])
    normalizer.normalize(support_feature_df)
    return normalizer


def apply_run_normalization(
    train_data: Any,
    test_data: Any,
    support_ids: List[Any],
    config: Dict[str, Any],
    run_dir: Path,
) -> None:
    normalizer = maybe_build_normalizer(config, train_data.feature_df.loc[support_ids].copy())
    if normalizer is None:
        return

    train_data.feature_df.loc[support_ids] = normalizer.normalize(train_data.feature_df.loc[support_ids].copy())
    test_data.feature_df.loc[test_data.all_IDs] = normalizer.normalize(test_data.feature_df.loc[test_data.all_IDs].copy())

    if config["norm_from"] is None and config["normalization"] and not config["normalization"].startswith("per_sample"):
        with open(run_dir / "normalization.pickle", "wb") as f:
            pickle.dump(normalizer.__dict__, f, pickle.HIGHEST_PROTOCOL)


def freeze_backbone_except_head(model: torch.nn.Module) -> None:
    for name, param in model.named_parameters():
        if name.startswith("out_layer") or name.startswith("output_layer"):
            param.requires_grad = True
        else:
            param.requires_grad = False


def create_dataloader(
    dataset,
    *,
    batch_size: int,
    max_len: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
):
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=lambda batch: collate_superv(batch, max_len=max_len),
    )


def build_optimizer(model: torch.nn.Module, config: Dict[str, Any]):
    if config["global_reg"]:
        weight_decay = config["l2_reg"]
        output_reg = None
    else:
        weight_decay = 0.0
        output_reg = config["l2_reg"]

    optim_class = get_optimizer(config["optimizer"])
    optimizer = optim_class(model.parameters(), lr=config["lr"], weight_decay=weight_decay)
    return optimizer, output_reg


def build_runner(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_module,
    config: Dict[str, Any],
    optimizer=None,
    l2_reg=None,
) -> SupervisedRunner:
    return SupervisedRunner(
        model,
        loader,
        device,
        loss_module,
        optimizer,
        l2_reg=l2_reg,
        print_interval=config["print_interval"],
        console=config["console"],
    )


def evaluate_model(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    config: Dict[str, Any],
    loss_module,
) -> Dict[str, Any]:
    evaluator = build_runner(model, test_loader, device, loss_module, config)
    with torch.no_grad():
        metrics, per_batch = evaluator.evaluate(epoch_num=None, keep_all=True)

    logits = np.concatenate(per_batch["predictions"], axis=0)
    labels = np.concatenate(per_batch["targets"], axis=0).reshape(-1)
    predictions = np.argmax(logits, axis=1)

    return {
        "metrics": {key: float(value) for key, value in metrics.items() if value is not None},
        "prediction_ids": predictions.tolist(),
        "label_ids": labels.astype(int).tolist(),
        "logits": logits.tolist(),
    }


def run_single_experiment(
    *,
    args: argparse.Namespace,
    config: Dict[str, Any],
    dataset_name: str,
    shot: ShotType,
    shot_idx: int,
    run_id: int,
    run_seed: int,
    save_root: Path,
    raw_train_data: Any,
    raw_test_data: Any,
    train_label_to_indices: Dict[int, List[Any]],
    test_label_to_indices: Dict[int, List[Any]],
    device: torch.device,
) -> Dict[str, Any]:
    shot_name = shot_to_name(shot)
    run_dir = save_root / f"shot_{shot_name}" / f"run_{run_id:02d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "model_last.pth"
    run_metrics_path = run_dir / "run_metrics.json"
    support_info_path = run_dir / "fewshot_indices.json"
    train_history_path = run_dir / "train_loss_history.json"

    completed_run_exists = (
        args.resume
        and run_metrics_path.exists()
        and (args.cleanup_checkpoints or checkpoint_path.exists())
    )
    if completed_run_exists:
        with open(run_metrics_path, "r", encoding="utf-8") as f:
            cached_metrics = json.load(f)
        print(f"[shot={shot_name} run={run_id:02d}] reuse completed run: {run_metrics_path}")
        return cached_metrics

    set_seed(run_seed)
    if args.resume and support_info_path.exists():
        with open(support_info_path, "r", encoding="utf-8") as f:
            support_info = json.load(f)
    else:
        support_info = sample_support_info(
            train_label_to_indices,
            shot=shot,
            seed=run_seed,
            way=args.way,
        )

    support_ids = support_info["selected_indices"]
    query_ids = filter_indices_by_class_ids(test_label_to_indices, support_info["selected_class_ids"])
    if not query_ids:
        raise RuntimeError("Query split is empty for the sampled class subset.")

    write_json(
        support_info_path,
        {
            "dataset": dataset_name,
            "shot": shot_name,
            "run_id": run_id,
            "seed": run_seed,
            "way": support_info["way"],
            "selected_class_ids": support_info["selected_class_ids"],
            "selected_indices": support_info["selected_indices"],
            "selected_by_class": support_info["selected_by_class"],
            "k_eff_per_class": support_info["k_eff_per_class"],
            "class_train_counts": support_info["class_train_counts"],
            "classes_with_shortage": support_info["classes_with_shortage"],
            "query_indices": query_ids,
        },
    )

    train_data = clone_data_object(raw_train_data)
    test_data = clone_data_object(raw_test_data)
    apply_run_normalization(train_data, test_data, support_ids, config, run_dir)

    train_dataset = ClassiregressionDataset(train_data, support_ids)
    test_dataset = ClassiregressionDataset(test_data, query_ids)

    pin_memory = device.type == "cuda"
    train_loader = create_dataloader(
        train_dataset,
        batch_size=min(args.batch_size, max(1, len(train_dataset))),
        max_len=train_data.max_seq_len,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = create_dataloader(
        test_dataset,
        batch_size=args.eval_batch_size,
        max_len=test_data.max_seq_len,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model = gpt4ts(config, train_data)
    if args.freeze:
        freeze_backbone_except_head(model)
    model.to(device)

    optimizer, output_reg = build_optimizer(model, config)
    loss_module = get_loss_module(config)
    trainer = build_runner(
        model=model,
        loader=train_loader,
        device=device,
        loss_module=loss_module,
        config=config,
        optimizer=optimizer,
        l2_reg=output_reg,
    )

    train_losses: List[float] = []
    start_epoch = 0
    if args.resume and checkpoint_path.exists():
        model, optimizer, start_epoch = onefitsall_utils.load_model(
            model,
            str(checkpoint_path),
            optimizer,
            resume=True,
            change_output=False,
            lr=config["lr"],
            lr_step=config["lr_step"],
            lr_factor=config["lr_factor"],
        )
        if train_history_path.exists():
            with open(train_history_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            train_losses = [float(item) for item in payload.get("train_loss_curve", [])]
        print(
            f"[shot={shot_name} run={run_id:02d}] "
            f"resume from epoch {start_epoch}/{args.epochs}"
        )

    for epoch_idx in range(start_epoch + 1, args.epochs + 1):
        epoch_metrics = trainer.train_epoch(epoch_num=epoch_idx)
        train_losses.append(float(epoch_metrics["loss"]))
        print(
            f"[shot={shot_name} run={run_id:02d}] "
            f"epoch {epoch_idx}/{args.epochs} train_loss={epoch_metrics['loss']:.6f}"
        )
        onefitsall_utils.save_model(str(checkpoint_path), epoch_idx, model, optimizer)
        write_json(run_dir / "train_loss_history.json", {"train_loss_curve": train_losses})

    if not checkpoint_path.exists():
        onefitsall_utils.save_model(str(checkpoint_path), args.epochs, model, optimizer)

    test_results = evaluate_model(model, test_loader, device, config, loss_module)
    test_metrics = test_results["metrics"]

    class_names = [str(item) for item in raw_train_data.class_names]
    run_metrics = {
        "dataset": dataset_name,
        "protocol": args.protocol,
        "shot": shot_name,
        "shot_index": shot_idx,
        "run_id": run_id,
        "seed": run_seed,
        "way": support_info["way"],
        "selected_class_ids": support_info["selected_class_ids"],
        "class_names": class_names,
        "support_size": len(support_ids),
        "query_size": len(query_ids),
        "k_eff_per_class": support_info["k_eff_per_class"],
        "class_train_counts": support_info["class_train_counts"],
        "classes_with_shortage": support_info["classes_with_shortage"],
        "any_shortage": support_info["any_shortage"],
        "epochs": args.epochs,
        "train_batch_size": min(args.batch_size, max(1, len(train_dataset))),
        "eval_batch_size": args.eval_batch_size,
        "train_last_loss": train_losses[-1] if train_losses else None,
        "train_loss_curve": train_losses,
        "test_loss": test_metrics["loss"],
        "test_accuracy": test_metrics.get("accuracy"),
        "test_precision": test_metrics.get("precision"),
        "test_AUROC": test_metrics.get("AUROC"),
        "test_AUPRC": test_metrics.get("AUPRC"),
        "model_checkpoint": checkpoint_path.name,
        "comment": args.comment,
    }

    write_json(run_metrics_path, run_metrics)
    write_json(
        run_dir / "test_predictions.json",
        {
            "class_names": class_names,
            "selected_class_ids": support_info["selected_class_ids"],
            "prediction_ids": test_results["prediction_ids"],
            "label_ids": test_results["label_ids"],
            "logits": test_results["logits"],
        },
    )

    print(
        f"[shot={shot_name} run={run_id:02d}] "
        f"test_acc={test_metrics.get('accuracy', 0.0):.4f} "
        f"test_loss={test_metrics['loss']:.4f}"
    )
    if args.cleanup_checkpoints:
        cleanup_checkpoint_files([checkpoint_path])
    return run_metrics


def main() -> None:
    args = normalize_protocol_args(parse_args())
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")
    if args.num_runs < 1:
        raise ValueError("--num_runs must be >= 1")
    if args.way is not None and args.way < 1:
        raise ValueError("--way must be >= 1 when provided")

    if args.protocol == "fewshot":
        shots: List[ShotType] = parse_shots(args.shots)
        num_runs = args.num_runs
    else:
        shots = ["full"]
        num_runs = 1

    dataset_name = infer_dataset_name(args)
    device = resolve_device(args.gpu)
    config = build_config(args)
    set_seed(args.seed)

    raw_train_data, raw_test_data = load_raw_splits(config, args, dataset_name)
    dataset_name = str(config.get("dataset", dataset_name))
    num_classes = len(raw_train_data.class_names)
    if args.way is not None and args.way > num_classes:
        raise ValueError(f"--way ({args.way}) cannot exceed num_classes ({num_classes})")

    train_label_to_indices = build_label_to_indices(raw_train_data.labels_df, raw_train_data.all_IDs)
    test_label_to_indices = build_label_to_indices(raw_test_data.labels_df, raw_test_data.all_IDs)

    save_root = Path(args.save_dir) / dataset_name
    save_root.mkdir(parents=True, exist_ok=True)
    write_json(
        save_root / "config.json",
        {
            **vars(args),
            "dataset": dataset_name,
            "num_classes": num_classes,
            "device": str(device),
            "temp_source": str(ONEFITSALL_SRC),
        },
    )

    print("=" * 80)
    print("One-Fits-All: Few-shot Supervised Classification")
    print("=" * 80)
    print(f"time: {datetime.datetime.now()}")
    print(f"dataset: {dataset_name}")
    resolved_data_source = config.get("data_dir") or args.data_dir or args.data_path
    if resolved_data_source is not None:
        print(f"data_source: {Path(resolved_data_source).resolve()}")
    print(f"protocol: {args.protocol}")
    print(f"shots: {[shot_to_name(shot) for shot in shots]}")
    print(f"way: {args.way if args.way is not None else 'all'}")
    print(f"num_runs: {num_runs}")
    print(f"device: {device}")
    print(f"num_classes: {num_classes}")
    print(f"train_size: {len(raw_train_data.all_IDs)} | test_size: {len(raw_test_data.all_IDs)}")
    print("=" * 80)

    shot_summaries = []
    for shot_idx, shot in enumerate(shots):
        shot_run_metrics: List[Dict[str, Any]] = []
        for run_id in range(1, num_runs + 1):
            run_seed = args.fewshot_seed_base + shot_idx * 1000 + run_id
            run_metrics = run_single_experiment(
                args=args,
                config=config,
                dataset_name=dataset_name,
                shot=shot,
                shot_idx=shot_idx,
                run_id=run_id,
                run_seed=run_seed,
                save_root=save_root,
                raw_train_data=raw_train_data,
                raw_test_data=raw_test_data,
                train_label_to_indices=train_label_to_indices,
                test_label_to_indices=test_label_to_indices,
                device=device,
            )
            shot_run_metrics.append(run_metrics)

        shot_summary = aggregate_shot_results(shot=shot, run_metrics=shot_run_metrics)
        shot_summaries.append(shot_summary)

        shot_dir = save_root / f"shot_{shot_to_name(shot)}"
        shot_dir.mkdir(parents=True, exist_ok=True)
        write_json(shot_dir / "shot_summary.json", shot_summary)
        print(
            f"[shot={shot_summary['shot']}] "
            f"acc={shot_summary['accuracy_mean']:.4f}±{shot_summary['accuracy_std']:.4f}"
        )

    overall_summary = {
        "dataset": dataset_name,
        "protocol": args.protocol,
        "way": args.way if args.way is not None else num_classes,
        "num_classes": num_classes,
        "shots": [shot_to_name(shot) for shot in shots],
        "num_runs": num_runs,
        "timestamp": str(datetime.datetime.now()),
        "shot_summaries": shot_summaries,
    }
    write_json(save_root / "fewshot_summary.json", overall_summary)
    save_shot_summary_csv(save_root / "fewshot_summary.csv", shot_summaries)

    if args.protocol == "full" and shot_summaries:
        full_summary = shot_summaries[0]
        final_results = {
            "dataset": dataset_name,
            "protocol": args.protocol,
            "test_loss": full_summary.get("loss_mean"),
            "test_accuracy": full_summary.get("accuracy_mean"),
            "epochs_trained": args.epochs,
        }
        write_json(save_root / "final_results.json", final_results)

    print("=" * 80)
    print(f"Done. Results saved to: {save_root}")
    print("=" * 80)


if __name__ == "__main__":
    main()
