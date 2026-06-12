"""Novel anomaly type holdout experiment runner.

This module intentionally keeps heavyweight training imports lazy so manifest and
aggregation commands can run on machines without the Mamba/CUDA stack.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import pickle
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np

MAIN_ANOMALY_LABELS = ("Engine Noise", "Tonal", "Dropout", "Data Gap", "Rain")
DEFAULT_SEEDS = (42, 43, 44)
DEFAULT_METHODS = ("ssl_finetune", "supervised_scratch")
DEFAULT_OUTPUT_ROOT = Path("results/anomaly_holdout")
DEFAULT_SEED42_PRETRAIN = Path(
    "data/trained_models/pretrain/"
    "amba-base-f16-t16-b16-lr1e-4-m300-custom-tr0.8-full_dataset_hydrophones_FINAL/"
    "models/pretrain-joint_best_checkpoint.pth"
)


@dataclass(frozen=True)
class ExperimentRow:
    row_id: str
    method: str
    seed: int
    k: int
    exclusion_id: str
    exclude_labels: list[str]
    data_path: str
    output_dir: str
    pretrain_checkpoint: str
    command: str


def normalize_labels(labels: Iterable[str] | None) -> list[str]:
    """Return stable, de-duplicated labels while preserving user order."""
    seen: set[str] = set()
    normalized: list[str] = []
    for label in labels or []:
        if label is None:
            continue
        value = str(label).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        normalized.append(value)
    return normalized


def label_slug(label: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")
    return slug or "none"


def exclusion_id(labels: Iterable[str] | None) -> str:
    labels = normalize_labels(labels)
    return "none" if not labels else "__".join(label_slug(label) for label in labels)


def exclusion_sets(labels: Iterable[str], max_k: int | None = None) -> list[tuple[str, ...]]:
    labels = tuple(normalize_labels(labels))
    if max_k is None:
        max_k = len(labels)
    max_k = min(max_k, len(labels))
    combos: list[tuple[str, ...]] = []
    for k in range(max_k + 1):
        combos.extend(itertools.combinations(labels, k))
    return combos


def stable_hash(values: Iterable[Any]) -> str:
    payload = json.dumps(list(values), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def split_signature(dataset: Any) -> str:
    return stable_hash(int(sample["index"]) for sample in dataset.sample_info)


def sample_has_any_label(sample: dict[str, Any], labels: Iterable[str]) -> bool:
    label_set = set(labels)
    return any(label in label_set for label in sample.get("labels", []))


def pretrain_checkpoint_for_seed(output_root: Path, seed: int, seed42_checkpoint: Path | None) -> Path:
    if seed == 42 and seed42_checkpoint and seed42_checkpoint.exists():
        return seed42_checkpoint.resolve()
    return output_root / "pretrain" / f"seed_{seed}" / "models" / "pretrain-joint_best_checkpoint.pth"


def build_manifest_rows(
    output_root: Path,
    data_path: Path,
    labels: Iterable[str] = MAIN_ANOMALY_LABELS,
    seeds: Iterable[int] = DEFAULT_SEEDS,
    methods: Iterable[str] = DEFAULT_METHODS,
    max_k: int | None = None,
    seed42_checkpoint: Path | None = DEFAULT_SEED42_PRETRAIN,
) -> list[ExperimentRow]:
    output_root = Path(output_root)
    data_path = Path(data_path)
    manifest_path = output_root / "manifest.csv"
    rows: list[ExperimentRow] = []
    for excluded in exclusion_sets(labels, max_k=max_k):
        excluded_list = list(excluded)
        k = len(excluded_list)
        excl_id = exclusion_id(excluded_list)
        for seed in seeds:
            seed = int(seed)
            pretrain_checkpoint = pretrain_checkpoint_for_seed(output_root, seed, seed42_checkpoint)
            for method in methods:
                method = str(method)
                row_id = f"{method}__seed-{seed}__k-{k}__{excl_id}"
                output_dir = output_root / "runs" / method / f"seed_{seed}" / f"k{k}_{excl_id}"
                command = (
                    "${PYTHON:-python3} -m onc_ssamba.experiments.anomaly_holdout run "
                    f"--manifest {manifest_path} --row-id {row_id} --data-path {data_path}"
                )
                rows.append(
                    ExperimentRow(
                        row_id=row_id,
                        method=method,
                        seed=seed,
                        k=k,
                        exclusion_id=excl_id,
                        exclude_labels=excluded_list,
                        data_path=str(data_path),
                        output_dir=str(output_dir),
                        pretrain_checkpoint=str(pretrain_checkpoint if method == "ssl_finetune" else ""),
                        command=command,
                    )
                )
    return rows


def write_manifest(rows: list[ExperimentRow], output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.csv"
    with manifest_path.open("w", newline="") as f:
        fieldnames = list(asdict(rows[0]).keys()) if rows else list(ExperimentRow.__dataclass_fields__)
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload = asdict(row)
            payload["exclude_labels"] = json.dumps(row.exclude_labels)
            writer.writerow(payload)
    command_path = output_root / "launch_commands.sh"
    with command_path.open("w") as f:
        f.write("#!/usr/bin/env bash\nset -euo pipefail\nPYTHON=\"${PYTHON:-python3}\"\n\n")
        for seed in sorted({row.seed for row in rows}):
            seed_rows = [row for row in rows if row.seed == seed and row.method == "ssl_finetune"]
            data_path = seed_rows[0].data_path if seed_rows else ""
            checkpoint = seed_rows[0].pretrain_checkpoint if seed_rows else ""
            existing_arg = f" --existing-checkpoint {json.dumps(checkpoint)}" if checkpoint and Path(checkpoint).exists() else ""
            f.write(
                "$PYTHON -m onc_ssamba.experiments.anomaly_holdout pretrain "
                f"--seed {seed} --output-root {output_root} --data-path {json.dumps(data_path)}{existing_arg}\n"
            )
        f.write("\n")
        for row in rows:
            f.write(row.command + "\n")
    command_path.chmod(0o755)
    return manifest_path


def read_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(newline="") as f:
        for row in csv.DictReader(f):
            row["seed"] = int(row["seed"])
            row["k"] = int(row["k"])
            raw_labels = row.get("exclude_labels", "[]")
            row["exclude_labels"] = json.loads(raw_labels) if raw_labels else []
            rows.append(row)
    return rows


def find_manifest_row(path: Path, row_id: str) -> dict[str, Any]:
    for row in read_manifest(path):
        if row["row_id"] == row_id:
            return row
    raise ValueError(f"Row id not found in manifest: {row_id}")


def build_training_args(
    *,
    task: str,
    data_path: Path,
    exp_dir: Path,
    seed: int,
    batch_size: int,
    num_workers: int,
    n_epochs: int,
    use_wandb: bool,
    wandb_project: str,
    wandb_group: str | None,
    wandb_entity: str | None,
    pretrained_path: str | None = None,
    allow_random_init_finetune: bool = False,
    resume: bool = False,
    early_stopping_patience: int = 0,
    early_stopping_metric: str | None = None,
    early_stopping_min_delta: float = 0.0,
    early_stopping_mode: str | None = None,
    debug: bool = False,
) -> SimpleNamespace:
    is_pretrain = task.startswith("pretrain_")
    args = SimpleNamespace()
    args.data_train = str(data_path)
    args.data_eval = None
    args.n_class = 2
    args.train_ratio = 0.8
    args.val_ratio = 0.1
    args.split_seed = int(seed)
    args.exclude_labels = None
    args.dataset = "custom"
    args.dataset_mean = 51.506817
    args.dataset_std = 13.638703
    args.target_length = 512
    args.num_mel_bins = 512
    args.exp_dir = str(exp_dir)
    args.lr = 1e-4 if is_pretrain else 5e-5
    args.head_lr = 10
    args.warmup = True
    args.optim = "adam"
    args.loss = "BCE"
    args.batch_size = int(batch_size)
    args.num_workers = int(num_workers)
    args.n_epochs = int(n_epochs)
    args.lr_patience = 2 if is_pretrain else 3
    args.adaptschedule = False
    args.n_print_steps = 100
    args.save_model = True
    args.save_every_epoch = False
    args.freqm = 0 if is_pretrain else 48
    args.timem = 0 if is_pretrain else 192
    args.mixup = 0.0 if is_pretrain else 0.5
    args.bal = "none" if is_pretrain else "balanced"
    args.fstride = 16 if is_pretrain else 10
    args.tstride = 16 if is_pretrain else 10
    args.fshape = 16
    args.tshape = 16
    args.model_size = "base"
    args.patch_size = 16
    args.embed_dim = 768
    args.depth = 24
    args.rms_norm = False
    args.residual_in_fp32 = False
    args.fused_add_norm = False
    args.if_rope = False
    args.if_rope_residual = False
    args.bimamba_type = "v2"
    args.drop_path_rate = 0.1
    args.stride = 16
    args.channels = 1
    args.num_classes = 2 if is_pretrain else 1
    args.multiclass = False
    args.drop_rate = 0.0
    args.norm_epsilon = 1e-5
    args.if_bidirectional = True
    args.final_pool_type = "none"
    args.if_abs_pos_embed = True
    args.if_bimamba = False
    args.if_cls_token = True
    args.if_divide_out = True
    args.use_double_cls_token = False
    args.use_middle_cls_token = False
    args.task = task
    args.mask_patch = 300 if is_pretrain else 0
    args.epoch_iter = 1
    args.main_metric = "acc" if is_pretrain else "auc"
    args.use_wandb = bool(use_wandb)
    args.wandb_entity = wandb_entity
    args.wandb_group = wandb_group
    args.wandb_project = wandb_project
    args.resume = bool(resume)
    args.early_stopping_patience = int(early_stopping_patience)
    args.early_stopping_metric = early_stopping_metric
    args.early_stopping_min_delta = float(early_stopping_min_delta)
    args.early_stopping_mode = early_stopping_mode
    args.pretrained_path = pretrained_path
    args.allow_random_init_finetune = bool(allow_random_init_finetune)
    args.use_tqdm = True
    args.debug = bool(debug)
    args.lrscheduler_start = 20
    args.lrscheduler_step = 10
    args.lrscheduler_decay = 0.5
    return args


def ensure_output_dirs(exp_dir: Path) -> None:
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "models").mkdir(parents=True, exist_ok=True)


def save_config(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def load_datasets(
    data_path: Path,
    seed: int,
    exclude_labels: list[str],
    mixup: float,
    *,
    freqm: int = 48,
    timem: int = 192,
):
    from onc_ssamba.dataset import get_onc_spectrogram_data

    datasets = get_onc_spectrogram_data(
        data_path=str(data_path),
        seed=seed,
        train_ratio=0.8,
        val_ratio=0.1,
        target_length=512,
        num_mel_bins=512,
        freqm=freqm,
        timem=timem,
        dataset_mean=51.506817,
        dataset_std=13.638703,
        mixup=mixup,
        ood=-1,
        amount=1.0,
        subsample_test=False,
        exclude_labels=exclude_labels or None,
        multiclass=False,
        num_classes=2,
    )
    if len(datasets) == 6:
        ssl_train, ssl_val, test, train, val, excluded = datasets
    else:
        ssl_train, ssl_val, test, train, val = datasets
        excluded = None
    return ssl_train, ssl_val, test, train, val, excluded


def audit_splits(train_dataset: Any, val_dataset: Any, eval_datasets: list[Any], exclude_labels: list[str]) -> dict[str, Any]:
    def count_intersections(dataset: Any) -> int:
        return sum(1 for sample in dataset.sample_info if sample_has_any_label(sample, exclude_labels))

    normal_eval = 0
    novel_eval = 0
    in_distribution_eval = 0
    full_anomalies = 0
    for dataset in eval_datasets:
        if dataset is None:
            continue
        for sample in dataset.sample_info:
            if not sample["is_anomalous"]:
                normal_eval += 1
            else:
                full_anomalies += 1
                if not exclude_labels or sample_has_any_label(sample, exclude_labels):
                    novel_eval += 1
                else:
                    in_distribution_eval += 1

    train_hits = count_intersections(train_dataset) if exclude_labels else 0
    val_hits = count_intersections(val_dataset) if exclude_labels else 0
    ok = train_hits == 0 and val_hits == 0 and normal_eval > 0 and full_anomalies > 0
    if exclude_labels:
        ok = ok and novel_eval > 0
    return {
        "ok": ok,
        "train_excluded_label_hits": train_hits,
        "val_excluded_label_hits": val_hits,
        "normal_eval_samples": normal_eval,
        "novel_eval_samples": novel_eval,
        "in_distribution_eval_samples": in_distribution_eval,
        "full_anomaly_eval_samples": full_anomalies,
        "train_signature": split_signature(train_dataset),
        "val_signature": split_signature(val_dataset),
        "eval_signature": stable_hash(
            int(sample["index"])
            for dataset in eval_datasets
            if dataset is not None
            for sample in dataset.sample_info
        ),
    }


def train_pretrain(args: argparse.Namespace) -> Path:
    output_root = Path(args.output_root)
    exp_dir = Path(args.exp_dir) if args.exp_dir else output_root / "pretrain" / f"seed_{args.seed}"
    ensure_output_dirs(exp_dir)
    if args.existing_checkpoint:
        checkpoint = Path(args.existing_checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(f"Existing checkpoint not found: {checkpoint}")
        training_args = build_training_args(
            task="pretrain_joint",
            data_path=Path(args.data_path),
            exp_dir=exp_dir,
            seed=args.seed,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            n_epochs=args.n_epochs,
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project,
            wandb_group=args.wandb_group or f"anomaly_holdout_pretrain_seed_{args.seed}",
            wandb_entity=args.wandb_entity,
            resume=args.resume,
            debug=args.debug,
        )
        config = vars(training_args).copy()
        config["mode"] = "pretrain"
        config["verified_checkpoint"] = str(checkpoint)
        save_config(
            exp_dir / "config.json",
            config,
        )
        print(f"Verified existing pretrain checkpoint: {checkpoint}")
        return checkpoint

    checkpoint = exp_dir / "models" / "pretrain-joint_best_checkpoint.pth"
    if checkpoint.exists() and args.skip_existing:
        print(f"Pretrain checkpoint already exists: {checkpoint}")
        return checkpoint

    training_args = build_training_args(
        task="pretrain_joint",
        data_path=Path(args.data_path),
        exp_dir=exp_dir,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        n_epochs=args.n_epochs,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_group=args.wandb_group or f"anomaly_holdout_pretrain_seed_{args.seed}",
        wandb_entity=args.wandb_entity,
        resume=args.resume,
        debug=args.debug,
    )
    save_config(exp_dir / "config.json", vars(training_args))
    with (exp_dir / "args.pkl").open("wb") as f:
        pickle.dump(training_args, f)

    ssl_train, ssl_val, _, _, _, _ = load_datasets(
        Path(args.data_path), args.seed, exclude_labels=[], mixup=0.0, freqm=0, timem=0
    )
    import torch
    from onc_ssamba.traintest_mask import trainmask

    train_loader = torch.utils.data.DataLoader(
        ssl_train,
        batch_size=training_args.batch_size,
        shuffle=True,
        num_workers=training_args.num_workers,
        pin_memory=False,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        ssl_val,
        batch_size=training_args.batch_size * 2,
        shuffle=False,
        num_workers=training_args.num_workers,
        pin_memory=False,
    )
    trainmask(None, train_loader, val_loader, training_args)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Expected pretrain checkpoint was not created: {checkpoint}")
    return checkpoint


def best_finetune_checkpoint(run_dir: Path) -> Path:
    return run_dir / "models" / "ft-cls_best_checkpoint.pth"


def train_run(args: argparse.Namespace) -> Path:
    row = None
    if args.manifest and args.row_id:
        row = find_manifest_row(Path(args.manifest), args.row_id)
        args.method = row["method"]
        args.seed = row["seed"]
        args.output_dir = row["output_dir"]
        args.exclude_labels = row["exclude_labels"]
        if row.get("pretrain_checkpoint") and not args.pretrained_checkpoint:
            args.pretrained_checkpoint = row["pretrain_checkpoint"]

    method = args.method
    if method not in DEFAULT_METHODS:
        raise ValueError(f"Unknown method: {method}")
    args.device = resolve_device(args.device)

    exclude_labels = normalize_labels(args.exclude_labels)
    run_dir = Path(args.output_dir) if args.output_dir else (
        Path(args.output_root) / "runs" / method / f"seed_{args.seed}" / f"k{len(exclude_labels)}_{exclusion_id(exclude_labels)}"
    )
    ensure_output_dirs(run_dir)
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists() and args.skip_existing and not args.force:
        print(f"Metrics already exist, skipping: {metrics_path}")
        return metrics_path

    pretrained_path = None
    if method == "ssl_finetune":
        pretrained_path = args.pretrained_checkpoint
        if not pretrained_path:
            candidate = pretrain_checkpoint_for_seed(
                Path(args.output_root),
                int(args.seed),
                DEFAULT_SEED42_PRETRAIN,
            )
            pretrained_path = str(candidate)
        if not Path(pretrained_path).exists():
            raise FileNotFoundError(f"SSL fine-tune requires a pretrain checkpoint: {pretrained_path}")

    config = {
        "row_id": args.row_id or f"{method}__seed-{args.seed}__k-{len(exclude_labels)}__{exclusion_id(exclude_labels)}",
        "method": method,
        "seed": int(args.seed),
        "k": len(exclude_labels),
        "exclude_labels": exclude_labels,
        "data_path": str(args.data_path),
        "output_dir": str(run_dir),
        "pretrained_checkpoint": pretrained_path,
    }
    save_config(run_dir / "config.json", config)

    training_args = build_training_args(
        task="ft_cls",
        data_path=Path(args.data_path),
        exp_dir=run_dir,
        seed=int(args.seed),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        n_epochs=args.n_epochs,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_group=args.wandb_group or f"anomaly_holdout_{method}",
        wandb_entity=args.wandb_entity,
        pretrained_path=pretrained_path,
        allow_random_init_finetune=(method == "supervised_scratch"),
        resume=args.resume,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_metric=args.early_stopping_metric,
        early_stopping_min_delta=args.early_stopping_min_delta,
        early_stopping_mode=args.early_stopping_mode,
        debug=args.debug,
    )
    training_args.exclude_labels = exclude_labels or None
    with (run_dir / "args.pkl").open("wb") as f:
        pickle.dump(training_args, f)

    ssl_train, ssl_val, test_dataset, train_dataset, val_dataset, excluded_dataset = load_datasets(
        Path(args.data_path), int(args.seed), exclude_labels, mixup=training_args.mixup
    )
    eval_datasets = [test_dataset] + ([excluded_dataset] if excluded_dataset is not None else [])
    split_audit = audit_splits(train_dataset, val_dataset, eval_datasets, exclude_labels)
    save_config(run_dir / "split_audit.json", split_audit)
    if not split_audit["ok"]:
        raise RuntimeError(f"Split audit failed. See {run_dir / 'split_audit.json'}")

    checkpoint = best_finetune_checkpoint(run_dir)
    if not args.eval_only:
        import torch
        from onc_ssamba.traintest import train

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=training_args.batch_size,
            shuffle=True,
            num_workers=training_args.num_workers,
            pin_memory=False,
            drop_last=True,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=training_args.batch_size * 2,
            shuffle=False,
            num_workers=training_args.num_workers,
            pin_memory=False,
        )
        train(None, train_loader, val_loader, training_args)

    if not checkpoint.exists():
        raise FileNotFoundError(f"Expected fine-tune checkpoint not found: {checkpoint}")
    evaluate_run(
        run_dir=run_dir,
        checkpoint_path=checkpoint,
        training_args=training_args,
        val_dataset=val_dataset,
        eval_datasets=eval_datasets,
        exclude_labels=exclude_labels,
        config=config,
        split_audit=split_audit,
        batch_size=args.eval_batch_size or args.batch_size * 2,
        num_workers=args.num_workers,
        device=args.device,
    )
    return metrics_path


def load_model_for_eval(training_args: SimpleNamespace, checkpoint_path: Path, device: str):
    import torch
    from onc_ssamba.utilities.checkpoint_utils import load_checkpoint
    from onc_ssamba.utilities.training_utils import create_model

    model = create_model(training_args)
    checkpoint = load_checkpoint(str(checkpoint_path), torch.device(device))
    state_dict = checkpoint["model_state_dict"]
    if any(key.startswith("module.") for key in state_dict):
        state_dict = {key[7:] if key.startswith("module.") else key: value for key, value in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[WARN] Missing keys while loading eval checkpoint: {missing[:10]}")
    if unexpected:
        print(f"[WARN] Unexpected keys while loading eval checkpoint: {unexpected[:10]}")
    model = model.to(device)
    model.eval()
    return model


def resolve_device(requested: str) -> str:
    import torch

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable; using CPU for evaluation")
        return "cpu"
    return requested


def predict_dataset(model: Any, dataset: Any, *, batch_size: int, num_workers: int, device: str) -> list[dict[str, Any]]:
    import torch

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.startswith("cuda")),
    )
    scores: list[float] = []
    with torch.no_grad():
        for data, _, _ in loader:
            data = data.to(device)
            logits = model(data, "ft_cls").squeeze(-1)
            probabilities = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
            scores.extend(float(value) for value in probabilities)
    if len(scores) != len(dataset.sample_info):
        raise RuntimeError("Prediction count does not match dataset sample count")
    rows: list[dict[str, Any]] = []
    for score, sample in zip(scores, dataset.sample_info):
        labels = normalize_labels(sample.get("labels", []))
        rows.append(
            {
                "index": int(sample["index"]),
                "source": sample.get("source"),
                "labels": ";".join(labels),
                "is_anomalous": bool(sample["is_anomalous"]),
                "is_excluded": bool(sample.get("is_excluded", False)),
                "score": float(score),
            }
        )
    return rows


def threshold_for_fpr(normal_scores: np.ndarray, target_fpr: float) -> float:
    if len(normal_scores) == 0:
        return math.inf
    q = max(0.0, min(1.0, 1.0 - target_fpr))
    try:
        return float(np.quantile(normal_scores, q, method="higher"))
    except TypeError:
        return float(np.quantile(normal_scores, q, interpolation="higher"))


def binary_metrics(scores: np.ndarray, labels: np.ndarray, normal_thresholds: dict[str, float]) -> dict[str, Any]:
    from sklearn import metrics as sk_metrics

    result: dict[str, Any] = {
        "n": int(len(labels)),
        "positives": int(np.sum(labels == 1)),
        "negatives": int(np.sum(labels == 0)),
    }
    if len(np.unique(labels)) == 2:
        result["auroc"] = float(sk_metrics.roc_auc_score(labels, scores))
    else:
        result["auroc"] = None
    for name, threshold in normal_thresholds.items():
        pred = scores >= threshold
        pos_mask = labels == 1
        neg_mask = labels == 0
        result[f"threshold_{name}"] = float(threshold) if math.isfinite(threshold) else None
        result[f"recall_at_{name}"] = float(np.mean(pred[pos_mask])) if np.any(pos_mask) else None
        result[f"actual_fpr_at_{name}"] = float(np.mean(pred[neg_mask])) if np.any(neg_mask) else None
    return result


def rows_to_arrays(rows: list[dict[str, Any]], mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    selected = [row for row, keep in zip(rows, mask) if keep]
    return (
        np.array([row["score"] for row in selected], dtype=float),
        np.array([1 if row["is_anomalous"] else 0 for row in selected], dtype=int),
    )


def evaluate_predictions(
    eval_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    exclude_labels: list[str],
) -> dict[str, Any]:
    val_normal_scores = np.array([row["score"] for row in val_rows if not row["is_anomalous"]], dtype=float)
    thresholds = {
        "fpr_1pct": threshold_for_fpr(val_normal_scores, 0.01),
        "fpr_5pct": threshold_for_fpr(val_normal_scores, 0.05),
    }
    normal_mask = np.array([not row["is_anomalous"] for row in eval_rows], dtype=bool)
    anomaly_mask = np.array([row["is_anomalous"] for row in eval_rows], dtype=bool)
    if exclude_labels:
        novel_pos_mask = np.array(
            [
                row["is_anomalous"]
                and bool(set(filter(None, row["labels"].split(";"))).intersection(exclude_labels))
                for row in eval_rows
            ],
            dtype=bool,
        )
    else:
        novel_pos_mask = anomaly_mask.copy()
    in_distribution_pos_mask = anomaly_mask & ~novel_pos_mask

    subset_masks = {
        "novel": normal_mask | novel_pos_mask,
        "in_distribution": normal_mask | in_distribution_pos_mask,
        "full": normal_mask | anomaly_mask,
    }
    metrics: dict[str, Any] = {}
    for subset, mask in subset_masks.items():
        scores, labels = rows_to_arrays(eval_rows, mask)
        metrics[subset] = binary_metrics(scores, labels, thresholds)
    return metrics


def write_predictions_csv(path: Path, rows: list[dict[str, Any]], exclude_labels: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        fieldnames = [
            "index",
            "source",
            "labels",
            "is_anomalous",
            "is_excluded",
            "is_novel",
            "score",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            label_set = set(filter(None, row["labels"].split(";")))
            is_novel = row["is_anomalous"] and (not exclude_labels or bool(label_set.intersection(exclude_labels)))
            payload = dict(row)
            payload["is_novel"] = is_novel
            writer.writerow(payload)


def evaluate_run(
    *,
    run_dir: Path,
    checkpoint_path: Path,
    training_args: SimpleNamespace,
    val_dataset: Any,
    eval_datasets: list[Any],
    exclude_labels: list[str],
    config: dict[str, Any],
    split_audit: dict[str, Any],
    batch_size: int,
    num_workers: int,
    device: str,
) -> dict[str, Any]:
    model = load_model_for_eval(training_args, checkpoint_path, device)
    val_rows = predict_dataset(
        model,
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )
    eval_rows: list[dict[str, Any]] = []
    for dataset in eval_datasets:
        eval_rows.extend(
            predict_dataset(
                model,
                dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                device=device,
            )
        )
    metrics = evaluate_predictions(eval_rows, val_rows, exclude_labels)
    write_predictions_csv(run_dir / "predictions.csv", eval_rows, exclude_labels)
    result = {
        **config,
        "checkpoint_path": str(checkpoint_path),
        "predictions_path": str(run_dir / "predictions.csv"),
        "split_audit": split_audit,
        "metrics": metrics,
    }
    save_config(run_dir / "metrics.json", result)
    return result


def flatten_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "row_id": payload.get("row_id"),
        "method": payload.get("method"),
        "seed": payload.get("seed"),
        "k": payload.get("k"),
        "exclusion_id": exclusion_id(payload.get("exclude_labels", [])),
        "exclude_labels": ";".join(payload.get("exclude_labels", [])),
        "data_path": payload.get("data_path"),
        "output_dir": payload.get("output_dir"),
        "checkpoint_path": payload.get("checkpoint_path"),
    }
    for subset, subset_metrics in payload.get("metrics", {}).items():
        for key, value in subset_metrics.items():
            row[f"{subset}_{key}"] = value
    audit = payload.get("split_audit", {})
    for key in ("ok", "train_signature", "val_signature", "eval_signature"):
        row[f"audit_{key}"] = audit.get(key)
    return row


def collect_metric_files(output_root: Path) -> list[Path]:
    return sorted((output_root / "runs").glob("*/*/*/metrics.json"))


def write_metrics_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(args: argparse.Namespace) -> Path:
    output_root = Path(args.output_root)
    metric_files = collect_metric_files(output_root)
    rows = []
    for path in metric_files:
        with path.open() as f:
            rows.append(flatten_metrics(json.load(f)))
    metrics_csv = output_root / "metrics.csv"
    write_metrics_csv(rows, metrics_csv)
    audit = build_aggregate_audit(output_root, rows, Path(args.manifest) if args.manifest else None)
    save_config(output_root / "audit.json", audit)
    if rows:
        make_plots(metrics_csv, output_root / "plots")
    print(f"Wrote {len(rows)} metric rows to {metrics_csv}")
    print(f"Wrote audit to {output_root / 'audit.json'}")
    return metrics_csv


def build_aggregate_audit(output_root: Path, rows: list[dict[str, Any]], manifest_path: Path | None) -> dict[str, Any]:
    completed = {row["row_id"] for row in rows}
    missing: list[str] = []
    missing_commands: list[str] = []
    if manifest_path and manifest_path.exists():
        for row in read_manifest(manifest_path):
            if row["row_id"] not in completed:
                missing.append(row["row_id"])
                if row.get("command"):
                    missing_commands.append(row["command"])
    failed_logs = []
    for path in sorted((output_root / "runs").glob("*/*/*")):
        if (path / "metrics.json").exists():
            continue
        for candidate in (path / "error.txt", path / "stderr.txt"):
            if candidate.exists():
                failed_logs.append(str(candidate))
    signature_mismatches = split_signature_mismatches(rows)
    return {
        "completed_rows": len(completed),
        "missing_rows": missing,
        "missing_commands": missing_commands,
        "failed_logs": failed_logs,
        "split_signature_mismatches": signature_mismatches,
    }


def split_signature_mismatches(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_experiment: dict[tuple[Any, Any, Any], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row.get("seed"), row.get("k"), row.get("exclusion_id"))
        by_experiment.setdefault(key, []).append(row)

    mismatches: list[dict[str, Any]] = []
    for (seed, k, excl_id), group in sorted(by_experiment.items(), key=lambda item: item[0]):
        signatures = {
            row.get("method"): (
                row.get("audit_train_signature"),
                row.get("audit_val_signature"),
                row.get("audit_eval_signature"),
            )
            for row in group
        }
        present_methods = set(signatures)
        if not set(DEFAULT_METHODS).issubset(present_methods):
            continue
        if len(set(signatures.values())) > 1:
            mismatches.append(
                {
                    "seed": seed,
                    "k": k,
                    "exclusion_id": excl_id,
                    "signatures": signatures,
                }
            )
    return mismatches


def make_plots(metrics_csv: Path, plots_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(metrics_csv)
    plots_dir.mkdir(parents=True, exist_ok=True)
    if "novel_auroc" in df:
        plot_metric_by_k(df, "novel_auroc", plots_dir / "novel_auroc_by_k.png", "Novel anomaly AUROC", "AUROC")
    if "in_distribution_auroc" in df:
        plot_metric_by_k(
            df,
            "in_distribution_auroc",
            plots_dir / "in_distribution_auroc_by_k.png",
            "In-distribution anomaly AUROC",
            "AUROC",
        )
    if "novel_recall_at_fpr_1pct" in df:
        plot_metric_by_k(
            df,
            "novel_recall_at_fpr_1pct",
            plots_dir / "novel_recall_at_1pct_fpr_by_k.png",
            "Novel anomaly recall at 1% validation-normal FPR",
            "Recall",
        )
    if "novel_recall_at_fpr_5pct" in df:
        plot_metric_by_k(
            df,
            "novel_recall_at_fpr_5pct",
            plots_dir / "novel_recall_at_5pct_fpr_by_k.png",
            "Novel anomaly recall at 5% validation-normal FPR",
            "Recall",
        )
    make_delta_plot(df, plots_dir / "ssl_minus_supervised_delta_by_k.png")
    make_leave_one_out_plot(df, plots_dir / "leave_one_type_out_novel_auroc.png")


def plot_metric_by_k(df: Any, column: str, output_path: Path, title: str, ylabel: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method, group in df.dropna(subset=[column]).groupby("method"):
        summary = group.groupby("k")[column].agg(["mean", "std"]).reset_index()
        ax.errorbar(summary["k"], summary["mean"], yerr=summary["std"], marker="o", capsize=4, label=method)
    ax.set_xlabel("# anomaly types removed from training")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(sorted(df["k"].dropna().unique()))
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def make_delta_plot(df: Any, output_path: Path) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd

    if "novel_auroc" not in df:
        return
    paired = df.pivot_table(
        index=["seed", "exclusion_id", "k"],
        columns="method",
        values="novel_auroc",
        aggfunc="mean",
    ).reset_index()
    if not {"ssl_finetune", "supervised_scratch"}.issubset(paired.columns):
        return
    paired["delta"] = paired["ssl_finetune"] - paired["supervised_scratch"]
    summary = paired.groupby("k")["delta"].agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(summary["k"], summary["mean"], yerr=summary["std"], marker="o", capsize=4, color="#2b6cb0")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("# anomaly types removed from training")
    ax.set_ylabel("SSL AUROC - supervised AUROC")
    ax.set_title("Novel anomaly generalization delta")
    ax.set_xticks(sorted(paired["k"].dropna().unique()))
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def make_leave_one_out_plot(df: Any, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    if "novel_auroc" not in df:
        return
    loo = df[df["k"] == 1].dropna(subset=["novel_auroc"]).copy()
    if loo.empty:
        return
    loo["held_out"] = loo["exclude_labels"].str.replace(";", "", regex=False)
    summary = loo.groupby(["held_out", "method"])["novel_auroc"].mean().reset_index()
    labels = sorted(summary["held_out"].unique())
    methods = [method for method in DEFAULT_METHODS if method in set(summary["method"])]
    x = np.arange(len(labels))
    width = 0.8 / max(1, len(methods))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, method in enumerate(methods):
        values = []
        method_df = summary[summary["method"] == method]
        for label in labels:
            series = method_df[method_df["held_out"] == label]["novel_auroc"]
            values.append(float(series.iloc[0]) if len(series) else np.nan)
        ax.bar(x + (i - (len(methods) - 1) / 2) * width, values, width=width, label=method)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("AUROC")
    ax.set_title("Leave-one-anomaly-type-out novel AUROC")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def parse_label_args(args: argparse.Namespace) -> list[str]:
    labels = []
    for value in args.exclude_label or []:
        labels.append(value)
    for value in args.exclude_labels or []:
        labels.extend(part for part in value.split(",") if part.strip())
    return normalize_labels(labels)


def command_manifest(args: argparse.Namespace) -> Path:
    labels = args.labels or list(MAIN_ANOMALY_LABELS)
    seeds = args.seeds or list(DEFAULT_SEEDS)
    methods = args.methods or list(DEFAULT_METHODS)
    rows = build_manifest_rows(
        output_root=Path(args.output_root),
        data_path=Path(args.data_path),
        labels=labels,
        seeds=seeds,
        methods=methods,
        max_k=args.max_k,
        seed42_checkpoint=Path(args.seed42_checkpoint) if args.seed42_checkpoint else None,
    )
    manifest_path = write_manifest(rows, Path(args.output_root))
    print(f"Wrote {len(rows)} rows to {manifest_path}")
    print(f"Wrote launch commands to {Path(args.output_root) / 'launch_commands.sh'}")
    return manifest_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Novel anomaly type holdout experiment runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest = subparsers.add_parser("manifest", help="Generate the full experiment manifest")
    manifest.add_argument("--data-path", required=True)
    manifest.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    manifest.add_argument("--labels", nargs="+", default=list(MAIN_ANOMALY_LABELS))
    manifest.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    manifest.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    manifest.add_argument("--max-k", type=int, default=None)
    manifest.add_argument("--seed42-checkpoint", default=str(DEFAULT_SEED42_PRETRAIN))
    manifest.set_defaults(func=command_manifest)

    pretrain = subparsers.add_parser("pretrain", help="Train or verify one SSL pretrain checkpoint")
    pretrain.add_argument("--data-path", default="")
    pretrain.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    pretrain.add_argument("--exp-dir", default="")
    pretrain.add_argument("--seed", type=int, required=True)
    pretrain.add_argument("--existing-checkpoint", default="")
    pretrain.add_argument("--skip-existing", action="store_true")
    pretrain.add_argument("--resume", action="store_true")
    pretrain.add_argument("--batch-size", type=int, default=16)
    pretrain.add_argument("--num-workers", type=int, default=8)
    pretrain.add_argument("--n-epochs", type=int, default=200)
    pretrain.add_argument("--use-wandb", action="store_true")
    pretrain.add_argument("--wandb-project", default="amba_spectrogram_pretrain")
    pretrain.add_argument("--wandb-group", default=None)
    pretrain.add_argument("--wandb-entity", default=None)
    pretrain.add_argument("--debug", action="store_true")
    pretrain.set_defaults(func=train_pretrain)

    run = subparsers.add_parser("run", help="Run one fine-tune/evaluation row")
    run.add_argument("--data-path", required=True)
    run.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    run.add_argument("--manifest", default="")
    run.add_argument("--row-id", default="")
    run.add_argument("--method", choices=list(DEFAULT_METHODS), default="ssl_finetune")
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--output-dir", default="")
    run.add_argument("--exclude-label", action="append", default=[])
    run.add_argument("--exclude-labels", nargs="*", default=[])
    run.add_argument("--pretrained-checkpoint", default="")
    run.add_argument("--eval-only", action="store_true")
    run.add_argument("--skip-existing", action="store_true")
    run.add_argument("--force", action="store_true")
    run.add_argument("--resume", action="store_true")
    run.add_argument("--batch-size", type=int, default=16)
    run.add_argument("--eval-batch-size", type=int, default=0)
    run.add_argument("--num-workers", type=int, default=8)
    run.add_argument("--n-epochs", type=int, default=200)
    run.add_argument("--early-stopping-patience", type=int, default=0)
    run.add_argument("--early-stopping-metric", default=None)
    run.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    run.add_argument("--early-stopping-mode", choices=("min", "max"), default=None)
    run.add_argument("--device", default="auto")
    run.add_argument("--use-wandb", action="store_true")
    run.add_argument("--wandb-project", default="amba_spectrogram_finetune")
    run.add_argument("--wandb-group", default=None)
    run.add_argument("--wandb-entity", default=None)
    run.add_argument("--debug", action="store_true")
    run.set_defaults(func=lambda ns: train_run(_with_parsed_labels(ns)))

    aggregate_parser = subparsers.add_parser("aggregate", help="Aggregate completed runs and create plots")
    aggregate_parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    aggregate_parser.add_argument("--manifest", default="")
    aggregate_parser.set_defaults(func=aggregate)
    return parser


def _with_parsed_labels(args: argparse.Namespace) -> argparse.Namespace:
    args.exclude_labels = parse_label_args(args)
    return args


def main(argv: list[str] | None = None) -> Any:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    main()
