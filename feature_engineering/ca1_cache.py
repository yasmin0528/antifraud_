import hashlib
import os
from typing import Dict, Iterable

import pandas as pd
import torch


INPUT_FIELDS = ["amount_norm", "log_amount", "tx_type_id", "time_diff"]


def source_fingerprint(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_aml_ca1_cache(processed_path: str, cache_path: str, k: int = 10,
                        amount_mean=None, amount_std=None) -> Dict:
    data = pd.read_csv(processed_path)
    required = {"TX_ID", "Source", "Type", "Time", "AmountNorm", "LogAmount", "TimeDiff"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"AML CA1 source is missing columns: {sorted(missing)}")
    if amount_mean is not None and "Amount" not in data.columns:
        raise ValueError("Train-split normalization requires the Amount column")

    # Stable, label-free encoding. The original row order remains the graph-node order.
    type_values = sorted(data["Type"].astype(str).unique().tolist())
    type_ids = {value: idx for idx, value in enumerate(type_values)}
    if amount_mean is not None and amount_std is not None:
        amount_norm = (data["Amount"].astype(float) - float(amount_mean)) / float(amount_std)
        normalization = {"scope": "train_split", "amount_mean": float(amount_mean),
                         "amount_std": float(amount_std)}
    else:
        amount_norm = data["AmountNorm"].astype(float)
        normalization = {"scope": "source_file"}
    features = torch.tensor(
        list(zip(
            amount_norm,
            data["LogAmount"].astype(float),
            data["Type"].astype(str).map(type_ids).astype(float),
            data["TimeDiff"].astype(float),
        )), dtype=torch.float32,
    )
    n_rows = len(data)
    sequence = torch.zeros((n_rows, k, len(INPUT_FIELDS)), dtype=torch.float32)
    sequence_len = torch.zeros(n_rows, dtype=torch.long)
    padding_mask = torch.ones((n_rows, k), dtype=torch.bool)

    ordered = data.assign(_row=range(n_rows)).sort_values(
        ["Source", "Time", "TX_ID", "_row"], kind="mergesort"
    )
    histories = {}
    for _, row in ordered.iterrows():
        row_idx = int(row["_row"])
        sender = str(row["Source"])
        history = histories.setdefault(sender, [])
        selected = history[-k:]
        length = len(selected)
        if length:
            start = k - length
            sequence[row_idx, start:] = features[selected]
            padding_mask[row_idx, start:] = False
            sequence_len[row_idx] = length
        history.append(row_idx)

    artifact = {
        "sequence": sequence,
        "sequence_len": sequence_len,
        "padding_mask": padding_mask,
        "sample_ids": data["TX_ID"].astype(str).tolist(),
        "input_fields": list(INPUT_FIELDS),
        "k": int(k),
        "dataset": "aml",
        "source_file": os.path.basename(processed_path),
        "source_fingerprint": source_fingerprint(processed_path),
        "num_rows": n_rows,
        "padding_mask_true_is_pad": True,
        "type_vocabulary": type_values,
        "normalization": normalization,
    }
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    torch.save(artifact, cache_path)
    return artifact


def load_or_build_aml_ca1_cache(
    processed_path: str,
    cache_path: str,
    sample_ids: Iterable,
    k: int = 10,
    amount_mean=None,
    amount_std=None,
) -> Dict:
    expected_ids = [str(value) for value in sample_ids]
    fingerprint = source_fingerprint(processed_path)
    expected_normalization = (
        {"scope": "train_split", "amount_mean": float(amount_mean), "amount_std": float(amount_std)}
        if amount_mean is not None and amount_std is not None else {"scope": "source_file"})
    cache = None
    if os.path.exists(cache_path):
        cache = torch.load(cache_path, map_location="cpu")
        valid = (
            cache.get("dataset") == "aml"
            and cache.get("k") == k
            and cache.get("input_fields") == INPUT_FIELDS
            and cache.get("source_fingerprint") == fingerprint
            and cache.get("num_rows") == len(expected_ids)
            and cache.get("sample_ids") == expected_ids
            and cache.get("padding_mask_true_is_pad") is True
            and cache.get("normalization") == expected_normalization
        )
        if not valid:
            cache = None
    if cache is None:
        cache = build_aml_ca1_cache(processed_path, cache_path, k, amount_mean, amount_std)
    if cache["sample_ids"] != expected_ids:
        raise ValueError("CA1 cache rows do not match feat_data/graph TX_ID order")
    return cache
