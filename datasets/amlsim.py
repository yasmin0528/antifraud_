"""
AMLSIM dataset — loading, preprocessing, temporal splitting, CA1 caching.

Usage:
    from datasets.amlsim import load_amlsim_data
    feat_df, labels, train_idx, val_idx, test_idx, g, cat_features, neigh_features = \
        load_amlsim_data(args)

Data flow:
    Trans.csv + _accounts.csv
        → preprocess_amlsim()    (writes processed CSV + graph + CA1 cache)
        → load_amlsim_data()     (reads artifacts, temporal split, encoding)
        → training loop
"""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import dgl
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

CA1_INPUT_FIELDS = [
    "log_amount_paid", "log_amount_received", "amount_norm",
    "time_diff", "payment_format_encoded", "cross_bank", "time_hour",
]

LABEL_COL = "Is Laundering"
CAT_FEATURES_BASE = ["PaymentFormat", "CurrencyPaid", "CurrencyReceived",
                     "FromBank", "ToBank"]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Preprocessing pipeline
# ─────────────────────────────────────────────────────────────────────────────


def preprocess_amlsim(
    trans_path: str,
    accounts_path: str,
    output_dir: str,
    edge_per_trans: int = 3,
    force: bool = False,
) -> Dict[str, str]:
    """Read raw AMLSIM CSV, build processed artifacts.

    Parameters
    ----------
    trans_path:
        Path to ``HI-Small_Trans.csv`` (or LI variant).
    accounts_path:
        Path to ``HI-Small_accounts.csv``.
    output_dir:
        Directory to write artifacts into.
    edge_per_trans:
        Graph temporal edge count per transaction.
    force:
        Overwrite existing artifacts.

    Returns
    -------
    dict with keys:
        processed_path, feat_path, label_path, graph_path, ca1_cache_path
    """
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(trans_path))[0].replace("_Trans", "")
    processed_path = os.path.join(output_dir, f"{base}_processed.csv")
    feat_path = os.path.join(output_dir, f"{base}_feat_data.csv")
    label_path = os.path.join(output_dir, f"{base}_label_data.csv")
    graph_path = os.path.join(output_dir, f"graph-{base}.bin")
    ca1_cache_path = os.path.join(output_dir, f"{base}_ca1_k10.pt")

    if not force and all(os.path.exists(p) for p in
                         [processed_path, feat_path, label_path, graph_path]):
        return {
            "processed_path": processed_path,
            "feat_path": feat_path,
            "label_path": label_path,
            "graph_path": graph_path,
            "ca1_cache_path": ca1_cache_path,
        }

    # ── Read raw ──────────────────────────────────────────────────────
    logger.info("Reading transactions from %s ...", trans_path)
    data = pd.read_csv(trans_path)
    n_raw = len(data)
    logger.info("  rows: %d", n_raw)

    if data.columns[0].startswith("Unnamed"):
        data = data.drop(columns=[data.columns[0]])

    # Read accounts for entity mapping (metadata only)
    accounts_df = None
    if os.path.exists(accounts_path):
        accounts_df = pd.read_csv(accounts_path)
        logger.info("  accounts: %d", len(accounts_df))

    # ── Timestamp parsing ─────────────────────────────────────────────
    data["_datetime"] = pd.to_datetime(data["Timestamp"], format="%Y/%m/%d %H:%M")
    data["_day"] = (data["_datetime"] - data["_datetime"].min()).dt.days + 1  # 1-indexed
    data["_hour"] = data["_datetime"].dt.hour.astype(int)
    data["_dow"] = data["_datetime"].dt.dayofweek.astype(int)
    data["_timestamp_s"] = data["_datetime"].view(np.int64) // 10**9

    # ── Account key (From side, used for sorting & CA1 grouping) ──────
    data["AccountKey"] = data["Account"].astype(str)  # From account

    # ── Sort by account + time ────────────────────────────────────────
    data = data.sort_values(
        by=["AccountKey", "_timestamp_s", "Amount Paid"],
        kind="mergesort",
    ).reset_index(drop=True)

    # ── Feature engineering ───────────────────────────────────────────
    data["LogAmountPaid"] = np.log1p(data["Amount Paid"].clip(lower=0.0).astype(float))
    data["LogAmountReceived"] = np.log1p(data["Amount Received"].clip(lower=0.0).astype(float))
    data["CrossBank"] = (data["From Bank"].astype(str) != data["To Bank"].astype(str)).astype(float)

    # TimeDiff (per-account, sorted)
    data["TimeDiff"] = (
        data.groupby("AccountKey")["_timestamp_s"].diff().fillna(0.0).astype(float)
    )

    # ── Label encoding (fit on all data, stored for reuse) ────────────
    encoders = {}
    for col in CAT_FEATURES_BASE:
        mapped_col = col  # some columns need mapping from raw names
        # Map raw column names to Trans.csv columns
        col_map = {
            "PaymentFormat": "Payment Format",
            "CurrencyPaid": "Payment Currency",
            "CurrencyReceived": "Receiving Currency",
            "FromBank": "From Bank",
            "ToBank": "To Bank",
        }
        raw_col = col_map.get(col, col)
        if raw_col not in data.columns:
            logger.warning("Column %s not found, skipping encoding", raw_col)
            continue
        le = LabelEncoder()
        data[col] = le.fit_transform(data[raw_col].astype(str))
        encoders[col] = le

    # ── Build processed DataFrame ─────────────────────────────────────
    processed = pd.DataFrame({
        "TX_ID": data.index.astype(int),
        "FromBank": data["From Bank"].astype(int),
        "FromAccount": data["Account"].astype(str),
        "ToBank": data["To Bank"].astype(int),
        "ToAccount": data["Account.1"].astype(str),  # second Account column
        "AmountPaid": data["Amount Paid"].astype(float),
        "AmountReceived": data["Amount Received"].astype(float),
        "LogAmountPaid": data["LogAmountPaid"].astype(float),
        "LogAmountReceived": data["LogAmountReceived"].astype(float),
        "CrossBank": data["CrossBank"].astype(float),
        "TimeDiff": data["TimeDiff"].astype(float),
        "Time": data["_timestamp_s"].astype(float),
        "Day": data["_day"].astype(int),
        "TimeHour": data["_hour"].astype(int),
        "TimeDayOfWeek": data["_dow"].astype(int),
        "AccountKey": data["AccountKey"],
        "Labels": data["Is Laundering"].astype(int),
    })
    for col in CAT_FEATURES_BASE:
        if col in data.columns:
            processed[col] = data[col].astype(int)

    # ── Build graph ───────────────────────────────────────────────────
    logger.info("Building transaction graph ...")
    g = _build_amlsim_transaction_graph(processed, edge_per_trans)
    logger.info("  graph: %d nodes, %d edges", g.num_nodes(), g.num_edges())

    # ── Save ──────────────────────────────────────────────────────────
    processed_cols = [c for c in processed.columns if c != "AccountKey"]
    processed[processed_cols].to_csv(processed_path, index=False)
    labels = processed["Labels"].astype(int)
    labels.to_frame(name="Labels").to_csv(label_path, index=False)
    feat_data = processed[processed_cols].drop(columns=["Labels"])
    feat_data.to_csv(feat_path, index=False)
    g.ndata["label"] = torch.from_numpy(labels.to_numpy()).to(torch.long)
    dgl.data.utils.save_graphs(graph_path, [g])

    # ── CA1 cache ─────────────────────────────────────────────────────
    build_amlsim_ca1_cache(processed, ca1_cache_path)

    logger.info("Preprocessing complete → %s", output_dir)
    return {
        "processed_path": processed_path,
        "feat_path": feat_path,
        "label_path": label_path,
        "graph_path": graph_path,
        "ca1_cache_path": ca1_cache_path,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. Graph builder
# ─────────────────────────────────────────────────────────────────────────────


def _build_amlsim_transaction_graph(
    processed: pd.DataFrame, edge_per_trans: int = 3,
) -> dgl.DGLGraph:
    """Build a transaction-level DGL graph.

    Each node = one transaction.
    Edges connect transactions from the same sender account (AccountKey)
    within a local temporal window.

    Self-loops are added with zero-weight edge features.
    """
    all_src: List[int] = []
    all_tgt: List[int] = []

    for _, group_df in tqdm(
        processed.groupby("AccountKey"), desc="graph edges", leave=False,
    ):
        group_df = group_df.sort_values(by="Time")
        idxs = group_df.index.to_list()
        n = len(idxs)
        for i in range(n):
            upper = min(i + 1 + edge_per_trans, n)
            src = [idxs[i]] * (upper - i - 1)
            tgt = idxs[i + 1:upper]
            all_src.extend(src)
            all_tgt.extend(tgt)

    g = dgl.graph(
        (np.array(all_src, dtype=np.int64), np.array(all_tgt, dtype=np.int64)),
        num_nodes=len(processed),
    )
    g = dgl.add_self_loop(g)
    return g


# ─────────────────────────────────────────────────────────────────────────────
# 3. CA1 cache
# ─────────────────────────────────────────────────────────────────────────────


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_amlsim_ca1_cache(
    processed: pd.DataFrame,
    cache_path: str,
    k: int = 10,
) -> dict:
    """Build CA1 sequence cache from processed DataFrame.

    For each transaction (row), records the last ``k`` transactions
    from the same ``AccountKey``, right-aligned.

    Fields stored: see ``CA1_INPUT_FIELDS``.
    """
    logger.info("Building CA1 cache → %s ...", cache_path)

    account_col = "AccountKey"
    time_col = "Time"

    col_map = {
        "log_amount_paid": "LogAmountPaid",
        "log_amount_received": "LogAmountReceived",
        "amount_norm": "LogAmountPaid",  # placeholder, fine-tune later
        "time_diff": "TimeDiff",
        "payment_format_encoded": "PaymentFormat",
        "cross_bank": "CrossBank",
        "time_hour": "TimeHour",
    }
    features_list = []
    for field in CA1_INPUT_FIELDS:
        src_col = col_map[field]
        features_list.append(
            torch.from_numpy(processed[src_col].astype(float).values).unsqueeze(1)
        )
    features = torch.cat(features_list, dim=1).float()  # [N, D]
    n_rows, feat_dim = features.shape

    sequence = torch.zeros((n_rows, k, feat_dim), dtype=torch.float32)
    sequence_len = torch.zeros(n_rows, dtype=torch.long)
    padding_mask = torch.ones((n_rows, k), dtype=torch.bool)

    # Sort by account + time to build histories
    ordered = processed.assign(_row=range(n_rows)).sort_values(
        [account_col, time_col, "_row"], kind="mergesort",
    )
    histories: dict = {}
    for _, row in tqdm(ordered.iterrows(), total=n_rows, desc="CA1 cache",
                       leave=False):
        row_idx = int(row["_row"])
        acct = str(row[account_col])
        history = histories.setdefault(acct, [])

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
        "input_fields": list(CA1_INPUT_FIELDS),
        "k": int(k),
        "dataset": "amlsim",
        "num_rows": n_rows,
        "padding_mask_true_is_pad": True,
    }
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    torch.save(artifact, cache_path)
    logger.info("  CA1 cache saved: %d rows x k=%d x %d dims", n_rows, k, feat_dim)
    return artifact


def load_or_build_amlsim_ca1_cache(
    processed: pd.DataFrame,
    cache_path: str,
    sample_ids: List,
    k: int = 10,
) -> dict:
    """Load or build AMLSIM CA1 cache with fingerprint validation."""
    expected_ids = [str(v) for v in sample_ids]
    fps = _sha256_file(cache_path.replace("_ca1_k10.pt", "_feat_data.csv"))
    cache = None
    if os.path.exists(cache_path):
        cache = torch.load(cache_path, map_location="cpu")
        valid = (
            cache.get("dataset") == "amlsim"
            and cache.get("k") == k
            and cache.get("input_fields") == CA1_INPUT_FIELDS
            and cache.get("num_rows") == len(expected_ids)
        )
        if valid:
            return cache
    if cache is None:
        cache = build_amlsim_ca1_cache(processed, cache_path, k)
    return cache


# ─────────────────────────────────────────────────────────────────────────────
# 4. Training data loader
# ─────────────────────────────────────────────────────────────────────────────


def _amlsim_sourcetest_split(
    group_ids: pd.Series,
    labels: pd.Series,
    test_size: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Account-level split: ensures no account crosses train/test."""
    from sklearn.model_selection import train_test_split
    group_df = pd.DataFrame({
        "group_id": group_ids.astype(str),
        "label": labels.astype(int),
    })
    group_level = group_df.groupby("group_id")["label"].max().reset_index()
    group_list = group_level["group_id"].to_numpy()
    group_labels = group_level["label"]

    stratify = None
    if group_labels.nunique() >= 2 and group_labels.value_counts().min() >= 2:
        stratify = group_labels

    train_groups, test_groups = train_test_split(
        group_list, test_size=test_size, random_state=seed,
        shuffle=True, stratify=stratify,
    )
    train_mask = group_ids.astype(str).isin(train_groups)
    test_mask = group_ids.astype(str).isin(test_groups)
    return np.flatnonzero(train_mask.to_numpy()), np.flatnonzero(test_mask.to_numpy())


def _amlsim_temporal_split(
    processed: pd.DataFrame,
    train_days: int = 14,
    val_days: int = 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Temporal split by Day column.

    Day 1 → train_days:               Training
    train_days+1 → train_days+val_days:  Validation
    train_days+val_days+1 → end:      Test
    """
    days = processed["Day"].values.astype(int)
    day_min = int(days.min())
    day_max = int(days.max())

    train_end = day_min + train_days - 1
    val_end = train_end + val_days

    train_idx = np.flatnonzero(days <= train_end)
    val_idx = np.flatnonzero((days > train_end) & (days <= val_end))
    test_idx = np.flatnonzero(days > val_end)

    logger.info(
        "Temporal split: days %d-%d train (%d), %d-%d val (%d), %d-%d test (%d)",
        day_min, train_end, len(train_idx),
        train_end + 1, val_end, len(val_idx),
        val_end + 1, day_max, len(test_idx),
    )

    # Leakage audit: count accounts spanning splits (expected in temporal split)
    account_col = "FromAccount" if "FromAccount" in processed.columns else "AccountKey"
    train_accounts = set(processed.iloc[train_idx][account_col].astype(str))
    test_accounts = set(processed.iloc[test_idx][account_col].astype(str))
    overlap = train_accounts & test_accounts
    if overlap:
        logger.info(
            "Leakage audit: %d/%d train accounts also appear in test "
            "(expected in temporal split)", len(overlap), len(train_accounts),
        )

    return train_idx, val_idx, test_idx


# ── Neighbor features ───────────────────────────────────────────────────


def _build_amlsim_neigh_features(graph: dgl.DGLGraph) -> pd.DataFrame:
    """Per-node (transaction) graph structural features."""
    in_deg = graph.in_degrees().cpu().numpy().astype(np.float32)
    out_deg = graph.out_degrees().cpu().numpy().astype(np.float32)
    return pd.DataFrame({
        "degree": in_deg,
        "out_degree": out_deg,
    })


# ── Main load function ──────────────────────────────────────────────────


def resolve_amlsim_paths(args: dict) -> Dict[str, str]:
    """Resolve AMLSIM file paths from args (or defaults for HI-Small)."""
    variant = args.get("amlsim_variant", "HI-Small")
    data_root = args.get("data_path", "data/AMLSIM")
    if not os.path.isabs(data_root):
        project_root = os.path.abspath(os.path.join(
            os.path.dirname(__file__), ".."))
        data_root = os.path.join(project_root, data_root)

    trans_path = os.path.join(data_root, f"{variant}_Trans.csv")
    accounts_path = os.path.join(data_root, f"{variant}_accounts.csv")
    output_dir = os.path.join(data_root, f"{variant}_processed")
    return {
        "trans_path": trans_path,
        "accounts_path": accounts_path,
        "output_dir": output_dir,
        "variant": variant,
        "data_root": data_root,
    }


def load_amlsim_data(args: dict) -> Tuple:
    """Load AMLSIM data, perform temporal split, return training interface.

    Mutates ``args`` with internal keys ``_val_idx``, ``_amlsim_processed_path``,
    ``_amlsim_sample_ids``, ``_amlsim_amount_mean``, ``_amlsim_amount_std``.

    Returns
    -------
    (feat_df, labels, train_idx, val_idx, test_idx, g, cat_features, neigh_features)
    """
    paths = resolve_amlsim_paths(args)

    # 1. Preprocess if needed
    artifacts = preprocess_amlsim(
        trans_path=paths["trans_path"],
        accounts_path=paths["accounts_path"],
        output_dir=paths["output_dir"],
        force=args.get("force_preprocess", False),
    )

    # 2. Load processed data
    processed = pd.read_csv(artifacts["processed_path"])
    feat_data = pd.read_csv(artifacts["feat_path"])
    labels = pd.read_csv(artifacts["label_path"])["Labels"].astype(int)
    g = dgl.load_graphs(artifacts["graph_path"])[0][0]

    alignment_ok = (
        len(processed) == len(feat_data) == len(labels) == g.num_nodes()
    )
    if not alignment_ok:
        raise ValueError(
            f"AMLSIM row count mismatch: processed={len(processed)}, "
            f"feat={len(feat_data)}, labels={len(labels)}, "
            f"graph={g.num_nodes()}"
        )
    logger.info("Loaded %d transactions, graph: %d nodes, %d edges",
                len(processed), g.num_nodes(), g.num_edges())

    # 3. Label encode categorical features
    cat_features = CAT_FEATURES_BASE
    for col in cat_features:
        if col in feat_data.columns and feat_data[col].dtype == object:
            le = LabelEncoder()
            feat_data[col] = le.fit_transform(feat_data[col].astype(str))

    # 4. Drop forbidden/leakage columns
    forbidden = {"Labels", "TX_ID", "AccountKey", "FromAccount", "ToAccount",
                 "Day", "Time"}
    present_forbidden = forbidden.intersection(feat_data.columns)
    if present_forbidden:
        feat_data = feat_data.drop(columns=list(present_forbidden))
        logger.info("Dropped forbidden columns: %s", sorted(present_forbidden))

    # 5. Temporal split
    train_days = int(args.get("amlsim_train_days", 14))
    val_days = int(args.get("amlsim_val_days", 2))
    train_idx, val_idx, test_idx = _amlsim_temporal_split(
        processed, train_days=train_days, val_days=val_days)

    # 5a. Leakage audit
    try:
        from audit.leakage_check import run_amlsim_leakage_audit
        audit_report = run_amlsim_leakage_audit(
            processed, train_idx, val_idx, test_idx,
            account_col="FromAccount" if "FromAccount" in processed.columns else "AccountKey",
        )
        logger.info("Leakage audit: CA1=%s, account_overlap=%s",
                    "PASS" if audit_report["summary"]["ca1_leakage_free"] else "FAIL",
                    {k: v for k, v in audit_report["summary"].items()
                     if "overlap" in k})
    except ImportError:
        logger.info("Leakage audit skipped (audit module not available)")

    # 6. Set internal args (for CA1 cache loading in trainer)
    args["_val_idx"] = val_idx
    args["_amlsim_processed_path"] = artifacts["processed_path"]
    args["_amlsim_sample_ids"] = processed["TX_ID"].astype(str).tolist()
    args["_amlsim_ca1_cache_path"] = artifacts["ca1_cache_path"]

    # 7. Per-split amount statistics (train only)
    amount_train = processed.iloc[train_idx]["AmountPaid"].astype(float)
    amount_mean = float(amount_train.mean())
    amount_std = float(amount_train.std() + 1e-6)
    # Compute normalized amount using train stats
    feat_data["AmountNorm"] = (
        processed["AmountPaid"].astype(float) - amount_mean
    ) / amount_std

    args["_amlsim_amount_mean"] = amount_mean
    args["_amlsim_amount_std"] = amount_std

    # 8. Neighbor features
    neigh_features = _build_amlsim_neigh_features(g)

    # 9. Validate shapes
    logger.info("Final shapes: feat=%s, labels=%s, train=%d, val=%d, test=%d",
                feat_data.shape, labels.shape,
                len(train_idx), len(val_idx), len(test_idx))

    return feat_data, labels, train_idx, test_idx, g, cat_features, neigh_features
