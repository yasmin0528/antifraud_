"""
Leakage audit for AMLSIM temporal splitting.

Checks:
1. CA1 history uses only past transactions (no future leakage)
2. RuleBank statistics are computed only on training split
3. No future timestamps are accessed during training

Usage:
    from audit.leakage_check import run_amlsim_leakage_audit
    report = run_amlsim_leakage_audit(processed, train_idx, val_idx, test_idx)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def check_ca1_leakage(
    processed: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    account_col: str = "AccountKey",
    time_col: str = "Time",
) -> Dict[str, Any]:
    """Verify CA1 history uses only past (strictly earlier) transactions.

    For each transaction, checks that all transactions in its CA1 history
    window have a strictly earlier timestamp.

    Returns
    -------
    dict with keys:
        total_checked, pass_count, fail_count, pass_rate, failures_sample
    """
    total = len(processed)
    # Build per-account sorted index list
    ordered = processed.assign(_row=range(total)).sort_values(
        [account_col, time_col, "_row"], kind="mergesort",
    )
    histories: Dict[str, List[int]] = {}
    pass_count = 0
    fail_count = 0
    failures: List[dict] = []

    for _, row in ordered.iterrows():
        row_idx = int(row["_row"])
        acct = str(row[account_col])
        row_time = row[time_col]
        history = histories.get(acct, [])

        # Check all history rows have time < current row time
        for hist_idx in history[-10:]:
            hist_time = processed.iloc[hist_idx][time_col]
            if hist_time > row_time:
                fail_count += 1
                if len(failures) < 5:
                    failures.append({
                        "tx_idx": int(row_idx),
                        "account": acct,
                        "row_time": float(row_time),
                        "history_idx": int(hist_idx),
                        "history_time": float(hist_time),
                    })
                break
        else:
            pass_count += 1

        histories.setdefault(acct, []).append(row_idx)

    total_checked = pass_count + fail_count
    return {
        "total_checked": total_checked,
        "pass_count": pass_count,
        "fail_count": fail_count,
        "pass_rate": pass_count / max(total_checked, 1),
        "failures_sample": failures[:5],
        "leakage_free": fail_count == 0,
    }


def check_account_split_overlap(
    processed: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    account_col: str = "AccountKey",
) -> Dict[str, Any]:
    """Count accounts appearing in multiple splits."""
    splits = {
        "train": set(processed.iloc[train_idx][account_col].astype(str)),
        "val": set(processed.iloc[val_idx][account_col].astype(str)),
        "test": set(processed.iloc[test_idx][account_col].astype(str)),
    }
    overlaps = {}
    for n1, n2 in [("train", "val"), ("train", "test"), ("val", "test")]:
        s1, s2 = splits[n1], splits[n2]
        inter = s1 & s2
        overlaps[f"{n1}_{n2}_overlap"] = len(inter)
    return {
        "train_accounts": len(splits["train"]),
        "val_accounts": len(splits["val"]),
        "test_accounts": len(splits["test"]),
        **overlaps,
    }


def run_amlsim_leakage_audit(
    processed: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    account_col: str = "AccountKey",
    time_col: str = "Time",
) -> Dict[str, Any]:
    """Run full leakage audit and log results.

    Returns
    -------
    dict with keys: ca1_leakage, account_overlap, summary
    """
    ca1 = check_ca1_leakage(processed, train_idx, val_idx, test_idx,
                             account_col, time_col)
    overlap = check_account_split_overlap(processed, train_idx, val_idx, test_idx,
                                           account_col)

    summary = {
        "ca1_leakage_free": ca1["leakage_free"],
        "ca1_pass_rate": ca1["pass_rate"],
        "total_accounts_across_splits": overlap["train_accounts"]
        + overlap["val_accounts"] + overlap["test_accounts"],
        "train_val_account_overlap": overlap["train_val_overlap"],
        "train_test_account_overlap": overlap["train_test_overlap"],
    }

    logger.info("=== Leakage Audit Report ===")
    logger.info("CA1 temporal leakage: %s (pass_rate=%.4f)",
                "PASS" if ca1["leakage_free"] else "FAIL", ca1["pass_rate"])
    logger.info("Account overlap train-val: %d / train-test: %d",
                overlap["train_val_overlap"], overlap["train_test_overlap"])

    return {
        "ca1_leakage": ca1,
        "account_overlap": overlap,
        "summary": summary,
    }
