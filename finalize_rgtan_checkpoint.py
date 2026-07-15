"""Finalize an interrupted AML RGTAN run from its best checkpoint.

The script rebuilds the deterministic sender-account split, selects the
macro-F1 threshold on validation scores, evaluates test exactly once, and
writes the missing standard run artifacts into the original run directory.
"""

import argparse
import json
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from dgl.dataloading import MultiLayerFullNeighborSampler, NodeDataLoader
from sklearn.metrics import precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from feature_engineering.ca1_cache import load_or_build_aml_ca1_cache
from methods.modules.ca1 import CA1Encoder
from methods.modules.ca3 import CA3PrototypeMemory
from methods.modules.mpfc import MPFCDecisionFusion, binary_logits_to_risk_logit
from methods.modules.rule_engine import RuleEngine
from methods.modules.rule_bank import RuleBank
from methods.rgtan.evaluation import best_macro_f1_threshold
from methods.rgtan.rgtan_lpa import load_lpa_subtensor
from methods.rgtan.rgtan_main import (
    _classification_metrics,
    _update_experiment_index,
    loda_rgtan_data,
)
from methods.rgtan.rgtan_model import RGTAN


SUPPORTED_METHODS = {"rgtan", "rgtan_ca1", "rgtan_mpfc"}


def _torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # PyTorch before weights_only was introduced.
        return torch.load(path, map_location=device)


def _find_checkpoint(path):
    path = Path(path).expanduser().resolve()
    if path.is_file():
        return path.parent, path
    if not path.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {path}")
    names = ("best_checkpoint.pt", "best.pt", "rgtan_aml_ca1_best.pt")
    found = [path / name for name in names if (path / name).is_file()]
    if len(found) != 1:
        raise FileNotFoundError(
            f"Expected exactly one checkpoint in {path}; checked: {', '.join(names)}")
    return path, found[0]


def _infer_method(checkpoint):
    if "mpfc_state_dict" in checkpoint:
        return "rgtan_mpfc"
    if "ca3_state_dict" in checkpoint:
        raise ValueError("CA3 checkpoint without MPFC is not handled by this recovery script")
    if "ca1_state_dict" in checkpoint or "ca1" in checkpoint:
        return "rgtan_ca1"
    if "rgtan_state_dict" in checkpoint or "model" in checkpoint:
        return "rgtan"
    raise ValueError("Unrecognized RGTAN checkpoint schema")


def _load_rgtan_state(model, state_dict):
    """Restore lazily registered categorical-embedding aliases before loading.

    ``TransEmbedding.forward_emb`` assigns ``emb_dict = cat_table`` on the
    first forward. Checkpoints saved after training therefore contain both
    names, while a freshly constructed model initially contains only
    ``cat_table``. Recreate that exact registered alias and retain strict state
    loading for every real parameter.
    """
    if state_dict is None:
        raise ValueError("Checkpoint does not contain an RGTAN state_dict")
    has_lazy_alias = any(key.startswith("n2v_mlp.emb_dict.") for key in state_dict)
    embedding_module = getattr(model, "n2v_mlp", None)
    if has_lazy_alias and hasattr(embedding_module, "cat_table"):
        if getattr(embedding_module, "emb_dict", None) is None:
            embedding_module.emb_dict = embedding_module.cat_table
    model.load_state_dict(state_dict, strict=True)


def _load_run_args(run_dir, checkpoint, method, device_override):
    config_path = run_dir / "config_resolved.yaml"
    if config_path.is_file():
        with config_path.open(encoding="utf-8") as handle:
            args = yaml.safe_load(handle) or {}
    else:
        args = dict(checkpoint.get("config") or checkpoint.get("args") or {})
    if not args:
        config_map = {"rgtan": "rgtan_aml_cfg.yaml", "rgtan_ca1": "rgtan_aml_ca1.yaml",
                      "rgtan_mpfc": "rgtan_aml_mpfc.yaml"}
        default_name = config_map.get(method, "rgtan_aml_cfg.yaml")
        with (Path(__file__).parent / "config" / default_name).open(encoding="utf-8") as handle:
            args = yaml.safe_load(handle)
    args["method"] = method
    args["dataset"] = "aml"
    args["results_dir"] = str(run_dir.parent)
    if device_override:
        args["device"] = device_override
    return args


def _atomic_csv(row, path):
    fd, temporary = tempfile.mkstemp(prefix=path.stem + "_", suffix=".csv", dir=path.parent)
    os.close(fd)
    try:
        pd.DataFrame([row]).to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def _atomic_json(value, path):
    fd, temporary = tempfile.mkstemp(prefix=path.stem + "_", suffix=".json", dir=path.parent)
    os.close(fd)
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def _atomic_yaml(value, path):
    fd, temporary = tempfile.mkstemp(prefix=path.stem + "_", suffix=".yaml", dir=path.parent)
    os.close(fd)
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            yaml.safe_dump(value, handle, allow_unicode=True, sort_keys=False)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def _history_summary(run_dir, checkpoint):
    history_path = run_dir / "epoch_history.csv"
    history = pd.read_csv(history_path) if history_path.is_file() else pd.DataFrame()
    if "best_epoch" in checkpoint:
        best_epoch = int(checkpoint["best_epoch"])
    else:
        best_epoch = int(checkpoint.get("epoch", -1)) + 1
    if "best_val_metric" in checkpoint:
        best_val_loss = float(checkpoint["best_val_metric"])
    elif not history.empty and "val_main_loss" in history:
        best_val_loss = float(history["val_main_loss"].min())
    else:
        best_val_loss = float("nan")
    training_seconds = (float(history["epoch_seconds"].sum())
                        if not history.empty and "epoch_seconds" in history else 0.0)
    started_at = (str(history.iloc[0]["timestamp"])
                  if not history.empty and "timestamp" in history else None)
    return best_epoch, best_val_loss, training_seconds, started_at


def finalize_run(run_or_checkpoint, method_override=None, device_override=None, force=False):
    run_dir, checkpoint_path = _find_checkpoint(run_or_checkpoint)
    final_path = run_dir / "final_metrics.csv"
    if final_path.exists() and not force:
        raise FileExistsError(
            f"{final_path} already exists; refusing to evaluate test again (use --force explicitly)")

    # Keep optimizer/scheduler tensors on CPU; only model weights are copied to
    # the evaluation device below. This avoids wasting GPU memory on recovery.
    checkpoint = _torch_load(checkpoint_path, "cpu")
    method = _infer_method(checkpoint)
    if method_override and method_override != method:
        raise ValueError(f"Checkpoint is {method}, not requested method {method_override}")
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unsupported method: {method}")
    args = _load_run_args(run_dir, checkpoint, method, device_override)
    device = torch.device(args["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Checkpoint/config requests CUDA, but CUDA is unavailable; pass --device cpu")

    project_root = Path(__file__).parent.resolve()
    previous_cwd = Path.cwd()
    os.chdir(project_root)
    evaluation_started = time.perf_counter()
    try:
        feat_df, labels, train_idx, test_idx, graph, cat_features, neigh_features = loda_rgtan_data(args)
        val_idx = args["_val_idx"]
        graph = graph.to(device)
        num_feat = torch.from_numpy(feat_df.values).float().to(device)
        cat_feat = {column: torch.from_numpy(feat_df[column].values).long().to(device)
                    for column in cat_features}
        nei_feat = ({column: torch.from_numpy(neigh_features[column].values).float().to(device)
                     for column in neigh_features.columns}
                    if isinstance(neigh_features, pd.DataFrame) else [])
        label_tensor = torch.from_numpy(labels.values).long().to(device)
        known_labels = torch.full_like(label_tensor, 2)
        train_nodes = torch.as_tensor(train_idx, dtype=torch.long, device=device)
        known_labels[train_nodes] = label_tensor[train_nodes]

        def make_loader(indices):
            return NodeDataLoader(
                graph, torch.as_tensor(indices, dtype=torch.long, device=device),
                MultiLayerFullNeighborSampler(args["n_layers"]), device=device, use_ddp=False,
                batch_size=args["batch_size"], shuffle=False, drop_last=False, num_workers=0)

        val_loader, test_loader = make_loader(val_idx), make_loader(test_idx)
        model_kwargs = dict(
            in_feats=feat_df.shape[1], hidden_dim=args["hid_dim"] // 4, n_classes=2,
            heads=[4] * args["n_layers"], activation=nn.PReLU(), n_layers=args["n_layers"],
            drop=args["dropout"], device=device, gated=args["gated"], ref_df=feat_df,
            cat_features=cat_feat, neigh_features=nei_feat,
            nei_att_head=args["nei_att_heads"]["aml"],
        )
        # ── 模型加载（按 method 分支） ──────────────────────────────────
        ca1, ca3, mpfc, rule_engine, field_tensors = None, None, None, None, None
        needs_ca1 = method in ("rgtan_ca1", "rgtan_mpfc")
        model_kwargs = dict(
            in_feats=feat_df.shape[1], hidden_dim=args["hid_dim"] // 4, n_classes=2,
            heads=[4] * args["n_layers"], activation=nn.PReLU(), n_layers=args["n_layers"],
            drop=args["dropout"], device=device, gated=args["gated"], ref_df=feat_df,
            cat_features=cat_feat, neigh_features=nei_feat,
            nei_att_head=args["nei_att_heads"]["aml"],
        )
        if needs_ca1:
            model_kwargs["ca1_hidden_dim"] = args["ca1_hidden_dim"]
            ca1 = CA1Encoder(4, args["ca1_hidden_dim"], args["ca1_dropout"],
                             args["ca1_encoder_type"], args["ca1_pooling"]).to(device)
            cache = load_or_build_aml_ca1_cache(
                args["_aml_processed_path"], args["ca1_cache_path"], args["_aml_sample_ids"],
                args["ca1_k"], args["_aml_amount_mean"], args["_aml_amount_std"])
            ca1_sequence, ca1_len, ca1_mask = (
                cache["sequence"], cache["sequence_len"], cache["padding_mask"])
            ca1.load_state_dict(checkpoint.get("ca1_state_dict", checkpoint.get("ca1")))
            ca1.eval()

            if method == "rgtan_mpfc":
                # CA3
                ca3 = CA3PrototypeMemory(
                    args["ca1_hidden_dim"], args["ca3_num_prototypes"], args["ca3_temperature"],
                    args["ca3_top_k"], args["ca3_fusion"],
                    gate_bias_init=args.get("ca3_gate_bias_init", -2.0),
                    gate_bias_final=args.get("ca3_gate_bias_final", -1.0),
                    anneal_epochs=args.get("ca3_anneal_epochs", 10),
                    dead_epoch_threshold=args.get("ca3_dead_threshold", 3),
                    entropy_gate_beta=args.get("ca3_entropy_gate_beta", 1.0),
                    contrastive_temperature=args.get("ca3_contrastive_temperature", 0.1),
                    diversity_margin=args.get("ca3_diversity_margin", 0.5),
                ).to(device)
                ca3.load_state_dict(checkpoint["ca3_state_dict"])
                ca3.eval()

                # MPFC
                mpfc = MPFCDecisionFusion(
                    input_dim=model_kwargs["heads"][-1] * (model_kwargs["hidden_dim"]),
                    hidden_dim=args.get("mpfc_hidden_dim", 64),
                    dropout=args.get("mpfc_dropout", 0.1),
                    gate_bias_init=args.get("mpfc_gate_bias_init", -2.0),
                    confidence_constrained=args.get("mpfc_confidence_constrained", True),
                ).to(device)
                mpfc.load_state_dict(checkpoint["mpfc_state_dict"])
                mpfc.eval()

                # RuleEngine
                processed = pd.read_csv(args["_aml_processed_path"])
                encoding_map = {}
                for ef in ("Type", "Target"):
                    raw = processed[ef].astype(str)
                    le = LabelEncoder()
                    le.fit(raw)
                    encoding_map[ef] = dict(zip(le.classes_, le.transform(le.classes_)))
                rbp = args.get("rulebank_path", "config/rulebank/aml_rulebank_v1.yaml")
                rulebank = RuleBank.load(rbp)
                rule_engine = RuleEngine(
                    rulebank, encoding_map=encoding_map,
                    aggregation=args.get("mpfc_aggregation", "noisy_or"),
                    device=device)
                # Pre-build full field tensors (used in collect)
                field_tensors = {}
                for field in ("Amount", "TimeDiff", "SenderHistCount",
                              "SenderHistAmountSum", "SenderHistAmountMean", "Time"):
                    field_tensors[field] = (
                        torch.from_numpy(processed[field].astype(float).values)
                        .float().unsqueeze(1).to(device))
                for field in ("Type", "Target"):
                    raw = processed[field].astype(str)
                    le = LabelEncoder()
                    field_tensors[field] = (
                        torch.from_numpy(le.fit_transform(raw)).long().unsqueeze(1).to(device))

        model = RGTAN(**model_kwargs).to(device)
        _load_rgtan_state(model, checkpoint.get("rgtan_state_dict", checkpoint.get("model")))
        model.eval()

        # ── 评估 forward ──────────────────────────────────────────────
        def collect(loader, description):
            truths, scores = [], []
            with torch.no_grad():
                for input_nodes, seeds, blocks in tqdm(loader, desc=description, unit="batch"):
                    inputs, work_inputs, neigh_inputs, batch_labels, lpa_labels = load_lpa_subtensor(
                        num_feat, cat_feat, nei_feat, {}, label_tensor, seeds, input_nodes, device,
                        blocks, known_labels)
                    blocks_gpu = [block.to(device) for block in blocks]

                    if method == "rgtan_mpfc":
                        cpu_nodes = input_nodes.detach().cpu().long()
                        ca1_emb, _, _ = ca1(
                            ca1_sequence[cpu_nodes].to(device), ca1_len[cpu_nodes].to(device),
                            ca1_mask[cpu_nodes].to(device))
                        ca3_out = ca3(ca1_emb, enabled=True)
                        class_logits, hidden = model(
                            blocks_gpu, inputs, lpa_labels, work_inputs, neigh_inputs,
                            ca1_embedding=ca3_out.enhanced_embedding, return_hidden=True)
                        neural_logit = binary_logits_to_risk_logit(class_logits)
                        seed_idx = seeds.detach().cpu().long()
                        rb = {n: t[seed_idx] for n, t in field_tensors.items()}
                        rule_score, rule_confidence = rule_engine.evaluate(rb)
                        mpfc_out = mpfc(hidden, neural_logit, rule_score, rule_confidence)
                        logits = torch.cat(
                            [-mpfc_out.final_logit, mpfc_out.final_logit], dim=1)
                    else:
                        embedding = None
                        if ca1 is not None:
                            cpu_nodes = input_nodes.detach().cpu().long()
                            embedding, _, _ = ca1(
                                ca1_sequence[cpu_nodes].to(device), ca1_len[cpu_nodes].to(device),
                                ca1_mask[cpu_nodes].to(device))
                            if len(embedding) != blocks[0].num_src_nodes():
                                raise RuntimeError("CA1 embedding misaligned")
                        logits = model(blocks_gpu, inputs, lpa_labels, work_inputs, neigh_inputs,
                                       ca1_embedding=embedding)

                    valid = batch_labels != 2
                    probabilities = torch.softmax(logits[valid], dim=1)[:, 1]
                    truths.extend(batch_labels[valid].cpu().tolist())
                    scores.extend(probabilities.cpu().tolist())
            return truths, scores

        # Validation is always evaluated before test and is the only source of the threshold.
        val_y, val_scores = collect(val_loader, "checkpoint validation")
        threshold, val_f1 = best_macro_f1_threshold(val_y, val_scores)
        val_preds = (np.asarray(val_scores) >= threshold).astype(int)
        val_metrics = _classification_metrics(val_y, val_scores, val_preds)

        # Test is deliberately evaluated once, after the threshold has been frozen.
        test_y, test_scores = collect(test_loader, "checkpoint test (single pass)")
        test_preds = (np.asarray(test_scores) >= threshold).astype(int)
        test_metrics = _classification_metrics(test_y, test_scores, test_preds)
        best_epoch, best_val_loss, training_seconds, started_at = _history_summary(run_dir, checkpoint)
        finalization_seconds = time.perf_counter() - evaluation_started
        run_id = run_dir.name
        final = {
            "run_id": run_id, "method": method, "dataset": "aml", "seed": args["seed"],
            "test_auc": test_metrics["auc"], "test_ap": test_metrics["ap"],
            "test_f1": test_metrics["f1"],
            "test_precision": float(precision_score(test_y, test_preds, zero_division=0)),
            "test_recall": float(recall_score(test_y, test_preds, zero_division=0)),
            "test_threshold": threshold, "val_auc_at_best": val_metrics["auc"],
            "val_ap_at_best": val_metrics["ap"], "val_f1_at_best": val_f1,
            "best_epoch": best_epoch, "best_val_loss": best_val_loss,
            "duration_seconds": training_seconds + finalization_seconds,
            "finalization_seconds": finalization_seconds,
            "train_size": len(train_idx), "val_size": len(val_idx), "test_size": len(test_idx),
            "ca1_enabled": method in ("rgtan_ca1", "rgtan_mpfc"),
            "ca3_enabled": method == "rgtan_mpfc",
            "mpfc_enabled": method == "rgtan_mpfc",
        }
        metadata = {
            **final, "run_dir": str(run_dir), "train_mode": "single_split",
            "split_mode": "sender_account", "label_propagation_scope": "train_only",
            "label_derived_neighbor_features": False, "amount_normalization_scope": "train_split",
            "graph_feature_setting": "transductive_features_train_labels_only",
            "alert_id_used_in_forward": False, "finalized_from_checkpoint": True,
            "checkpoint_path": str(checkpoint_path), "training_started_at": started_at,
            "finished_at": datetime.now().astimezone().isoformat(),
            "test_evaluation_passes": 1,
        }
        if method == "rgtan_ca1":
            metadata.update({
                "ca1_k": args["ca1_k"], "ca1_hidden_dim": args["ca1_hidden_dim"],
                "ca1_aux_weight": args["ca1_aux_weight"],
                "ca1_cache_fingerprint": cache["source_fingerprint"],
            })
        resolved_path = run_dir / "config_resolved.yaml"
        if not resolved_path.exists():
            resolved = {key: value for key, value in args.items() if not key.startswith("_")}
            resolved.update({"run_id": run_id, "run_dir": str(run_dir),
                             "train_mode": "single_split"})
            _atomic_yaml(resolved, resolved_path)
        _atomic_csv(final, final_path)
        _atomic_json(metadata, run_dir / "metadata.json")
        _atomic_json({
            "run_id": run_id, "checkpoint": str(checkpoint_path),
            "recovered_at": metadata["finished_at"], "test_evaluation_passes": 1,
        }, run_dir / "recovery.json")
        _update_experiment_index(str(run_dir.parent), {
            "run_id": run_id, "method": method, "dataset": "aml", "seed": args["seed"],
            "auc": final["test_auc"], "ap": final["test_ap"], "f1": final["test_f1"],
            "best_epoch": best_epoch, "best_val_loss": best_val_loss,
            "duration_seconds": final["duration_seconds"],
            "ca1_enabled": method in ("rgtan_ca1", "rgtan_mpfc"),
            "ca3_enabled": method == "rgtan_mpfc",
            "mpfc_enabled": method == "rgtan_mpfc",
            "run_dir": str(run_dir), "created_at": started_at, "status": "success",
        })
        print(json.dumps(final, ensure_ascii=False, indent=2))
        return final
    finally:
        os.chdir(previous_cwd)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Complete validation threshold selection and one test pass from best.pt")
    parser.add_argument("--run-dir", required=True,
                        help="Run directory, or the checkpoint file itself")
    parser.add_argument("--method", choices=sorted(SUPPORTED_METHODS), default=None,
                        help="Optional safety check; normally inferred from checkpoint keys")
    parser.add_argument("--device", default=None, help="Override checkpoint device, e.g. cuda:0 or cpu")
    parser.add_argument("--force", action="store_true",
                        help="Allow replacing existing final metrics and evaluating test again")
    return parser.parse_args()


if __name__ == "__main__":
    cli = parse_args()
    finalize_run(cli.run_dir, cli.method, cli.device, cli.force)
