import os
import copy
import json
import time
import tempfile
from datetime import datetime
from dgl.dataloading import MultiLayerFullNeighborSampler
from dgl.dataloading import NodeDataLoader
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import dgl
import yaml
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
from scipy.io import loadmat
from tqdm import tqdm
from . import *
from .rgtan_lpa import load_lpa_subtensor
from .rgtan_model import RGTAN
from .evaluation import best_macro_f1_threshold
from feature_engineering.data_process import preprocess_aml_for_gtan
from feature_engineering.ca1_cache import load_or_build_aml_ca1_cache
from methods.modules.ca1 import CA1Encoder
from methods.modules.ca3 import CA3PrototypeMemory


def _resolve_dataset_path(data_path: str) -> str:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    candidate_paths = []
    if os.path.isabs(data_path):
        candidate_paths.append(data_path)
    else:
        candidate_paths.append(os.path.abspath(os.path.join(project_root, data_path)))
        candidate_paths.append(os.path.abspath(os.path.join(os.getcwd(), data_path)))
        candidate_paths.append(os.path.abspath(data_path))

    for candidate in candidate_paths:
        if os.path.exists(candidate):
            return candidate
    return candidate_paths[0]


def _sender_account_train_test_split(
    group_ids: pd.Series,
    labels: pd.Series,
    test_size: float,
    seed: int,
):
    group_df = pd.DataFrame({"group_id": group_ids.astype(str), "label": labels.astype(int)})
    group_level = group_df.groupby("group_id")["label"].max().reset_index()
    group_list = group_level["group_id"].to_numpy()
    group_labels = group_level["label"]

    stratify = None
    if group_labels.nunique() >= 2 and group_labels.value_counts().min() >= 2:
        stratify = group_labels
    else:
        print("Warning: AML sender-account stratified split unavailable, falling back to non-stratified split.")

    train_groups, test_groups = train_test_split(
        group_list,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )
    train_mask = group_ids.astype(str).isin(train_groups)
    test_mask = group_ids.astype(str).isin(test_groups)
    return np.flatnonzero(train_mask.to_numpy()), np.flatnonzero(test_mask.to_numpy())


def _build_aml_neigh_features(graph: dgl.DGLGraph, labels: pd.Series) -> pd.DataFrame:
    in_degree = graph.in_degrees().cpu().numpy().astype(np.float32)
    out_degree = graph.out_degrees().cpu().numpy().astype(np.float32)
    return pd.DataFrame({"degree": in_degree, "out_degree": out_degree})


def _seed_local_indices(input_nodes, seeds):
    positions = {int(node): idx for idx, node in enumerate(input_nodes.detach().cpu().tolist())}
    try:
        return torch.tensor([positions[int(node)] for node in seeds.detach().cpu().tolist()],
                            dtype=torch.long, device=input_nodes.device)
    except KeyError as exc:
        raise ValueError(f"Seed node {exc.args[0]} is absent from input_nodes") from exc


def _classification_metrics(truths, scores, preds):
    if not truths:
        return {"auc": float("nan"), "ap": float("nan"), "f1": float("nan")}
    truths = np.asarray(truths)
    result = {
        "ap": float(average_precision_score(truths, scores)),
        "f1": float(f1_score(truths, preds, average="macro")),
    }
    result["auc"] = (float(roc_auc_score(truths, scores))
                     if np.unique(truths).size > 1 else float("nan"))
    return result


def _save_ca1_results(args, metrics, cache, run_dir, run_id, started_at):
    final = {
        "run_id": run_id, "method": "rgtan_ca1", "dataset": "aml", "seed": args["seed"],
        "test_auc": metrics["auc"], "test_ap": metrics["ap"], "test_f1": metrics["f1"],
        "test_precision": metrics["precision"], "test_recall": metrics["recall"],
        "test_threshold": metrics["test_threshold"],
        "val_auc_at_best": metrics["val_auc_at_best"],
        "val_ap_at_best": metrics["val_ap_at_best"],
        "val_f1_at_best": metrics["val_f1_at_best"],
        "best_epoch": metrics["best_epoch"], "best_val_loss": metrics["best_val_loss"],
        "duration_seconds": metrics["duration_seconds"],
        "train_size": metrics["train_size"], "val_size": metrics["val_size"],
        "test_size": metrics["test_size"], "ca1_enabled": True, "ca3_enabled": False,
    }
    pd.DataFrame([final]).to_csv(os.path.join(run_dir, "final_metrics.csv"), index=False)
    metadata = {
        "run_id": run_id, "method": "rgtan_ca1", "dataset": "aml",
        "started_at": started_at, "finished_at": datetime.now().astimezone().isoformat(),
        "ca1_enabled": True,
        "train_mode": "single_split", "ca1_k": args["ca1_k"],
        "ca1_hidden_dim": args["ca1_hidden_dim"],
        "ca1_aux_weight": args["ca1_aux_weight"],
        "ca1_input_fields": cache["input_fields"],
        "ca1_encoder_type": args["ca1_encoder_type"],
        "ca1_pooling": args["ca1_pooling"],
        "ca1_cache_fingerprint": cache["source_fingerprint"],
        "alert_id_used_in_forward": False,
        "label_derived_neighbor_features": False,
        "label_propagation_scope": "train_only",
        "amount_normalization_scope": "train_split",
        "graph_feature_setting": "transductive_features_train_labels_only",
        **final,
    }
    with open(os.path.join(run_dir, "metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    results_dir = args.get("results_dir", "results")
    _update_experiment_index(results_dir, {
        "run_id": run_id, "method": "rgtan_ca1", "dataset": "aml", "seed": args["seed"],
        "auc": final["test_auc"], "ap": final["test_ap"], "f1": final["test_f1"],
        "best_epoch": final["best_epoch"], "best_val_loss": final["best_val_loss"],
        "duration_seconds": final["duration_seconds"], "ca1_enabled": True,
        "ca3_enabled": False, "run_dir": os.path.abspath(run_dir),
        "created_at": started_at, "status": "success",
    })


def rgtan_ca1_main(feat_df, graph, train_idx, val_idx, test_idx, labels, args,
                   cat_features, neigh_features, nei_att_head):
    started_at = datetime.now().astimezone().isoformat()
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"rgtan_ca1_aml_seed{args['seed']}_{run_stamp}"
    results_dir = args.get("results_dir", "results")
    os.makedirs(results_dir, exist_ok=True)
    run_dir = os.path.join(results_dir, run_id)
    os.makedirs(run_dir, exist_ok=False)
    checkpoint_path = os.path.join(run_dir, "best_checkpoint.pt")
    history_path = os.path.join(run_dir, "epoch_history.csv")
    print(f"Experiment run: {run_id}")
    print(f"Artifacts: {os.path.abspath(run_dir)}")
    resolved = {k: v for k, v in args.items() if not k.startswith("_")}
    resolved.update({"run_id": run_id, "run_dir": os.path.abspath(run_dir),
                     "train_mode": "single_split"})
    with open(os.path.join(run_dir, "config_resolved.yaml"), "w", encoding="utf-8") as handle:
        yaml.safe_dump(resolved, handle, allow_unicode=True, sort_keys=False)
    _update_experiment_index(results_dir, {
        "run_id": run_id, "method": "rgtan_ca1", "dataset": "aml", "seed": args["seed"],
        "ca1_enabled": True, "ca3_enabled": False, "run_dir": os.path.abspath(run_dir),
        "created_at": started_at, "status": "running",
    })
    device = torch.device(args["device"])
    graph = graph.to(device)
    num_feat = torch.from_numpy(feat_df.values).float().to(device)
    cat_feat = {col: torch.from_numpy(feat_df[col].values).long().to(device)
                for col in cat_features}
    nei_feat = ({col: torch.from_numpy(neigh_features[col].values).float().to(device)
                 for col in neigh_features.columns}
                if isinstance(neigh_features, pd.DataFrame) else [])
    label_tensor = torch.from_numpy(labels.values).long().to(device)
    known_label_tensor = torch.full_like(label_tensor, 2)
    train_nodes_for_labels = torch.as_tensor(train_idx, dtype=torch.long, device=device)
    known_label_tensor[train_nodes_for_labels] = label_tensor[train_nodes_for_labels]
    cache = load_or_build_aml_ca1_cache(
        args["_aml_processed_path"], args["ca1_cache_path"],
        args["_aml_sample_ids"], args["ca1_k"], args["_aml_amount_mean"], args["_aml_amount_std"])
    if cache["num_rows"] != len(feat_df) or cache["num_rows"] != graph.num_nodes():
        raise ValueError("CA1 cache, feature rows, and graph nodes are not aligned")
    ca1_sequence = cache["sequence"]
    ca1_len = cache["sequence_len"]
    ca1_mask = cache["padding_mask"]

    def loader(indices, shuffle):
        sampler = MultiLayerFullNeighborSampler(args["n_layers"])
        nodes = torch.as_tensor(indices, dtype=torch.long, device=device)
        return NodeDataLoader(graph, nodes, sampler, device=device, use_ddp=False,
                              batch_size=args["batch_size"], shuffle=shuffle,
                              drop_last=False, num_workers=0)

    train_loader, val_loader, test_loader = (
        loader(train_idx, True), loader(val_idx, False), loader(test_idx, False))
    model = RGTAN(
        in_feats=feat_df.shape[1], hidden_dim=args["hid_dim"] // 4, n_classes=2,
        heads=[4] * args["n_layers"], activation=nn.PReLU(), n_layers=args["n_layers"],
        drop=args["dropout"], device=device, gated=args["gated"], ref_df=feat_df,
        cat_features=cat_feat, neigh_features=nei_feat, nei_att_head=nei_att_head,
        ca1_hidden_dim=args["ca1_hidden_dim"],
    ).to(device)
    ca1 = CA1Encoder(4, args["ca1_hidden_dim"], args["ca1_dropout"],
                     args["ca1_encoder_type"], args["ca1_pooling"]).to(device)
    lr = args["lr"] * np.sqrt(args["batch_size"] / 1024)
    optimizer = optim.Adam(list(model.parameters()) + list(ca1.parameters()),
                           lr=lr, weight_decay=args["wd"])
    scheduler = MultiStepLR(optimizer, milestones=[4000, 12000], gamma=0.3)
    main_loss_fn = nn.CrossEntropyLoss().to(device)
    aux_loss_fn = nn.BCEWithLogitsLoss().to(device)
    best_loss, stale, best_state = float("inf"), 0, None

    def forward_batch(input_nodes, seeds, blocks):
        batch = load_lpa_subtensor(num_feat, cat_feat, nei_feat, {}, label_tensor,
                                   seeds, input_nodes, device, blocks, known_label_tensor)
        inputs, work_inputs, neigh_inputs, batch_labels, lpa_labels = batch
        cpu_nodes = input_nodes.detach().cpu().long()
        embedding, micro_logits, _ = ca1(
            ca1_sequence[cpu_nodes].to(device), ca1_len[cpu_nodes].to(device),
            ca1_mask[cpu_nodes].to(device))
        local_idx = _seed_local_indices(input_nodes, seeds)
        logits = model([block.to(device) for block in blocks], inputs, lpa_labels,
                       work_inputs, neigh_inputs, ca1_embedding=embedding)
        return logits, micro_logits[local_idx], batch_labels

    epoch_bar = tqdm(range(args["max_epochs"]), desc="epochs", unit="epoch")
    for epoch in epoch_bar:
        epoch_started = time.perf_counter()
        model.train(); ca1.train()
        train_main, train_aux, train_total = [], [], []
        train_truths, train_scores, train_preds = [], [], []
        train_bar = tqdm(train_loader, desc=f"epoch {epoch + 1:02d} train",
                         unit="batch", leave=False)
        for input_nodes, seeds, blocks in train_bar:
            logits, micro_logits, batch_labels = forward_batch(input_nodes, seeds, blocks)
            valid = batch_labels != 2
            if not valid.any():
                continue
            main_loss = main_loss_fn(logits[valid], batch_labels[valid])
            aux_loss = aux_loss_fn(micro_logits[valid].squeeze(-1), batch_labels[valid].float())
            total_loss = main_loss + args["ca1_aux_weight"] * aux_loss
            optimizer.zero_grad(); total_loss.backward(); optimizer.step(); scheduler.step()
            train_main.append(main_loss.item()); train_aux.append(aux_loss.item()); train_total.append(total_loss.item())
            probabilities = torch.softmax(logits[valid].detach(), dim=1)[:, 1]
            train_truths.extend(batch_labels[valid].detach().cpu().tolist())
            train_scores.extend(probabilities.cpu().tolist())
            train_preds.extend(torch.argmax(logits[valid].detach(), dim=1).cpu().tolist())
            train_bar.set_postfix(loss=f"{total_loss.item():.4f}")

        model.eval(); ca1.eval(); val_sum, val_count = 0.0, 0
        val_aux_sum, val_total_sum = 0.0, 0.0
        val_truths, val_scores, val_preds = [], [], []
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"epoch {epoch + 1:02d} val",
                           unit="batch", leave=False)
            for input_nodes, seeds, blocks in val_bar:
                logits, micro_logits, batch_labels = forward_batch(input_nodes, seeds, blocks)
                valid = batch_labels != 2
                if valid.any():
                    count = int(valid.sum())
                    batch_main = main_loss_fn(logits[valid], batch_labels[valid])
                    batch_aux = aux_loss_fn(micro_logits[valid].squeeze(-1), batch_labels[valid].float())
                    batch_total = batch_main + args["ca1_aux_weight"] * batch_aux
                    val_sum += batch_main.item() * count
                    val_aux_sum += batch_aux.item() * count
                    val_total_sum += batch_total.item() * count
                    val_count += count
                    probabilities = torch.softmax(logits[valid], dim=1)[:, 1]
                    val_truths.extend(batch_labels[valid].cpu().tolist())
                    val_scores.extend(probabilities.cpu().tolist())
                    val_preds.extend(torch.argmax(logits[valid], dim=1).cpu().tolist())
        val_loss = val_sum / max(val_count, 1)
        improved = val_loss < best_loss
        train_metrics = _classification_metrics(train_truths, train_scores, train_preds)
        val_metrics = _classification_metrics(val_truths, val_scores, val_preds)
        epoch_row = {
            "run_id": run_id, "timestamp": datetime.now().astimezone().isoformat(),
            "epoch": epoch + 1, "epoch_seconds": time.perf_counter() - epoch_started,
            "train_main_loss": float(np.mean(train_main)),
            "train_ca1_aux_loss": float(np.mean(train_aux)),
            "train_total_loss": float(np.mean(train_total)),
            "train_auc": train_metrics["auc"], "train_ap": train_metrics["ap"],
            "train_f1": train_metrics["f1"], "val_main_loss": val_loss,
            "val_ca1_aux_loss": val_aux_sum / max(val_count, 1),
            "val_total_loss": val_total_sum / max(val_count, 1),
            "val_auc": val_metrics["auc"], "val_ap": val_metrics["ap"],
            "val_f1": val_metrics["f1"], "is_best": improved,
        }
        pd.DataFrame([epoch_row]).to_csv(
            history_path, mode="a", header=not os.path.exists(history_path), index=False)
        epoch_bar.set_postfix(val_loss=f"{val_loss:.4f}", val_auc=f"{val_metrics['auc']:.4f}",
                              val_ap=f"{val_metrics['ap']:.4f}", best=improved)
        print(f"[{epoch_row['timestamp']}] epoch={epoch + 1:03d} "
              f"train_auc={train_metrics['auc']:.4f} train_ap={train_metrics['ap']:.4f} "
              f"train_f1={train_metrics['f1']:.4f} val_auc={val_metrics['auc']:.4f} "
              f"val_ap={val_metrics['ap']:.4f} val_f1={val_metrics['f1']:.4f}")
        if improved:
            best_loss, stale = val_loss, 0
            best_state = {
                # Keep legacy keys so existing tooling remains compatible.
                "epoch": epoch, "model": copy.deepcopy(model.state_dict()),
                "ca1": copy.deepcopy(ca1.state_dict()),
                "optimizer": copy.deepcopy(optimizer.state_dict()), "args": dict(args),
                "rgtan_state_dict": copy.deepcopy(model.state_dict()),
                "ca1_state_dict": copy.deepcopy(ca1.state_dict()),
                "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
                "scheduler_state_dict": copy.deepcopy(scheduler.state_dict()),
                "best_epoch": epoch + 1, "best_val_metric": val_loss, "config": resolved,
            }
            torch.save(best_state, checkpoint_path)
        else:
            stale += 1
            if stale >= args["early_stopping"]:
                break

    if best_state is None:
        raise RuntimeError("No valid CA1 training/validation batch was produced")
    model.load_state_dict(best_state["model"]); ca1.load_state_dict(best_state["ca1"])
    optimizer.load_state_dict(best_state["optimizer"])
    if "scheduler_state_dict" in best_state:
        scheduler.load_state_dict(best_state["scheduler_state_dict"])
    model.eval(); ca1.eval()
    def collect_scores(data_loader, description):
        collected_scores, collected_truths = [], []
        with torch.no_grad():
            for input_nodes, seeds, blocks in tqdm(data_loader, desc=description, unit="batch", leave=False):
                logits, _, batch_labels = forward_batch(input_nodes, seeds, blocks)
                valid = batch_labels != 2
                probabilities = torch.softmax(logits[valid], dim=1)[:, 1]
                collected_scores.extend(probabilities.cpu().tolist())
                collected_truths.extend(batch_labels[valid].cpu().tolist())
        return collected_truths, collected_scores
    val_truths, val_scores = collect_scores(val_loader, "best checkpoint val")
    threshold, val_f1 = best_macro_f1_threshold(val_truths, val_scores)
    val_threshold_metrics = _classification_metrics(
        val_truths, val_scores, (np.asarray(val_scores) >= threshold).astype(int))
    truths, scores = collect_scores(test_loader, "best checkpoint test")
    preds = (np.asarray(scores) >= threshold).astype(int)
    test_metrics = _classification_metrics(truths, scores, preds)
    metrics = {
        "auc": test_metrics["auc"], "f1": test_metrics["f1"], "ap": test_metrics["ap"],
        "precision": float(precision_score(truths, preds, zero_division=0)),
        "recall": float(recall_score(truths, preds, zero_division=0)),
        "test_threshold": threshold, "val_auc_at_best": val_threshold_metrics["auc"],
        "val_ap_at_best": val_threshold_metrics["ap"], "val_f1_at_best": val_f1,
        "train_size": len(train_idx), "val_size": len(val_idx), "test_size": len(test_idx),
        "best_epoch": int(best_state["epoch"]) + 1, "best_val_loss": float(best_loss),
        "duration_seconds": (datetime.now().astimezone()
                             - datetime.fromisoformat(started_at)).total_seconds(),
    }
    print("test AUC:", metrics["auc"]); print("test f1:", metrics["f1"]); print("test AP:", metrics["ap"])
    _save_ca1_results(args, metrics, cache, run_dir, run_id, started_at)
    return metrics


def _update_experiment_index(results_dir, row):
    columns = [
        "run_id", "method", "dataset", "seed", "auc", "ap", "f1", "best_epoch",
        "best_val_loss", "duration_seconds", "ca1_enabled", "ca3_enabled",
        "ca3_num_prototypes", "ca3_warmup_epochs", "run_dir", "created_at", "status",
    ]
    path = os.path.join(results_dir, "experiment_index.csv")
    current = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
    if "run_id" in current.columns and row["run_id"] in set(current["run_id"].astype(str)):
        current = current[current["run_id"].astype(str) != row["run_id"]]
    for column in columns:
        if column not in current.columns:
            current[column] = np.nan
    normalized = {column: row.get(column, np.nan) for column in columns}
    updated = pd.concat([current[columns], pd.DataFrame([normalized])], ignore_index=True)
    fd, temporary = tempfile.mkstemp(prefix="experiment_index_", suffix=".csv", dir=results_dir)
    os.close(fd)
    try:
        updated.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def _rgtan_ca1_ca3_main_impl(feat_df, graph, train_idx, val_idx, test_idx, labels, args,
                             cat_features, neigh_features, nei_att_head):
    warmup = int(args["ca3_warmup_epochs"])
    if not 0 <= warmup < int(args["max_epochs"]):
        raise ValueError("ca3_warmup_epochs must satisfy 0 <= warmup < max_epochs")
    positive_train = np.asarray(train_idx)[labels.iloc[train_idx].to_numpy() == 1]
    if len(positive_train) < int(args["ca3_num_prototypes"]):
        raise ValueError("Training positives are fewer than ca3_num_prototypes")

    started_dt = datetime.now().astimezone()
    started_at = started_dt.isoformat()
    stamp = started_dt.strftime("%Y%m%d_%H%M%S")
    run_id = f"rgtan_ca1_ca3_aml_seed{args['seed']}_{stamp}"
    results_dir = args.get("results_dir", "results")
    run_dir = os.path.join(results_dir, run_id)
    os.makedirs(run_dir, exist_ok=False)
    resolved = {k: v for k, v in args.items() if not k.startswith("_")}
    resolved.update({"run_id": run_id, "run_dir": os.path.abspath(run_dir),
                     "train_mode": "single_split"})
    with open(os.path.join(run_dir, "config_resolved.yaml"), "w", encoding="utf-8") as handle:
        yaml.safe_dump(resolved, handle,
                       allow_unicode=True, sort_keys=False)
    index_base = {
        "run_id": run_id, "method": "rgtan_ca1_ca3", "dataset": "aml",
        "seed": args["seed"], "ca1_enabled": True, "ca3_enabled": True,
        "ca3_num_prototypes": args["ca3_num_prototypes"],
        "ca3_warmup_epochs": warmup, "run_dir": os.path.abspath(run_dir),
        "created_at": started_at,
    }
    _update_experiment_index(results_dir, {**index_base, "status": "running"})

    device = torch.device(args["device"])
    graph = graph.to(device)
    num_feat = torch.from_numpy(feat_df.values).float().to(device)
    cat_feat = {col: torch.from_numpy(feat_df[col].values).long().to(device) for col in cat_features}
    nei_feat = ({col: torch.from_numpy(neigh_features[col].values).float().to(device)
                 for col in neigh_features.columns} if isinstance(neigh_features, pd.DataFrame) else [])
    label_tensor = torch.from_numpy(labels.values).long().to(device)
    known_label_tensor = torch.full_like(label_tensor, 2)
    train_nodes_for_labels = torch.as_tensor(train_idx, dtype=torch.long, device=device)
    known_label_tensor[train_nodes_for_labels] = label_tensor[train_nodes_for_labels]
    cache = load_or_build_aml_ca1_cache(args["_aml_processed_path"], args["ca1_cache_path"],
                                        args["_aml_sample_ids"], args["ca1_k"],
                                        args["_aml_amount_mean"], args["_aml_amount_std"])
    ca1_sequence, ca1_len, ca1_mask = cache["sequence"], cache["sequence_len"], cache["padding_mask"]

    def make_loader(indices, shuffle=False):
        return NodeDataLoader(
            graph, torch.as_tensor(indices, dtype=torch.long, device=device),
            MultiLayerFullNeighborSampler(args["n_layers"]), device=device, use_ddp=False,
            batch_size=args["batch_size"], shuffle=shuffle, drop_last=False, num_workers=0)

    train_loader, val_loader, test_loader = (
        make_loader(train_idx, True), make_loader(val_idx), make_loader(test_idx))
    model = RGTAN(
        in_feats=feat_df.shape[1], hidden_dim=args["hid_dim"] // 4, n_classes=2,
        heads=[4] * args["n_layers"], activation=nn.PReLU(), n_layers=args["n_layers"],
        drop=args["dropout"], device=device, gated=args["gated"], ref_df=feat_df,
        cat_features=cat_feat, neigh_features=nei_feat, nei_att_head=nei_att_head,
        ca1_hidden_dim=args["ca1_hidden_dim"]).to(device)
    ca1 = CA1Encoder(4, args["ca1_hidden_dim"], args["ca1_dropout"],
                     args["ca1_encoder_type"], args["ca1_pooling"]).to(device)
    ca3 = CA3PrototypeMemory(
        args["ca1_hidden_dim"], args["ca3_num_prototypes"], args["ca3_temperature"],
        args["ca3_top_k"], args["ca3_fusion"]).to(device)
    optimizer = optim.Adam(
        list(model.parameters()) + list(ca1.parameters()) + list(ca3.parameters()),
        lr=args["lr"] * np.sqrt(args["batch_size"] / 1024), weight_decay=args["wd"])
    scheduler = MultiStepLR(optimizer, milestones=[4000, 12000], gamma=0.3)
    main_loss_fn, binary_loss_fn = nn.CrossEntropyLoss().to(device), nn.BCEWithLogitsLoss().to(device)

    def initialize_ca3():
        model.eval(); ca1.eval(); embeddings = []
        with torch.no_grad():
            for start in range(0, len(positive_train), args["batch_size"]):
                nodes = torch.as_tensor(positive_train[start:start + args["batch_size"]], dtype=torch.long)
                emb, _, _ = ca1(ca1_sequence[nodes].to(device), ca1_len[nodes].to(device),
                                ca1_mask[nodes].to(device))
                embeddings.append(emb.cpu())
        matrix = torch.cat(embeddings).numpy()
        kmeans = MiniBatchKMeans(n_clusters=args["ca3_num_prototypes"], random_state=args["seed"],
                                batch_size=max(args["batch_size"], args["ca3_num_prototypes"]),
                                n_init=10)
        kmeans.fit(matrix)
        ca3.initialize_prototypes(kmeans.cluster_centers_)

    def forward_batch(input_nodes, seeds, blocks, ca3_enabled):
        inputs, work_inputs, neigh_inputs, batch_labels, lpa_labels = load_lpa_subtensor(
            num_feat, cat_feat, nei_feat, {}, label_tensor, seeds, input_nodes, device, blocks,
            known_label_tensor)
        cpu_nodes = input_nodes.detach().cpu().long()
        ca1_embedding, ca1_logit, ca1_score = ca1(
            ca1_sequence[cpu_nodes].to(device), ca1_len[cpu_nodes].to(device), ca1_mask[cpu_nodes].to(device))
        ca3_out = ca3(ca1_embedding, enabled=ca3_enabled)
        local = _seed_local_indices(input_nodes, seeds)
        logits = model([block.to(device) for block in blocks], inputs, lpa_labels,
                       work_inputs, neigh_inputs, ca1_embedding=ca3_out.enhanced_embedding)
        return logits, ca1_logit[local], ca1_score[local], ca3_out, local, batch_labels

    if warmup == 0:
        initialize_ca3()
    best_loss, best_state, stale = float("inf"), None, 0
    history_path = os.path.join(run_dir, "epoch_history.csv")
    checkpoint_path = os.path.join(run_dir, "best_checkpoint.pt")
    for epoch in tqdm(range(args["max_epochs"]), desc="epochs", unit="epoch"):
        ca3_enabled = bool(ca3.initialized.item())
        epoch_started = time.perf_counter()
        model.train(); ca1.train(); ca3.train()
        accum = {key: [] for key in ("main", "ca1", "ca3", "div", "total")}
        train_y, train_score, train_pred, train_proto = [], [], [], []
        train_entropy, train_ids, train_gates = [], [], []
        for input_nodes, seeds, blocks in tqdm(train_loader, desc=f"epoch {epoch + 1:02d} train",
                                               leave=False, unit="batch"):
            logits, ca1_logit, _, ca3_out, local, batch_labels = forward_batch(
                input_nodes, seeds, blocks, ca3_enabled)
            valid = batch_labels != 2
            if not valid.any():
                continue
            main_loss = main_loss_fn(logits[valid], batch_labels[valid])
            ca1_loss = binary_loss_fn(ca1_logit[valid].squeeze(-1), batch_labels[valid].float())
            if ca3_enabled:
                ca3_loss = binary_loss_fn(ca3_out.proto_risk_logit[local][valid].squeeze(-1),
                                          batch_labels[valid].float())
                diversity = ca3.diversity_loss()
            else:
                ca3_loss = main_loss.new_zeros(()); diversity = main_loss.new_zeros(())
            total = (main_loss + args["ca1_aux_weight"] * ca1_loss
                     + args["ca3_aux_weight"] * ca3_loss
                     + args["ca3_diversity_weight"] * diversity)
            optimizer.zero_grad(); total.backward(); optimizer.step(); scheduler.step()
            for key, value in zip(accum, (main_loss, ca1_loss, ca3_loss, diversity, total)):
                accum[key].append(value.item())
            probs = torch.softmax(logits[valid].detach(), dim=1)[:, 1]
            train_y += batch_labels[valid].cpu().tolist(); train_score += probs.cpu().tolist()
            train_pred += (probs >= 0.5).long().cpu().tolist()
            if ca3_enabled:
                train_proto += ca3_out.proto_score[local][valid].detach().squeeze(-1).cpu().tolist()
                train_entropy += ca3_out.assignment_entropy[local][valid].detach().cpu().tolist()
                train_ids += ca3_out.top_proto_id[local][valid].detach().cpu().tolist()
                train_gates += ca3_out.gate[local][valid].detach().squeeze(-1).cpu().tolist()

        def evaluate(loader, description):
            model.eval(); ca1.eval(); ca3.eval(); result = {key: [] for key in ("main", "ca1", "ca3", "div", "total")}
            ys, scores, preds, proto_scores, entropies, proto_ids, gates = [], [], [], [], [], [], []
            with torch.no_grad():
                for input_nodes, seeds, blocks in tqdm(loader, desc=description, leave=False, unit="batch"):
                    logits, ca1_logit, _, out, local, batch_labels = forward_batch(input_nodes, seeds, blocks, ca3_enabled)
                    valid = batch_labels != 2
                    if not valid.any(): continue
                    ml = main_loss_fn(logits[valid], batch_labels[valid])
                    c1 = binary_loss_fn(ca1_logit[valid].squeeze(-1), batch_labels[valid].float())
                    c3 = (binary_loss_fn(out.proto_risk_logit[local][valid].squeeze(-1), batch_labels[valid].float())
                          if ca3_enabled else ml.new_zeros(()))
                    div = ca3.diversity_loss() if ca3_enabled else ml.new_zeros(())
                    total = ml + args["ca1_aux_weight"] * c1 + args["ca3_aux_weight"] * c3 + args["ca3_diversity_weight"] * div
                    for key, value in zip(result, (ml, c1, c3, div, total)): result[key].append(value.item())
                    prob = torch.softmax(logits[valid], dim=1)[:, 1]
                    ys += batch_labels[valid].cpu().tolist(); scores += prob.cpu().tolist(); preds += (prob >= 0.5).long().cpu().tolist()
                    if ca3_enabled:
                        proto_scores += out.proto_score[local][valid].squeeze(-1).cpu().tolist()
                        entropies += out.assignment_entropy[local][valid].cpu().tolist()
                        proto_ids += out.top_proto_id[local][valid].cpu().tolist()
                        gates += out.gate[local][valid].squeeze(-1).cpu().tolist()
            return result, ys, scores, preds, proto_scores, entropies, proto_ids, gates

        val_losses, val_y, val_scores, val_preds, val_proto, val_entropy, val_ids, val_gates = evaluate(
            val_loader, f"epoch {epoch + 1:02d} val")
        train_metrics = _classification_metrics(train_y, train_score, train_pred)
        val_metrics = _classification_metrics(val_y, val_scores, val_preds)
        train_proto_metrics = _classification_metrics(train_y, train_proto, [s >= .5 for s in train_proto]) if ca3_enabled else {"auc": np.nan, "ap": np.nan, "f1": np.nan}
        val_proto_metrics = _classification_metrics(val_y, val_proto, [s >= .5 for s in val_proto]) if ca3_enabled else {"auc": np.nan, "ap": np.nan, "f1": np.nan}
        val_main = float(np.mean(val_losses["main"]))
        row = {
            "run_id": run_id, "timestamp": datetime.now().astimezone().isoformat(), "epoch": epoch + 1,
            "epoch_seconds": time.perf_counter() - epoch_started, "ca3_enabled": ca3_enabled,
            "ca3_initialized": bool(ca3.initialized.item()),
        }
        for prefix, values in (("train", accum), ("val", val_losses)):
            for name, values_list in values.items(): row[f"{prefix}_{name}_loss"] = float(np.mean(values_list))
        for prefix, values in (("train", train_metrics), ("val", val_metrics)):
            for name, value in values.items(): row[f"{prefix}_{name}"] = value
        for prefix, values in (("train_proto_score", train_proto_metrics), ("val_proto_score", val_proto_metrics)):
            for name, value in values.items(): row[f"{prefix}_{name}"] = value
        row.update({
            "train_assignment_entropy": float(np.mean(train_entropy)) if train_entropy else np.nan,
            "val_assignment_entropy": float(np.mean(val_entropy)) if val_entropy else np.nan,
            "train_prototype_used_count": len(set(train_ids)), "val_prototype_used_count": len(set(val_ids)),
            "train_gate_mean": float(np.mean(train_gates)) if train_gates else np.nan,
            "val_gate_mean": float(np.mean(val_gates)) if val_gates else np.nan,
            "train_gate_max": float(np.max(train_gates)) if train_gates else np.nan,
            "val_gate_max": float(np.max(val_gates)) if val_gates else np.nan,
            "is_best": ca3_enabled and val_main < best_loss,
        })
        pd.DataFrame([row]).to_csv(history_path, mode="a", header=not os.path.exists(history_path), index=False)
        if row["is_best"]:
            best_loss, stale = val_main, 0
            best_state = {
                "rgtan_state_dict": copy.deepcopy(model.state_dict()),
                "ca1_state_dict": copy.deepcopy(ca1.state_dict()), "ca3_state_dict": copy.deepcopy(ca3.state_dict()),
                "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
                "scheduler_state_dict": copy.deepcopy(scheduler.state_dict()), "best_epoch": epoch + 1,
                "best_val_metric": val_main, "ca3_initialized": True, "ca3_init_epoch": warmup,
                "config": resolved,
            }
            torch.save(best_state, checkpoint_path)
        elif ca3_enabled:
            stale += 1
            if stale >= args["early_stopping"]: break
        if not ca3_enabled and epoch + 1 == warmup:
            initialize_ca3(); best_loss, stale = float("inf"), 0

    if best_state is None or not best_state["ca3_initialized"]:
        raise RuntimeError("No initialized CA3 checkpoint was selected")
    model.load_state_dict(best_state["rgtan_state_dict"]); ca1.load_state_dict(best_state["ca1_state_dict"])
    ca3.load_state_dict(best_state["ca3_state_dict"]); optimizer.load_state_dict(best_state["optimizer_state_dict"])
    scheduler.load_state_dict(best_state["scheduler_state_dict"])
    ca3_enabled = True
    val_eval = evaluate(val_loader, "best checkpoint val")
    threshold, val_f1 = best_macro_f1_threshold(val_eval[1], val_eval[2])

    model.eval(); ca1.eval(); ca3.eval(); assignments = []
    test_y, test_scores = [], []
    with torch.no_grad():
        for input_nodes, seeds, blocks in tqdm(test_loader, desc="best checkpoint test", leave=False, unit="batch"):
            logits, _, ca1_score, out, local, batch_labels = forward_batch(input_nodes, seeds, blocks, True)
            valid = batch_labels != 2; seed_ids = seeds[valid].detach().cpu().tolist()
            score = torch.softmax(logits[valid], dim=1)[:, 1]
            test_y += batch_labels[valid].cpu().tolist(); test_scores += score.cpu().tolist()
            for j, node_id in enumerate(seed_ids):
                assignments.append({
                    "TX_ID": args["_aml_sample_ids"][node_id], "label": int(batch_labels[valid][j].item()),
                    "rgtan_logit": float(logits[valid][j, 1].item()), "rgtan_score": float(score[j].item()),
                    "ca1_score_micro": float(ca1_score[valid][j].item()),
                    "ca3_proto_risk_logit": float(out.proto_risk_logit[local][valid][j].item()),
                    "ca3_proto_score": float(out.proto_score[local][valid][j].item()),
                    "top_proto_id": int(out.top_proto_id[local][valid][j].item()),
                    "top_proto_sim": float(out.top_proto_sim[local][valid][j].item()),
                    "assignment_entropy": float(out.assignment_entropy[local][valid][j].item()),
                    "gate": float(out.gate[local][valid][j].item()), "node_id": node_id,
                })
    test_preds = (np.asarray(test_scores) >= threshold).astype(int)
    test_metric = _classification_metrics(test_y, test_scores, test_preds)
    final = {
        "run_id": run_id, "method": "rgtan_ca1_ca3", "dataset": "aml", "seed": args["seed"],
        "test_auc": test_metric["auc"], "test_ap": test_metric["ap"], "test_f1": test_metric["f1"],
        "test_precision": precision_score(test_y, test_preds, zero_division=0),
        "test_recall": recall_score(test_y, test_preds, zero_division=0), "test_threshold": threshold,
        "val_auc_at_best": _classification_metrics(val_eval[1], val_eval[2], np.asarray(val_eval[2]) >= threshold)["auc"],
        "val_ap_at_best": _classification_metrics(val_eval[1], val_eval[2], np.asarray(val_eval[2]) >= threshold)["ap"],
        "val_f1_at_best": val_f1, "best_epoch": best_state["best_epoch"], "best_val_loss": best_loss,
        "duration_seconds": (datetime.now().astimezone() - started_dt).total_seconds(),
        "ca1_enabled": True, "ca3_enabled": True, "ca3_num_prototypes": args["ca3_num_prototypes"],
        "ca3_warmup_epochs": warmup, "ca3_temperature": args["ca3_temperature"],
        "ca3_aux_weight": args["ca3_aux_weight"], "ca3_diversity_weight": args["ca3_diversity_weight"],
    }
    pd.DataFrame([final]).to_csv(os.path.join(run_dir, "final_metrics.csv"), index=False)
    assignment_df = pd.DataFrame(assignments)
    assignment_df.to_csv(os.path.join(run_dir, "prototype_assignments.csv"), index=False)
    processed = pd.read_csv(args["_aml_processed_path"])[["TX_ID", "AlertID"]]
    assignment_df["TX_ID"] = assignment_df["TX_ID"].astype(str); processed["TX_ID"] = processed["TX_ID"].astype(str)
    explained = assignment_df.merge(processed, on="TX_ID", how="left")
    summaries = []
    for proto_id in range(args["ca3_num_prototypes"]):
        group = explained[explained["top_proto_id"] == proto_id]
        alert_counts = group["AlertID"].value_counts(dropna=True)
        summaries.append({
            "prototype_id": proto_id, "assigned_sample_count": len(group),
            "positive_sample_count": int(group["label"].sum()) if len(group) else 0,
            "positive_rate": float(group["label"].mean()) if len(group) else np.nan,
            "avg_similarity": float(group["top_proto_sim"].mean()) if len(group) else np.nan,
            "avg_gate": float(group["gate"].mean()) if len(group) else np.nan,
            "avg_proto_score": float(group["ca3_proto_score"].mean()) if len(group) else np.nan,
            "unique_alert_id_count": int(group["AlertID"].nunique()),
            "top_alert_id": alert_counts.index[0] if len(alert_counts) else None,
            "top_alert_id_count": int(alert_counts.iloc[0]) if len(alert_counts) else 0,
            "top_alert_id_ratio": float(alert_counts.iloc[0] / len(group)) if len(group) and len(alert_counts) else 0.0,
        })
    pd.DataFrame(summaries).to_csv(os.path.join(run_dir, "prototype_summary.csv"), index=False)
    usage = torch.bincount(torch.tensor(assignment_df["top_proto_id"].tolist()), minlength=args["ca3_num_prototypes"])
    positive_count = torch.zeros(args["ca3_num_prototypes"], dtype=torch.long)
    avg_similarity = torch.zeros(args["ca3_num_prototypes"], dtype=torch.float32)
    avg_gate = torch.zeros(args["ca3_num_prototypes"], dtype=torch.float32)
    for proto_id in range(args["ca3_num_prototypes"]):
        proto_rows = assignment_df[assignment_df["top_proto_id"] == proto_id]
        if len(proto_rows):
            positive_count[proto_id] = int(proto_rows["label"].sum())
            avg_similarity[proto_id] = float(proto_rows["top_proto_sim"].mean())
            avg_gate[proto_id] = float(proto_rows["gate"].mean())
    torch.save({
        "final_prototypes": ca3.prototypes.detach().cpu(), "init_prototypes": ca3.init_prototypes.cpu(),
        "prototype_usage_count": usage, "prototype_positive_count": positive_count,
        "prototype_avg_similarity": avg_similarity, "prototype_avg_gate": avg_gate,
        "ca3_num_prototypes": args["ca3_num_prototypes"],
        "ca3_init_method": args["ca3_init_method"], "ca3_init_epoch": warmup,
    }, os.path.join(run_dir, "prototype_bank.pt"))
    metadata = {**index_base, "base_method": "rgtan", "train_mode": "single_split",
                "split_mode": args["split_mode"], "ca3_initialized": True, "ca3_init_epoch": warmup,
                "ca3_init_data": "train_positive_only", "ca3_use_group_id": False,
                "alert_id_used_in_forward": False, "label_derived_neighbor_features": False,
                "label_propagation_scope": "train_only",
                "amount_normalization_scope": "train_split",
                "graph_feature_setting": "transductive_features_train_labels_only",
                "finished_at": datetime.now().astimezone().isoformat()}
    with open(os.path.join(run_dir, "metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    _update_experiment_index(results_dir, {**index_base, "auc": final["test_auc"], "ap": final["test_ap"],
                             "f1": final["test_f1"], "best_epoch": final["best_epoch"],
                             "best_val_loss": best_loss, "duration_seconds": final["duration_seconds"],
                             "status": "success"})
    return final


def rgtan_ca1_ca3_main(feat_df, graph, train_idx, val_idx, test_idx, labels, args,
                       cat_features, neigh_features, nei_att_head):
    results_dir = args.get("results_dir", "results")
    os.makedirs(results_dir, exist_ok=True)
    prefix = f"rgtan_ca1_ca3_aml_seed{args['seed']}_"
    before = {name for name in os.listdir(results_dir) if name.startswith(prefix)}
    try:
        return _rgtan_ca1_ca3_main_impl(
            feat_df, graph, train_idx, val_idx, test_idx, labels, args,
            cat_features, neigh_features, nei_att_head)
    except BaseException as exc:
        after = {name for name in os.listdir(results_dir) if name.startswith(prefix)}
        created = sorted(after - before)
        if created:
            run_id = created[-1]
            status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            failure = {
                "run_id": run_id, "method": "rgtan_ca1_ca3", "dataset": "aml",
                "seed": args["seed"], "ca1_enabled": True, "ca3_enabled": True,
                "ca3_num_prototypes": args["ca3_num_prototypes"],
                "ca3_warmup_epochs": args["ca3_warmup_epochs"],
                "run_dir": os.path.abspath(os.path.join(results_dir, run_id)),
                "created_at": datetime.now().astimezone().isoformat(), "status": status,
                "error_type": type(exc).__name__, "error_message": str(exc),
            }
            _update_experiment_index(results_dir, failure)
            with open(os.path.join(results_dir, run_id, "failure.json"), "w", encoding="utf-8") as handle:
                json.dump(failure, handle, ensure_ascii=False, indent=2)
        raise


def _rgtan_aml_single_main_impl(feat_df, graph, train_idx, val_idx, test_idx, labels, args,
                                cat_features, neigh_features, nei_att_head):
    started_dt = datetime.now().astimezone()
    run_id = f"rgtan_aml_seed{args['seed']}_{started_dt.strftime('%Y%m%d_%H%M%S')}"
    results_dir = args.get("results_dir", "results")
    run_dir = os.path.join(results_dir, run_id)
    os.makedirs(run_dir, exist_ok=False)
    resolved = {k: v for k, v in args.items() if not k.startswith("_")}
    resolved.update({"run_id": run_id, "run_dir": os.path.abspath(run_dir),
                     "train_mode": "single_split"})
    with open(os.path.join(run_dir, "config_resolved.yaml"), "w", encoding="utf-8") as handle:
        yaml.safe_dump(resolved, handle, allow_unicode=True, sort_keys=False)
    index_base = {
        "run_id": run_id, "method": "rgtan", "dataset": "aml", "seed": args["seed"],
        "ca1_enabled": False, "ca3_enabled": False, "ca3_num_prototypes": np.nan,
        "ca3_warmup_epochs": np.nan, "run_dir": os.path.abspath(run_dir),
        "created_at": started_dt.isoformat(),
    }
    _update_experiment_index(results_dir, {**index_base, "status": "running"})
    device = torch.device(args["device"])
    graph = graph.to(device)
    num_feat = torch.from_numpy(feat_df.values).float().to(device)
    cat_feat = {col: torch.from_numpy(feat_df[col].values).long().to(device) for col in cat_features}
    nei_feat = ({col: torch.from_numpy(neigh_features[col].values).float().to(device)
                 for col in neigh_features.columns} if isinstance(neigh_features, pd.DataFrame) else [])
    label_tensor = torch.from_numpy(labels.values).long().to(device)
    known_labels = torch.full_like(label_tensor, 2)
    train_nodes = torch.as_tensor(train_idx, dtype=torch.long, device=device)
    known_labels[train_nodes] = label_tensor[train_nodes]

    def make_loader(indices, shuffle=False):
        return NodeDataLoader(
            graph, torch.as_tensor(indices, dtype=torch.long, device=device),
            MultiLayerFullNeighborSampler(args["n_layers"]), device=device, use_ddp=False,
            batch_size=args["batch_size"], shuffle=shuffle, drop_last=False, num_workers=0)

    train_loader, val_loader, test_loader = (
        make_loader(train_idx, True), make_loader(val_idx), make_loader(test_idx))
    model = RGTAN(
        in_feats=feat_df.shape[1], hidden_dim=args["hid_dim"] // 4, n_classes=2,
        heads=[4] * args["n_layers"], activation=nn.PReLU(), n_layers=args["n_layers"],
        drop=args["dropout"], device=device, gated=args["gated"], ref_df=feat_df,
        cat_features=cat_feat, neigh_features=nei_feat, nei_att_head=nei_att_head).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args["lr"] * np.sqrt(args["batch_size"] / 1024),
                           weight_decay=args["wd"])
    scheduler = MultiStepLR(optimizer, milestones=[4000, 12000], gamma=0.3)
    loss_fn = nn.CrossEntropyLoss().to(device)

    def forward_batch(input_nodes, seeds, blocks):
        inputs, work_inputs, neigh_inputs, batch_labels, lpa_labels = load_lpa_subtensor(
            num_feat, cat_feat, nei_feat, {}, label_tensor, seeds, input_nodes, device, blocks,
            known_labels)
        logits = model([block.to(device) for block in blocks], inputs, lpa_labels,
                       work_inputs, neigh_inputs)
        return logits, batch_labels

    def evaluate(loader, description):
        model.eval(); losses, ys, scores = [], [], []
        with torch.no_grad():
            for input_nodes, seeds, blocks in tqdm(loader, desc=description, leave=False, unit="batch"):
                logits, batch_labels = forward_batch(input_nodes, seeds, blocks)
                valid = batch_labels != 2
                if not valid.any():
                    continue
                losses.append(loss_fn(logits[valid], batch_labels[valid]).item())
                probabilities = torch.softmax(logits[valid], dim=1)[:, 1]
                ys += batch_labels[valid].cpu().tolist(); scores += probabilities.cpu().tolist()
        preds = (np.asarray(scores) >= 0.5).astype(int).tolist()
        return losses, ys, scores, preds

    best_loss, best_state, stale = float("inf"), None, 0
    history_path = os.path.join(run_dir, "epoch_history.csv")
    checkpoint_path = os.path.join(run_dir, "best_checkpoint.pt")
    for epoch in tqdm(range(args["max_epochs"]), desc="epochs", unit="epoch"):
        epoch_started = time.perf_counter(); model.train()
        train_losses, train_y, train_scores = [], [], []
        for input_nodes, seeds, blocks in tqdm(train_loader, desc=f"epoch {epoch + 1:02d} train",
                                               leave=False, unit="batch"):
            logits, batch_labels = forward_batch(input_nodes, seeds, blocks)
            valid = batch_labels != 2
            if not valid.any():
                continue
            loss = loss_fn(logits[valid], batch_labels[valid])
            optimizer.zero_grad(); loss.backward(); optimizer.step(); scheduler.step()
            train_losses.append(loss.item())
            probabilities = torch.softmax(logits[valid].detach(), dim=1)[:, 1]
            train_y += batch_labels[valid].cpu().tolist(); train_scores += probabilities.cpu().tolist()
        val_losses, val_y, val_scores, val_preds = evaluate(val_loader, f"epoch {epoch + 1:02d} val")
        train_preds = (np.asarray(train_scores) >= 0.5).astype(int).tolist()
        train_metrics = _classification_metrics(train_y, train_scores, train_preds)
        val_metrics = _classification_metrics(val_y, val_scores, val_preds)
        train_loss, val_loss = float(np.mean(train_losses)), float(np.mean(val_losses))
        improved = val_loss < best_loss
        row = {
            "run_id": run_id, "timestamp": datetime.now().astimezone().isoformat(),
            "epoch": epoch + 1, "epoch_seconds": time.perf_counter() - epoch_started,
            "train_main_loss": train_loss, "train_ca1_aux_loss": 0.0,
            "train_total_loss": train_loss, "train_auc": train_metrics["auc"],
            "train_ap": train_metrics["ap"], "train_f1": train_metrics["f1"],
            "val_main_loss": val_loss, "val_ca1_aux_loss": 0.0, "val_total_loss": val_loss,
            "val_auc": val_metrics["auc"], "val_ap": val_metrics["ap"],
            "val_f1": val_metrics["f1"], "is_best": improved,
        }
        pd.DataFrame([row]).to_csv(history_path, mode="a", header=not os.path.exists(history_path), index=False)
        if improved:
            best_loss, stale = val_loss, 0
            best_state = {
                "rgtan_state_dict": copy.deepcopy(model.state_dict()),
                "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
                "scheduler_state_dict": copy.deepcopy(scheduler.state_dict()),
                "best_epoch": epoch + 1, "best_val_metric": val_loss, "config": resolved,
            }
            torch.save(best_state, checkpoint_path)
        else:
            stale += 1
            if stale >= args["early_stopping"]:
                break
    if best_state is None:
        raise RuntimeError("No valid AML baseline checkpoint was selected")
    model.load_state_dict(best_state["rgtan_state_dict"])
    optimizer.load_state_dict(best_state["optimizer_state_dict"])
    scheduler.load_state_dict(best_state["scheduler_state_dict"])
    val_eval = evaluate(val_loader, "best checkpoint val")
    threshold, val_f1 = best_macro_f1_threshold(val_eval[1], val_eval[2])
    test_eval = evaluate(test_loader, "best checkpoint test")
    test_preds = (np.asarray(test_eval[2]) >= threshold).astype(int)
    test_metrics = _classification_metrics(test_eval[1], test_eval[2], test_preds)
    val_threshold_metrics = _classification_metrics(
        val_eval[1], val_eval[2], (np.asarray(val_eval[2]) >= threshold).astype(int))
    final = {
        "run_id": run_id, "method": "rgtan", "dataset": "aml", "seed": args["seed"],
        "test_auc": test_metrics["auc"], "test_ap": test_metrics["ap"], "test_f1": test_metrics["f1"],
        "test_precision": precision_score(test_eval[1], test_preds, zero_division=0),
        "test_recall": recall_score(test_eval[1], test_preds, zero_division=0),
        "test_threshold": threshold, "val_auc_at_best": val_threshold_metrics["auc"],
        "val_ap_at_best": val_threshold_metrics["ap"], "val_f1_at_best": val_f1,
        "best_epoch": best_state["best_epoch"], "best_val_loss": best_loss,
        "duration_seconds": (datetime.now().astimezone() - started_dt).total_seconds(),
        "train_size": len(train_idx), "val_size": len(val_idx), "test_size": len(test_idx),
        "ca1_enabled": False, "ca3_enabled": False,
    }
    pd.DataFrame([final]).to_csv(os.path.join(run_dir, "final_metrics.csv"), index=False)
    metadata = {
        **index_base, "train_mode": "single_split", "split_mode": "sender_account",
        "label_propagation_scope": "train_only", "label_derived_neighbor_features": False,
        "amount_normalization_scope": "train_split",
        "graph_feature_setting": "transductive_features_train_labels_only",
        "alert_id_used_in_forward": False,
        "finished_at": datetime.now().astimezone().isoformat(),
    }
    with open(os.path.join(run_dir, "metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    _update_experiment_index(results_dir, {**index_base, "auc": final["test_auc"],
                             "ap": final["test_ap"], "f1": final["test_f1"],
                             "best_epoch": final["best_epoch"], "best_val_loss": best_loss,
                             "duration_seconds": final["duration_seconds"], "status": "success"})
    return final


def rgtan_aml_single_main(feat_df, graph, train_idx, val_idx, test_idx, labels, args,
                          cat_features, neigh_features, nei_att_head):
    results_dir = args.get("results_dir", "results")
    os.makedirs(results_dir, exist_ok=True)
    prefix = f"rgtan_aml_seed{args['seed']}_"
    before = {name for name in os.listdir(results_dir) if name.startswith(prefix)}
    try:
        return _rgtan_aml_single_main_impl(
            feat_df, graph, train_idx, val_idx, test_idx, labels, args,
            cat_features, neigh_features, nei_att_head)
    except BaseException as exc:
        created = sorted({name for name in os.listdir(results_dir) if name.startswith(prefix)} - before)
        if created:
            run_id = created[-1]
            status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            failure = {
                "run_id": run_id, "method": "rgtan", "dataset": "aml", "seed": args["seed"],
                "ca1_enabled": False, "ca3_enabled": False,
                "run_dir": os.path.abspath(os.path.join(results_dir, run_id)),
                "created_at": datetime.now().astimezone().isoformat(), "status": status,
                "error_type": type(exc).__name__, "error_message": str(exc),
            }
            _update_experiment_index(results_dir, failure)
            with open(os.path.join(results_dir, run_id, "failure.json"), "w", encoding="utf-8") as handle:
                json.dump(failure, handle, ensure_ascii=False, indent=2)
        raise


def rgtan_main(feat_df, graph, train_idx, test_idx, labels, args, cat_features, neigh_features: pd.DataFrame, nei_att_head):
    # torch.autograd.set_detect_anomaly(True)
    device = args['device']
    graph = graph.to(device)
    oof_predictions = torch.from_numpy(
        np.zeros([len(feat_df), 2])).float().to(device)
    test_predictions = torch.from_numpy(
        np.zeros([len(feat_df), 2])).float().to(device)
    kfold = StratifiedKFold(
        n_splits=args['n_fold'], shuffle=True, random_state=args['seed'])

    y_target = labels.iloc[train_idx].values
    num_feat = torch.from_numpy(feat_df.values).float().to(device)
    cat_feat = {col: torch.from_numpy(feat_df[col].values).long().to(
        device) for col in cat_features}

    neigh_padding_dict = {}
    nei_feat = []
    if isinstance(neigh_features, pd.DataFrame):  # otherwise []
        # if null it is []
        nei_feat = {col: torch.from_numpy(neigh_features[col].values).to(torch.float32).to(
            device) for col in neigh_features.columns}
        
    y = labels
    labels = torch.from_numpy(y.values).long().to(device)
    loss_fn = nn.CrossEntropyLoss().to(device)
    for fold, (trn_idx, val_idx) in enumerate(kfold.split(feat_df.iloc[train_idx], y_target)):
        print(f'Training fold {fold + 1}')
        trn_ind, val_ind = torch.from_numpy(np.array(train_idx)[trn_idx]).long().to(
            device), torch.from_numpy(np.array(train_idx)[val_idx]).long().to(device)
        propagation_labels = None
        if args.get("dataset") == "aml":
            propagation_labels = torch.full_like(labels, 2)
            propagation_labels[trn_ind] = labels[trn_ind]

        train_sampler = MultiLayerFullNeighborSampler(args['n_layers'])
        train_dataloader = NodeDataLoader(graph,
                                          trn_ind,
                                          train_sampler,
                                          device=device,
                                          use_ddp=False,
                                          batch_size=args['batch_size'],
                                          shuffle=True,
                                          drop_last=False,
                                          num_workers=0
                                          )
        val_sampler = MultiLayerFullNeighborSampler(args['n_layers'])
        val_dataloader = NodeDataLoader(graph,
                                        val_ind,
                                        val_sampler,
                                        use_ddp=False,
                                        device=device,
                                        batch_size=args['batch_size'],
                                        shuffle=True,
                                        drop_last=False,
                                        num_workers=0,
                                        )
        model = RGTAN(in_feats=feat_df.shape[1],
                      hidden_dim=args['hid_dim']//4,
                      n_classes=2,
                      heads=[4]*args['n_layers'],
                      activation=nn.PReLU(),
                      n_layers=args['n_layers'],
                      drop=args['dropout'],
                      device=device,
                      gated=args['gated'],
                      ref_df=feat_df,
                      cat_features=cat_feat,
                      neigh_features=nei_feat,
                      nei_att_head=nei_att_head).to(device)
        lr = args['lr'] * np.sqrt(args['batch_size']/1024)
        optimizer = optim.Adam(model.parameters(), lr=lr,
                               weight_decay=args['wd'])
        lr_scheduler = MultiStepLR(optimizer=optimizer, milestones=[
                                   4000, 12000], gamma=0.3)

        earlystoper = early_stopper(
            patience=args['early_stopping'], verbose=True)
        start_epoch, max_epochs = 0, 2000
        for epoch in range(start_epoch, args['max_epochs']):
            train_loss_list = []
            # train_acc_list = []
            model.train()
            for step, (input_nodes, seeds, blocks) in enumerate(train_dataloader):
                # print(f"loading batch data...")
                batch_inputs, batch_work_inputs, batch_neighstat_inputs, batch_labels, lpa_labels = load_lpa_subtensor(num_feat, cat_feat, nei_feat, neigh_padding_dict, labels,
                                                                                                                       seeds, input_nodes, device, blocks, propagation_labels)
                # print(f"load {step}")

                # batch_neighstat_inputs: {"degree":(|batch|, degree_dim)}

                blocks = [block.to(device) for block in blocks]
                train_batch_logits = model(
                    blocks, batch_inputs, lpa_labels, batch_work_inputs, batch_neighstat_inputs)
                mask = batch_labels == 2
                train_batch_logits = train_batch_logits[~mask]
                batch_labels = batch_labels[~mask]
                # batch_labels[mask] = 0

                train_loss = loss_fn(train_batch_logits, batch_labels)
                # backward
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                lr_scheduler.step()
                train_loss_list.append(train_loss.cpu().detach().numpy())

                if step % 10 == 0:
                    tr_batch_pred = torch.sum(torch.argmax(train_batch_logits.clone(
                    ).detach(), dim=1) == batch_labels) / batch_labels.shape[0]
                    score = torch.softmax(train_batch_logits.clone().detach(), dim=1)[
                        :, 1].cpu().numpy()
                    try:
                        print('In epoch:{:03d}|batch:{:04d}, train_loss:{:4f}, '
                              'train_ap:{:.4f}, train_acc:{:.4f}, train_auc:{:.4f}'.format(epoch, step,
                                                                                           np.mean(
                                                                                               train_loss_list),
                                                                                           average_precision_score(
                                                                                               batch_labels.cpu().numpy(), score),
                                                                                           tr_batch_pred.detach(),
                                                                                           roc_auc_score(batch_labels.cpu().numpy(), score)))
                    except:
                        pass

            # mini-batch for validation
            val_loss_list = 0
            val_acc_list = 0
            val_all_list = 0
            model.eval()
            with torch.no_grad():
                for step, (input_nodes, seeds, blocks) in enumerate(val_dataloader):
                    batch_inputs, batch_work_inputs, batch_neighstat_inputs, batch_labels, lpa_labels = load_lpa_subtensor(num_feat, cat_feat, nei_feat, neigh_padding_dict, labels,
                                                                                                                           seeds, input_nodes, device, blocks, propagation_labels)

                    blocks = [block.to(device) for block in blocks]
                    val_batch_logits = model(
                        blocks, batch_inputs, lpa_labels, batch_work_inputs, batch_neighstat_inputs)
                    oof_predictions[seeds] = val_batch_logits
                    mask = batch_labels == 2
                    val_batch_logits = val_batch_logits[~mask]
                    batch_labels = batch_labels[~mask]
                    # batch_labels[mask] = 0
                    val_loss_list = val_loss_list + \
                        loss_fn(val_batch_logits, batch_labels)
                    # val_all_list += 1
                    val_batch_pred = torch.sum(torch.argmax(
                        val_batch_logits, dim=1) == batch_labels) / torch.tensor(batch_labels.shape[0])
                    val_acc_list = val_acc_list + val_batch_pred * \
                        torch.tensor(
                            batch_labels.shape[0])  # how many in this batch is right!
                    val_all_list = val_all_list + \
                        batch_labels.shape[0]  # how many val nodes
                    if step % 10 == 0:
                        score = torch.softmax(val_batch_logits.clone().detach(), dim=1)[
                            :, 1].cpu().numpy()
                        try:
                            print('In epoch:{:03d}|batch:{:04d}, val_loss:{:4f}, val_ap:{:.4f}, '
                                  'val_acc:{:.4f}, val_auc:{:.4f}'.format(epoch,
                                                                          step,
                                                                          val_loss_list/val_all_list,
                                                                          average_precision_score(
                                                                              batch_labels.cpu().numpy(), score),
                                                                          val_batch_pred.detach(),
                                                                          roc_auc_score(batch_labels.cpu().numpy(), score)))
                        except:
                            pass

            # val_acc_list/val_all_list, model)
            earlystoper.earlystop(val_loss_list/val_all_list, model)
            if earlystoper.is_earlystop:
                print("Early Stopping!")
                break
        print("Best val_loss is: {:.7f}".format(earlystoper.best_cv))
        test_ind = torch.from_numpy(np.array(test_idx)).long().to(device)
        test_sampler = MultiLayerFullNeighborSampler(args['n_layers'])
        test_dataloader = NodeDataLoader(graph,
                                         test_ind,
                                         test_sampler,
                                         use_ddp=False,
                                         device=device,
                                         batch_size=args['batch_size'],
                                         shuffle=True,
                                         drop_last=False,
                                         num_workers=0,
                                         )
        b_model = earlystoper.best_model.to(device)
        b_model.eval()
        with torch.no_grad():
            for step, (input_nodes, seeds, blocks) in enumerate(test_dataloader):
                # print(input_nodes)
                batch_inputs, batch_work_inputs, batch_neighstat_inputs, batch_labels, lpa_labels = load_lpa_subtensor(num_feat, cat_feat, nei_feat, neigh_padding_dict, labels,
                                                                                                                       seeds, input_nodes, device, blocks, propagation_labels)

                blocks = [block.to(device) for block in blocks]
                test_batch_logits = b_model(
                    blocks, batch_inputs, lpa_labels, batch_work_inputs, batch_neighstat_inputs)
                test_predictions[seeds] = test_batch_logits
                test_batch_pred = torch.sum(torch.argmax(
                    test_batch_logits, dim=1) == batch_labels) / torch.tensor(batch_labels.shape[0])
                if step % 10 == 0:
                    print('In test batch:{:04d}'.format(step))
    mask = y_target == 2
    y_target[mask] = 0
    my_ap = average_precision_score(y_target, torch.softmax(
        oof_predictions, dim=1).cpu()[train_idx, 1])
    print("NN out of fold AP is:", my_ap)
    b_models, val_gnn_0, test_gnn_0 = earlystoper.best_model.to(
        'cpu'), oof_predictions, test_predictions

    test_score = torch.softmax(test_gnn_0, dim=1)[test_idx, 1].cpu().numpy()
    y_target = labels[test_idx].cpu().numpy()
    test_score1 = torch.argmax(test_gnn_0, dim=1)[test_idx].cpu().numpy()

    mask = y_target != 2
    test_score = test_score[mask]
    y_target = y_target[mask]
    test_score1 = test_score1[mask]

    print("test AUC:", roc_auc_score(y_target, test_score))
    print("test f1:", f1_score(y_target, test_score1, average="macro"))
    print("test AP:", average_precision_score(y_target, test_score))


def loda_rgtan_data(args):
    # prefix = "./antifraud/data/"
    prefix = "data/"
    dataset = args["dataset"]
    test_size = args["test_size"]
    if dataset == 'S-FFSD':
        cat_features = ["Target", "Location", "Type"]

        
        df = pd.read_csv(prefix + "S-FFSDneofull.csv")
        df = df.loc[:, ~df.columns.str.contains('Unnamed')]
        #####
        neigh_features = []
        #####
        data = df[df["Labels"] <= 2]
        data = data.reset_index(drop=True)
        out = []
        alls = []
        allt = []
        pair = ["Source", "Target", "Location", "Type"]
        for column in pair:
            src, tgt = [], []
            edge_per_trans = 3
            for c_id, c_df in tqdm(data.groupby(column), desc=column):
                c_df = c_df.sort_values(by="Time")
                df_len = len(c_df)
                sorted_idxs = c_df.index
                src.extend([sorted_idxs[i] for i in range(df_len)
                            for j in range(edge_per_trans) if i + j < df_len])
                tgt.extend([sorted_idxs[i+j] for i in range(df_len)
                            for j in range(edge_per_trans) if i + j < df_len])
            alls.extend(src)
            allt.extend(tgt)
        alls = np.array(alls)
        allt = np.array(allt)
        g = dgl.graph((alls, allt))
        cal_list = ["Source", "Target", "Location", "Type"]
        for col in cal_list:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].apply(str).values)
        feat_data = data.drop("Labels", axis=1)
        labels = data["Labels"]

        #######
        g.ndata['label'] = torch.from_numpy(
            labels.to_numpy()).to(torch.long)
        g.ndata['feat'] = torch.from_numpy(
            feat_data.to_numpy()).to(torch.float32)
        #######

        graph_path = prefix+"graph-{}.bin".format(dataset)
        dgl.data.utils.save_graphs(graph_path, [g])
        index = list(range(len(labels)))

        train_idx, test_idx, y_train, y_test = train_test_split(index, labels, stratify=labels, test_size=0.6,
                                                                random_state=2, shuffle=True)
        feat_neigh = pd.read_csv(
            prefix + "S-FFSD_neigh_feat.csv")
        print("neighborhood feature loaded for nn input.")
        neigh_features = feat_neigh

    elif dataset == 'yelp':
        cat_features = []
        neigh_features = []
        data_file = loadmat(prefix + 'YelpChi.mat')
        labels = pd.DataFrame(data_file['label'].flatten())[0]
        feat_data = pd.DataFrame(data_file['features'].todense().A)
        # load the preprocessed adj_lists
        with open(prefix + 'yelp_homo_adjlists.pickle', 'rb') as file:
            homo = pickle.load(file)
        file.close()
        index = list(range(len(labels)))
        train_idx, test_idx, y_train, y_test = train_test_split(index, labels, stratify=labels, test_size=test_size,
                                                                random_state=2, shuffle=True)
        src = []
        tgt = []
        for i in homo:
            for j in homo[i]:
                src.append(i)
                tgt.append(j)
        src = np.array(src)
        tgt = np.array(tgt)
        g = dgl.graph((src, tgt))
        g.ndata['label'] = torch.from_numpy(labels.to_numpy()).to(torch.long)
        g.ndata['feat'] = torch.from_numpy(
            feat_data.to_numpy()).to(torch.float32)
        graph_path = prefix + "graph-{}.bin".format(dataset)
        dgl.data.utils.save_graphs(graph_path, [g])

        try:
            feat_neigh = pd.read_csv(
                prefix + "yelp_neigh_feat.csv")
            print("neighborhood feature loaded for nn input.")
            neigh_features = feat_neigh
        except:
            print("no neighbohood feature used.")

    elif dataset == 'amazon':
        cat_features = []
        neigh_features = []
        data_file = loadmat(prefix + 'Amazon.mat')
        labels = pd.DataFrame(data_file['label'].flatten())[0]
        feat_data = pd.DataFrame(data_file['features'].todense().A)
        # load the preprocessed adj_lists
        with open(prefix + 'amz_homo_adjlists.pickle', 'rb') as file:
            homo = pickle.load(file)
        file.close()
        index = list(range(3305, len(labels)))
        train_idx, test_idx, y_train, y_test = train_test_split(index, labels[3305:], stratify=labels[3305:],
                                                                test_size=test_size, random_state=2, shuffle=True)
        src = []
        tgt = []
        for i in homo:
            for j in homo[i]:
                src.append(i)
                tgt.append(j)
        src = np.array(src)
        tgt = np.array(tgt)
        g = dgl.graph((src, tgt))
        g.ndata['label'] = torch.from_numpy(labels.to_numpy()).to(torch.long)
        g.ndata['feat'] = torch.from_numpy(
            feat_data.to_numpy()).to(torch.float32)
        graph_path = prefix + "graph-{}.bin".format(dataset)
        dgl.data.utils.save_graphs(graph_path, [g])
        try:
            feat_neigh = pd.read_csv(
                prefix + "amazon_neigh_feat.csv")
            print("neighborhood feature loaded for nn input.")
            neigh_features = feat_neigh
        except:
            print("no neighbohood feature used.")

    elif dataset == 'aml':
        cat_features = ["Target", "Type"]
        data_path = args.get("data_path", "../AMLdataset.csv")
        raw_csv_path = _resolve_dataset_path(data_path)
        artifact_paths = preprocess_aml_for_gtan(raw_csv_path, output_dir=prefix)

        processed = pd.read_csv(artifact_paths["processed_path"])
        feat_data = pd.read_csv(artifact_paths["feat_path"])
        labels = pd.read_csv(artifact_paths["label_path"])["Labels"].astype(int)
        g = dgl.load_graphs(artifact_paths["graph_path"])[0][0]
        processed_ids = processed["TX_ID"].astype(str).tolist()
        feature_ids = feat_data["TX_ID"].astype(str).tolist()
        if processed_ids != feature_ids or len(processed_ids) != g.num_nodes():
            raise ValueError("AML processed rows, feature rows, and graph node IDs are misaligned")

        for col in cat_features:
            le = LabelEncoder()
            feat_data[col] = le.fit_transform(feat_data[col].astype(str))

        feat_data = feat_data.drop(
            columns=[col for col in ["Source", "AlertID", "SenderPrevFraudCount", "TX_ID"]
                     if col in feat_data.columns])
        forbidden_features = {"Labels", "AlertID", "SenderPrevFraudCount", "TX_ID", "Source"}
        leaked_features = forbidden_features.intersection(feat_data.columns)
        if leaked_features:
            raise RuntimeError(f"Forbidden AML forward features remain: {sorted(leaked_features)}")

        train_idx, test_idx = _sender_account_train_test_split(
            processed["Source"],
            processed["Labels"],
            test_size=test_size,
            seed=args["seed"],
        )
        if args.get("method") in {"rgtan", "rgtan_ca1", "rgtan_ca1_ca3"}:
            remaining_groups = processed.iloc[train_idx]["Source"].reset_index(drop=True)
            remaining_labels = processed.iloc[train_idx]["Labels"].reset_index(drop=True)
            relative_train, relative_val = _sender_account_train_test_split(
                remaining_groups, remaining_labels, test_size=args["val_size"], seed=args["seed"] + 1)
            original_train = np.asarray(train_idx)
            train_idx, val_idx = original_train[relative_train], original_train[relative_val]
            train_senders = set(processed.iloc[train_idx]["Source"].astype(str))
            val_senders = set(processed.iloc[val_idx]["Source"].astype(str))
            test_senders = set(processed.iloc[test_idx]["Source"].astype(str))
            if train_senders & val_senders or train_senders & test_senders or val_senders & test_senders:
                raise RuntimeError("AML sender-account split leakage detected")
            args["_val_idx"] = val_idx
            args["_aml_processed_path"] = artifact_paths["processed_path"]
            args["_aml_sample_ids"] = processed_ids
        normalization_idx = train_idx
        amount_train = processed.iloc[normalization_idx]["Amount"].astype(float)
        amount_mean = float(amount_train.mean())
        amount_std = float(amount_train.std() + 1e-6)
        feat_data["AmountNorm"] = (processed["Amount"].astype(float) - amount_mean) / amount_std
        args["_aml_amount_mean"] = amount_mean
        args["_aml_amount_std"] = amount_std
        neigh_features = _build_aml_neigh_features(g, labels)
        print("AML neighborhood features generated for RGTAN input.")
    else:
        raise NotImplementedError(f"Unsupported RGTAN dataset: {dataset}")

    return feat_data, labels, train_idx, test_idx, g, cat_features, neigh_features
