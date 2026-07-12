import numpy as np
import dgl
import torch
import os
import json
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
import torch.optim as optim
from scipy.io import loadmat
import pandas as pd
import pickle
from sklearn.model_selection import StratifiedKFold, train_test_split
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder, QuantileTransformer
from dgl.dataloading import MultiLayerFullNeighborSampler
from dgl.dataloading import NodeDataLoader
from torch.optim.lr_scheduler import MultiStepLR
from .gtan_model import GraphAttnModel
from . import *
from feature_engineering.data_process import preprocess_aml_for_gtan


def _resolve_dataset_path(data_path: str) -> str:
    """Resolve dataset paths relative to the project root first."""
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


def _group_train_test_val_split(
    group_ids: pd.Series,
    labels: pd.Series,
    val_size: float,
    test_size: float,
    seed: int,
):
    group_label_df = pd.DataFrame({"group_id": group_ids.astype(str), "label": labels.astype(int)})
    group_level = group_label_df.groupby("group_id")["label"].max().reset_index()
    stratify_labels = group_level["label"]

    def _safe_stratify(values: pd.Series):
        if values.nunique() < 2:
            return None
        min_count = values.value_counts().min()
        return values if min_count >= 2 else None

    group_list = group_level["group_id"].to_numpy()
    stratify = _safe_stratify(stratify_labels)
    holdout_size = val_size + test_size
    if holdout_size <= 0:
        train_groups = group_list
        temp_groups = np.array([], dtype=object)
    else:
        train_groups, temp_groups = train_test_split(
            group_list,
            test_size=holdout_size,
            random_state=seed,
            shuffle=True,
            stratify=stratify,
        )
        if stratify is None:
            print("Warning: sender-account stratified split unavailable, falling back to non-stratified group split.")

    val_groups = np.array([], dtype=object)
    test_groups = np.array([], dtype=object)
    if len(temp_groups) > 0:
        temp_level = group_level[group_level["group_id"].isin(temp_groups)]
        temp_stratify = _safe_stratify(temp_level["label"])
        relative_test_size = test_size / (val_size + test_size) if (val_size + test_size) > 0 else 0.0
        if relative_test_size > 0 and len(temp_groups) >= 2:
            val_groups, test_groups = train_test_split(
                temp_groups,
                test_size=relative_test_size,
                random_state=seed,
                shuffle=True,
                stratify=temp_stratify,
            )
            if temp_stratify is None:
                print("Warning: val/test group stratification unavailable, using non-stratified sender-account split.")
        elif relative_test_size > 0:
            split_at = max(1, int(round(len(temp_groups) * (1 - relative_test_size))))
            val_groups = temp_groups[:split_at]
            test_groups = temp_groups[split_at:]
        else:
            val_groups = temp_groups

    train_mask = group_ids.astype(str).isin(train_groups)
    val_mask = group_ids.astype(str).isin(val_groups)
    test_mask = group_ids.astype(str).isin(test_groups)
    return (
        np.flatnonzero(train_mask.to_numpy()),
        np.flatnonzero(val_mask.to_numpy()),
        np.flatnonzero(test_mask.to_numpy()),
        {
            "group_count": int(len(group_list)),
            "train_group_count": int(len(train_groups)),
            "val_group_count": int(len(val_groups)),
            "test_group_count": int(len(test_groups)),
            "used_group_stratify": bool(stratify is not None),
        },
    )


def _save_metrics(args, graph, train_idx, val_idx, test_idx, labels, test_auc, test_f1, test_ap, split_meta):
    results_dir = os.path.join(os.path.dirname(__file__), "..", "..", "results")
    os.makedirs(results_dir, exist_ok=True)

    labels_np = labels.to_numpy().astype(int)
    metrics = {
        "dataset": args["dataset"],
        "seed": int(args["seed"]),
        "split_mode": split_meta.get("split_mode", "random"),
        "semantic": split_meta.get("semantic", "transaction-sample graph experiment"),
        "num_nodes": int(graph.num_nodes()),
        "num_edges": int(graph.num_edges()),
        "train_size": int(len(train_idx)),
        "val_size": int(len(val_idx)),
        "test_size": int(len(test_idx)),
        "train_pos_ratio": float(labels_np[train_idx].mean()) if len(train_idx) else 0.0,
        "val_pos_ratio": float(labels_np[val_idx].mean()) if len(val_idx) else 0.0,
        "test_pos_ratio": float(labels_np[test_idx].mean()) if len(test_idx) else 0.0,
        "auc": float(test_auc),
        "f1": float(test_f1),
        "ap": float(test_ap),
        "group_count": int(split_meta.get("group_count", 0)),
        "train_group_count": int(split_meta.get("train_group_count", 0)),
        "val_group_count": int(split_meta.get("val_group_count", 0)),
        "test_group_count": int(split_meta.get("test_group_count", 0)),
        "used_group_stratify": bool(split_meta.get("used_group_stratify", False)),
    }

    metrics_path = os.path.join(results_dir, f"gtan_{args['dataset']}_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    if metrics["auc"] <= 0.55 and metrics["ap"] <= 0.55 and metrics["f1"] <= 0.55:
        print("Warning: AML metrics are close to random. Check feature quality and class balance.")
    if metrics["auc"] >= 0.995 and metrics["ap"] >= 0.995 and metrics["f1"] >= 0.995:
        print("Warning: AML metrics are unusually high. Check for leakage in graph construction or splits.")
    print(f"Saved metrics to {metrics_path}")
    return metrics_path


def gtan_main(feat_df, graph, train_idx, val_idx, test_idx, labels, args, cat_features, split_meta=None):
    device = args['device']
    graph = graph.to(device)
    if split_meta is None:
        split_meta = {}
    holdout_val_idx = np.array(val_idx, copy=True)
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

    y = labels
    labels = torch.from_numpy(y.values).long().to(device)
    loss_fn = nn.CrossEntropyLoss().to(device)
    for fold, (trn_idx, fold_val_idx) in enumerate(kfold.split(feat_df.iloc[train_idx], y_target)):
        print(f'Training fold {fold + 1}')
        trn_ind, val_ind = torch.from_numpy(np.array(train_idx)[trn_idx]).long().to(
            device), torch.from_numpy(np.array(train_idx)[fold_val_idx]).long().to(device)

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
        # TODO
        model = GraphAttnModel(in_feats=feat_df.shape[1],
                               # 为什么要整除4？
                               hidden_dim=args['hid_dim']//4,
                               n_classes=2,
                               heads=[4]*args['n_layers'],  # [4,4,4]
                               activation=nn.PReLU(),
                               n_layers=args['n_layers'],
                               drop=args['dropout'],
                               device=device,
                               gated=args['gated'],
                               ref_df=feat_df,
                               cat_features=cat_feat).to(device)
        lr = args['lr'] * np.sqrt(args['batch_size']/1024)  # 0.00075
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
                batch_inputs, batch_work_inputs, batch_labels, lpa_labels = load_lpa_subtensor(num_feat, cat_feat, labels,
                                                                                               seeds, input_nodes, device)
                # (|input|, feat_dim); null; (|batch|,); (|input|,)
                blocks = [block.to(device) for block in blocks]
                train_batch_logits = model(
                    blocks, batch_inputs, lpa_labels, batch_work_inputs)
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

                    # if (len(np.unique(score)) == 1):
                    #     print("all same prediction!")
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
                    batch_inputs, batch_work_inputs, batch_labels, lpa_labels = load_lpa_subtensor(num_feat, cat_feat, labels,
                                                                                                   seeds, input_nodes, device)

                    blocks = [block.to(device) for block in blocks]
                    val_batch_logits = model(
                        blocks, batch_inputs, lpa_labels, batch_work_inputs)
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
                        torch.tensor(batch_labels.shape[0])
                    val_all_list = val_all_list + batch_labels.shape[0]
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
                batch_inputs, batch_work_inputs, batch_labels, lpa_labels = load_lpa_subtensor(num_feat, cat_feat, labels,
                                                                                               seeds, input_nodes, device)

                blocks = [block.to(device) for block in blocks]
                test_batch_logits = b_model(
                    blocks, batch_inputs, lpa_labels, batch_work_inputs)
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
    metrics_path = _save_metrics(
        args,
        graph,
        train_idx,
        holdout_val_idx,
        test_idx,
        y,
        roc_auc_score(y_target, test_score),
        f1_score(y_target, test_score1, average="macro"),
        average_precision_score(y_target, test_score),
        split_meta,
    )
    print(f"AML GTAN experiment semantics: {split_meta.get('semantic', 'transaction-sample graph experiment')}")
    return metrics_path


def load_gtan_data(args: dict):
    """
    Load graph, feature, and label given dataset name
    :param args: config dictionary
    :returns: feature, label, graph, category features
    """
    # prefix = './antifraud/data/'
    prefix = os.path.join(os.path.dirname(__file__), "..", "..", "data/")
    dataset = args["dataset"]
    test_size = args["test_size"]
    val_size = args.get("val_size", 0.0)
    split_meta = {"split_mode": "random", "semantic": "benchmark dataset graph experiment"}
    if dataset == "S-FFSD":
        cat_features = ["Target", "Location", "Type"]

        df = pd.read_csv(prefix + "S-FFSDneofull.csv")
        df = df.loc[:, ~df.columns.str.contains('Unnamed')]
        data = df[df["Labels"] <= 2]
        data = data.reset_index(drop=True)
        out = []
        alls = []
        allt = []
        pair = ["Source", "Target", "Location", "Type"]
        for column in pair:
            src, tgt = [], []
            edge_per_trans = 3
            for c_id, c_df in data.groupby(column):
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
        ###
        feat_data.to_csv(prefix + "S-FFSD_feat_data.csv", index=None)
        labels.to_csv(prefix + "S-FFSD_label_data.csv", index=None)
        ###
        index = list(range(len(labels)))
        g.ndata['label'] = torch.from_numpy(
            labels.to_numpy()).to(torch.long)
        g.ndata['feat'] = torch.from_numpy(
            feat_data.to_numpy()).to(torch.float32)
        graph_path = prefix+"graph-{}.bin".format(dataset)
        dgl.data.utils.save_graphs(graph_path, [g])

        train_idx, test_idx, y_train, y_test = train_test_split(index, labels, stratify=labels, test_size=test_size/2,
                                                                random_state=2, shuffle=True)
        val_idx = np.array([], dtype=np.int64)

    elif dataset == "yelp":
        cat_features = []
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
                src.append(i)  # src是出发点
                tgt.append(j)  # tgt是被指向点
        src = np.array(src)
        tgt = np.array(tgt)
        g = dgl.graph((src, tgt))
        g.ndata['label'] = torch.from_numpy(labels.to_numpy()).to(torch.long)
        g.ndata['feat'] = torch.from_numpy(
            feat_data.to_numpy()).to(torch.float32)
        graph_path = prefix + "graph-{}.bin".format(dataset)
        dgl.data.utils.save_graphs(graph_path, [g])
        val_idx = np.array([], dtype=np.int64)

    elif dataset == "amazon":
        cat_features = []
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
        val_idx = np.array([], dtype=np.int64)

    elif dataset == "aml":
        cat_features = ["Target", "Type"]
        data_path = args.get("data_path", "../AMLdataset.csv")
        raw_csv_path = _resolve_dataset_path(data_path)
        artifact_paths = preprocess_aml_for_gtan(raw_csv_path, output_dir=prefix)

        processed = pd.read_csv(artifact_paths["processed_path"])
        feat_data = pd.read_csv(artifact_paths["feat_path"])
        labels = pd.read_csv(artifact_paths["label_path"])["Labels"].astype(int)
        g = dgl.load_graphs(artifact_paths["graph_path"])[0][0]

        for col in cat_features:
            encoder = LabelEncoder()
            feat_data[col] = encoder.fit_transform(feat_data[col].astype(str))

        feat_data = feat_data.drop(columns=["Source"])
        if "AlertID" in feat_data.columns:
            feat_data["AlertID"] = feat_data["AlertID"].astype(int)

        train_idx, val_idx, test_idx, split_stats = _group_train_test_val_split(
            processed["Source"],
            processed["Labels"],
            val_size=val_size,
            test_size=test_size,
            seed=args["seed"],
        )
        split_meta.update(
            {
                "split_mode": args.get("split_mode", "sender_account"),
                "semantic": "transaction-sample graph experiment with sender-account split",
                **split_stats,
            }
        )
        print(
            "AML split summary:",
            {
                "train_size": len(train_idx),
                "val_size": len(val_idx),
                "test_size": len(test_idx),
                "train_group_count": split_meta["train_group_count"],
                "val_group_count": split_meta["val_group_count"],
                "test_group_count": split_meta["test_group_count"],
            },
        )
    else:
        raise NotImplementedError(f"Unsupported GTAN dataset: {dataset}")

    return feat_data, labels, train_idx, val_idx, test_idx, g, cat_features, split_meta
