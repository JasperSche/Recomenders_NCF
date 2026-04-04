import torch
import torch.nn as nn
import torch.optim as optim

from time import time
from torch.utils.data import DataLoader
from Dataset import NCFDataset
from GMF_torch import GMF
from MLP_torch import MLP
from NeuMF_torch import NeuMF

import os
import csv
import json
import copy
import math
import random
import numpy as np


data_path = "Data/ratings.dat"
results_dir = "comparison_results"

groups_to_run = [
    # "mlp_architecture",
    "pretraining_ablation",
    # "learning_rate",
    "finetune_learning_rate",
    "latent_dimension",
    "negative_sampling_ratio",
]

# Training settings:
seed = 42
threshold = 4 # for positives
epochs = 50
batch_size = 256
patience = 5
num_val_neg = 100
k_eval = 10

# Base configuration
base_mlp_layers = [64, 32, 16, 8]
base_mlp_reg_layers = [0, 0, 0, 0]
base_pretrain_lr = 0.001
base_finetune_lr = 0.001
base_latent_dim = 8
base_num_neg = 4
base_pretraining = "both"

# Parameter sweeps:
mlp_architecture_values = [
    [32, 16, 8],
    [64, 32, 16, 8], # base
    [128, 64, 32, 16],
]
pretraining_values = [
    # "none",
    # "gmf",
    # "mlp",
    "both", # base
]
learning_rate_values = [
    0.0005,
    0.001, # base
    0.005,
]
finetune_learning_rate_values = [
    # 0.0005,
    # 0.0001,
    # 0.00005,
    0.001, # base
]
latent_dimension_values = [
    # 4,
    8, # base   
    # 16,
    # 24,
]
negative_sampling_values = [
    # 1,
    4, # base
    # 8,
    # 12,
]


def train(model, dataset, epochs, batch_size, optimizer, loss_func, device,
          stage_name, patience=5):
    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    wait = 0

    history_stage = []
    history_epoch = []
    history_train_loss = []
    history_val_loss = []

    for epoch in range(epochs):
        dataset.resample_negatives()

        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=8
        )

        model.train()
        total_loss = 0.0
        start = time()

        for user, item, label in train_loader:
            user = user.to(device, non_blocking = True)
            item = item.to(device, non_blocking = True)
            label = label.to(device, non_blocking = True)

            prediction = model(user, item)
            loss = loss_func(prediction, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * user.size(0)

        train_loss = total_loss / len(dataset)
        val_loss = compute_validation_loss(model, dataset, device, num_val_neg=num_val_neg)

        history_stage.append(stage_name)
        history_epoch.append(epoch + 1)
        history_train_loss.append(train_loss)
        history_val_loss.append(val_loss)

        print(f"{stage_name} epoch {epoch+1} [{time()-start:.3f}]:\ntrain loss = {train_loss:.6f}, val loss = {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping triggered.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, {
        "stage": history_stage,
        "epoch": history_epoch,
        "train_loss": history_train_loss,
        "val_loss": history_val_loss,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
    }


@torch.no_grad()
def compute_validation_loss(model, dataset, device, num_val_neg=100):
    model.eval()
    total_loss = 0.0
    total_count = 0
    bce = nn.BCELoss()

    for user in range(dataset.num_users):
        pos_items = list(dataset.val_ground_truth[user])
        if len(pos_items) == 0:
            continue

        all_pos = (
            dataset.train_matrix[user]
            | dataset.val_ground_truth[user]
            | dataset.test_ground_truth[user]
        )

        for pos_item in pos_items:
            users = [user]
            items = [pos_item]
            labels = [1.0]

            negs = set()
            while len(negs) < num_val_neg:
                neg = random.randint(0, dataset.num_items - 1)
                if neg not in all_pos:
                    negs.add(neg)

            for neg in negs:
                users.append(user)
                items.append(neg)
                labels.append(0.0)

            users = torch.tensor(users, device=device)
            items = torch.tensor(items, device=device)
            labels = torch.tensor(labels, device=device)

            preds = model(users, items)
            loss = bce(preds, labels)

            total_loss += loss.item()
            total_count += 1

    return total_loss / total_count

@torch.no_grad()
def evaluate_ranking(model, dataset, device, split="test", k=10):
    model.eval()

    if split == "test":
        ground_truth = dataset.test_ground_truth
    else:
        ground_truth = dataset.val_ground_truth

    recalls = []
    ndcgs = []

    all_items = torch.arange(dataset.num_items, device=device)

    for user in range(dataset.num_users):
        ground_truth_items = ground_truth[user]
        if len(ground_truth_items) == 0:
            continue

        train_items = dataset.train_matrix[user]

        # score all items
        user_tensor = torch.full((dataset.num_items,), user, device=device)
        scores = model(user_tensor, all_items).clone()

        # exclude train items
        if len(train_items) > 0:
            train_idx = torch.tensor(list(train_items), device=device)
            scores[train_idx] = -np.inf

        # top-k ranked items
        _, topk_idx = torch.topk(scores, k)
        topk_items = topk_idx.tolist()

        # recall
        hits = [1 if item in ground_truth_items else 0 for item in topk_items]
        num_hits = sum(hits)
        recall = num_hits / len(ground_truth_items)
        recalls.append(recall)

        # NDCG
        dcg = 0.0
        for rank, hit in enumerate(hits):
            if hit:
                dcg += 1.0 / math.log2(rank + 2)

        ideal_hits = min(len(ground_truth_items), k)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcgs.append(ndcg)

    return {
        f"Recall@{k}": sum(recalls) / len(recalls) if recalls else 0.0,
        f"NDCG@{k}": sum(ndcgs) / len(ndcgs) if ndcgs else 0.0,
        "num_eval_users": len(recalls)
    }

if __name__ == "__main__":
    # Build experiment list
    experiments = []

    if "mlp_architecture" in groups_to_run:
        for layers in mlp_architecture_values:
            experiments.append({
                "group": "mlp_architecture",
                "run_name": "mlp_" + "-".join(str(x) for x in layers),
                "mlp_layers": layers,
                "mlp_reg_layers": [0] * len(layers),
                "pretraining": base_pretraining,
                "pretrain_lr": base_pretrain_lr,
                "finetune_lr": base_finetune_lr,
                "latent_dim": base_latent_dim,
                "num_neg": base_num_neg,
            })

    if "pretraining_ablation" in groups_to_run:
        for pretraining in pretraining_values:
            experiments.append({
                "group": "pretraining_ablation",
                "run_name": "pretrain_" + pretraining,
                "mlp_layers": base_mlp_layers,
                "mlp_reg_layers": base_mlp_reg_layers,
                "pretraining": pretraining,
                "pretrain_lr": base_pretrain_lr,
                "finetune_lr": base_finetune_lr,
                "latent_dim": base_latent_dim,
                "num_neg": base_num_neg,
            })

    if "learning_rate" in groups_to_run:
        for lr in learning_rate_values:
            experiments.append({
                "group": "learning_rate",
                "run_name": "lr_" + str(lr).replace(".", "p"),
                "mlp_layers": base_mlp_layers,
                "mlp_reg_layers": base_mlp_reg_layers,
                "pretraining": base_pretraining,
                "pretrain_lr": lr,
                "finetune_lr": lr,
                "latent_dim": base_latent_dim,
                "num_neg": base_num_neg,
            })
    
    if "finetune_learning_rate" in groups_to_run:
        for finetune_lr in finetune_learning_rate_values:
            experiments.append({
                "group": "finetune_learning_rate",
                "run_name": "pretrainlr_" + str(base_pretrain_lr).replace(".", "p") +
                            "_finetunelr_" + str(finetune_lr).replace(".", "p"),
                "mlp_layers": base_mlp_layers,
                "mlp_reg_layers": base_mlp_reg_layers,
                "pretraining": "both",
                "pretrain_lr": base_pretrain_lr,
                "finetune_lr": finetune_lr,
                "latent_dim": base_latent_dim,
                "num_neg": base_num_neg,
            })

    if "latent_dimension" in groups_to_run:
        for latent_dim in latent_dimension_values:
            experiments.append({
                "group": "latent_dimension",
                "run_name": "latent_" + str(latent_dim),
                "mlp_layers": base_mlp_layers,
                "mlp_reg_layers": base_mlp_reg_layers,
                "pretraining": base_pretraining,
                "pretrain_lr": base_pretrain_lr,
                "finetune_lr": finetune_lr,
                "latent_dim": latent_dim,
                "num_neg": base_num_neg,
            })

    if "negative_sampling_ratio" in groups_to_run:
        for num_neg in negative_sampling_values:
            experiments.append({
                "group": "negative_sampling_ratio",
                "run_name": "numneg_" + str(num_neg),
                "mlp_layers": base_mlp_layers,
                "mlp_reg_layers": base_mlp_reg_layers,
                "pretraining": base_pretraining,
                "pretrain_lr": base_pretrain_lr,
                "finetune_lr": finetune_lr,
                "latent_dim": base_latent_dim,
                "num_neg": num_neg,
            })

    # Run experiments
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    os.makedirs(results_dir, exist_ok=True)

    summary_rows = []

    print(f"Running experiments on {device}")
    print(f"Number of runs: {len(experiments)}")

    for run_idx, config in enumerate(experiments, start=1):
        print("=" * 80)
        print(f"Run {run_idx}/{len(experiments)}")
        print(json.dumps(config, indent=2))

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        dataset = NCFDataset(
            path=data_path,
            num_neg=config["num_neg"],
            threshold=threshold,
            seed=seed
        )

        num_users = dataset.num_users
        num_items = dataset.num_items

        all_stage = []
        all_epoch = []
        all_train_loss = []
        all_val_loss = []

        gmf_best_val = None
        gmf_best_epoch = None
        mlp_best_val = None
        mlp_best_epoch = None
        neumf_best_val = None
        neumf_best_epoch = None

        model_gmf = None
        model_mlp = None

        if config["pretraining"] in ["gmf", "both"]:
            print("Pretraining GMF...")
            model_gmf = GMF(
                num_users=num_users,
                num_items=num_items,
                latent_dim=config["latent_dim"]
            ).to(device)

            model_gmf, gmf_history = train(
                model=model_gmf,
                dataset=dataset,
                epochs=epochs,
                batch_size=batch_size,
                optimizer=optim.Adam(model_gmf.parameters(), lr=config["pretrain_lr"]),
                loss_func=nn.BCELoss(),
                device=device,
                stage_name="gmf_pretrain",
                patience=patience,
            )

            all_stage.extend(gmf_history["stage"])
            all_epoch.extend(gmf_history["epoch"])
            all_train_loss.extend(gmf_history["train_loss"])
            all_val_loss.extend(gmf_history["val_loss"])

            gmf_best_val = gmf_history["best_val_loss"]
            gmf_best_epoch = gmf_history["best_epoch"]

        if config["pretraining"] in ["mlp", "both"]:
            print("Pretraining MLP...")
            model_mlp = MLP(
                num_users=num_users,
                num_items=num_items,
                layers=config["mlp_layers"],
                reg_layers=config["mlp_reg_layers"]
            ).to(device)

            model_mlp, mlp_history = train(
                model=model_mlp,
                dataset=dataset,
                epochs=epochs,
                batch_size=batch_size,
                optimizer=optim.Adam(model_mlp.parameters(), lr=config["pretrain_lr"]),
                loss_func=nn.BCELoss(),
                device=device,
                stage_name="mlp_pretrain",
                patience=patience,
            )

            all_stage.extend(mlp_history["stage"])
            all_epoch.extend(mlp_history["epoch"])
            all_train_loss.extend(mlp_history["train_loss"])
            all_val_loss.extend(mlp_history["val_loss"])

            mlp_best_val = mlp_history["best_val_loss"]
            mlp_best_epoch = mlp_history["best_epoch"]

        print("Training NeuMF...")
        model_neumf = NeuMF(
            num_users=num_users,
            num_items=num_items,
            mf_dim=config["latent_dim"],
            layers=config["mlp_layers"],
            reg_layers=config["mlp_reg_layers"]
        ).to(device)

        if config["pretraining"] == "both":
            model_neumf.load_pretrained_model(
                gmf_model=model_gmf,
                mlp_model=model_mlp
            )

        elif config["pretraining"] == "gmf":
            model_neumf.MF_Embedding_Item = model_gmf.MF_Embedding_Item
            model_neumf.MF_Embedding_User = model_gmf.MF_Embedding_User

        elif config["pretraining"] == "mlp":
            model_neumf.MLP_Embedding_Item = model_mlp.MLP_Embedding_Item
            model_neumf.MLP_Embedding_User = model_mlp.MLP_Embedding_User

            for idx in range(1, len(config["mlp_layers"])):
                setattr(model_neumf, f"layer{idx}", getattr(model_mlp, f"layer{idx}"))

        model_neumf, neumf_history = train(
            model=model_neumf,
            dataset=dataset,
            epochs=epochs,
            batch_size=batch_size,
            optimizer=optim.Adam(model_neumf.parameters(), lr=config["finetune_lr"]),
            loss_func=nn.BCELoss(),
            device=device,
            stage_name="neumf_train",
            patience=patience,
        )

        all_stage.extend(neumf_history["stage"])
        all_epoch.extend(neumf_history["epoch"])
        all_train_loss.extend(neumf_history["train_loss"])
        all_val_loss.extend(neumf_history["val_loss"])

        neumf_best_val = neumf_history["best_val_loss"]
        neumf_best_epoch = neumf_history["best_epoch"]

        test_results = evaluate_ranking(
            model_neumf,
            dataset,
            device,
            split="test",
            k=k_eval
        )

        run_dir = os.path.join(results_dir, config["group"], config["run_name"])
        os.makedirs(run_dir, exist_ok=True)

        np.savez(
            os.path.join(run_dir, "history_and_results.npz"),
            stage=np.array(all_stage),
            epoch=np.array(all_epoch),
            train_loss=np.array(all_train_loss, dtype=np.float32),
            val_loss=np.array(all_val_loss, dtype=np.float32),
            config_json=np.array(json.dumps(config)),
            gmf_best_val=np.array(-1.0 if gmf_best_val is None else gmf_best_val, dtype=np.float32),
            gmf_best_epoch=np.array(-1 if gmf_best_epoch is None else gmf_best_epoch, dtype=np.int32),
            mlp_best_val=np.array(-1.0 if mlp_best_val is None else mlp_best_val, dtype=np.float32),
            mlp_best_epoch=np.array(-1 if mlp_best_epoch is None else mlp_best_epoch, dtype=np.int32),
            neumf_best_val=np.array(neumf_best_val, dtype=np.float32),
            neumf_best_epoch=np.array(neumf_best_epoch, dtype=np.int32),
            test_recall=np.array(test_results[f"Recall@{k_eval}"], dtype=np.float32),
            test_ndcg=np.array(test_results[f"NDCG@{k_eval}"], dtype=np.float32),
            num_eval_users=np.array(test_results["num_eval_users"], dtype=np.int32),
        )

        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        with open(os.path.join(run_dir, "final_results.json"), "w") as f:
            json.dump({
                f"Recall@{k_eval}": test_results[f"Recall@{k_eval}"],
                f"NDCG@{k_eval}": test_results[f"NDCG@{k_eval}"],
                "num_eval_users": test_results["num_eval_users"],
                "gmf_best_val": gmf_best_val,
                "gmf_best_epoch": gmf_best_epoch,
                "mlp_best_val": mlp_best_val,
                "mlp_best_epoch": mlp_best_epoch,
                "neumf_best_val": neumf_best_val,
                "neumf_best_epoch": neumf_best_epoch,
            }, f, indent=2)

        summary_row = {
            "group": config["group"],
            "run_name": config["run_name"],
            "mlp_layers": str(config["mlp_layers"]),
            "pretraining": config["pretraining"],
            "pretrain_lr": config["pretrain_lr"],
            "finetune_lr": config["finetune_lr"],
            "latent_dim": config["latent_dim"],
            "num_neg": config["num_neg"],
            "gmf_best_val": gmf_best_val,
            "gmf_best_epoch": gmf_best_epoch,
            "mlp_best_val": mlp_best_val,
            "mlp_best_epoch": mlp_best_epoch,
            "neumf_best_val": neumf_best_val,
            "neumf_best_epoch": neumf_best_epoch,
            f"Recall@{k_eval}": test_results[f"Recall@{k_eval}"],
            f"NDCG@{k_eval}": test_results[f"NDCG@{k_eval}"],
            "num_eval_users": test_results["num_eval_users"],
            "npz_path": os.path.join(run_dir, "history_and_results.npz"),
        }

        summary_rows.append(summary_row)

        print("Final test results:")
        print(test_results)

    summary_csv_path = os.path.join(results_dir, "comparison_summary.csv")
    summary_json_path = os.path.join(results_dir, "comparison_summary.json")

    with open(summary_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    with open(summary_json_path, "w") as f:
        json.dump(summary_rows, f, indent=2)

    print("=" * 80)
    print("Done.")