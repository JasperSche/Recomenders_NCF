import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from time import time
from torch.utils.data import DataLoader
from Dataset import NCFDataset
from GMF_torch import GMF
from MLP_torch import MLP
from NeuMF_torch import NeuMF

import numpy as np
import random
import copy
import math

model_dir = 'Models/' # Directory to save trained models

num_neg = 4
epochs = 50
batch_size = 256
lr = 0.001
latent_dim = 10
mlp_layers = [64,32,16,8]
mlp_reg_layers = [0,0,0,0]

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

dataset = NCFDataset(
    path='Data/ratings.dat',
    num_neg=num_neg,
    threshold=4
)

num_users = dataset.num_users
num_items = dataset.num_items

def train(model:nn.Module, epochs:int, batch_size:int, optim, loss_func, device, patience=5):
    best_val_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(epochs):

        dataset.resample_negatives() # Don't forget to resample negatives every epoch for more variety!

        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=8
        )

        model.train()
        total_loss = 0
        start = time()

        for user, item, label in train_loader:
            user = user.to(device, non_blocking = True)
            item = item.to(device, non_blocking = True)
            label = label.to(device, non_blocking = True)

            prediction = model(user, item)
            loss = loss_func(prediction, label)

            optim.zero_grad()
            loss.backward()
            optim.step()

            total_loss += loss.item() * user.size(0)

        val_loss = compute_validation_loss(model, dataset, device)

        print(f"Epoch {epoch+1} [{time()-start:.3f}]:\ntrain loss = {total_loss/len(dataset):.6f}, val loss = {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping triggered.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


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
    model_gmf = GMF(
        num_users = num_users,
        num_items = num_items,
        latent_dim = latent_dim
    ).to(device)

    model_mlp = MLP(
        num_users = num_users, 
        num_items = num_items, 
        layers = mlp_layers,
        reg_layers = mlp_reg_layers
    ).to(device)

    model_NeuMF = NeuMF(
        num_users = num_users,
        num_items = num_items,
        mf_dim = latent_dim,
        layers = mlp_layers,
        reg_layers= mlp_reg_layers,
    ).to(device)

    print(f'Running experiment on {device} device:')

    # Pretrain GMF and MLP separately:

    # GMF
    print("Pretraining GMF...")
    train(
        model = model_gmf,
        epochs = epochs,
        batch_size = batch_size,
        optim = optim.Adam(model_gmf.parameters(), lr=lr),
        loss_func = nn.BCELoss(),
        device = device
    )
    torch.save(model_gmf.state_dict(), model_dir+'gmf.pt')

    gmf_results = evaluate_ranking(model_gmf, dataset, device, split="test", k=10)
    print("GMF:", gmf_results)

    # MLP
    print("Pretraining MLP...")
    train(
        model = model_mlp,
        epochs = epochs,
        batch_size = batch_size,
        optim = optim.Adam(model_mlp.parameters(), lr=lr),
        loss_func = nn.BCELoss(),
        device = device
    )
    torch.save(model_mlp.state_dict(), model_dir+'mlp.pt')

    mlp_results = evaluate_ranking(model_mlp, dataset, device, split="test", k=10)
    print("MLP:", mlp_results)

    # Load pretrained GMF and MLP into NeuMF:
    model_NeuMF.load_pretrained_model(
        gmf_model=model_gmf,
        mlp_model=model_mlp,
    )

    # Train NeuMF
    print("Training NeuMF...")
    train(
        model = model_NeuMF,
        epochs = epochs,
        batch_size = batch_size,
        optim = optim.Adam(model_NeuMF.parameters(), lr=lr),
        loss_func = nn.BCELoss(),
        device = device,
        patience = 5
    )

    neumf_results = evaluate_ranking(model_NeuMF, dataset, device, split="test", k=10)
    print("NeuMF:", neumf_results)
