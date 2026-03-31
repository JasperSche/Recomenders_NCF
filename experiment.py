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
train(
    model = model_gmf,
    epochs = epochs,
    batch_size = batch_size,
    optim = optim.Adam(model_gmf.parameters(), lr=lr),
    loss_func = nn.BCELoss(),
    device = device
)

# safe
torch.save(model_gmf.state_dict(), 'Models/gmf.pt')

train(
    model = model_mlp,
    epochs = epochs,
    batch_size = batch_size,
    optim = optim.Adam(model_mlp.parameters(), lr=lr),
    loss_func = nn.BCELoss(),
    device = device
)

torch.save(model_mlp.state_dict(), 'Models/mlp.pt')


model_NeuMF.load_pretrained_model(
    gmf_model=model_gmf,
    mlp_model=model_mlp,
)

# post_train
 
train(
    model = model_NeuMF,
    epochs = epochs,
    batch_size = batch_size,
    optim = optim.Adam(model_NeuMF.parameters(), lr=lr),
    loss_func = nn.BCELoss(),
    device = device
)

torch.save(model_NeuMF.state_dict(), 'Models/neu_mf.pt')
