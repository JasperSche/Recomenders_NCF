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

def train(model:nn.Module, epochs:int, batch_size:int, optim, loss_func, device):
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

        print(f"Epoch {epoch+1} [{time()-start:.3f}]: loss = {total_loss/len(dataset):.6f}")

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
