import argparse
import os
from time import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F

from torch.utils.data import DataLoader
from Dataset import NCFDataset


#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run MLP.")
    parser.add_argument('--path', nargs='?', default='Data', help='Input data path.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size.')
    parser.add_argument('--layers', nargs='?', default='[64,32,16,8]',
                        help='Size of each layer. Note that the first layer is the concatenation of user and item embeddings. So layers[0]/2 is the embedding size.')
    parser.add_argument('--reg_layers', nargs='?', default='[0,0,0,0]',
                        help='Regularization for each layer.')
    parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--out', type=int, default=1,
                        help='Whether to save the trained model.')
    return parser.parse_args()

class MLP(nn.Module):
    def __init__(self, num_users, num_items, layers, reg_layers):
        super(MLP, self).__init__()

        self.layers = layers
        self.reg_layers = reg_layers
        self.num_layer = len(layers)

        self.MLP_Embedding_User = nn.Embedding(num_users, layers[0] // 2)
        self.MLP_Embedding_Item = nn.Embedding(num_items, layers[0] // 2)

        init.normal_(self.MLP_Embedding_User.weight, std=0.01)
        init.normal_(self.MLP_Embedding_Item.weight, std=0.01)

        for idx in range(1, self.num_layer):
            setattr(self, 'layer%d' % idx, nn.Linear(layers[idx - 1], layers[idx]))
            init.xavier_uniform_(getattr(self, 'layer%d' % idx).weight)
            init.zeros_(getattr(self, 'layer%d' % idx).bias)

        self.prediction = nn.Linear(layers[-1], 1)
        init.kaiming_uniform_(self.prediction.weight, a=1, nonlinearity='sigmoid')
        init.zeros_(self.prediction.bias)

    def forward(self, user_input, item_input):
        user_latent = self.MLP_Embedding_User(user_input)
        item_latent = self.MLP_Embedding_Item(item_input)

        vector = torch.cat([user_latent, item_latent], dim=-1)

        for idx in range(1, self.num_layer):
            vector = torch.relu(getattr(self, 'layer%d' % idx)(vector))

        prediction = F.sigmoid(self.prediction(vector))
        return prediction.view(-1) 

if __name__ == '__main__':
    args = parse_args()

    # -----------------------------
    # Load dataset
    # -----------------------------
    data_path = os.path.join(args.path, "ratings.dat")

    dataset = NCFDataset(
        path=data_path,
        num_neg=args.num_neg,
        threshold=4
    )

    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True
    )

    num_users = dataset.num_users
    num_items = dataset.num_items

    # -----------------------------
    # Build model
    # -----------------------------
    layers = eval(args.layers)
    reg_layers = eval(args.reg_layers)

    model = MLP(num_users, num_items, layers, reg_layers)

    # -----------------------------
    # Optimizer
    # -----------------------------

    # Assignment requires ADAM optimizer with BCE loss.
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_function = nn.BCELoss()

    # -----------------------------
    # Training loop
    # -----------------------------
    for epoch in range(args.epochs):

        dataset.resample_negatives() # Don't forget to resample negatives every epoch for more variety!

        train_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True
        )

        model.train()
        total_loss = 0
        start = time()

        for user, item, label in train_loader:

            prediction = model(user, item)

            loss = loss_function(prediction, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(
            "Epoch %d [%.1fs]: loss = %.4f"
            % (epoch, time() - start, total_loss)
        )
    
    # TODO: Evaluation and/or model saving