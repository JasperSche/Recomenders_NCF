'''
Implementation of Generalized Matrix Factorization (GMF) recommender model, using pytorch
Based on: He Xiangnan et al. Neural Collaborative Filtering. In WWW 2017.
'''
from os.path import join
from Dataset import NCFDataset
from time import time
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import argparse

#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run GMF.")
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ml-1m',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--num_factors', type=int, default=8,
                        help='Embedding size.')
    parser.add_argument('--regs', nargs='?', default='[2,2]', #changed to [2,2] since it is the default in pytorch reffering to the l2 (euclidian) norm
                        help="Regularization for user and item embeddings.")
    parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--learner', nargs='?', default='adam',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X iterations')
    parser.add_argument('--out', type=int, default=1,
                        help='Whether to save the trained model.')
    return parser.parse_args()

def lecun_uniform(tensor):
    fan_in, _ = init._calculate_fan_in_and_fan_out(tensor)
    limit = (3 / fan_in) ** 0.5
    init.uniform_(tensor, -limit, limit)

class GMF(nn.Module):
    def __init__(self, num_users, num_items, latent_dim, regs=[2,2]):
        super().__init__()
        self.MF_Embedding_User =nn.Embedding(
            num_embeddings=num_users,
            embedding_dim=latent_dim,
            norm_type=regs[0]
        )
        self.MF_Embedding_Item = nn.Embedding(
            num_embeddings=num_items,
            embedding_dim=latent_dim,
            norm_type=regs[1]
        )

        init.normal_(self.MF_Embedding_User.weight)
        init.normal_(self.MF_Embedding_Item.weight)

        self.prediction = nn.Linear(latent_dim,1)
        lecun_uniform(self.pred_layer.weight)

    def forward(self, u_input, i_input):
        emb_user = self.MF_Embedding_Item(u_input)
        emb_item = self.MF_Embedding_Item(i_input)

        user_latent = nn.Flatten()(emb_user)
        item_latent = nn.Flatten()(emb_item)

        predict_vec = torch.mul(user_latent,item_latent)
        
        prediction = F.sigmoid(self.prediction(predict_vec))
        return prediction.view(-1)
    
#TODO: Test instance
if __name__ == "__main__":
    args = parse_args()

    data_path = join(args.path, "ratings.dat")

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

    #Load Model
    num_factors = args.num_factors
    regs = eval(args.regs)

    model = GMF(num_users,num_items,num_factors)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_function = nn.BCELoss()

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