import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.nn.functional as F

from os.path import join
from time import time
from GMF_torch import GMF
from MLP_torch import MLP
from Dataset import Dataset
from torch.utils.data import DataLoader
from Dataset import NCFDataset

import argparse

#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run NeuMF.")
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ml-1m',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--num_factors', type=int, default=8,
                        help='Embedding size of MF model.')
    parser.add_argument('--layers', nargs='?', default='[64,32,16,8]',
                        help="MLP layers. Note that the first layer is the concatenation of user and item embeddings. So layers[0]/2 is the embedding size.")
    parser.add_argument('--reg_mf', type=float, default=0,
                        help='Regularization for MF embeddings.')                    
    parser.add_argument('--reg_layers', nargs='?', default='[0,0,0,0]',
                        help="Regularization for each MLP layer. reg_layers[0] is the regularization for embeddings.")
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
    parser.add_argument('--mf_pretrain', nargs='?', default='',
                        help='Specify the pretrain model file for MF part. If empty, no pretrain will be used')
    parser.add_argument('--mlp_pretrain', nargs='?', default='',
                        help='Specify the pretrain model file for MLP part. If empty, no pretrain will be used')
    return parser.parse_args()


class NeuMF(nn.Module):
    def __init__(self, num_users, num_items, mf_dim=10, layers=[10], reg_layers=[0], reg_mf=0):
        super().__init__()
        assert len(layers) == len(reg_layers)
        self.layers = layers
        self.mf_dim = mf_dim
    
        self.MF_Embedding_User = nn.Embedding(
            num_embeddings = num_users,
            embedding_dim = mf_dim,
            norm_type=reg_mf
        )
        self.MF_Embedding_Item = nn.Embedding(
            num_embeddings = num_items,
            embedding_dim = mf_dim,
            norm_type=reg_mf
        )

        init.normal_(self.MF_Embedding_Item.weight)
        init.normal_(self.MF_Embedding_User.weight)

        self.MLP_Embedding_User = nn.Embedding(
            num_embeddings = num_users,
            embedding_dim = layers[0]//2,
            norm_type=reg_layers[0]
        )
        self.MLP_Embedding_Item = nn.Embedding(
            num_embeddings = num_items,
            embedding_dim = layers[0]//2,
            norm_type=reg_layers[0]
        )

        init.normal_(self.MLP_Embedding_Item.weight)
        init.normal_(self.MLP_Embedding_User.weight)

        for idx in range(1, len(layers)):
            setattr(self, 'layer%d' % idx, nn.Linear(layers[idx - 1], layers[idx]))

        self.pred_layer = nn.Linear(layers[-1]+mf_dim,1)

    def forward(self, user_input, item_input):
        mf_emb_user = self.MF_Embedding_User(user_input)
        mf_emb_item = self.MF_Embedding_Item(item_input)

        mlp_emb_user = self.MLP_Embedding_User(user_input)
        mlp_emb_item = self.MLP_Embedding_Item(item_input)

        mf_emb_user = nn.Flatten()(mf_emb_user)
        mf_emb_item = nn.Flatten()(mf_emb_item)
        mlp_emb_user = nn.Flatten()(mlp_emb_user)
        mlp_emb_item = nn.Flatten()(mlp_emb_item)

        mlp_vec = torch.cat([mlp_emb_user,mlp_emb_item], dim = 1)
        mf_vec = torch.mul(mf_emb_user,mf_emb_item)

        for idx in range(1, len(self.layers)):
            mlp_vec = torch.relu(getattr(self, 'layer%d' % idx)(mlp_vec))
        

        vector = torch.cat([mlp_vec,mf_vec], dim = 1)

        prediction = F.sigmoid(self.pred_layer(vector))

        return prediction.view(-1)

    def load_pretrained_model(self, gmf_model:GMF, mlp_model:MLP):
        self.MF_Embedding_item = gmf_model.MF_embedding_item
        self.MF_Embedding_user = gmf_model.MF_embedding_user

        self.MLP_Embedding_item = mlp_model.MLP_Embedding_User
        self.MLP_Embedding_user = mlp_model.MLP_Embedding_User

        for idx in range(1, len(self.layers)):
            setattr(self, 'layer%d' % idx, getattr(mlp_model, 'layer%d' % idx))

        #Prediction weights
        gmf_prediction = gmf_model.pred_layer
        mlp_predicion = mlp_model.prediction
        self.pred_layer.weight = torch.cat([gmf_prediction.weight, mlp_predicion.weight], dim = 0) #mybe mult by 0.5 since thats how they do it in the sample code but i am not sure how that makes sense yet
        self.pred_layer.bias = 0.5*(gmf_prediction.bias+mlp_predicion.bias)






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
    

    model = NeuMF(num_users,num_items)

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

