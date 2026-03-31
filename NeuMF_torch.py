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
    parser.add_argument('--out', type=int, default=1,
                        help='Whether to save the trained model.')
    parser.add_argument('--out_path', nargs='?', default='Models/',
                        help='Input output path for the trained model.')
    parser.add_argument('--mf_pretrain', nargs='?', default='',
                        help='Specify the pretrain model file for MF part. If empty, no pretrain will be used')
    parser.add_argument('--mlp_pretrain', nargs='?', default='',
                        help='Specify the pretrain model file for MLP part. If empty, no pretrain will be used')
    return parser.parse_args()


class NeuMF(nn.Module):
    def __init__(self, num_users, num_items, mf_dim=10, layers=[10], reg_layers=[0], reg_mf=0):
        super(NeuMF,self).__init__()
        assert len(layers) == len(reg_layers)
        self.layers = layers
    
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

        self.prediction = nn.Linear(layers[-1]+mf_dim,1)

    def forward(self, user_input, item_input):
        mf_emb_user = self.MF_Embedding_User(user_input)
        mf_emb_item = self.MF_Embedding_Item(item_input)

        mlp_emb_user = self.MLP_Embedding_User(user_input)
        mlp_emb_item = self.MLP_Embedding_Item(item_input)

        mf_emb_user = nn.Flatten()(mf_emb_user)
        mf_emb_item = nn.Flatten()(mf_emb_item)
        mlp_emb_user = nn.Flatten()(mlp_emb_user)
        mlp_emb_item = nn.Flatten()(mlp_emb_item)

        mf_vec = torch.mul(mf_emb_user,mf_emb_item)
        mlp_vec = torch.cat([mlp_emb_user,mlp_emb_item], dim = 1)

        for idx in range(1, len(self.layers)):
            mlp_vec = F.relu(getattr(self, 'layer%d' % idx)(mlp_vec))
        
        vector = torch.cat([mlp_vec,mf_vec], dim = 1)

        prediction = F.sigmoid(self.prediction(vector))

        return prediction.view(-1)

    def load_pretrained_model(self, gmf_model:GMF, mlp_model:MLP):
        self.MF_Embedding_Item = gmf_model.MF_Embedding_Item
        self.MF_Embedding_User = gmf_model.MF_Embedding_User

        self.MLP_Embedding_Item = mlp_model.MLP_Embedding_Item
        self.MLP_Embedding_User = mlp_model.MLP_Embedding_User

        for idx in range(1, len(self.layers)):
            setattr(self, 'layer%d' % idx, getattr(mlp_model, 'layer%d' % idx))

        #Prediction weights
        gmf_prediction = gmf_model.prediction
        mlp_prediction = mlp_model.prediction
        self.prediction.weight = nn.Parameter(torch.cat([gmf_prediction.weight, mlp_prediction.weight], dim = 1)) #mybe mult by 0.5 since thats how they do it in the sample code but i am not sure how that makes sense yet
        self.prediction.bias = nn.Parameter(0.5*(gmf_prediction.bias+mlp_prediction.bias))






if __name__ == "__main__":
    args = parse_args()

    data_path = join(args.path, "ratings.dat")
    out_path = join(args.out_path, "default_NeuMF.pt")
    epochs = args.epochs
    batch_size = args.batch_size
    num_factors = args.num_factors
    layers = eval(args.layers)
    reg_mf  = args.reg_mf
    reg_layers = eval(args.reg_layers)
    num_neg = args.num_neg
    lr = args.lr
    learner = args.learner
    out = args.out
    mf_pretrain  = args.mf_pretrain
    mlp_pretrain = args.mlp_pretrain


    dataset = NCFDataset(
        path=data_path,
        num_neg=num_neg,
        threshold=4
    )

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    num_users = dataset.num_users
    num_items = dataset.num_items

    model = NeuMF(
        num_items = num_items,
        num_users = num_users,
        layers = layers,
        reg_layers = reg_layers,
        reg_mf = reg_mf 
    )

    if type(mf_pretrain) == GMF and type(mlp_pretrain) == MLP:
        model.load_pretrained_model(mf_pretrain,mlp_pretrain)

    if learner == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError(f'{learner} is not implemented')
    
    loss_function = nn.BCELoss()

    for epoch in range(epochs):

        dataset.resample_negatives() # Don't forget to resample negatives every epoch for more variety!

        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
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
            % (epoch+1, time() - start, total_loss)
        )
    
    if out == 1: torch.save(model.state_dict(),out_path)

