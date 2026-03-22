'''
Implementation of Generalized Matrix Factorization (GMF) recommender model, using pytorch
Based on: He Xiangnan et al. Neural Collaborative Filtering. In WWW 2017.
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GMF(nn.Module):
    def __init__(self, num_users, num_items, latent_dim, regs=[2,2]):
        super().__init__()
        self.MF_embedding_user = nn.Embedding(
            num_embeddings=num_users,
            embedding_dim=latent_dim,
            norm_type=regs[0]
        )
        self.MF_embedding_item = nn.Embedding(
            num_embeddings=num_items,
            embedding_dim=latent_dim,
            norm_type=regs[1]
        )
        self.pred_layer = nn.Linear()

    def forward(self, u_input, i_input):
        emb_user = self.MF_embedding_user(u_input)
        emb_item = self.MF_embedding_item(i_input)

        user_latent = nn.Flatten(emb_user)
        item_latent = nn.Flatten(emb_item)

        predict_vec = torch.cat(user_latent,item_latent)
        prediction = self.pred_layer(predict_vec)

        output = F.softmax(prediction)
        return output

#TODO: Test instance