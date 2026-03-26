import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class NeuMF(nn.Module):
    def __init__(self, num_users, num_items, mf_dim=10, layers=[10], reg_layers=[0], reg_mf=0):
        super().__init__()
        assert len(layers) == len(reg_layers)
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
            embedding_dim = mf_dim,
            norm_type=reg_layers[0]
        )
        self.MLP_Embedding_Item = nn.Embedding(
            num_embeddings = num_items,
            embedding_dim = mf_dim,
            norm_type=reg_layers[0]
        )

        init.normal_(self.MLP_Embedding_Item.weight)
        init.normal_(self.MLP_Embedding_User.weight)

        for idx in range(1, len(layers)):
            setattr(self, 'layer%d' % idx, nn.Linear(layers[idx - 1], layers[idx]))

        self.pred_layer = nn.Linear(layers[::-1],1)

    def forward(self, user_input, item_input):
        mf_emb_user = self.MF_Embedding_User(user_input)
        mf_emb_item = self.MF_Embedding_Item(item_input)


        mlp_emb_user = self.MF_Embedding_User(user_input)
        mlp_emb_item = self.MF_Embedding_Item(item_input)

        mlp_vec = torch.cat([mlp_emb_user,mlp_emb_item], dim = 1)
        mf_vec = torch.cat([mf_emb_user,mf_emb_item], dim = 1)

        for idx in range(1, (self.layers)):
            mlp_vec = torch.relu(getattr(self, 'layer%d' % idx)(mlp_vec))
        
        vector = torch.cat([mlp_vec,mf_vec], dim = 1)

        prediction = nn.Softmax(self.pred_layer(vector))

        return prediction.view(-1)


if __name__ == "__main__":
    pass
    #I guess this is pretty much the same as the other files

