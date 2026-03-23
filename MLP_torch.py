import argparse
import os
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


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
    parser.add_argument('--learner', nargs='?', default='adam',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
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

        for idx in range(1, self.num_layer):
            setattr(self, 'layer%d' % idx, nn.Linear(layers[idx - 1], layers[idx]))

        self.prediction = nn.Linear(layers[-1], 1)

    def forward(self, user_input, item_input):
        user_latent = self.MLP_Embedding_User(user_input)
        item_latent = self.MLP_Embedding_Item(item_input)

        vector = torch.cat([user_latent, item_latent], dim=-1)

        for idx in range(1, self.num_layer):
            vector = torch.relu(getattr(self, 'layer%d' % idx)(vector))

        prediction = torch.sigmoid(self.prediction(vector))
        return prediction.view(-1)


def get_model(num_users, num_items, layers=[20, 10], reg_layers=[0, 0]):
    model = MLP(num_users, num_items, layers, reg_layers)

    nn.init.normal_(model.MLP_Embedding_User.weight, std=0.01)
    nn.init.normal_(model.MLP_Embedding_Item.weight, std=0.01)

    for idx in range(1, len(layers)):
        layer = getattr(model, 'layer%d' % idx)
        nn.init.xavier_uniform_(layer.weight)
        nn.init.zeros_(layer.bias)

    nn.init.kaiming_uniform_(model.prediction.weight, a=1, nonlinearity='sigmoid')
    nn.init.zeros_(model.prediction.bias)

    return model


def get_train_instances(train, all_user_items, num_items, num_negatives):
    user_input, item_input, labels = [], [], []

    for (u, i) in train:
        user_input.append(u)
        item_input.append(i)
        labels.append(1)

        for _ in range(num_negatives):
            j = np.random.randint(num_items)
            while j in all_user_items[u]:
                j = np.random.randint(num_items)

            user_input.append(u)
            item_input.append(j)
            labels.append(0)

    return user_input, item_input, labels


if __name__ == '__main__':
   pass