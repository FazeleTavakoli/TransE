import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from torch.nn.init import xavier_normal,xavier_uniform
from torch.autograd import Variable
from utils import get_minibatches, sample_negatives, accuracy, auc
from time import time

import os
import matplotlib.pyplot as plt
import random
########################
class TransE(nn.Module):
    """
    TransE embedding model
    ----------------------
    Bordes, Antoine, et al.
    "Translating embeddings for modeling multi-relational data." NIPS. 2013.
    """

    def __init__(self, n_e, n_r, k, margin, distance='l2', gpu=False):
        """
        TransE embedding model
        ----------------------

        Params:
        -------
            n_e: int
                Number of entities in dataset.

            n_r: int
                Number of relationships in dataset.

            k: int
                Embedding size.

            margin: float
                Margin size for TransE's hinge loss.

            distance: {'l1', 'l2'}
                Distance measure to be used in the loss.

            gpu: bool, default: False
                Whether to use GPU or not.
        """
        super(TransE, self).__init__()
        # Parameters
        self.n_e = n_e
        self.n_r = n_r
        self.k = k
        self.gpu = gpu
        self.distance = distance
        self.gamma = margin
        # Embedding Layer
        self.emb_E = nn.Embedding(self.n_e, self.k)
        self.emb_R = nn.Embedding(self.n_r, self.k)
        # Initialization
        r = 6 / np.sqrt(self.k)
        self.emb_E.weight.data.uniform_(-r, r)
        self.emb_R.weight.data.uniform_(-r, r)
        # Copy all params to GPU if specified
        if self.gpu:
            self.cuda()
        # self.train_tiple = 1
        # self.read_data()

    #     test_triple & train_triple



    def forward(self, X):
        X = Variable(torch.from_numpy(X)).long()
        X = X.cuda() if self.gpu else X
        # Decompose X into head, relationship, tail
        hs, ls, ts = X[:, 0], X[:, 1], X[:, 2]
        # print(hs.shape)
        e_hs = self.emb_E(hs)
        e_ts = self.emb_E(ts)
        e_ls = self.emb_R(ls)
        f = self.energy(e_hs, e_ls, e_ts).view(-1, 1)
        return f

    def energy(self, h, l, t):
        if self.distance == 'l1':
            out = torch.sum(torch.abs(h + l - t), 1)
        else:
            out = torch.sqrt(torch.sum((h + l - t) ** 2, 1))
        return out

    def ranking_loss(self, y_pos, y_neg, C=1, average=True):
        """
        Compute loss max margin ranking loss.

        Params:
        -------
        y_pos: vector of size Mx1
            Contains scores for positive samples.

        y_neg: np.array of size Mx1 (binary)
            Contains the true labels.

        margin: float, default: 1
            Margin used for the loss.

        C: int, default: 1
            Number of negative samples per positive sample.

        average: bool, default: True
            Whether to average the loss or just summing it.

        Returns:
        --------
        loss: float
        """
        M = y_pos.size(0)

        y_pos = y_pos.view(-1).repeat(C)  # repeat to match y_neg
        y_neg = y_neg.view(-1)
        target_ = torch.from_numpy(-np.ones(M * C, dtype=np.float32))
        if self.gpu:
            target = Variable(target_.cuda())
        else:
            target = Variable(target_)
        loss = nn.MarginRankingLoss(margin=self.gamma)
        loss = loss(y_pos, y_neg, target)
        # print(loss)
        return loss

    def normalize_embeddings(self):
        self.emb_E.weight.data.renorm_(p=2, dim=0, maxnorm=1)

    def predict(self, X, sigmoid=False):

        y_pred = self.forward(X).view(-1, 1)

        if sigmoid:
            y_pred = torch.sigmoid(y_pred)

        if self.gpu:
            return y_pred.cpu().data.numpy()
        else:
            return y_pred.data.numpy()



    def hit_at_ten(self,X, n_e):

        X = Variable(torch.from_numpy(X)).long()
        X = X.cuda() if self.gpu else X
        # Decompose X into head, relationship, tail
        hs, ls, ts = X[:, 0], X[:, 1], X[:, 2]
        e_hs = self.emb_E(hs)
        e_ts = self.emb_E(ts)
        e_ls = self.emb_R(ls)
        f1 = self.energy(e_hs, e_ls, e_ts).view(-1, 1)


        # # e_ts_corrupted = self.emb_E(torch.from_numpy(np.arange(n_e)))
        # uniq_tail = torch.from_numpy(np.unique(ts))
        # if self.gpu:
        #     uniq_tail = uniq_tail.cuda()
        # e_ts_corrupted = self.emb_E(uniq_tail)
        # # e_ts_corrupted = nn.Embedding(self.n_e, self.k)
        # e_ts_cor_size = np.shape(e_ts_corrupted)
        # counter = 0
        # counter1 = 0
        # for item in e_hs:
        #     # print(item.dtype, item.device)
        #     single_head = item
        #     # single_head_repeated = single_head.repeat(e_ts_cor_size[0]).view(104,50)
        #     single_head_repeated = single_head.repeat(e_ts_cor_size[0]).view(14558, 50)
        #     single_relation = e_ls[counter]
        #     # single_relation_repeated = single_relation.repeat(e_ts_cor_size[0]).view(104,50)
        #     single_relation_repeated = single_relation.repeat(e_ts_cor_size[0]).view(14558, 50)
        #     f = self.energy(single_head_repeated, single_relation_repeated, e_ts_corrupted)
        #     sorted_f, indices = torch.sort(f)
        #     f1[counter]
        #     # top_ten = torch.topk(f,10)
        #     np.shape(sorted_f)
        #     if f1[counter] in sorted_f[:10]:
        #         counter1 = counter1 + 1
        #     counter = counter + 1


            #creating clusters
        X_test = np.load('/data/fazele/workplace/data/wordnet/bin/test.npy')
        rels = np.unique(X_test[:, 1])
        rel_1 = X_test[X_test[:, 1] == 1]
        rel_groups = {}
        for rel in rels:
            rel_groups[rel] = X_test[X_test[:, 1] == rel]
            rel_groups.keys()

        counter =0
        counter1 = 0
        for h in hs:
            single_head = h
            single_relation = ls[counter]
            relation_triple = rel_groups[single_relation.item()]

            random_number = torch.LongTensor(20).random_(0, np.shape(relation_triple)[0]-1)
            random_relation_triple = relation_triple[random_number]
            random_tails = random_relation_triple[:,2]

            single_head_repeated = single_head.repeat(20)
            single_relation_repeated = single_relation.repeat(20)

            random_tails = Variable(torch.from_numpy(random_tails)).long()
            if self.gpu:
                random_tails = random_tails.cuda()
            # print(random_tails.dtype, random_tails.device)
            e_hs_corrupted = self.emb_E(single_head_repeated)
            e_ts_corrupted = self.emb_E(random_tails)
            e_ls_corrupted = self.emb_R(single_relation_repeated)
            f = self.energy(e_hs_corrupted, e_ls_corrupted, e_ts_corrupted).view(-1, 1)

            sorted_f, indices = torch.sort(f, descending=True)
            f1[counter]
            # top_ten = torch.topk(f,10)
            np.shape(sorted_f)
            if f1[counter] in sorted_f[:10]:
                counter1 = counter1 + 1
            counter = counter + 1


        hit_at_ten =  counter1
        return hit_at_ten





##################
# print(torch.cuda.is_available())
# Set random seed
randseed = 9999
np.random.seed(randseed)
torch.manual_seed(randseed)
####################

# Data Loading
# Load dictionary lookups
idx2ent = np.load('/data/fazele/workplace/data/wordnet/bin/idx2ent.npy')
idx2rel = np.load('/data/fazele/workplace/data/wordnet/bin/idx2rel.npy')

n_e = len(idx2ent)
n_r = len(idx2rel)

# Load dataset
X_train = np.load('/data/fazele/workplace/data/wordnet/bin/train.npy')
X_val = np.load('/data/fazele/workplace/data/wordnet/bin/val.npy')
y_val = np.load('/data/fazele/workplace/data/wordnet/bin/y_val.npy')
print(len(np.unique(X_train[:, 0])), len(np.unique(X_train[:, 2])))
X_val_pos = X_val[y_val.ravel() == 1, :]  # Take only positive samples

M_train = X_train.shape[0]
M_val = X_val.shape[0]

# Model Parameters
k = 50
distance = 'l2'
margin = 1.0
model = TransE(n_e=n_e, n_r=n_r, k=k, margin=margin, distance=distance, gpu= True)

# ########
# normalize_embed = True
# C = 5 # Negative Samples
# n_epoch = 20
# lr = 0.1
# lr_decay_every = 20
# #weight_decay = 1e-4
# mb_size = 100
# print_every = 100
# average = False
# loss_list = []
#
# # Optimizer Initialization
# #solver = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
# solver = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
# # Begin training
# for epoch in range(n_epoch):
#     print('Epoch-{}'.format(epoch+1))
#     print('----------------')
#     it = 0
#     # Shuffle and chunk data into minibatches
#     mb_iter = get_minibatches(X_train, mb_size, shuffle=True)
#
#     # Anneal learning rate
#     lr = lr * (0.5 ** (epoch // lr_decay_every))
#     for param_group in solver.param_groups:
#         param_group['lr'] = lr
#
#     #average loss
#     total_loss = 0
#     average_loss = 0
#
#     for X_mb in mb_iter:
#         start = time()
#
#         # Build batch with negative sampling
#         m = X_mb.shape[0]
#         # C x M negative samples
#         X_neg_mb = np.vstack([sample_negatives(X_mb, n_e) for _ in range(C)])
#         X_train_mb = np.vstack([X_mb, X_neg_mb])
#
#         y_true_mb = np.vstack([np.zeros([m, 1]), np.ones([C*m, 1])])
#
#         # Training step
#
#         y = model.forward(X_train_mb)
#         y_pos, y_neg = y[:m], y[m:]
#         loss = model.ranking_loss(y_pos, y_neg, C=C, average=average)
#         loss.backward()
#         solver.step()
#         solver.zero_grad()
#
#         end = time()
#         if normalize_embed:
#             model.normalize_embeddings()
#
#         end = time()
#
#         # Ploting loss
#         total_loss = total_loss + loss
#         # Training logs
#         if it % print_every == 0:
#             # Training auc
#             pred = model.predict(X_train_mb, sigmoid=True)
#             train_acc = auc(pred, y_true_mb)
#
#             # Validation auc
#             y_pred_val = model.forward(X_val)
#             y_prob_val = torch.sigmoid(y_pred_val)
#             y_prob_val = 1 - y_prob_val
#             val_acc = auc(y_prob_val.cpu().data.numpy(), y_val)
#
#             print('Iter-{}; loss: {:.4f}; train_auc: {:.4f}; val_auc: {:.4f}; time per batch: {:.2f}s'
#                     .format(it, loss.data.item(), train_acc, val_acc, end-start))
#
#
#         it += 1
#
#     average_loss = total_loss / it
#     loss_list.append(average_loss)
#


# plt.plot(range(n_epoch), loss_list, 'ro')
# # plt.axis([0, 6, 0, 20])
# # plt.plot([1,2,3,4])
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.show()
#
# os.makedirs("saved_models", exist_ok=True)
# torch.save(model.state_dict(), "saved_models/transE.pth")


model.load_state_dict(torch.load("saved_models/transE.pth"))
# n_e=38692

# model = TransE(n_e=40000, n_r=12, k=k, margin=margin, distance=distance, gpu= False)
# model = TransE(n_e=n_e, n_r=n_r, k=k, margin=margin, distance=distance, gpu= False)
X_test = np.load('/data/fazele/workplace/data/wordnet/bin/test.npy')
print(len(np.unique(np.unique(X_test[:, 2]))) + len(np.unique(np.unique(X_test[:, 0]))))
print(len(np.unique(X_test[:, 2])) + len(np.unique(X_test[:, 0])))
print(X_test.min(), X_test.max())
print(len(np.unique(X_test[:,1]) ))
mb_iter = get_minibatches(X_test, 400, shuffle=False)
# for X_mb in mb_iter:
mb_test = next(iter(mb_iter))
# y = model.forward(mb_test)
# print(X_mb.shape)
# y = model.forward(mb_test)
y = model.forward(X_test)
hit_rate = model.hit_at_ten(X_test, 40000)
print("Hit@10 is:", hit_rate)



# model_t = TransE(n_e=n_e, n_r=n_r, k=k, margin=margin, distance=distance, gpu= False)

# y = model(X_test[:600, :])
# pred = model.predict(X_test, sigmoid=True)
# train_acc = auc(pred, y_true_mb)
