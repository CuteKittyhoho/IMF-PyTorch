import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.layer import *
import matplotlib.pyplot as plt
import math
import logging
import random
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
from pykalman import KalmanFilter
import pywt
from torch.autograd import Variable
# from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from PIL import Image
import matplotlib.pyplot as plt
from models.SpecturalFusion import *
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BaseModel(nn.Module):
    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.device = args.device

    @staticmethod
    def format_metrics(metrics, split):
        return " ".join(
            ["{}_{}: {:.4f}".format(split, metric_name, metric_val) for metric_name, metric_val in metrics.items()])

    @staticmethod
    def has_improved(m1, m2):
        return (m1["Mean Rank"] > m2["Mean Rank"]) or (m1["Mean Reciprocal Rank"] < m2["Mean Reciprocal Rank"])

    @staticmethod
    def init_metric_dict():
        return {"Hits@100": -1, "Hits@10": -1, "Hits@3": -1, "Hits@1": -1,
                "Mean Rank": 100000, "Mean Reciprocal Rank": -1}


class GAT(nn.Module):
    def __init__(self, args):
        '''Sparse version of KBGAT
        entity_in_dim -> Entity Input Embedding dimensions
        entity_out_dim  -> Entity Output Embedding dimensions, passed as a list
        num_relation -> number of unique relations
        relation_dim -> Relation Embedding dimensions
        num_nodes -> number of nodes in the Graph
        nheads_GAT -> Used for Multihead attention, passed as a list '''

        super().__init__()
        self.device = args.device
        self.num_nodes = len(args.entity2id)
        self.entity_in_dim = args.gat_dim
        self.entity_out_dim = args.gat_dim
        self.num_relation = len(args.relation2id)
        self.relation_in_dim = args.gat_dim
        self.relation_out_dim = args.gat_dim
        self.nheads_GAT = args.n_heads
        self.neg_num = args.neg_num_gat

        self.drop_GAT = args.dropout_gat
        self.alpha = args.alpha_gat # For leaky relu

        # Initial Embedding
        self.entity_embeddings = nn.Parameter(torch.randn(self.num_nodes, self.entity_in_dim))
        self.relation_embeddings = nn.Parameter(torch.randn(self.num_relation, self.relation_in_dim))
        #if args.pre_trained_encoder:
        if args.pre_trained_encoder:
            self.entity_embeddings = nn.Parameter(torch.from_numpy(pickle.load(open('datasets/' + args.dataset + '/entity2vec.pkl', 'rb'))).float())
            self.relation_embeddings = nn.Parameter(torch.from_numpy(pickle.load(open('datasets/' + args.dataset + '/relation2vec.pkl', 'rb'))).float())
        # Final output Embedding
        self.final_entity_embeddings = nn.Parameter(
            torch.randn(self.num_nodes, self.entity_out_dim * self.nheads_GAT))
        self.final_relation_embeddings = nn.Parameter(
            torch.randn(self.num_relation, self.relation_out_dim * self.nheads_GAT))

        self.spgat = SpGAT(self.num_nodes, self.entity_in_dim, self.entity_out_dim,
                                  self.drop_GAT, self.alpha, self.nheads_GAT)

        self.W_entities = nn.Parameter(torch.zeros(size=(self.entity_in_dim, self.entity_out_dim * self.nheads_GAT)))
        nn.init.xavier_uniform_(self.W_entities.data, gain=1.414)

    def forward(self, adj, train_indices):
        edge_list = adj[0]
        if(CUDA):
            edge_list = edge_list.to(self.device)

        self.entity_embeddings.data = F.normalize(
            self.entity_embeddings.data, p=2, dim=1).detach()

        self.relation_embeddings.data = F.normalize(
            self.relation_embeddings.data, p=2, dim=1).detach()

        mask_indices = torch.unique(train_indices[:, 2]).to(self.device)
        mask = torch.zeros(self.entity_embeddings.shape[0]).to(self.device)
        mask[mask_indices] = 1.0

        out_entity, out_relation = self.spgat(self.entity_embeddings, self.relation_embeddings, edge_list)
        out_entity = F.normalize(self.entity_embeddings.mm(self.W_entities)
                                 + mask.unsqueeze(-1).expand_as(out_entity) * out_entity, p=2, dim=1)

        self.final_entity_embeddings.data = out_entity.data
        self.final_relation_embeddings.data = out_relation.data

        return out_entity, out_relation

    def loss_func(self, train_indices, entity_embeddings, relation_embeddings):
        len_pos_triples = int(train_indices.shape[0] / (int(self.neg_num) + 1))
        pos_triples = train_indices[:len_pos_triples]
        neg_triples = train_indices[len_pos_triples:]
        pos_triples = pos_triples.repeat(int(self.neg_num), 1)
        print(pos_triples[:, 1].max())
        source_embeds = entity_embeddings[pos_triples[:, 0]]

        relation_embeds = relation_embeddings[pos_triples[:, 1]]

        tail_embeds = entity_embeddings[pos_triples[:, 2]]
        x = source_embeds + relation_embeds - tail_embeds
        pos_norm = torch.norm(x, p=1, dim=1)

        source_embeds = entity_embeddings[neg_triples[:, 0]]
        relation_embeds = relation_embeddings[neg_triples[:, 1]]
        tail_embeds = entity_embeddings[neg_triples[:, 2]]
        x = source_embeds + relation_embeds - tail_embeds
        neg_norm = torch.norm(x, p=1, dim=1)

        y = -torch.ones(int(self.neg_num) * len_pos_triples).to(self.device)
        loss = F.margin_ranking_loss(pos_norm, neg_norm, y, margin=1.0)
        return loss


class OnlyConvKB(BaseModel):
    def __init__(self, args):
        super(OnlyConvKB, self).__init__(args)
        self.entityEmbed = nn.Parameter(torch.randn(len(args.entity2id), args.dim * args.n_heads))
        self.relationEmbed = nn.Parameter(torch.randn(len(args.relation2id), args.dim * args.n_heads))
        self.ConvKB = ConvKBLayer(args.dim * args.n_heads, 3, 1, args.out_channels, args.dropout, args.alpha)

    def forward(self, batch_inputs):
        head = self.entityEmbed[batch_inputs[:, 0]]
        relation = self.relationEmbed[batch_inputs[:, 1]]
        tail = self.entityEmbed[batch_inputs[:, 2]]
        conv_input = torch.cat((head.unsqueeze(1), relation.unsqueeze(1), tail.unsqueeze(1)), dim=1)
        conv_out = self.ConvKB(conv_input)
        return conv_out

    def loss_func(self, output, target):
        return F.soft_margin_loss(output, target)


class IKRLConvKB(BaseModel):
    def __init__(self, args):
        super(IKRLConvKB, self).__init__(args)
        self.imgEmbed = nn.Linear(args.img.shape[1], args.dim * args.n_heads)
        self.relationImgEmbed = nn.Parameter(torch.randn(len(args.relation2id), args.dim * args.n_heads))
        self.entityEmbed = nn.Parameter(torch.randn(len(args.entity2id), args.dim * args.n_heads))
        self.relationEmbed = nn.Parameter(torch.randn(len(args.relation2id), args.dim * args.n_heads))
        self.ConvKB = ConvKBLayer(args.dim * args.n_heads, 2*3, 1, args.out_channels, args.dropout, args.alpha)
        self.img = args.img.to(self.device)

    def forward(self, batch_inputs):
        head = torch.cat((self.entityEmbed[batch_inputs[:, 0]].unsqueeze(1),
                          self.imgEmbed(self.img[batch_inputs[:, 0]]).unsqueeze(1)), dim=1)
        relation = torch.cat((self.relationEmbed[batch_inputs[:, 1]].unsqueeze(1),
                              self.relationImgEmbed[batch_inputs[:, 1]].unsqueeze(1)), dim=1)
        tail = torch.cat((self.entityEmbed[batch_inputs[:, 2]].unsqueeze(1),
                          self.imgEmbed(self.img[batch_inputs[:, 2]]).unsqueeze(1)), dim=1)
        conv_input = torch.cat((head, relation, tail), dim=1)
        conv_out = self.ConvKB(conv_input)
        return conv_out

    def loss_func(self, output, target):
        return F.soft_margin_loss(output, target)


class IKRL(BaseModel):
    def __init__(self, args):
        super(IKRL, self).__init__(args)
        self.neg_num = args.neg_num
        self.entity_embeddings = nn.Embedding(len(args.entity2id), args.dim, padding_idx=None)
        nn.init.xavier_normal_(self.entity_embeddings.weight)
        self.relation_embeddings = nn.Embedding(len(args.relation2id), args.r_dim, padding_idx=None)
        nn.init.xavier_normal_(self.relation_embeddings.weight)

        img_pool = torch.nn.AvgPool2d(4, stride=4)
        img = img_pool(args.img.to(self.device).view(-1, 64, 64))
        img = img.view(img.size(0), -1)
        self.img_entity_embeddings = nn.Embedding.from_pretrained(img, freeze=False)

    def forward(self, batch_inputs):
        len_pos_triples = int(batch_inputs.shape[0] / (int(self.neg_num) + 1))
        pos_triples = batch_inputs[:len_pos_triples]
        neg_triples = batch_inputs[len_pos_triples:]
        pos_triples = pos_triples.repeat(int(self.neg_num), 1)

        relation_pos = self.relation_embeddings(pos_triples[:, 1])
        relation_neg = self.relation_embeddings(neg_triples[:, 1])

        head_s_pos = self.entity_embeddings(pos_triples[:, 0])
        tail_s_pos = self.entity_embeddings(pos_triples[:, 2])

        head_s_neg = self.entity_embeddings(neg_triples[:, 0])
        tail_s_neg = self.entity_embeddings(neg_triples[:, 2])

        head_i_pos = self.img_entity_embeddings(pos_triples[:, 0])
        tail_i_pos = self.img_entity_embeddings(pos_triples[:, 2])

        head_i_neg = self.img_entity_embeddings(neg_triples[:, 0])
        tail_i_neg = self.img_entity_embeddings(neg_triples[:, 2])

        energy_ss_pos = torch.norm(head_s_pos + relation_pos - tail_s_pos, p=1, dim=1)
        energy_ss_neg = torch.norm(head_s_neg + relation_neg - tail_s_neg, p=1, dim=1)

        energy_si_pos = torch.norm(head_s_pos + relation_pos - tail_i_pos, p=1, dim=1)
        energy_si_neg = torch.norm(head_s_neg + relation_neg - tail_i_neg, p=1, dim=1)

        energy_is_pos = torch.norm(head_i_pos + relation_pos - tail_s_pos, p=1, dim=1)
        energy_is_neg = torch.norm(head_i_neg + relation_neg - tail_s_neg, p=1, dim=1)

        energy_ii_pos = torch.norm(head_i_pos + relation_pos - tail_i_pos, p=1, dim=1)
        energy_ii_neg = torch.norm(head_i_neg + relation_neg - tail_i_neg, p=1, dim=1)

        energy_pos = energy_ss_pos + energy_si_pos + energy_is_pos + energy_ii_pos
        energy_neg = energy_ss_neg + energy_si_neg + energy_is_neg + energy_ii_neg

        y = -torch.ones(int(self.neg_num) * len_pos_triples).to(self.device)
        loss = F.margin_ranking_loss(energy_pos, energy_neg, y, margin=10.0)

        return loss

    def predict(self, batch_inputs):
        pos_triples = batch_inputs

        relation_pos = self.relation_embeddings(pos_triples[:, 1])

        head_s_pos = self.entity_embeddings(pos_triples[:, 0])
        tail_s_pos = self.entity_embeddings(pos_triples[:, 2])

        head_i_pos = self.img_entity_embeddings(pos_triples[:, 0])
        tail_i_pos = self.img_entity_embeddings(pos_triples[:, 2])

        energy_ss_pos = torch.norm(head_s_pos + relation_pos - tail_s_pos, p=1, dim=1)
        energy_si_pos = torch.norm(head_s_pos + relation_pos - tail_i_pos, p=1, dim=1)
        energy_is_pos = torch.norm(head_i_pos + relation_pos - tail_s_pos, p=1, dim=1)
        energy_ii_pos = torch.norm(head_i_pos + relation_pos - tail_i_pos, p=1, dim=1)

        energy_pos = energy_ss_pos + energy_si_pos + energy_is_pos + energy_ii_pos

        return energy_pos


class MKGC(BaseModel):
    def __init__(self, args):
        super(MKGC, self).__init__(args)
        self.neg_num = args.neg_num
        self.entity_embeddings = nn.Embedding(len(args.entity2id), 2 * args.dim, padding_idx=None)
        nn.init.xavier_normal_(self.entity_embeddings.weight)
        self.relation_embeddings = nn.Embedding(len(args.relation2id), 2 * args.r_dim, padding_idx=None)
        nn.init.xavier_normal_(self.relation_embeddings.weight)

        img_pool = torch.nn.AvgPool2d(4, stride=4)
        img = img_pool(args.img.to(self.device).view(-1, 64, 64))
        img = img.view(img.size(0), -1)

        txt_pool = torch.nn.AdaptiveAvgPool2d(output_size=(4, 64))
        txt = txt_pool(args.desp.to(self.device).view(-1, 12, 64))
        txt = txt.view(txt.size(0), -1)

        self.img_entity_embeddings = nn.Embedding.from_pretrained(torch.cat([img, txt], dim=1), freeze=False)

    def forward(self, batch_inputs):
        len_pos_triples = int(batch_inputs.shape[0] / (int(self.neg_num) + 1))
        pos_triples = batch_inputs[:len_pos_triples]
        neg_triples = batch_inputs[len_pos_triples:]
        pos_triples = pos_triples.repeat(int(self.neg_num), 1)

        relation_pos = self.relation_embeddings(pos_triples[:, 1])
        relation_neg = self.relation_embeddings(neg_triples[:, 1])

        head_s_pos = self.entity_embeddings(pos_triples[:, 0])
        tail_s_pos = self.entity_embeddings(pos_triples[:, 2])

        head_s_neg = self.entity_embeddings(neg_triples[:, 0])
        tail_s_neg = self.entity_embeddings(neg_triples[:, 2])

        head_i_pos = self.img_entity_embeddings(pos_triples[:, 0])
        tail_i_pos = self.img_entity_embeddings(pos_triples[:, 2])

        head_i_neg = self.img_entity_embeddings(neg_triples[:, 0])
        tail_i_neg = self.img_entity_embeddings(neg_triples[:, 2])

        energy_ss_pos = torch.norm(head_s_pos + relation_pos - tail_s_pos, p=1, dim=1)
        energy_ss_neg = torch.norm(head_s_neg + relation_neg - tail_s_neg, p=1, dim=1)

        energy_si_pos = torch.norm(head_s_pos + relation_pos - tail_i_pos, p=1, dim=1)
        energy_si_neg = torch.norm(head_s_neg + relation_neg - tail_i_neg, p=1, dim=1)

        energy_is_pos = torch.norm(head_i_pos + relation_pos - tail_s_pos, p=1, dim=1)
        energy_is_neg = torch.norm(head_i_neg + relation_neg - tail_s_neg, p=1, dim=1)

        energy_ii_pos = torch.norm(head_i_pos + relation_pos - tail_i_pos, p=1, dim=1)
        energy_ii_neg = torch.norm(head_i_neg + relation_neg - tail_i_neg, p=1, dim=1)

        energy_sisi_pos = torch.norm((head_s_pos + head_i_pos) + relation_pos - (tail_s_pos + tail_i_pos), p=1, dim=1)
        energy_sisi_neg = torch.norm((head_s_neg + head_i_neg) + relation_neg - (tail_s_neg + tail_i_neg), p=1, dim=1)

        energy_pos = energy_ss_pos + energy_si_pos + energy_is_pos + energy_ii_pos + energy_sisi_pos
        energy_neg = energy_ss_neg + energy_si_neg + energy_is_neg + energy_ii_neg + energy_sisi_neg

        y = -torch.ones(int(self.neg_num) * len_pos_triples).to(self.device)
        loss = F.margin_ranking_loss(energy_pos, energy_neg, y, margin=10.0)

        return loss

    def predict(self, batch_inputs):
        pos_triples = batch_inputs

        relation_pos = self.relation_embeddings(pos_triples[:, 1])

        head_s_pos = self.entity_embeddings(pos_triples[:, 0])
        tail_s_pos = self.entity_embeddings(pos_triples[:, 2])

        head_i_pos = self.img_entity_embeddings(pos_triples[:, 0])
        tail_i_pos = self.img_entity_embeddings(pos_triples[:, 2])

        energy_ss_pos = torch.norm(head_s_pos + relation_pos - tail_s_pos, p=1, dim=1)
        energy_si_pos = torch.norm(head_s_pos + relation_pos - tail_i_pos, p=1, dim=1)
        energy_is_pos = torch.norm(head_i_pos + relation_pos - tail_s_pos, p=1, dim=1)
        energy_ii_pos = torch.norm(head_i_pos + relation_pos - tail_i_pos, p=1, dim=1)
        energy_sisi_pos = torch.norm((head_s_pos + head_i_pos) + relation_pos - (tail_s_pos + tail_i_pos), p=1, dim=1)

        energy_pos = energy_ss_pos + energy_si_pos + energy_is_pos + energy_ii_pos + energy_sisi_pos

        return energy_pos


class Mutan(BaseModel):
    def __init__(self, args):
        super(Mutan, self).__init__(args)
        self.entity_embeddings = nn.Embedding(len(args.entity2id), args.dim, padding_idx=None)
        nn.init.xavier_normal_(self.entity_embeddings.weight)
        self.relation_embeddings = nn.Embedding(2 * len(args.relation2id), args.r_dim, padding_idx=None)
        nn.init.xavier_normal_(self.relation_embeddings.weight)
        self.dim = args.dim
        self.Mutan = MutanLayer(args.dim, 5)
        self.bias = nn.Parameter(torch.zeros(len(args.entity2id)))
        self.bceloss = nn.BCELoss()

    def forward(self, batch_inputs):
        head = batch_inputs[:, 0]
        relation = batch_inputs[:, 1]
        e_embed = self.entity_embeddings(head)
        r_embed = self.relation_embeddings(relation)
        pred = self.Mutan(e_embed, r_embed)
        pred = torch.mm(pred, self.entity_embeddings.weight.transpose(1, 0))
        pred = torch.sigmoid(pred)
        return pred

    def loss_func(self, output, target):
        return self.bceloss(output, target)


class TuckER(BaseModel):
    def __init__(self, args):
        super(TuckER, self).__init__(args)
        self.entity_embeddings = nn.Embedding(len(args.entity2id), args.dim, padding_idx=None)
        nn.init.xavier_normal_(self.entity_embeddings.weight)
        self.relation_embeddings = nn.Embedding(2 * len(args.relation2id), args.r_dim, padding_idx=None)
        nn.init.xavier_normal_(self.relation_embeddings.weight)
        if args.pre_trained:
            self.entity_embeddings = nn.Embedding.from_pretrained(
                torch.from_numpy(pickle.load(open('datasets/' + args.dataset + '/gat_entity_vec.pkl', 'rb'))).float(), freeze=False)
            self.relation_embeddings = nn.Embedding.from_pretrained(torch.cat((
                torch.from_numpy(pickle.load(open('datasets/' + args.dataset + '/gat_relation_vec.pkl', 'rb'))).float(),
                -1 * torch.from_numpy(pickle.load(open('datasets/' + args.dataset + '/gat_relation_vec.pkl', 'rb'))).float()), dim=0), freeze=False)
        self.dim = args.dim
        self.TuckER = TuckERLayer(args.dim, args.r_dim)
        self.bias = nn.Parameter(torch.zeros(len(args.entity2id)))
        self.bceloss = nn.BCELoss()

    def forward(self, batch_inputs, lookup=None):
        head = batch_inputs[:, 0]
        relation = batch_inputs[:, 1]
        e_embed = self.entity_embeddings(head)
        r_embed = self.relation_embeddings(relation)
        pred = self.TuckER(e_embed, r_embed)
        if lookup is None:
            pred = torch.mm(pred, self.entity_embeddings.weight.transpose(1, 0))
        else:
            pred = torch.bmm(pred.unsqueeze(1), self.entity_embeddings.weight[lookup].transpose(1, 2)).squeeze(1)
        pred = torch.sigmoid(pred)
        return pred

    def loss_func(self, output, target):
        return self.bceloss(output, target)


class ConvE(BaseModel):
    def __init__(self, args):
        super(ConvE, self).__init__(args)
        self.entity_embeddings = nn.Embedding(len(args.entity2id), args.dim, padding_idx=None)
        nn.init.xavier_normal_(self.entity_embeddings.weight)
        self.relation_embeddings = nn.Embedding(2 * len(args.relation2id), args.dim, padding_idx=None)
        nn.init.xavier_normal_(self.relation_embeddings.weight)
        self.dim = args.dim
        self.k_w = args.k_w
        self.k_h = args.k_h
        self.ConvE = ConvELayer(args.dim, args.out_channels, args.kernel_size, args.k_h, args.k_w)
        self.bias = nn.Parameter(torch.zeros(len(args.entity2id)))
        self.bceloss = nn.BCELoss()

    def forward(self, batch_inputs):
        head = batch_inputs[:, 0]
        relation = batch_inputs[:, 1]
        e_embed = self.entity_embeddings(head)
        r_embed = self.relation_embeddings(relation)
        e_embed = e_embed.view(-1, 1, self.dim)
        r_embed = r_embed.view(-1, 1, self.dim)
        embed = torch.cat([e_embed, r_embed], dim=1)
        embed = torch.transpose(embed, 2, 1).reshape((-1, 1, 2 * self.k_w, self.k_h))

        pred = self.ConvE(embed)
        pred = torch.mm(pred, self.entity_embeddings.weight.transpose(1, 0))
        pred += self.bias.expand_as(pred)
        pred = torch.sigmoid(pred)
        return pred

    def loss_func(self, output, target):
        return self.bceloss(output, target)


class IMFFFT(BaseModel):
    def __init__(self, args):
        super(IMFFFT, self).__init__(args)
        self.entity_embeddings = nn.Embedding(len(args.entity2id), args.dim, padding_idx=None)
        #self.entity_embeddings = nn.Embedding(len(args.entity2id), 256, padding_idx=None)
        nn.init.xavier_normal_(self.entity_embeddings.weight)
        self.relation_embeddings = nn.Embedding(2 * len(args.relation2id), args.r_dim, padding_idx=None)
        self.text_projection = nn.Linear(768, 256)   # Project image embeddings to 768 (text dim)
        self.image_projection = nn.Linear(1000, 256)     # Project image embeddings to 768

        nn.init.xavier_normal_(self.relation_embeddings.weight)
        if args.pre_trained:

            self.entity_embeddings = nn.Embedding.from_pretrained(
                torch.from_numpy(pickle.load(open('datasets/' + args.dataset + '/gat_entity_vec.pkl', 'rb'))).float(), freeze=False)

            self.relation_embeddings = nn.Embedding.from_pretrained(torch.cat((
                torch.from_numpy(pickle.load(open('datasets/' + args.dataset + '/gat_relation_vec.pkl', 'rb'))).float(),
                -1 * torch.from_numpy(pickle.load(open('datasets/' + args.dataset + '/gat_relation_vec.pkl', 'rb'))).float()), dim=0), freeze=False)
            print('YangZi')
        # image and text projection
        # args.img=args.img[:100,:,:]
        img= args.img.to('cpu')
        img= self.image_projection(img)
        # args.desp=args.desp[:100,:,:]
        txt = args.desp.to('cpu')
        txt = self.text_projection(txt)

        txt, img = self.fft_and_ifft(txt, img, d_model=256, num_filter=2, dropout=0.0)


        self.img_entity_embeddings = nn.Embedding.from_pretrained(img, freeze=False)
        self.img_relation_embeddings = nn.Embedding(2 * len(args.relation2id), args.r_dim, padding_idx=None)
        nn.init.xavier_normal_(self.img_relation_embeddings.weight)

        # txt_pool = torch.nn.AdaptiveAvgPool2d(output_size=(4, 64))
        # txt = txt_pool(args.desp.to(self.device).view(-1, 12, 64))#args.desp torch.Size([14951, 100, 768]), txt:torch.Size([1495100, 4, 64])
        # txt = txt_pool(args.desp.to(self.device))#args.desp torch.Size([14951, 100, 768]), txt:torch.Size([1495100, 4, 64])
        # txt = txt.view(args.desp.size(0), -1)
        # text projection
        #txt = self.text_projection(txt) #torch.Size([14951, 768])
        self.txt_entity_embeddings = nn.Embedding.from_pretrained(txt, freeze=False)
        self.txt_relation_embeddings = nn.Embedding(2 * len(args.relation2id), args.r_dim, padding_idx=None)
        nn.init.xavier_normal_(self.txt_relation_embeddings.weight)

        self.dim = args.dim
        self.TuckER_S = TuckERLayer(args.dim, args.r_dim) #args.dim, args.r_dim:256
        self.TuckER_I = TuckERLayer(args.dim, args.r_dim)
        self.TuckER_D = TuckERLayer(args.dim, args.r_dim)
        self.TuckER_MM = TuckERLayer(args.dim, args.r_dim)
        self.Mutan_MM_E = MutanLayer(args.dim, 2)
        self.Mutan_MM_R = MutanLayer(args.dim, 2)
        self.bias = nn.Parameter(torch.zeros(len(args.entity2id)))
        self.bceloss = nn.BCELoss()

    def fft_and_ifft(self,text,image,d_model=256,num_filter=2,dropout=0.0):

        # fft
        x_text = text
        B, S, D = text.shape
        s= S // 2 + 1

        B, N, C = image.shape
        n = N // 2 + 1
        self.text_weight = nn.Parameter(torch.randn(s, d_model, 2, dtype=torch.float32))
        self.text_filter_bank = nn.Parameter(torch.randn(num_filter, s, d_model, 2, dtype=torch.float32))

        self.image_weight = nn.Parameter(torch.randn(n, d_model, 2, dtype=torch.float32))
        self.image_filter_bank = nn.Parameter(torch.randn(num_filter, n, d_model, 2, dtype=torch.float32))

        self.text_frequency_select = TextFrequencySelection(n, d_model)
        self.image_frequency_select = ImageFrequencySelection(s, d_model)

        self.text_add_norm = AddNorm(d_model, dropout)
        self.image_add_norm = AddNorm(d_model, dropout)


        x_image = image
        x_text = text
        _text = torch.fft.rfft(text, dim=1, norm='ortho')
        _image = torch.fft.rfft(image, dim=1, norm='ortho')

        # frequency filter
        _text = self.filter(_text, s, torch.view_as_complex(self.text_filter_bank),
                            torch.view_as_complex(self.text_weight))
        _image = self.filter(_image, n, torch.view_as_complex(self.image_filter_bank),
                             torch.view_as_complex(self.image_weight))

        # frequency select
        _text = self.text_frequency_select(_text, _image)
        _image = self.image_frequency_select(_image, _text)

        # ifft (inverse Fourier transform)
        text = torch.fft.irfft(_text, n=S, dim=1, norm='ortho')
        image = torch.fft.irfft(_image, n=N, dim=1, norm='ortho')

        # add & norm
        # x_text = x_text
        # x_image = x_image
        text = self.text_add_norm((text + x_text))
        image = self.image_add_norm((image + x_image))

        text = torch.max(text, dim=1)[0] #[?, 768]
        image = torch.max(image, dim=1)[0] #[?, 768]

        return text, image

    def filter(self, x, length, filter_bank, weight, num_filter=2,use_bank=True):
        pi=3.14159265358979323846
        if use_bank:
            power = (x * x) / length
            Y = []
            for k in range(num_filter):
                cos = torch.cos(torch.as_tensor((2 * (k + 1) - 1) * pi / 2 * num_filter))
                Y.append(power * filter_bank[k] * cos)
            C = torch.stack(Y)  # (filter, batch, s, dim)
            x = torch.sum(C, dim=0)  # (batch, s, dim)
        else:
            x = x * weight

        return x


    def forward(self, batch_inputs):

        head = batch_inputs[:, 0]
        relation = batch_inputs[:, 1]
        e_embed = self.entity_embeddings(head)
        r_embed = self.relation_embeddings(relation)
        #linear_layer6 = torch.nn.Linear(512, 256).to(device)
        #e_embed = linear_layer6(e_embed)      #torch.Size([256, 256])
        #r_embed = linear_layer6(r_embed)
        #print('hello', e_embed.shape)
        e_img_embed = self.img_entity_embeddings(head)  #torch.Size([256, 256])
        r_img_embed = self.img_relation_embeddings(relation) #torch.Size([256, 256])
        e_txt_embed = self.txt_entity_embeddings(head) #torch.Size([256, 256])
        r_txt_embed = self.txt_relation_embeddings(relation) #torch.Size([256, 256])
        #融合向量

        e_mm_embed = self.Mutan_MM_E(e_embed, e_img_embed, e_txt_embed)


        # e_mm_embed = torch.cat((e_embed, e_txt_embed, e_img_embed), dim=1)  # 形状：[256, 768]
        # linear_layer5 = torch.nn.Linear(768, 256).to(device)
        # e_mm_embed = linear_layer5(e_mm_embed)  # 形状：[256, 256]

        r_mm_embed = self.Mutan_MM_R(r_embed, r_img_embed, r_txt_embed)
        # r_mm_embed = torch.cat((r_embed, r_txt_embed, r_img_embed), dim=1)  # 形状：[256, 768]
        # r_mm_embed = linear_layer5(r_mm_embed)  # 形状：[256, 256]

        pred_s = self.TuckER_S(e_embed, r_embed)
        #print('112',pred_s.shape)
        pred_i = self.TuckER_I(e_img_embed, r_img_embed)
        pred_d = self.TuckER_D(e_txt_embed, r_txt_embed)
        pred_mm = self.TuckER_MM(e_mm_embed, r_mm_embed)
        #pred_mm = self.TuckER_MM(e_mm_embed, r_embed)

        pred_s = torch.mm(pred_s, self.entity_embeddings.weight.transpose(1, 0))
        pred_i = torch.mm(pred_i, self.img_entity_embeddings.weight.transpose(1, 0))
        pred_d = torch.mm(pred_d, self.txt_entity_embeddings.weight.transpose(1, 0))
        pred_mm = torch.mm(pred_mm, self.Mutan_MM_E(self.entity_embeddings.weight,
                                                    self.img_entity_embeddings.weight,
                                                    self.txt_entity_embeddings.weight).transpose(1, 0))

        pred_s = torch.sigmoid(pred_s)
        pred_i = torch.sigmoid(pred_i)
        pred_d = torch.sigmoid(pred_d)
        pred_mm = torch.sigmoid(pred_mm)

        return [pred_s, pred_i, pred_d, pred_mm]

    def loss_func(self, output, target):
        loss_s = self.bceloss(output[0], target)
        loss_i = self.bceloss(output[1], target)
        loss_d = self.bceloss(output[2], target)
        loss_mm = self.bceloss(output[3], target)
        return loss_s + loss_i + loss_d + loss_mm
        #return self.bceloss(output, target)

class IMF(BaseModel):
    def __init__(self, args):
        super(IMF, self).__init__(args)
        self.entity_embeddings = nn.Embedding(len(args.entity2id), args.dim, padding_idx=None)
        nn.init.xavier_normal_(self.entity_embeddings.weight)
        self.relation_embeddings = nn.Embedding(2 * len(args.relation2id), args.r_dim, padding_idx=None)
        nn.init.xavier_normal_(self.relation_embeddings.weight)
        if args.pre_trained:
            self.entity_embeddings = nn.Embedding.from_pretrained(
                torch.from_numpy(pickle.load(open('datasets/' + args.dataset + '/gat_entity_vec.pkl', 'rb'))).float(), freeze=False)
            self.relation_embeddings = nn.Embedding.from_pretrained(torch.cat((
                torch.from_numpy(pickle.load(open('datasets/' + args.dataset + '/gat_relation_vec.pkl', 'rb'))).float(),
                -1 * torch.from_numpy(pickle.load(open('datasets/' + args.dataset + '/gat_relation_vec.pkl', 'rb'))).float()), dim=0), freeze=False)

        img_pool = torch.nn.AvgPool2d(4, stride=4)
        img = img_pool(args.img.to(self.device).view(-1, 64, 64))
        img = img.view(img.size(0), -1)
        self.img_entity_embeddings = nn.Embedding.from_pretrained(img, freeze=False)
        self.img_relation_embeddings = nn.Embedding(2 * len(args.relation2id), args.r_dim, padding_idx=None)
        nn.init.xavier_normal_(self.img_relation_embeddings.weight)

        txt_pool = torch.nn.AdaptiveAvgPool2d(output_size=(4, 64))
        txt = txt_pool(args.desp.to(self.device).view(-1, 12, 64))
        txt = txt.view(txt.size(0), -1)
        self.txt_entity_embeddings = nn.Embedding.from_pretrained(txt, freeze=False)
        self.txt_relation_embeddings = nn.Embedding(2 * len(args.relation2id), args.r_dim, padding_idx=None)
        nn.init.xavier_normal_(self.txt_relation_embeddings.weight)

        self.dim = args.dim
        self.TuckER_S = TuckERLayer(args.dim, args.r_dim)
        self.TuckER_I = TuckERLayer(args.dim, args.r_dim)
        self.TuckER_D = TuckERLayer(args.dim, args.r_dim)
        self.TuckER_MM = TuckERLayer(args.dim, args.r_dim)
        self.Mutan_MM_E = MutanLayer(args.dim, 2)
        self.Mutan_MM_R = MutanLayer(args.dim, 2)
        self.bias = nn.Parameter(torch.zeros(len(args.entity2id)))
        self.bceloss = nn.BCELoss()




    def contrastive_loss(self, s_embed, v_embed, t_embed):
        #print("hello kitty1")
        s_embed, v_embed, t_embed = s_embed / torch.norm(s_embed), v_embed / torch.norm(v_embed), t_embed / torch.norm(t_embed)
        pos_sv = torch.sum(s_embed * v_embed, dim=1, keepdim=True)
        pos_st = torch.sum(s_embed * t_embed, dim=1, keepdim=True)
        pos_vt = torch.sum(v_embed * t_embed, dim=1, keepdim=True)
        neg_s = torch.matmul(s_embed, s_embed.t())
        neg_v = torch.matmul(v_embed, v_embed.t())
        neg_t = torch.matmul(t_embed, t_embed.t())
        neg_s = neg_s - torch.diag_embed(torch.diag(neg_s))
        neg_v = neg_v - torch.diag_embed(torch.diag(neg_v))
        neg_t = neg_t - torch.diag_embed(torch.diag(neg_t))
        pos = torch.mean(torch.cat([pos_sv, pos_st, pos_vt], dim=1), dim=1)
        neg = torch.mean(torch.cat([neg_s, neg_v, neg_t], dim=1), dim=1)
        loss = torch.mean(F.softplus(neg - pos))
        return loss

    def contrastive_loss(self, output):
        #print("hello kitty1")
        s_embed, v_embed, t_embed = output[0], output[1], output[2]
        s_embed, v_embed, t_embed = s_embed / torch.norm(s_embed), v_embed / torch.norm(v_embed), t_embed / torch.norm(t_embed)
        pos_sv = torch.sum(s_embed * v_embed, dim=1, keepdim=True)
        pos_st = torch.sum(s_embed * t_embed, dim=1, keepdim=True)
        pos_vt = torch.sum(v_embed * t_embed, dim=1, keepdim=True)
        neg_s = torch.matmul(s_embed, s_embed.t())
        neg_v = torch.matmul(v_embed, v_embed.t())
        neg_t = torch.matmul(t_embed, t_embed.t())
        neg_s = neg_s - torch.diag_embed(torch.diag(neg_s))
        neg_v = neg_v - torch.diag_embed(torch.diag(neg_v))
        neg_t = neg_t - torch.diag_embed(torch.diag(neg_t))
        pos = torch.mean(torch.cat([pos_sv, pos_st, pos_vt], dim=1), dim=1)
        neg = torch.mean(torch.cat([neg_s, neg_v, neg_t], dim=1), dim=1)
        loss = torch.mean(F.softplus(neg - pos))
        return loss

    def forward(self, batch_inputs):
        head = batch_inputs[:, 0]
        relation = batch_inputs[:, 1]
        e_embed = self.entity_embeddings(head)
        r_embed = self.relation_embeddings(relation)
        e_img_embed = self.img_entity_embeddings(head)
        r_img_embed = self.img_relation_embeddings(relation)
        e_txt_embed = self.txt_entity_embeddings(head)
        r_txt_embed = self.txt_relation_embeddings(relation)
        e_mm_embed = self.Mutan_MM_E(e_embed, e_img_embed, e_txt_embed)
        r_mm_embed = self.Mutan_MM_R(r_embed, r_img_embed, r_txt_embed)

        pred_s = self.TuckER_S(e_embed, r_embed)
        pred_i = self.TuckER_I(e_img_embed, r_img_embed)
        pred_d = self.TuckER_D(e_txt_embed, r_txt_embed)
        pred_mm = self.TuckER_MM(e_mm_embed, r_mm_embed)

        pred_s = torch.mm(pred_s, self.entity_embeddings.weight.transpose(1, 0))
        pred_i = torch.mm(pred_i, self.img_entity_embeddings.weight.transpose(1, 0))
        pred_d = torch.mm(pred_d, self.txt_entity_embeddings.weight.transpose(1, 0))
        pred_mm = torch.mm(pred_mm, self.Mutan_MM_E(self.entity_embeddings.weight,
                                                    self.img_entity_embeddings.weight,
                                                    self.txt_entity_embeddings.weight).transpose(1, 0))

        pred_s = torch.sigmoid(pred_s)
        pred_i = torch.sigmoid(pred_i)
        pred_d = torch.sigmoid(pred_d)
        pred_mm = torch.sigmoid(pred_mm)
        return [pred_s, pred_i, pred_d, pred_mm]

    def loss_func(self, output, target):
        #print("hello kitty2")
        loss_s = self.bceloss(output[0], target)
        loss_i = self.bceloss(output[1], target)
        loss_d = self.bceloss(output[2], target)
        loss_mm = self.bceloss(output[3], target)
        return loss_s + loss_i + loss_d + loss_mm
        #return self.bceloss(output, target)



class IMFWavelet(BaseModel):
    def __init__(self, args):
        super(IMFWavelet, self).__init__(args)
        self.entity_embeddings = nn.Embedding(len(args.entity2id), args.dim, padding_idx=None)
        nn.init.xavier_normal_(self.entity_embeddings.weight)
        self.relation_embeddings = nn.Embedding(2 * len(args.relation2id), args.r_dim, padding_idx=None)
        self.text_projection = nn.Linear(768, 256)
        self.image_projection = nn.Linear(1000, 256)

        nn.init.xavier_normal_(self.relation_embeddings.weight)
        if args.pre_trained:
            self.entity_embeddings = nn.Embedding.from_pretrained(
                torch.from_numpy(pickle.load(open('datasets/' + args.dataset + '/gat_entity_vec.pkl', 'rb'))).float(),
                freeze=False)
            self.relation_embeddings = nn.Embedding.from_pretrained(torch.cat((
                torch.from_numpy(pickle.load(open('datasets/' + args.dataset + '/gat_relation_vec.pkl', 'rb'))).float(),
                -1 * torch.from_numpy(
                    pickle.load(open('datasets/' + args.dataset + '/gat_relation_vec.pkl', 'rb'))).float()), dim=0),
                freeze=False)

        img = args.img.to('cpu')
        img = self.image_projection(img)
        txt = args.desp.to('cpu')
        txt = self.text_projection(txt)
        txt, img = self.wavelet_transform(txt, img, d_model=256, num_filter=2, dropout=0.0)

        self.img_entity_embeddings = nn.Embedding.from_pretrained(img, freeze=False)
        self.img_relation_embeddings = nn.Embedding(2 * len(args.relation2id), args.r_dim, padding_idx=None)
        nn.init.xavier_normal_(self.img_relation_embeddings.weight)

        self.txt_entity_embeddings = nn.Embedding.from_pretrained(txt, freeze=False)
        self.txt_relation_embeddings = nn.Embedding(2 * len(args.relation2id), args.r_dim, padding_idx=None)
        nn.init.xavier_normal_(self.txt_relation_embeddings.weight)

        self.dim = args.dim
        self.TuckER_S = TuckERLayer(args.dim, args.r_dim)  # args.dim, args.r_dim:256
        self.TuckER_I = TuckERLayer(args.dim, args.r_dim)
        self.TuckER_D = TuckERLayer(args.dim, args.r_dim)
        self.TuckER_MM = TuckERLayer(args.dim, args.r_dim)
        self.Mutan_MM_E = MutanLayer(args.dim, 2)
        self.Mutan_MM_R = MutanLayer(args.dim, 2)
        self.bias = nn.Parameter(torch.zeros(len(args.entity2id)))
        self.bceloss = nn.BCELoss()

    def wavelet_transform(self, text, image, d_model=256, num_filter=2, dropout=0.0):
        x_text = text

        B, S, D = text.shape

        B, N, C = image.shape

        # Define wavelet transform parameters
        wavelet = 'db4'
        level = 2

        # Decompose text and image using wavelet transform
        self.text_add_norm = AddNorm(d_model, dropout)
        self.image_add_norm = AddNorm(d_model, dropout)

        #text_coeffs = [pywt.wavedec(text[i].detach().cpu().numpy(), wavelet, level=level) for i in range(B)]
        text_coeffs = [
            [pywt.wavedec(text[i, :, j].detach().cpu().numpy(), wavelet, level=level) for j in range(D)]
            for i in range(B)
        ]
        #text_coeffs = np.array(text_coeffs, dtype=object)
        #print('hello',text_coeffs.shape)
        #image_coeffs = [pywt.wavedec(image[i].detach().cpu().numpy(), wavelet, level=level) for i in range(B)]
        image_coeffs = [
            [pywt.wavedec(image[i, :, j].detach().cpu().numpy(), wavelet, level=level) for j in range(C)]
            for i in range(B)
        ]
        # Frequency filter (update this according to wavelet domain logic)
        # For demonstration, simple filtering without actual filters
        #text_filtered = text_coeffs
        #image_filtered = image_coeffs
        #text_filtered = []
        #image_filtered = []
        text_filtered = []
        for sample in text_coeffs:
            filtered_sample = []
            for coeffs in sample:
                filtered_coeffs = []
                for i, coeff in enumerate(coeffs):
                    # 根据频率划分
                    if i == 0 or i==1:  # 低频分量
                        filtered_coeffs.append(coeff)
                    else:  # 高频分量
                        #print(f"原始高频分量 {i}: {coeff}")
                        filtered_coeffs.append(coeff * 0.2)  # 高频大幅缩小
                filtered_sample.append(filtered_coeffs)

            text_filtered.append(filtered_sample)

        # Apply filtering logic to image coefficients
        image_filtered = []
        for sample in image_coeffs:
            filtered_sample = []
            for coeffs in sample:
                filtered_coeffs = []
                for i, coeff in enumerate(coeffs):
                    if i == 0 or i==1:  # 低频分量
                        filtered_coeffs.append(coeff)
                    else:  # 高频分量
                        #print(f"原始高频分量 {i}: {coeff}")
                        filtered_coeffs.append(coeff * 0.2)

                filtered_sample.append(filtered_coeffs)
            image_filtered.append(filtered_sample)


        # Frequency selection (update this according to wavelet domain logic)
        # You would normally perform some selection here as per the model's needs.

        # Inverse wavelet transform
        #text_reconstructed = torch.tensor([pywt.waverec(c, wavelet) for c in text_filtered], dtype=torch.float32)
        #image_reconstructed = torch.tensor([pywt.waverec(c, wavelet) for c in image_filtered], dtype=torch.float32)

        text_reconstructed = torch.stack([
            torch.stack([torch.tensor(pywt.waverec(c, wavelet), dtype=torch.float32) for c in sample], dim=1)
            for sample in text_filtered
        ], dim=0)
        image_reconstructed = torch.stack([
            torch.stack([torch.tensor(pywt.waverec(c, wavelet), dtype=torch.float32) for c in sample], dim=1)
            for sample in image_filtered
        ], dim=0)
        # Ensure dimensions are consistent after reconstruction
        text_reconstructed = text_reconstructed[:, :S, :D]
        image_reconstructed = image_reconstructed[:, :N, :C]

        # Add & normalize
        text_out = self.text_add_norm((text_reconstructed.to(text.device) + x_text))
        image_out = self.image_add_norm((image_reconstructed.to(image.device) + image))

        text_out = torch.max(text_out, dim=1)[0]
        image_out = torch.max(image_out, dim=1)[0]

        # # Plot low and high frequency information for text
        # # 选择一个批次中的样本（例如第一个样本）
        # sample_index = 0  # 选择第一个样本进行可视化
        # sample_filtered = text_filtered[sample_index]
        # D = 8
        # # 设置子图布局
        # fig, axs = plt.subplots(2, D, figsize=(20, 8))
        # fig.suptitle('Low and High Frequency Components of Text Features')
        #
        # # 设置子图布局：1行8列图，表示每个特征维度的低频分量
        # fig, axs = plt.subplots(1, D, figsize=(20, 5))
        # fig.suptitle('Low Frequency Components of Text Features', fontsize=16)
        #
        # # 遍历每个特征维度，绘制低频分量
        # for j in range(D):
        #     low_freq = sample_filtered[j][0]  # 提取第j个特征的低频分量（第0个小波分量）
        #     axs[j].plot(range(S), low_freq)  # 横轴为序列长度S，纵轴为低频幅值
        #     axs[j].set_title(f'Feature {j + 1} Low Frequency')
        #     axs[j].set_xlabel('Sequence Index (S)')
        #     axs[j].set_ylabel('Low Frequency Amplitude')
        #     axs[j].tick_params(axis='x', rotation=45)
        #     axs[j].tick_params(axis='y', rotation=45)
        #     axs[j].grid(True)
        #
        # # 调整布局以防止标题重叠
        # plt.tight_layout(rect=[0, 0, 1, 0.95])
        # plt.show()

        return text_out, image_out

    # def wavelet_transform(self, text, image, d_model=256, num_filter=2, dropout=0.0):
    #     x_text = text
    #     B, S, D = text.shape
    #     B, N, C = image.shape
    #
    #     # Define wavelet transform parameters
    #     wavelet = 'db4'
    #     level = 2
    #
    #     # Decompose text and image using wavelet transform
    #     self.text_add_norm = AddNorm(d_model, dropout)
    #     self.image_add_norm = AddNorm(d_model, dropout)
    #
    #     # Perform wavelet decomposition without filtering
    #     text_coeffs = [
    #         [pywt.wavedec(text[i, :, j].detach().cpu().numpy(), wavelet, level=level) for j in range(D)]
    #         for i in range(B)
    #     ]
    #
    #     image_coeffs = [
    #         [pywt.wavedec(image[i, :, j].detach().cpu().numpy(), wavelet, level=level) for j in range(C)]
    #         for i in range(B)
    #     ]
    #
    #     # Directly reconstruct from the original coefficients (no filtering)
    #     text_reconstructed = torch.stack([
    #         torch.stack([torch.tensor(pywt.waverec(c, wavelet), dtype=torch.float32) for c in sample], dim=1)
    #         for sample in text_coeffs
    #     ], dim=0)
    #
    #     image_reconstructed = torch.stack([
    #         torch.stack([torch.tensor(pywt.waverec(c, wavelet), dtype=torch.float32) for c in sample], dim=1)
    #         for sample in image_coeffs
    #     ], dim=0)
    #
    #     # Ensure dimensions are consistent after reconstruction
    #     text_reconstructed = text_reconstructed[:, :S, :D]
    #     image_reconstructed = image_reconstructed[:, :N, :C]
    #
    #     # Add & normalize
    #     text_out = self.text_add_norm((text_reconstructed.to(text.device) + x_text))
    #     image_out = self.image_add_norm((image_reconstructed.to(image.device) + image))
    #
    #     text_out = torch.max(text_out, dim=1)[0]
    #     image_out = torch.max(image_out, dim=1)[0]
    #
    #     return text_out, image_out



    def filter(self, x, length, filter_bank, weight, num_filter=2, use_bank=True):
        pi = 3.14159265358979323846
        if use_bank:
            power = (x * x) / length
            Y = []
            for k in range(num_filter):
                cos = torch.cos(torch.as_tensor((2 * (k + 1) - 1) * pi / 2 * num_filter))
                Y.append(power * filter_bank[k] * cos)
            C = torch.stack(Y)  # (filter, batch, s, dim)
            x = torch.sum(C, dim=0)  # (batch, s, dim)
        else:
            x = x * weight

        return x

    def forward(self, batch_inputs):

        head = batch_inputs[:, 0]
        relation = batch_inputs[:, 1]
        e_embed = self.entity_embeddings(head)
        r_embed = self.relation_embeddings(relation)
        #linear_layer6 = torch.nn.Linear(512, 256).to(device)
        #e_embed = linear_layer6(e_embed)  # torch.Size([256, 256])
        #r_embed = linear_layer6(r_embed)
        #print('hello', e_embed.shape)
        e_img_embed = self.img_entity_embeddings(head)  # torch.Size([256, 256])
        r_img_embed = self.img_relation_embeddings(relation)  # torch.Size([256, 256])
        e_txt_embed = self.txt_entity_embeddings(head)  # torch.Size([256, 256])
        r_txt_embed = self.txt_relation_embeddings(relation)  # torch.Size([256, 256])
        # 融合向量
        # e_mm_embed = self.Mutan_MM_E(e_embed, e_img_embed, e_txt_embed)

        e_mm_embed = torch.cat((e_embed, e_txt_embed, e_img_embed), dim=1)  # 形状：[256, 768]
        linear_layer5 = torch.nn.Linear(768, 256).to(device)
        e_mm_embed = linear_layer5(e_mm_embed)  # 形状：[256, 256]

        # r_mm_embed = self.Mutan_MM_R(r_embed, r_img_embed, r_txt_embed)
        r_mm_embed = torch.cat((r_embed, r_txt_embed, r_img_embed), dim=1)  # 形状：[256, 768]
        r_mm_embed = linear_layer5(r_mm_embed)  # 形状：[256, 256]

        pred_s = self.TuckER_S(e_embed, r_embed)
        # print('112',pred_s.shape)
        pred_i = self.TuckER_I(e_img_embed, r_img_embed)
        pred_d = self.TuckER_D(e_txt_embed, r_txt_embed)
        pred_mm = self.TuckER_MM(e_mm_embed, r_mm_embed)
        # pred_mm = self.TuckER_MM(e_mm_embed, r_embed)

        pred_s = torch.mm(pred_s, self.entity_embeddings.weight.transpose(1, 0))
        pred_i = torch.mm(pred_i, self.img_entity_embeddings.weight.transpose(1, 0))
        pred_d = torch.mm(pred_d, self.txt_entity_embeddings.weight.transpose(1, 0))
        pred_mm = torch.mm(pred_mm, self.Mutan_MM_E(self.entity_embeddings.weight,
                                                    self.img_entity_embeddings.weight,
                                                    self.txt_entity_embeddings.weight).transpose(1, 0))

        pred_s = torch.sigmoid(pred_s)
        pred_i = torch.sigmoid(pred_i)
        pred_d = torch.sigmoid(pred_d)
        pred_mm = torch.sigmoid(pred_mm)

        return [pred_s, pred_i, pred_d, pred_mm]

    def loss_func(self, output, target):
        loss_s = self.bceloss(output[0], target)
        loss_i = self.bceloss(output[1], target)
        loss_d = self.bceloss(output[2], target)
        loss_mm = self.bceloss(output[3], target)
        return loss_s + loss_i + loss_d + loss_mm


