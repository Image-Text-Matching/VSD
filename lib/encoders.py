import os
import torch
import torch.nn as nn
import numpy as np
from transformers import BertModel
import logging
from lib.gpo import GPO
from lib.mlp import FC_MLP
from lib.mlp import MLP_
from transformers import AutoModel

logger = logging.getLogger(__name__)


# 'True' represents to be masked （Do not participate in the calculation of attention）
# 'False' represents not to be masked
def padding_mask(embs, lengths):
    mask = torch.ones(len(lengths), embs.shape[1], device=lengths.device)
    for i in range(mask.shape[0]):
        end = int(lengths[i])
        mask[i, :end] = 0.

    return mask.bool()


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.)


def l2norm(X, dim, eps=1e-8):
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def maxk_pool1d(x, dim, k):
    max_k = maxk(x, dim, k)
    return max_k.mean(dim)


def maxk(x, dim, k):
    _x, index = x.topk(k, dim=dim)
    return _x


# uncertain length
def maxk_pool1d_var(x, dim, k, lengths):
    # k >= 1
    results = []
    # assert len(lengths) == x.size(0)

    for idx in range(x.size(0)):
        # keep use all number of features
        k = min(k, int(lengths[idx].item()))

        tmp = torch.split(x[idx], split_size_or_sections=lengths[idx], dim=dim - 1)[0]

        max_k_i = maxk_pool1d(tmp, dim - 1, k)
        results.append(max_k_i)

    # construct with the batch
    results = torch.stack(results, dim=0)

    return results


def avg_pool1d_var(x, dim, lengths):
    results = []
    # assert len(lengths) == x.size(0)

    for idx in range(x.size(0)):
        # keep use all number of features
        tmp = torch.split(x[idx], split_size_or_sections=lengths[idx], dim=dim - 1)[0]
        avg_i = tmp.mean(dim - 1)

        results.append(avg_i)

    # construct with the batch
    results = torch.stack(results, dim=0)

    return results


class Maxk_Pooling_Variable(nn.Module):
    def __init__(self, dim=1, k=2):
        super(Maxk_Pooling_Variable, self).__init__()

        self.dim = dim
        self.k = k

    def forward(self, features, lengths):
        pool_weights = None
        pooled_features = maxk_pool1d_var(features, dim=self.dim, k=self.k, lengths=lengths)

        return pooled_features, pool_weights


class Avg_Pooling_Variable(nn.Module):
    def __init__(self, dim=1):
        super(Avg_Pooling_Variable, self).__init__()

        self.dim = dim

    def forward(self, features, lengths):
        pool_weights = None
        pooled_features = avg_pool1d_var(features, dim=self.dim, lengths=lengths)

        return pooled_features, pool_weights


def get_text_encoder(opt, embed_size, no_txtnorm=False):
    text_encoder = EncoderText_BERT(opt, embed_size, no_txtnorm=no_txtnorm)

    return text_encoder


def get_image_encoder(opt, img_dim, embed_size, no_imgnorm=False):
    img_enc = EncoderImageAggr(opt, img_dim, embed_size, no_imgnorm)

    return img_enc


class GatedFusion(nn.Module):
    def __init__(self, embed_dim):
        super(GatedFusion, self).__init__()
        self.fc_gate = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, feat1, feat2):
        # 将两个特征向量拼接
        concat_feat = torch.cat([feat1, feat2], dim=-1)  # (batch_size, 2 * embed_dim)

        # 通过前馈神经网络计算门控值
        gate_values = torch.sigmoid(self.fc_gate(concat_feat))  # (batch_size, embed_dim)

        # 使用门控值对两个特征向量进行加权求和
        fused_feat = gate_values * feat1 + (1 - gate_values) * feat2  # (batch_size, embed_dim)

        return fused_feat


class EncoderImageAggr(nn.Module):
    def __init__(self, opt, img_dim=2048, embed_size=1024, no_imgnorm=False):
        super(EncoderImageAggr, self).__init__()

        self.opt = opt

        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm

        # B * N * 2048 -> B * N * 1024
        # N = 36 for region features
        self.fc = nn.Linear(img_dim, self.embed_size)
        self.mlp = MLP_(img_dim, embed_size // 2, embed_size, 2)

        self.gpo = GPO(32, 32)
        self.gated_fusion = GatedFusion(embed_size)
        self.init_weights()

    def init_weights(self):
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images, teacher_aux_captions, image_lengths, graph=False):
        img_emb = self.fc(images)
        img_emb = self.mlp(images) + img_emb

        img_emb, pool_weights = self.gpo(img_emb, image_lengths)

        if not self.no_imgnorm:
            img_emb = l2norm(img_emb, dim=-1)

        # img_emb = self.gated_fusion(img_emb, teacher_aux_captions)
        img_emb = img_emb + teacher_aux_captions
        img_emb = l2norm(img_emb, dim=-1)

        return img_emb


# Language Model with BERT backbone
class EncoderText_BERT(nn.Module):
    def __init__(self, opt, embed_size=1024, no_txtnorm=False):
        super(EncoderText_BERT, self).__init__()

        self.opt = opt

        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm

        self.bert = BertModel.from_pretrained('./bert-base-uncased')

        # backbone features -> embbedings
        self.linear = nn.Linear(768, embed_size)

        self.gpo = GPO(32, 32)
        self.gated_fusion = GatedFusion(embed_size)

    def forward(self, x, teacher_captions, lengths, graph=False):
        # Embed word ids to vectors
        # pad 0 for redundant tokens in previous process
        bert_attention_mask = (x != 0).float()

        # N = max_cap_lengths, D = 768
        bert_emb = self.bert(input_ids=x, attention_mask=bert_attention_mask).last_hidden_state  # B x N x D
        cap_len = lengths

        # B x N x embed_size
        cap_emb = self.linear(bert_emb)

        cap_emb, pool_weights = self.gpo(cap_emb, cap_len.to(cap_emb.device))

        if not self.no_txtnorm:
            cap_emb = l2norm(cap_emb, dim=-1)

        # cap_emb = self.gated_fusion(cap_emb, teacher_captions)
        cap_emb = cap_emb + teacher_captions
        # cap_emb = cap_emb

        cap_emb = l2norm(cap_emb, dim=-1)

        return cap_emb


if __name__ == '__main__':
    pass
