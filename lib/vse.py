import arguments
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init
from torch.nn.utils import clip_grad_norm_

from lib.encoders import get_image_encoder, get_text_encoder
from lib.loss import *

logger = logging.getLogger(__name__)


class VSEModel(object):
    def __init__(self, opt, eval=False):

        self.opt = opt
        self.grad_clip = opt.grad_clip

        self.img_enc = get_image_encoder(opt, opt.img_dim, opt.embed_size, no_imgnorm=opt.no_imgnorm)
        self.txt_enc = get_text_encoder(opt, opt.embed_size, no_txtnorm=opt.no_txtnorm)

        self.criterion = loss_select(opt, loss_type=opt.loss_type)
        self.proto_criterion = ProtoContrastiveLoss(opt)

        # Set up the lr for different parts of the VSE model
        decay_factor = 1e-4

        self.params = list(self.txt_enc.parameters()) + list(self.img_enc.parameters())

        all_text_params = list(self.txt_enc.parameters())
        bert_params = list(self.txt_enc.bert.parameters())

        # Tensor.data_ptr() → int, Returns the address of the first element
        bert_params_ptr = [p.data_ptr() for p in bert_params]
        text_params_no_bert = list()

        # select other parameters except BERT
        for p in all_text_params:
            if p.data_ptr() not in bert_params_ptr:
                text_params_no_bert.append(p)

        self.optimizer = torch.optim.AdamW([
            {'params': text_params_no_bert, 'lr': opt.learning_rate},
            # {'params': bert_params, 'lr': opt.learning_rate * 0.1},
            {'params': bert_params, 'lr': opt.learning_rate * 0.05},

            {'params': self.img_enc.parameters(), 'lr': opt.learning_rate},
        ],
            lr=opt.learning_rate, weight_decay=decay_factor)

        # iteration
        self.Eiters = 0
        self.data_parallel = False

        # use the gpu
        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.txt_enc.cuda()
            torch.backends.cudnn.benchmark = True

    def set_max_violation(self, max_violation=True):
        if self.opt.loss_type == 'vse':
            if max_violation:
                self.criterion.max_violation_on()
            else:
                self.criterion.max_violation_off()

    def state_dict(self):
        state_dict = [
            self.img_enc.state_dict(),
            self.txt_enc.state_dict(),
        ]
        return state_dict

    def load_state_dict(self, state_dict, ):
        # strict=True, ensure keys match
        self.img_enc.load_state_dict(state_dict[0], strict=True)

        # Unexpected key(s) in state_dict: "bert.embeddings.position_ids". 
        # incompatible problem of transformers package version 
        self.txt_enc.load_state_dict(state_dict[1], strict=False)

    def train_start(self):
        self.img_enc.train()
        self.txt_enc.train()

    def val_start(self):
        self.img_enc.eval()
        self.txt_enc.eval()

    def make_data_parallel(self):
        self.img_enc = nn.DataParallel(self.img_enc)
        self.txt_enc = nn.DataParallel(self.txt_enc)
        self.data_parallel = True
        logger.info('Image/Text encoder is data paralleled (use multi GPUs).')

    @property
    def is_data_parallel(self):
        return self.data_parallel

    # Compute the image and caption embeddings
    def forward_emb(self, images, captions, teacher_aux_captions, teacher_captions, img_ids, lengths,
                    image_lengths=None):

        # compute images embs
        images = images.cuda()
        bge_aux_cap_emb = teacher_aux_captions.cuda()
        image_lengths = image_lengths.cuda()
        img_emb = self.img_enc(images, bge_aux_cap_emb, image_lengths)

        # compute caption embs
        captions = captions.cuda()
        bge_cap_emb = teacher_captions.cuda()
        lengths = lengths.cuda()
        cap_emb = self.txt_enc(captions, bge_cap_emb, lengths)

        return img_emb, cap_emb

    # One training step given images and captions
    def train_emb(self, images, captions, teacher_aux_captions, teacher_captions, opt, lengths, image_lengths=None,
                  img_ids=None):

        self.Eiters += 1
        self.logger.update('Iter', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # compute images embs
        images = images.cuda()
        bge_aux_cap_emb = teacher_aux_captions.cuda()
        image_lengths = image_lengths.cuda()
        img_emb = self.img_enc(images, bge_aux_cap_emb, image_lengths, graph=True)

        # compute caption embs
        captions = captions.cuda()
        bge_cap_emb = teacher_captions.cuda()
        lengths = lengths.cuda()
        cap_emb = self.txt_enc(captions, bge_cap_emb, lengths, graph=True)

        self.optimizer.zero_grad()

        # compute loss
        proto_loss = self.proto_criterion(img_emb, cap_emb)
        loss = self.criterion(img_emb, cap_emb, img_ids=img_ids) + proto_loss

        print('proto_loss', proto_loss.item())
        print('Loss', loss.item())
        self.logger.update('Loss', loss.item(), self.opt.batch_size)

        # compute gradient and update
        if torch.isnan(loss):
            logger.error("We have NaN numbers, ")
            return 0.

        loss.backward()

        if self.grad_clip > 0:
            clip_grad_norm_(self.params, self.grad_clip)

        self.optimizer.step()


if __name__ == '__main__':
    pass
