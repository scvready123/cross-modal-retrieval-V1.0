# -----------------------------------------------------------
# Stacked Cross Attention Network implementation based on 
# https://arxiv.org/abs/1803.08024.
# "Stacked Cross Attention for Image-Text Matching"
# Kuang-Huei Lee, Xi Chen, Gang Hua, Houdong Hu, Xiaodong He
#
# Writen by Kuang-Huei Lee, 2018
# ---------------------------------------------------------------
"""SCAN model"""

import torch
import torch.nn as nn
import torch.nn.init
import torchvision.models as models
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.weight_norm import weight_norm
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm
import numpy as np
from collections import OrderedDict
from transformers import ViTModel, BertModel
import torch.nn.functional as F


def l1norm_d(X, dim, eps=1e-8):  #l1正化  ?
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps  #为什么这里要加一个eps?
    X = torch.div(X, norm)     #X/norm
    return X

def l2norm(X, dim, eps=1e-8):   #l2正则化
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

#图像采用ViT
class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.vit = ViTModel.from_pretrained("google/vit-base-patch32-384")  #加载ViT模型
        self.vision_proj = nn.Linear(768,256)

    def forward(self, images):
        out = self.vit(**images)
        features = out[0].last_hidden_state   #batch*nums*dim
        features = F.normalize(self.vision_proj(features[:,0,:]),dim=-1)

        return features  #batch*dim

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(ImageEncoder, self).load_state_dict(new_state)


# 文本采用Bert
class TextEncoder(nn.Module):
    def __init__(self):
        super(TextEncoder, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")  #加载bert
        self.text_proj = nn.Linear(768,256)

    def forward(self, captions):

        out = self.bert(**captions)
        features = out[0].last_hidden_state   #这里还是不要用pooler_output,用全部的信息
        features = F.normalize(self.text_proj(features[:,0,:]),dim=-1)

        return features  #batch*nums*dim


def func_attention(query, context, opt, smooth, eps=1e-8):
    """
    query: (n_context, queryL, d)
    context: (n_context, sourceL, d)
    """
    #print('query',query)
    #print('context',context)
    #print('query shape0',query[0])
    #print('query shape1',query[1])
    #print('query shape2',query[2])
    #print('context_shape1',context[0])
    #print('context_shape1',context[1])

    batch_size_q, queryL = query.size(0), query.size(1)
    batch_size, sourceL = context.size(0), context.size(1)


    # Get attention
    # --> (batch, d, queryL)
    #print('query',query)
    queryT = torch.transpose(query, 1, 2)  #交换tensor的两个维度
    #print('queryT',queryT)

    # (batch, sourceL, d)(batch, d, queryL)     这里做了bmm乘法
    # --> (batch, sourceL, queryL)
    attn = torch.bmm(context, queryT)
    #print('attn',attn)
    if opt.raw_feature_norm == "softmax":
        # --> (batch*sourceL, queryL)
        attn = attn.view(batch_size*sourceL, queryL)
        attn = nn.Softmax()(attn)#默认按照行进行softmax
        # --> (batch, sourceL, queryL)
        attn = attn.view(batch_size, sourceL, queryL)
    elif opt.raw_feature_norm == "l2norm":
        attn = l2norm(attn, 2)
    elif opt.raw_feature_norm == "clipped_l2norm":   #默认norm是这个
        attn = nn.LeakyReLU(0.1)(attn)
        attn = l2norm(attn, 2)
    elif opt.raw_feature_norm == "l1norm":
        attn = l1norm_d(attn, 2)
    elif opt.raw_feature_norm == "clipped_l1norm":
        attn = nn.LeakyReLU(0.1)(attn)
        attn = l1norm_d(attn, 2)
    elif opt.raw_feature_norm == "clipped":
        attn = nn.LeakyReLU(0.1)(attn)
    elif opt.raw_feature_norm == "no_norm":
        pass
    else:
        raise ValueError("unknown first norm type:", opt.raw_feature_norm)
    # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous()  #交换1，2维度,contiguous深拷贝，拷贝一份tensor,但是这两者没有联系。
    # --> (batch*queryL, sourceL)
    attn = attn.view(batch_size*queryL, sourceL)
    #print('attn',attn)
    attn = nn.Softmax()(attn*smooth)
    #print('attn',attn)
    # --> (batch, queryL, sourceL)
    attn = attn.view(batch_size, queryL, sourceL)
    # --> (batch, sourceL, queryL)
    attnT = torch.transpose(attn, 1, 2).contiguous()

    # --> (batch, d, sourceL)
    contextT = torch.transpose(context, 1, 2)
    # (batch x d x sourceL)(batch x sourceL x queryL)
    # --> (batch, d, queryL)
    weightedContext = torch.bmm(contextT, attnT)
    # --> (batch, queryL, d)
    weightedContext = torch.transpose(weightedContext, 1, 2)
    #print('weightedContext',weightedContext)
    return weightedContext, attnT


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


def xattn_score_t2i(images, captions, cap_lens, opt):
    """
    Images: (n_image, n_regions, d) matrix of images
    Captions: (n_caption, max_n_word, d) matrix of captions  #每个caption有多少个词
    CapLens: (n_caption) array of caption lengths
    """
    similarities = []
    n_image = images.size(0)
    #print('n_image',n_image)
    n_caption = captions.size(0)
    #print('n_caption',n_caption)
    for i in range(n_caption):
        # Get the i-th text description
        n_word = cap_lens[i]
        cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
        # --> (n_image, n_word, d) !!!
        cap_i_expand = cap_i.repeat(n_image, 1, 1)
        """
            word(query): (n_image, n_word, d)
            image(context): (n_image, n_regions, d)
            weiContext: (n_image, n_word, d)
            attn: (n_image, n_region, n_word)
        """
        weiContext, attn = func_attention(cap_i_expand, images, opt, smooth=opt.lambda_softmax)
        cap_i_expand = cap_i_expand.contiguous()
        weiContext = weiContext.contiguous()
        # (n_image, n_word)
        row_sim = cosine_similarity(cap_i_expand, weiContext, dim=2)
        if opt.agg_func == 'LogSumExp':   #LSE方法
            row_sim.mul_(opt.lambda_lse).exp_()
            row_sim = row_sim.sum(dim=1, keepdim=True)
            row_sim = torch.log(row_sim)/opt.lambda_lse
        elif opt.agg_func == 'Max':
            row_sim = row_sim.max(dim=1, keepdim=True)[0]
        elif opt.agg_func == 'Sum':
            row_sim = row_sim.sum(dim=1, keepdim=True)
        elif opt.agg_func == 'Mean':
            row_sim = row_sim.mean(dim=1, keepdim=True)
        else:
            raise ValueError("unknown aggfunc: {}".format(opt.agg_func))
        similarities.append(row_sim)

    # (n_image, n_caption)
    similarities = torch.cat(similarities, 1)
    
    return similarities


def xattn_score_i2t(images, captions, cap_lens, opt):
    """
    Images: (batch_size, n_regions, d) matrix of images
    Captions: (batch_size, max_n_words, d) matrix of captions
    CapLens: (batch_size) array of caption lengths
    """
    similarities = []
    n_image = images.size(0)
    n_caption = captions.size(0)
    n_region = images.size(1)
    for i in range(n_caption):
        # Get the i-th text description
        n_word = cap_lens[i]
        cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
        # (n_image, n_word, d)
        cap_i_expand = cap_i.repeat(n_image, 1, 1)
        """
            word(query): (n_image, n_word, d)
            image(context): (n_image, n_region, d)
            weiContext: (n_image, n_region, d)
            attn: (n_image, n_word, n_region)
        """
        weiContext, attn = func_attention(images, cap_i_expand, opt, smooth=opt.lambda_softmax) #query-->images context-->cap_i_expand
        # (n_image, n_region)
        row_sim = cosine_similarity(images, weiContext, dim=2)
        if opt.agg_func == 'LogSumExp':
            row_sim.mul_(opt.lambda_lse).exp_()
            row_sim = row_sim.sum(dim=1, keepdim=True)
            row_sim = torch.log(row_sim)/opt.lambda_lse
        elif opt.agg_func == 'Max':
            row_sim = row_sim.max(dim=1, keepdim=True)[0]
        elif opt.agg_func == 'Sum':
            row_sim = row_sim.sum(dim=1, keepdim=True)
        elif opt.agg_func == 'Mean':
            row_sim = row_sim.mean(dim=1, keepdim=True)
        else:
            raise ValueError("unknown aggfunc: {}".format(opt.agg_func))
        similarities.append(row_sim)

    # (n_image, n_caption)
    similarities = torch.cat(similarities, 1)
    return similarities


class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """
    def __init__(self, opt, margin=0, max_violation=True):
        super(ContrastiveLoss, self).__init__()
        self.opt = opt
        self.margin = margin
        self.max_violation = max_violation

    def forward(self, im, s, s_l):
        # compute image-sentence score matrix
        if self.opt.cross_attn == 't2i':            #如何理解这个t2i和i2t
            scores = xattn_score_t2i(im, s, s_l, self.opt)
        elif self.opt.cross_attn == 'i2t':
            scores = xattn_score_i2t(im, s, s_l, self.opt)
        else:
            #raise ValueError("unknown first norm type:", opt.raw_feature_norm)
            pass
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]
        return cost_s.sum() + cost_im.sum()

class ContrastiveLoss1(nn.Module):
    """
    Compute contrastive loss
    """
    def __init__(self, opt, margin=0, max_violation=True):
        super(ContrastiveLoss1, self).__init__()
        self.opt = opt
        self.margin = margin
        self.max_violation = max_violation

    def forward(self, im, s, s_l):
        img = F.normalize(im.mean(dim=1),dim=-1)
        cap = F.normalize(s.mean(dim=1),dim=-1)

        scores = img @ cap.t()

        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]
        return cost_s.sum() + cost_im.sum()

class ContrastiveLoss2(nn.Module):  #batch*dim
    """
    Compute contrastive loss
    """
    def __init__(self, opt, margin=0, max_violation=True):
        super(ContrastiveLoss2, self).__init__()
        self.opt = opt
        self.margin = margin
        self.max_violation = max_violation

    def forward(self, im, s, s_l):

        scores = im @ s.t()

        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]
        return cost_s.sum() + cost_im.sum()

class ContrastiveLoss3(nn.Module):  #batch*dim
    """
    Compute contrastive loss
    """
    def __init__(self, opt, margin=0, max_violation=True):
        super(ContrastiveLoss3, self).__init__()
        self.opt = opt
        self.margin = margin
        self.max_violation = max_violation

    def forward(self, im, s, s_l):

        batch_num = im.shape[0] 
        L= []
        for i in range(batch_num):
            L.append([i])
        idx = torch.tensor(L)

        idx = idx.view(-1,1) #batch*1
        idx_all = idx.t()  
        pos_idx = torch.eq(idx, idx_all).float()      
        sim_targets = pos_idx / pos_idx.sum(1,keepdim=True)
        sim_targets = sim_targets.cuda()

        scores = im @ s.t()

        loss_i2t = -torch.sum(F.log_softmax(scores, dim=1)*sim_targets,dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(scores.t(), dim=1)*sim_targets,dim=1).mean() 
        loss_ita = (loss_i2t+loss_t2i)/2

        return loss_ita


class VAB(object):
    """
    Stacked Cross Attention Network (SCAN) model
    """
    def __init__(self, opt):
        # Build Models
        self.grad_clip = opt.grad_clip
        self.img_enc = ImageEncoder()
        self.txt_enc = TextEncoder()
        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.txt_enc.cuda()
            cudnn.benchmark = True

        # Loss and Optimizer
        self.criterion = ContrastiveLoss3(opt=opt,
                                         margin=opt.margin,
                                         max_violation=opt.max_violation)
        params = list(self.txt_enc.parameters())
        params += list(self.img_enc.parameters())
        params = list(filter(lambda p: p.requires_grad, params))

        self.params = params

        self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)

        self.Eiters = 0

    def state_dict(self):
        state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0])
        self.txt_enc.load_state_dict(state_dict[1])

    def train_start(self):
        """switch to train mode
        """
        self.img_enc.train()
        self.txt_enc.train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.img_enc.eval()
        self.txt_enc.eval()

    def forward_emb(self, images, captions):
        """Compute the image and caption embeddings
        """
        # Forward
        for key in images.keys():
            images[key] = images[key].cuda()
        for key in captions.keys():
            captions[key] = captions[key].cuda()

        img_emb = self.img_enc(images)
        # img_emb: batch*img_nums*dim
        cap_emb = self.txt_enc(captions)
        # cap_emb: batch*cap_nums*dim
        return img_emb, cap_emb

    def forward_loss(self, img_emb, cap_emb, cap_len, **kwargs):
        """Compute the loss given pairs of image and caption embeddings
        """
        loss = self.criterion(img_emb, cap_emb, cap_len)
        self.logger.update('Le', loss.item(), img_emb.size(0))
        return loss

    def train_emb(self, images, captions, lengths, ids=None, *args):
        """One training step given images and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # compute the embeddings
        img_emb, cap_emb = self.forward_emb(images, captions)

        # measure accuracy and record loss
        self.optimizer.zero_grad()
        loss = self.forward_loss(img_emb, cap_emb, lengths)

        # compute gradient and do SGD step
        loss.backward()
        if self.grad_clip > 0:    #梯度裁剪
            clip_grad_norm(self.params, self.grad_clip)
        self.optimizer.step()
