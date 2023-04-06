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


def l1norm_d(X, dim, eps=1e-8):  # l1正化  ?
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps  # 为什么这里要加一个eps?
    X = torch.div(X, norm)  # X/norm
    return X


def l2norm(X, dim, eps=1e-8):  # l2正则化
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def EncoderImage(data_name, img_dim, embed_size, precomp_enc_type='basic',
                 no_imgnorm=False):
    """A wrapper to image encoders. Chooses between an different encoders
    that uses precomputed image features.
    """
    if precomp_enc_type == 'basic':
        img_enc = EncoderImagePrecomp(
            img_dim, embed_size, no_imgnorm)
    elif precomp_enc_type == 'weight_norm':
        img_enc = EncoderImageWeightNormPrecomp(
            img_dim, embed_size, no_imgnorm)
    else:
        raise ValueError("Unknown precomp_enc_type: {}".format(precomp_enc_type))

    return img_enc


class EncoderImagePrecomp(nn.Module):

    def __init__(self, img_dim, embed_size, no_imgnorm=False):
        super(EncoderImagePrecomp, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.fc = nn.Linear(img_dim, embed_size)

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized

        features = self.fc(images)

        # normalize in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features, dim=-1)

        return features

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderImagePrecomp, self).load_state_dict(new_state)


class EncoderImageWeightNormPrecomp(nn.Module):

    def __init__(self, img_dim, embed_size, no_imgnorm=False):
        super(EncoderImageWeightNormPrecomp, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.fc = weight_norm(nn.Linear(img_dim, embed_size), dim=None)

    def forward(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized

        features = self.fc(images)

        # normalize in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features, dim=-1)

        return features

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderImageWeightNormPrecomp, self).load_state_dict(new_state)


# RNN Based Language Model
class EncoderText(nn.Module):

    def __init__(self, vocab_size, word_dim, embed_size, num_layers,
                 use_bi_gru=False, no_txtnorm=False):
        super(EncoderText, self).__init__()
        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm

        # word embedding
        self.embed = nn.Embedding(vocab_size, word_dim)

        # caption embedding
        self.use_bi_gru = use_bi_gru
        self.rnn = nn.GRU(word_dim, embed_size, num_layers, batch_first=True, bidirectional=use_bi_gru)

        self.init_weights()

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x, lengths):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        # print('x',x)
        # print('lengths',lengths)
        x = self.embed(x)
        packed = pack_padded_sequence(x, lengths, batch_first=True)

        # Forward propagate RNN
        out, _ = self.rnn(packed)

        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)
        cap_emb, cap_len = padded

        if self.use_bi_gru:
            cap_emb = (cap_emb[:, :, :int(cap_emb.size(2) / 2)] + cap_emb[:, :,
                                                                  int(cap_emb.size(2) / 2):]) / 2  # 组合双向GRU的特征

        # normalization in the joint embedding space
        if not self.no_txtnorm:
            cap_emb = l2norm(cap_emb, dim=-1)
        # print('cap_emb',cap_emb)
        print('cap_len', cap_len)

        return cap_emb, cap_len


def func_attention(query, context, opt, smooth, eps=1e-8):
    """
    query: (n_context, queryL, d)
    context: (n_context, sourceL, d)
    """
    # print('query',query)
    # print('context',context)
    # print('query shape0',query[0])
    # print('query shape1',query[1])
    # print('query shape2',query[2])
    # print('context_shape1',context[0])
    # print('context_shape1',context[1])

    batch_size_q, queryL = query.size(0), query.size(1)
    batch_size, sourceL = context.size(0), context.size(1)

    # Get attention
    # --> (batch, d, queryL)
    # print('query',query)
    queryT = torch.transpose(query, 1, 2)  # 交换tensor的两个维度
    # print('queryT',queryT)

    # (batch, sourceL, d)(batch, d, queryL)     这里做了bmm乘法
    # --> (batch, sourceL, queryL)
    attn = torch.bmm(context, queryT)
    # print('attn',attn)
    if opt.raw_feature_norm == "softmax":
        # --> (batch*sourceL, queryL)
        attn = attn.view(batch_size * sourceL, queryL)
        attn = nn.Softmax()(attn)  # 默认按照行进行softmax
        # --> (batch, sourceL, queryL)
        attn = attn.view(batch_size, sourceL, queryL)
    elif opt.raw_feature_norm == "l2norm":
        attn = l2norm(attn, 2)
    elif opt.raw_feature_norm == "clipped_l2norm":  # 默认norm是这个
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
    attn = torch.transpose(attn, 1, 2).contiguous()  # 交换1，2维度,contiguous深拷贝，拷贝一份tensor,但是这两者没有联系。
    # --> (batch*queryL, sourceL)
    attn = attn.view(batch_size * queryL, sourceL)
    # print('attn',attn)
    attn = nn.Softmax()(attn * smooth)
    # print('attn',attn)
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
    # print('weightedContext',weightedContext)
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
    # print('n_image',n_image)
    n_caption = captions.size(0)
    # print('n_caption',n_caption)
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
        if opt.agg_func == 'LogSumExp':  # LSE方法
            row_sim.mul_(opt.lambda_lse).exp_()
            row_sim = row_sim.sum(dim=1, keepdim=True)
            row_sim = torch.log(row_sim) / opt.lambda_lse
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
        weiContext, attn = func_attention(images, cap_i_expand, opt,
                                          smooth=opt.lambda_softmax)  # query-->images context-->cap_i_expand
        # (n_image, n_region)
        row_sim = cosine_similarity(images, weiContext, dim=2)
        if opt.agg_func == 'LogSumExp':
            row_sim.mul_(opt.lambda_lse).exp_()
            row_sim = row_sim.sum(dim=1, keepdim=True)
            row_sim = torch.log(row_sim) / opt.lambda_lse
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
        # compute image-sentence score matrix
        if self.opt.direction == 'i2t':  # 如何理解这个t2i和i2t,SCAN里面的这两个和DIME是反过来的
            scores = xattn_score_t2i(im, s, s_l, self.opt)
        elif self.opt.direction == 't2i':
            scores = xattn_score_i2t(im, s, s_l, self.opt)
        else:
            # raise ValueError("unknown first norm type:", opt.raw_feature_norm)
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

def calscores(images, captions):
    """
        Images: (n_image, n_regions, d) matrix of images
        Captions: (n_caption, max_n_word, d) matrix of captions  #每个caption有多少个词
        CapLens: (n_caption) array of caption lengths
    """
    avg_img = l2norm(images.mean(1),dim=1)
    avg_caps = l2norm(captions.mean(1),dim=1)
    similarities = torch.einsum("id,jd->ij",avg_img,avg_caps)

    return similarities


class feature_ContrastiveLoss1(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, opt, margin=0, max_violation=True):
        super(feature_ContrastiveLoss1, self).__init__()
        self.opt = opt
        self.margin = margin
        self.max_violation = max_violation

    def forward(self, im, s, s_l):
        # compute image-sentence score matrix
        scores = calscores(im, s)

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

