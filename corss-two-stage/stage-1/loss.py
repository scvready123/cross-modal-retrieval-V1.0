from torch import nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np


class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """
    def __init__(self, opt, margin=0, measure=False, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.opt = opt
        self.max_violation = max_violation
    
    def forward(self, scores):
        if self.opt.extra_stc > 0:
            return self.forward_extraStc(scores)
        elif self.opt.extra_img > 0:
            return self.forward_extraImg(scores)
        else:
            return self.forward_(scores)

    def forward_(self, scores):
        # compute image-sentence score matrix
        diagonal = scores.diag().view(scores.size(0), 1)
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

    def forward_extraStc(self, scores):
        n_img, n_stc = scores.size()
        # compute image-sentence score matrix
        diagonal = scores.diag().view(scores.size(0), 1) 
        d1 = diagonal 
        d2 = diagonal.t() 

        #d1 = diagonal.expand_as(scores)
        #d2 = diagonal.t().expand_as(scores[:, :n_img])  same as above

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0) 
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores[:, :n_img] - d2).clamp(min=0)  

        print(cost_s)
        print(cost_im)

        assert 1==0

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5  
        if torch.cuda.is_available():
            I = mask.cuda()
            extra_zeros = torch.zeros((n_img, n_stc - n_img)).byte().cuda()


        mask_s = torch.cat([I, extra_zeros], dim=1) > .5
        
        cost_s = cost_s.masked_fill_(mask_s, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return cost_s.sum() + cost_im.sum()

    def forward_extraImg(self, scores):
        n_img, n_stc = scores.size()
        # compute image-sentence score matrix
        diagonal = scores.diag().view(scores.size(1), 1)   
        d1 = diagonal 
        d2 = diagonal.t() 

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores[:n_stc, :] - d1).clamp(min=0)  
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)   

        # clear diagonals
        mask = torch.eye(scores.size(1)) > .5 
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
            extra_zeros = torch.zeros((n_img - n_stc, n_stc)).byte().cuda()

        mask_im = torch.cat([I, extra_zeros], dim=0)
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(mask_im, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            # print(cost_im.max(0)[1])
            cost_im = cost_im.max(0)[0]
            
        return cost_s.sum() + cost_im.sum()


class InfoNCELoss(nn.Module):
    """
    Compute triplet loss
    """

    def __init__(self, margin=0, max_violation=False):
        super(InfoNCELoss, self).__init__()
        self.margin = margin
        self.max_violation = max_violation
        self.cri = nn.CrossEntropyLoss(reduction='sum')

    def forward(self, scores, scores_img, scores_txt):
        '''# compute image-sentence score matrix
        
        # 0.1 is temperature parameter
        scores = torch.div(scores, 0.1)

        diagonal = scores.diag().view(scores.size(0), 1)
        
        
        logits_0 = torch.cat([diagonal, scores], dim=1)
        s_T = scores.t()

        
        logits_1 = torch.cat([diagonal, s_T], dim=1)
        labels = torch.zeros(logits_0.shape[0], dtype=torch.long).cuda()
        
        # logits_0.shape [b, b+1]

        cost_s = self.cri(logits_0, labels)
        cost_im = self.cri(logits_1, labels)

        #cost_s.shape cost_im.shape scale'''

        # compute image-sentence score matrix
        
        # 0.1 is temperature parameter
        scores = torch.div(scores, 0.1)
        scores_img = torch.div(scores_img, 0.1)
        scores_txt = torch.div(scores_txt, 0.1)
        
        #print(scores.shape)
        #print(scores_img.shape)
        #print(scores_txt.shape)        all [b, b]
        

        assert scores_img.shape == scores_txt.shape, 'wrong in loss 112'

        mask_diag = torch.ones_like(scores_txt)

        #print(mask_diag.shape) [b, b]

        for i in range(mask_diag.size(0)):
            mask_diag[i, i] = 0

        scores_img_diag = torch.masked_select(scores_img, mask_diag==1).view(scores.size(0), -1)
        scores_txt_diag = torch.masked_select(scores_txt, mask_diag==1).view(scores.size(0), -1)

        #print(scores_img_diag.shape)
        #print(scores_txt_diag.shape) all [b, b-1]

        #print()

        diagonal = scores.diag().view(scores.size(0), 1)

        logits_0 = torch.cat([diagonal, scores, scores_img_diag], dim=1)
        
        s_T = scores.t()
        logits_1 = torch.cat([diagonal, s_T, scores_txt_diag], dim=1)
        
        labels = torch.zeros(logits_0.shape[0], dtype=torch.long).cuda()
        
        # logits_0.shape [b, b+1]

        cost_s = self.cri(logits_0, labels)
        cost_im = self.cri(logits_1, labels)

        #cost_s.shape cost_im.shape scale

        return cost_s.sum() + cost_im.sum()
