import pickle
import os
import time
import shutil

import torch
import os 
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
totalsavecandi = 400
inf = 9999
def main():
    # img_embstrain: (n,2048) image features predicted by first-round VSRN
    # caps_embstrain: (5n,2048) caption features predicted by first-round VSRN    
    img_embs = np.load('/data2/lihaoxuan/DIME/two-stage-1-triplet/runs/image.npy')
    cap_embs = np.load('/data2/lihaoxuan/DIME/two-stage-1-triplet/runs/text.npy')
    
    print(np.shape(img_embs), np.shape(cap_embs))
    img_embs = torch.from_numpy(img_embs).half().cuda()
    cap_embs = torch.from_numpy(cap_embs).half().cuda()

    
    t2ihn = np.zeros((len(cap_embs),totalsavecandi)).astype('int32')
    for i in range(len(cap_embs)):
        cap = cap_embs[i:i+1]
        simi = (img_embs * cap).sum(1)
        if i%50==0:
            scoret, topt = torch.sort(simi,descending=True)
            topt = topt.cpu().numpy().copy()
            print(i, np.where(topt==(i/5))) 
        simi[i//5] = -inf
        score, top = torch.topk(simi,totalsavecandi)
        t2ihn[i] = top.cpu().numpy().copy().astype('int32')

    np.save('/data2/lihaoxuan/DIME/two-stage-1-triplet/runs/t2i_flickr.npy',t2ihn)
    print(t2ihn[0])


if __name__ == '__main__':
    main()
