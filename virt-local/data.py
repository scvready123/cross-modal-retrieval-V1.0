import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import nltk
from PIL import Image
from pycocotools.coco import COCO
import numpy as np
import json as jsonmod
import random
import numpy as np
import time
import pickle
from transformers import BertTokenizerFast, ViTFeatureExtractor

# convert_to_feature
# 输入 原始句子，长度，tokenizer
# 返回tokoen,以及对应的id

feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch32-384")
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

def get_paths(path, name='f30k', use_restval=False):
    """
    Returns paths to images and annotations for the given datasets. For MSCOCO
    indices are also returned to control the data split being used.
    The indices are extracted from the Karpathy et al. splits using this
    snippet:

    >>> import json
    >>> dataset=json.load(open('dataset_coco.json','r'))
    >>> A=[]
    >>> for i in range(len(D['images'])):
    ...   if D['images'][i]['split'] == 'val':
    ...     A+=D['images'][i]['sentids'][:5]
    ...

    :param name: Dataset names
    :param use_restval: If True, the the `restval` data is included in train.
    """
    roots = {}
    ids = {}
    if 'coco' == name:
        imgdir = os.path.join(path, 'images')
        capdir = os.path.join(path, 'annotations')
        roots['train'] = {
            'img': os.path.join(imgdir, 'train2014'),
            'cap': os.path.join(capdir, 'captions_train2014.json')
        }
        roots['val'] = {
            'img': os.path.join(imgdir, 'val2014'),
            'cap': os.path.join(capdir, 'captions_val2014.json')
        }
        roots['test'] = {
            'img': os.path.join(imgdir, 'val2014'),
            'cap': os.path.join(capdir, 'captions_val2014.json')
        }
        roots['trainrestval'] = {
            'img': (roots['train']['img'], roots['val']['img']),
            'cap': (roots['train']['cap'], roots['val']['cap'])
        }
        ids['train'] = np.load(os.path.join(capdir, 'coco_train_ids.npy'))
        ids['val'] = np.load(os.path.join(capdir, 'coco_dev_ids.npy'))[:5000]
        ids['test'] = np.load(os.path.join(capdir, 'coco_test_ids.npy'))
        ids['trainrestval'] = (
            ids['train'],
            np.load(os.path.join(capdir, 'coco_restval_ids.npy')))
        if use_restval:
            roots['train'] = roots['trainrestval']
            ids['train'] = ids['trainrestval']
    elif 'f8k' == name:
        imgdir = os.path.join(path, 'images')
        cap = os.path.join(path, 'dataset_flickr8k.json')
        roots['train'] = {'img': imgdir, 'cap': cap}
        roots['val'] = {'img': imgdir, 'cap': cap}
        roots['test'] = {'img': imgdir, 'cap': cap}
        ids = {'train': None, 'val': None, 'test': None}
    elif 'f30k' == name:
        imgdir = os.path.join(path, 'images')
        cap = os.path.join(path, 'dataset_flickr30k.json')
        roots['train'] = {'img': imgdir, 'cap': cap}
        roots['val'] = {'img': imgdir, 'cap': cap}
        roots['test'] = {'img': imgdir, 'cap': cap}
        ids = {'train': None, 'val': None, 'test': None}

    return roots, ids


class CocoDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(self, root, json, vocab, transform=None, ids=None):
        """
        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: transformer for image.
        """
        self.root = root
        # when using `restval`, two json files are needed
        if isinstance(json, tuple):
            self.coco = (COCO(json[0]), COCO(json[1]))
        else:
            self.coco = (COCO(json),)
            self.root = (root,)
        # if ids provided by get_paths, use split-specific ids
        if ids is None:
            self.ids = list(self.coco.anns.keys())
        else:
            self.ids = ids

        # if `restval` data is to be used, record the break point for ids
        if isinstance(self.ids, tuple):
            self.bp = len(self.ids[0])
            self.ids = list(self.ids[0]) + list(self.ids[1])
        else:
            self.bp = len(self.ids)
        self.vocab = vocab
        self.transform = transform

    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        """
        vocab = self.vocab
        root, caption, img_id, path, image = self.get_raw_item(index)

        if self.transform is not None:
            image = self.transform(image)

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(
            str(caption).lower().decode('utf-8'))
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return image, target, index, img_id

    def get_raw_item(self, index):
        if index < self.bp:
            coco = self.coco[0]
            root = self.root[0]
        else:
            coco = self.coco[1]
            root = self.root[1]
        ann_id = self.ids[index]
        caption = coco.anns[ann_id]['caption']
        img_id = coco.anns[ann_id]['image_id']
        path = coco.loadImgs(img_id)[0]['file_name']
        image = Image.open(os.path.join(root, path)).convert('RGB')

        return root, caption, img_id, path, image

    def __len__(self):
        return len(self.ids)


class FlickrDataset(data.Dataset):
    """
    Dataset loader for Flickr30k and Flickr8k full datasets.
    """

    def __init__(self, root, json, split):
        self.root = root
        self.split = split
        self.dataset = jsonmod.load(open(json, 'r'))['images']
        self.ids = []
        for i, d in enumerate(self.dataset):
            if d['split'] == split:
                self.ids += [(i, x) for x in range(len(d['sentences']))]
        self.length = len(self.ids)
        if split == 'val':       #验证集5000个句子
            self.length = 5000

    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        """
        #vocab = self.vocab
        root = self.root
        ann_id = self.ids[index]
        img_id = ann_id[0]
        caption = self.dataset[img_id]['sentences'][ann_id[1]]['raw']
        path = self.dataset[img_id]['filename']

        image = Image.open(os.path.join(root, path))
        image = image.resize((384, 384))

        return image, caption, index, img_id

    def __len__(self):
        return self.length

def collate_fn(data):
    images, captions, ids, img_ids = zip(*data)

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    img_list = []
    cap_list = []
    lengths =[]

    # Merget captions & images(put them into a list)
    for img in images:
        img_list.append(img)
    images = feature_extractor(images=img_list, return_tensors="pt")  #images: type:tensor ,size: batch*C*H*W

    for cap in captions:
        cap_list.append(cap)
    captions = tokenizer(cap_list, return_tensors="pt", padding=True)  #captions: type:dict 三个部分：input_id,token_ids,att_mask

    if captions['input_ids'].shape[1] > 40:
        captions['input_ids'] = captions['input_ids'][:,:40].type(torch.int64)
        captions['token_type_ids'] = captions['token_type_ids'][:,:40].type(torch.int64)
        captions['attention_mask'] = captions['attention_mask'][:,:40].type(torch.int64)
    else:
        pass

    for i in range(len(captions['attention_mask'])):
        length = int(captions['attention_mask'][i].sum())
        lengths.append(length)

    return images, captions, lengths, ids

def get_loaders(data_name, split, root, json,
                      batch_size=100, shuffle=True,
                      num_workers=2, collate_fn=collate_fn):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    if 'coco' in data_name:
        '''
        # COCO custom dataset
        dataset = CocoDataset(root=root,
                              json=json,
                              vocab=None,
                              transform=transform, ids=ids)
        '''
        pass

    elif 'f8k' in data_name or 'f30k' in data_name:
        dataset = FlickrDataset(root=root, split=split, json=json)

    # Data loader
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=True,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader
