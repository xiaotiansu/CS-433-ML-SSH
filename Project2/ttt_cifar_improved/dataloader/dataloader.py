import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from wilds.datasets.iwildcam_dataset import IWildCamDataset

from utils.prepare_dataset import prepare_transforms, TwoCropTransform, common_corruptions, seed_worker
# from .datasets import GeneralWilds_Batched_Dataset

# IMG_HEIGHT = 224
# NUM_CLASSES = 186
tesize = 10000

class IWildData():
    def __init__(self, args):
        self.dataset = IWildCamDataset(root_dir=os.path.join(args.data_dir, 'wilds'), download=True)
        self.tr_transforms, self.te_transforms, self.simclr_transforms = prepare_transforms(args.dataset)

    def get_train_dataloader(self, args, num_sample=None):
        if hasattr(args, 'ssl') and args.ssl == 'contrastive':
            train_data = self.dataset.get_subset('train', transform=TwoCropTransform(self.tr_transforms))
            if hasattr(args, 'corruption') and args.corruption in common_corruptions:
                print('Contrastive on %s level %d' %(args.corruption, args.level))
                tesize = 10000
                trset_raw = np.load(args.dataroot + '/iwildcam/%s.npy' %(args.corruption))
                trset_raw = trset_raw[(args.level-1)*tesize: args.level*tesize]
                train_data.data = trset_raw
        else:
            train_data = self.dataset.get_subset('train', transform=self.tr_transforms)

        if not hasattr(args, 'workers') or args.workers < 2:
            pin_memory = False
        else:
            pin_memory = True

        if num_sample and num_sample < train_data.data.shape[0]:
            train_data.data = train_data.data[:num_sample]
            print("Truncate the training set to {:d} samples".format(num_sample))

        trloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
                                               shuffle=True, num_workers=args.workers,
                                               worker_init_fn=seed_worker, pin_memory=pin_memory, drop_last=True)
        return train_data, trloader

    def get_test_dataloader(self, args, ttt=False, num_sample=None):
        if not hasattr(args, 'corruption') or args.corruption == 'original':
            print('Test on the original test set')
            teset = self.dataset.get_subset('test', transform=self.te_transforms)

        elif args.corruption in common_corruptions:
            print('Test on %s level %d' % (args.corruption, args.level))
            teset_raw = np.load(args.dataroot + '/iwildcam/%s.npy' % (args.corruption))
            teset_raw = teset_raw[(args.level - 1) * tesize: args.level * tesize]
            teset = self.dataset.get_subset('test', transform=self.te_transforms)
            teset.data = teset_raw
        else:
            raise Exception('Corruption not found!')

        if not hasattr(args, 'workers') or args.workers < 2:
            pin_memory = False
        else:
            pin_memory = True

        if ttt:
            shuffle = True
            drop_last = True
        else:
            shuffle = True
            drop_last = False

        if num_sample and num_sample < teset.data.shape[0]:
            teset.data = teset.data[:num_sample]
            print("Truncate the test set to {:d} samples".format(num_sample))

        teloader = torch.utils.data.DataLoader(teset, batch_size=args.batch_size,
                                               shuffle=shuffle, num_workers=args.workers,
                                               worker_init_fn=seed_worker, pin_memory=pin_memory, drop_last=drop_last)
        return teset, teloader