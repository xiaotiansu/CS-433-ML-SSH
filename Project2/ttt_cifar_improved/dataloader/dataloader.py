import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from wilds.datasets.iwildcam_dataset import IWildCamDataset

from dataloader.datasets import GeneralWilds_Batched_Dataset
from utils.prepare_dataset import prepare_transforms, TwoCropTransform, common_corruptions, seed_worker
# from .datasets import GeneralWilds_Batched_Dataset

# IMG_HEIGHT = 224
# NUM_CLASSES = 186
tesize = 10000

class IWildData():
    def __init__(self, args):
        self.dataset = IWildCamDataset(root_dir=os.path.join(args.dataroot, 'wilds'), download=True)
        # self.tr_transforms, self.te_transforms, self.simclr_transforms = prepare_transforms(args.dataset)
        self.tr_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            # transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        self.te_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        self.simclr_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def get_train_dataloader(self, args, num_sample=None):
        if hasattr(args, 'ssl') and args.ssl == 'contrastive':
            train_data = self.dataset.get_subset('train', frac = 1, transform=TwoCropTransform(self.simclr_transforms))
            train_sets = GeneralWilds_Batched_Dataset(train_data, args.batch_size, domain_idx=0)
            # if hasattr(args, 'corruption') and args.corruption in common_corruptions:
            #     print('Contrastive on %s level %d' %(args.corruption, args.level))
            #     tesize = 10000
            #     trset_raw = np.load(args.dataroot + '/iwildcam/%s.npy' %(args.corruption))
            #     trset_raw = trset_raw[(args.level-1)*tesize: args.level*tesize]
            #     train_data.data = trset_raw
        else:
            train_data = self.dataset.get_subset('train', frac = 1, transform=self.tr_transforms)
            train_sets = GeneralWilds_Batched_Dataset(train_data, args.batch_size, domain_idx=0)

        if not hasattr(args, 'workers') or args.workers < 2:
            pin_memory = False
        else:
            pin_memory = True

        # if num_sample and num_sample < train_data.data.shape[0]:
        #     train_data.data = train_data.data[:num_sample]
        #     print("Truncate the training set to {:d} samples".format(num_sample))
        print("here2")
        trloader = torch.utils.data.DataLoader(train_sets, batch_size=args.batch_size,
                                               shuffle=True, num_workers=args.workers,
                                               worker_init_fn=seed_worker, pin_memory=pin_memory, drop_last=False)
        return train_data, trloader

    def get_test_dataloader(self, args, ttt=False, num_sample=None):
        # if not hasattr(args, 'corruption') or args.corruption == 'original':
        #     print('Test on the original test set')
        test_data = self.dataset.get_subset('test', frac = 0.3, transform=self.te_transforms)
        teset = GeneralWilds_Batched_Dataset(test_data, args.batch_size, domain_idx=0)

        # elif args.corruption in common_corruptions:
        #     print('Test on %s level %d' % (args.corruption, args.level))
        #     teset_raw = np.load(args.dataroot + '/iwildcam/%s.npy' % (args.corruption))
        #     teset_raw = teset_raw[(args.level - 1) * tesize: args.level * tesize]
        #     teset = self.dataset.get_subset('test', frac = 0.01, transform=self.te_transforms)
        #     teset.data = teset_raw
        # else:
        #     raise Exception('Corruption not found!')

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

        # if num_sample and num_sample < teset.data.shape[0]:
        #     teset.data = teset.data[:num_sample]
        #     print("Truncate the test set to {:d} samples".format(num_sample))

        teloader = torch.utils.data.DataLoader(teset, batch_size=args.batch_size,
                                               shuffle=shuffle, num_workers=args.workers,
                                               worker_init_fn=seed_worker, pin_memory=pin_memory, drop_last=drop_last)
        return teset, teloader