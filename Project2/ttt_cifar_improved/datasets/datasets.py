import copy
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

class CivilComments_Batched_Dataset(Dataset):
    """
    Batched dataset for CivilComments. Allows for getting a batch of data given
    a specific domain index.
    """
    def __init__(self, train_data, batch_size=16):
        self.num_envs = 9 # civilcomments dataset has 8 attributes, plus 1 blank (no attribute)
        meta = torch.nonzero(train_data.metadata_array[:, :8] == 1)
        indices, domains = meta[:, 0],  meta[:, 1]
        blank_indices = torch.nonzero(train_data.metadata_array[:, :8].sum(-1) == 0).squeeze()
        self.domain_indices = [blank_indices] + [indices[domains == d] for d in domains.unique()]
        domain_indices_by_group = []
        for d_idx in self.domain_indices:
            domain_indices_by_group.append(d_idx[train_data.metadata_array[d_idx][:, -1]==0])
            domain_indices_by_group.append(d_idx[train_data.metadata_array[d_idx][:, -1]==1])
        self.domain_indices = domain_indices_by_group

        train_data._text_array = [train_data.dataset._text_array[i] for i in train_data.indices]
        self.metadata_array = train_data.metadata_array
        self.y_array = train_data.y_array
        self.data = train_data._text_array

        self.eval = train_data.eval
        self.collate = train_data.collate
        self.metadata_fields = train_data.metadata_fields
        self.data_dir = train_data.data_dir
        self.transform = train_data.transform

        self.data = train_data._text_array
        self.targets = self.y_array
        self.domains = self.metadata_array[:, :8]
        self.batch_size = batch_size

    def reset_batch(self):
        """Reset batch indices for each domain."""
        self.batch_indices, self.batches_left = {}, {}
        for loc, d_idx in enumerate(self.domain_indices):
            self.batch_indices[loc] = torch.split(d_idx[torch.randperm(len(d_idx))], self.batch_size)
            self.batches_left[loc] = len(self.batch_indices[loc])

    def get_batch(self, domain):
        """Return the next batch of the specified domain."""
        batch_index = self.batch_indices[domain][len(self.batch_indices[domain]) - self.batches_left[domain]]
        return torch.stack([self.transform(self.get_input(i)) for i in batch_index]), \
               self.targets[batch_index], self.domains[batch_index]

    def get_input(self, idx):
        """Returns x for a given idx."""
        return self.data[idx]

    def __getitem__(self, idx):
        return self.transform(self.get_input(idx)), \
               self.targets[idx], self.domains[idx]

    def __len__(self):
        return len(self.targets)