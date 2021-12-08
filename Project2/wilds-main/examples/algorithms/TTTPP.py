import torch
from initializer import initialize_model
import torch.optim as optim

class TTTPP():

    def __init__(self, config, d_out, criterion, lr, queue_size, scale):

        # initialize models

        # bert encoder
        featurizer, classifier = initialize_model(config, d_out=d_out, is_featurizer=True)
        featurizer = featurizer.to(config.device)
        
        # main task classifier
        classifier = classifier.to(config.device)

        # TODO: ssl task projector
        projector = 1

        # set model components
        self.featurizer = featurizer
        self.classifier = classifier
        self.projector = projector

        self.net = torch.nn.Sequential(featurizer, classifier).to(config.device)
        self.ssh = torch.nn.Sequential(featurizer, projector).to(config.device)

        self.mode = "train" # {"train", "summarize", "test_train", "eval"}
        
        self.lr = lr
        self.criterion = criterion
        self.queue_size = queue_size
        self.scale = scale

    def train(self):
        self.mode = "train"

        parameters = list(self.net.parameters()) + list(self.projector.parameters())
        self.optimizer = optim.SGD(parameters, lr=self.lr, momentum=0.9)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
    'min', factor=0.5, patience=10, cooldown=10,
    threshold=0.0001, threshold_mode='rel', min_lr=0.0001, verbose=True)
    
    def summarize(self):
        self.mode = "summarize"
    
    def test_train(self):
        self.mode = "test_train"
        self.optimizer = optim.SGD(self.ssh.parameters(), lr=self.lr, momentum=0.9)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
    'min', factor=0.5, patience=10, cooldown=10,
    threshold=0.0001, threshold_mode='rel', min_lr=0.0001, verbose=True)
    
    def eval(self):
        self.mode = "eval"

    def _train_process_batch(self, batch):
        self.optimizer.zero_grad()

        x, y_true, _ = batch
        x = x.to(self.device)
        y_true = y_true.to(self.device)
        
        y_pred = self.net(x)
        loss = self.criterion(y_pred, y_true)

        # TODO: compute embeddings and ssl loss


        loss.backward()
        self.optimizer.step()

    def _summarize_process_batch(self, batch):
        x, _, _ = batch
        x = x.to(self.device)

        # TODO: compute feature mean & covariance
        # align featurizer
        

        # align projector


    def _test_train_process_batch(self, batch):
        
    
    def _evaluate_process_batch(self, batch):




    def update(self, batch):
        

    def test_time_train(self):

    def evaluate(self):


    def process_batch(self, batch):
        """
        Override
        """
        # forward pass
        x, y_true, metadata = batch
        x = x.to(self.device)
        y_true = y_true.to(self.device)
        g = self.grouper.metadata_to_group(metadata).to(self.device)
        
        # bert encoder
        features = self.featurizer(x)

        # main task classifer
        outputs = self.classifier(features)

        # TODO: ssl task projector

        # package the results
        results = {
            'g': g,
            'y_true': y_true,
            'y_pred': outputs,
            'metadata': metadata,
            'features': features,
            }
        return results

    # TODO: how to compute loss for main task & ssl task separately
    # Option1: implement two loss functions and return one of them based on input arguments
    # Option2: 

    def objective(self, results):
        # extract features
        features = results.pop('features')

        # 1. compute loss for main task


        # 2. compute loss for ssl task

        # no need for groups
        if self.is_training:
            # split into groups
            unique_groups, group_indices, _ = split_into_groups(results['g'])
            # compute penalty
            n_groups_per_batch = unique_groups.numel()
            penalty = torch.zeros(1, device=self.device)
            for i_group in range(n_groups_per_batch):
                for j_group in range(i_group+1, n_groups_per_batch):
                    penalty += self.coral_penalty(features[group_indices[i_group]], features[group_indices[j_group]])
            if n_groups_per_batch > 1:
                penalty /= (n_groups_per_batch * (n_groups_per_batch-1) / 2) # get the mean penalty
            # save penalty
        else:
            penalty = 0.

        if isinstance(penalty, torch.Tensor):
            results['penalty'] = penalty.item()
        else:
            results['penalty'] = penalty


        avg_loss = self.loss.compute(results['y_pred'], results['y_true'], return_dict=False)

        return avg_loss + penalty * self.penalty_weight

class FeatureQueue():
    def __init__(self, dim, length):
        self.length = length
        self.queue = torch.zeros(length, dim)
        self.ptr = 0

    @torch.no_grad()
    def update(self, feat):

        batch_size = feat.shape[0]
        assert self.length % batch_size == 0  # for simplicity

        # replace the features at ptr (dequeue and enqueue)
        self.queue[self.ptr:self.ptr + batch_size] = feat
        self.ptr = (self.ptr + batch_size) % self.length  # move pointer

    def get(self):
        cnt = (self.queue[-1] != 0).sum()
        if cnt.item():
            return self.queue
        else:
            return None

def constrastive_transform(batch):
    # TODO: construct positive and negative samples for contrastive learning
    return batch, batch