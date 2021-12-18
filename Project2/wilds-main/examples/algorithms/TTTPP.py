import torch
from initializer import initialize_model
import torch.optim as optim
from ttt_helpers.offline import offline
from ttt_helpers.online import FeatureQueue

class TTTPP:

    def __init__(self, config, d_out, criterion, lr):

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

    def train(self):
        self.mode = "train"

        parameters = list(self.net.parameters()) + list(self.projector.parameters())
        self.optimizer = optim.SGD(parameters, lr=self.lr, momentum=0.9)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
    'min', factor=0.5, patience=10, cooldown=10,
    threshold=0.0001, threshold_mode='rel', min_lr=0.0001, verbose=True)

        self.featurizer.train()
        self.classifier.train()
        self.projector.train()
    
    def summarize(self, offlineloader, scale_ext, scale_ssh, queue_size, batch_size_align):
        assert queue_size % batch_size_align == 0
        assert queue_size > batch_size_align

        MMD_SCALE_FACTOR = 0.5

        # align featurizer
        cov_src_ext, coral_src_ext, mu_src_ext, mmd_src_ext = offline(offlineloader, self.featurizer, scale_ext)
        scale_coral_ext = scale_ext / coral_src_ext
        scale_mmd_ext = scale_ext / mmd_src_ext * MMD_SCALE_FACTOR
        queue_ext = FeatureQueue(dim=mu_src_ext.shape[0], length=queue_size-batch_size_align)
        
        self.cov_src_ext = cov_src_ext
        self.mu_src_ext = mu_src_ext
        self.scale_coral_ext = scale_coral_ext
        self.scale_mmd_ext = scale_mmd_ext
        self.queue_ext = queue_ext

        # align ssh
        cov_src_ssh, coral_src_ssh, mu_src_ssh, mmd_src_ssh = offline(offlineloader, self.ssh, scale_ssh)
        scale_coral_ssh = scale_ssh / coral_src_ssh
        scale_mmd_ssh = scale_ssh / mmd_src_ssh * MMD_SCALE_FACTOR
        queue_ssh = FeatureQueue(dim=mu_src_ssh.shape[0], length=queue_size-batch_size_align)

        self.cov_src_ssh = cov_src_ssh
        self.mu_src_ssh = mu_src_ssh
        self.scale_coral_ssh = scale_coral_ssh
        self.scale_mmd_ssh = scale_mmd_ssh
        self.queue_ssh = queue_ssh
            
    def test_train(self):
        self.mode = "test_train"
        self.optimizer = optim.SGD(self.ssh.parameters(), lr=self.lr, momentum=0.9)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
    'min', factor=0.5, patience=10, cooldown=10,
    threshold=0.0001, threshold_mode='rel', min_lr=0.0001, verbose=True)

        self.featurizer.train()
        self.projector.train()
    
    def eval(self):
        self.mode = "eval"

        self.featurizer.eval()
        self.classifier.eval()
        

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

    def _test_train_process_batch(self, batch):

        self.optimizer.zero_grad()

        # contrastive learning

        x, y, _ = batch
        texts = torch.cat([x[0], x[1]], dim=0)
        texts = texts.cuda(non_blocking=True)
        labels = y.cuda(non_blocking=True)

        sz = labels.shape[0]
        features = self.ssh(texts)
        f1, f2 = torch.split(features, [sz, sz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        loss = self.criterion(features)

        

        # TODO
        
    
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

def constrastive_transform(batch):
    # TODO: construct positive and negative samples for contrastive learning
    return batch, batch