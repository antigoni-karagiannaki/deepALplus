import numpy as np
import torch
from .strategy import Strategy

class BALDDropout(Strategy):
    def __init__(self, dataset, net, args_input, args_task, n_drop=10):
        super(BALDDropout, self).__init__(dataset, net, args_input, args_task)
        self.n_drop = n_drop

    def compute_class_weights(self):
        class_counts = np.bincount(self.dataset.Y_train)
        total_samples = len(self.dataset.Y_train)
        class_weights = total_samples / (len(class_counts) * class_counts)
        return class_weights
    
    def query(self, n):
        # which dataset is used -- ucmerced or ucmerced_Imb
        if self.args_input.dataset_name == 'uc_merced':

            unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
            probs = self.predict_prob_dropout_split(unlabeled_data, n_drop=self.n_drop)
            pb = probs.mean(0)
            entropy1 = (-pb*torch.log(pb)).sum(1)
            entropy2 = (-probs*torch.log(probs)).sum(2).mean(0)
            uncertainties = entropy2 - entropy1
            return unlabeled_idxs[uncertainties.sort()[1][:n]]
        
        elif self.args_input.dataset_name == 'uc_merced_Imb':
        
            unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
            probs = self.predict_prob_dropout_split(unlabeled_data, n_drop=self.n_drop)
            pb = probs.mean(0)
            entropy1 = (-pb * torch.log(pb)).sum(1)
            entropy2 = (-probs * torch.log(probs)).sum(2).mean(0)
            uncertainties = entropy2 - entropy1

            # Compute class weights
            class_weights = self.compute_class_weights()

            # Apply class weights to uncertainties
            weighted_uncertainties = uncertainties * torch.tensor([class_weights[y] for y in self.dataset.Y_train[unlabeled_idxs]])

            return unlabeled_idxs[weighted_uncertainties.sort()[1][:n]]
        
        