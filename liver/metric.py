import numpy as np
import sklearn
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score

"""
Pseudocode:
    1. For every label group (liver, bowel, etc):
        - Normalize the sum of each row's probabilities to 100%.
        - Calculate the sample weighted log loss.
    2. Derive a new any_injury label by taking the max of 1 - p(healthy) for each label group
    3. Calculate the sample weighted log loss for the new label group
    4. Return the average of all of the label group log losses as the final score.
"""
sample_weight = {
    "bowel_healthy" : 1,
    "bowel_injury" : 2,
    "extravasation_healthy" : 1,
    "extravasation_injury" : 6,
    "kidney_healthy" : 1,
    "kidney_low" : 2,
    "kidney_high": 4,
    "liver_healthy" : 1,
    "liver_low" : 2,
    "liver_high": 4,
    "spleen_healthy" : 1,
    "spleen_low" : 2,
    "spleen_high": 4,
    "any_injury" : 6,
}

def normalize_probabilities_to_one(data, col_group):
    # Normalize the sum of each row's probabilities to 100%.
    # 0.75, 0.75 => 0.5, 0.5
    # 0.1, 0.1 => 0.5, 0.5
    row_totals = data.sum(axis=1)
    for idx, col in enumerate(col_group):
        data[:, idx] /= row_totals
    return data

def make_sample_weight(data, col_group):
    weight_list = []
    for i in range(len(data)):
        weight_list.append(sample_weight[col_group[data[i].argmax()]])
        
    return weight_list

def calcuate_metric(pred, gt):


    # Calculate the label group log losses
    binary_targets = ['bowel', 'extravasation']
    triple_level_targets = ['kidney', 'liver', 'spleen']
    all_target_categories = binary_targets + triple_level_targets


    label_group_losses = []
    any_injury_labels = []
    any_injury_pred = []
    for category in all_target_categories:
        if category in binary_targets:
            col_group = [f'{category}_healthy', f'{category}_injury']
        else:
            col_group = [f'{category}_healthy', f'{category}_low', f'{category}_high']
        # norm_gt = normalize_probabilities_to_one(gt[category], col_group)
        norm_pred = normalize_probabilities_to_one(pred[category], col_group)

        label_group_losses.append(
            sklearn.metrics.log_loss(
                y_true = gt[category],
                y_pred = norm_pred,
                sample_weight = make_sample_weight(gt[category], col_group),
                labels=range(len(col_group))
            )
        )

        any_injury_labels.append(gt[category])
        any_injury_pred.append(norm_pred[:, 0])
    any_injury_labels = np.array(any_injury_labels).sum(axis=0)
    any_injury_pred = np.array(any_injury_pred)

    any_injury_labels = np.where(any_injury_labels >= 1, 1, 0)
    any_injury_pred = (1 - np.array(any_injury_pred)).max(axis=0)

    any_injury_loss = sklearn.metrics.log_loss(
        y_true=any_injury_labels,
        y_pred=any_injury_pred,
        sample_weight= np.where(any_injury_labels == 1, 6, 1),
        labels=[0,1]
    )

    label_group_losses.append(any_injury_loss)
    return np.mean(label_group_losses)


class MetricsCalculator:
    def __init__(self, mode='multi'):
        self.probabilities = []
        self.predictions = []
        self.targets = []
        self.mode = mode

    def update(self, logits, target):
        if self.mode == 'multi':
            probabilities = F.softmax(logits, dim = 1)
            predicted = torch.argmax(probabilities, dim=1)
        else:
            probabilities = F.softmax(logits, dim = 1)[:, 1]
            predicted = (probabilities > 0.5).long()

        self.probabilities.extend(probabilities.detach().cpu().numpy())
        self.predictions.extend(predicted.detach().cpu().numpy())
        self.targets.extend(target.detach().cpu().numpy())
    
    def reset(self):
        self.probabilities = []
        self.predictions = []
        self.targets = []
    
    def compute_accuracy(self):
        return accuracy_score(self.targets, self.predictions)
    
    def compute_auc(self):
        if self.mode == 'multi':
            return roc_auc_score(self.targets, self.probabilities, multi_class = 'ovo', labels=[0, 1, 2])
        else:
            return roc_auc_score(self.targets, self.probabilities)
    
    def reset(self):
        self.probabilities = []
        self.predictions = []
        self.targets = []
    
    def compute_accuracy(self):
        return accuracy_score(self.targets, self.predictions)
    
    def compute_auc(self):
        if self.mode == 'multi':
            return roc_auc_score(self.targets, self.probabilities, multi_class = 'ovo', labels=[0, 1, 2])
        else:
            return roc_auc_score(self.targets, self.probabilities)
    
from re import L
import torch
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def conf_matrix(args, acc_targets, acc_outputs, epoch):
    with torch.no_grad():
        cm = confusion_matrix(acc_targets, acc_outputs)
        group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
        group_percentages = ["{0:.2%}".format(value) for value in cm.flatten()/np.sum(cm)]
        labels = [f"{v2}\n{v3}" for v2, v3 in zip(group_counts, group_percentages)]
        labels = np.asarray(labels).reshape(3,3)
        sns.set(font_scale=0.5)
        disp = sns.heatmap(cm, annot=labels, fmt='', cmap='Blues')
        disp.plot()
        plt.savefig(args.checkpoint_dir + f'conf_mat__{epoch}.png')
        plt.close()
