import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from nbdt.data.custom import Node
import numpy as np
from nbdt.utils import Colors, MaskLoss
from saliency.Grad_CAM.gcam_loss import GradCAM

__all__ = names = ('HardTreeSupLoss', 'SoftTreeSupLoss', 'CrossEntropyLoss', 'HardTreeSupLossMultiPath',
                   'HardAttentionLoss', 'SoftTreeSupMaskLoss', 'ZSSoftTreeSupLoss')
keys = (
    'path_graph', 'path_wnids', 'max_leaves_supervised',
    'min_leaves_supervised', 'weighted_average', 'tree_supervision_weight',
    'classes'
)


def convert_to_onehot(indices, num_classes):
    vec = np.zeros(num_classes)
    for i in indices:
        vec[i] = 1
    return vec

def add_arguments(parser):
    parser.add_argument('--path-graph', help='Path to graph-*.json file.')  # WARNING: hard-coded suffix -build in generate_fname
    parser.add_argument('--path-wnids', help='Path to wnids.txt file.')
    parser.add_argument('--max-leaves-supervised', type=int, default=-1,
                        help='Maximum number of leaves a node can have to '
                        'contribute loss, in tree-supervised training.')
    parser.add_argument('--min-leaves-supervised', type=int, default=-1,
                        help='Minimum number of leaves a node must have to '
                        'contribute loss, in tree-supervised training.')
    parser.add_argument('--weighted-average', action='store_true',
                        help='Use weighted average instead of average, for cluster '
                        'centers.')
    parser.add_argument('--tree-supervision-weight', type=float, default=1,
                        help='Weight assigned to tree supervision losses')


CrossEntropyLoss = nn.CrossEntropyLoss


class TreeSupLoss(nn.Module):

    accepts_path_graph = True
    accepts_path_wnids = True
    accepts_classes = True
    accepts_max_leaves_supervised = True
    accepts_min_leaves_supervised = True
    accepts_tree_supervision_weight = True
    accepts_weighted_average = True
    accepts_classes = lambda trainset, **kwargs: trainset.classes

    def __init__(self, path_graph, path_wnids, classes,
            max_leaves_supervised=-1, min_leaves_supervised=-1,
            tree_supervision_weight=1., weighted_average=False,
            criterion=nn.CrossEntropyLoss()):
        super().__init__()

        self.num_classes = len(classes)
        self.nodes = Node.get_nodes(path_graph, path_wnids, classes)
        self.max_leaves_supervised = max_leaves_supervised
        self.min_leaves_supervised = min_leaves_supervised
        self.tree_supervision_weight = tree_supervision_weight
        self.weighted_average = weighted_average
        self.criterion = criterion
        self.tree_criterion = nn.BCEWithLogitsLoss()

class HardTreeSupLoss(TreeSupLoss):

    def forward(self, outputs, targets):
        """
        The supplementary losses are all uniformly down-weighted so that on
        average, each sample incurs half of its loss from standard cross entropy
        and half of its loss from all nodes.
        The code below is structured weirdly to minimize number of tensors
        constructed and moved from CPU to GPU or vice versa. In short,
        all outputs and targets for nodes with 2 children are gathered and
        moved onto GPU at once. Same with those with 3, with 4 etc. On CIFAR10,
        the max is 2. On CIFAR100, the max is 8.
        """
        loss = self.criterion(outputs, targets)
        num_losses = outputs.size(0) * len(self.nodes) / 2.

        outputs_subs = defaultdict(lambda: [])
        targets_subs = defaultdict(lambda: [])
        targets_ints = [int(target) for target in targets.cpu().long()]
        for node in self.nodes:
            if self.max_leaves_supervised > 0 and \
                    node.num_leaves > self.max_leaves_supervised:
                continue

            if self.min_leaves_supervised > 0 and \
                    node.num_leaves < self.min_leaves_supervised:
                continue

            _, outputs_sub, targets_sub = HardTreeSupLoss.inference(
                node, outputs, targets_ints, self.weighted_average)

            key = node.num_classes
            assert outputs_sub.size(0) == len(targets_sub)
            outputs_subs[key].append(outputs_sub)
            targets_subs[key].extend(targets_sub)

        for key in outputs_subs:
            outputs_sub = torch.cat(outputs_subs[key], dim=0)
            targets_sub = torch.Tensor(targets_subs[key]).long().to(outputs_sub.device)
            if not outputs_sub.size(0):
                continue
            fraction = outputs_sub.size(0) / float(num_losses) \
                * self.tree_supervision_weight
            loss += self.criterion(outputs_sub, targets_sub) * fraction
        return loss

    @classmethod
    def inference(cls, node, outputs, targets, weighted_average=False):
        classes = [node.old_to_new_classes[int(t)] for t in targets]
        selector = [bool(cls) for cls in classes]
        targets_sub = [cls[0] for cls in classes if cls]

        _outputs = outputs[selector]
        if _outputs.size(0) == 0:
            return selector, _outputs[:, :node.num_classes], targets_sub
        outputs_sub = cls.get_output_sub(_outputs, node, weighted_average)
        return selector, outputs_sub, targets_sub

    @staticmethod
    def get_output_sub(_outputs, node, weighted_average=False):
        if weighted_average:
            node.move_leaf_weights_to(_outputs.device)

        weights = [
            node.new_to_leaf_weights[new_label] if weighted_average else 1
            for new_label in range(node.num_classes)
        ]
        return torch.stack([
            (_outputs * weight).T
            [node.new_to_old_classes[new_label]].mean(dim=0)
            for new_label, weight in zip(range(node.num_classes), weights)
        ]).T

class SoftTreeSupLoss(HardTreeSupLoss):

    def forward(self, outputs, targets):
        loss = self.criterion(outputs, targets)
        bayesian_outputs = SoftTreeSupLoss.inference(
            self.nodes, outputs, self.num_classes, self.weighted_average)
        loss += self.criterion(bayesian_outputs, targets) * self.tree_supervision_weight
        return loss

    @classmethod
    def inference(cls, nodes, outputs, num_classes, weighted_average=False):
        """
        In theory, the loop over children below could be replaced with just a
        few lines:

            for index_child in range(len(node.children)):
                old_indexes = node.new_to_old_classes[index_child]
                class_probs[:,old_indexes] *= output[:,index_child][:,None]

        However, we collect all indices first, so that only one tensor operation
        is run.
        """
        class_probs = torch.ones((outputs.size(0), num_classes)).to(outputs.device)
        for node in nodes:
            output = cls.get_output_sub(outputs, node, weighted_average)
            output = F.softmax(output)

            old_indices, new_indices = [], []
            for index_child in range(len(node.children)):
                old = node.new_to_old_classes[index_child]
                old_indices.extend(old)
                new_indices.extend([index_child] * len(old))

            assert len(set(old_indices)) == len(old_indices), (
                'All old indices must be unique in order for this operation '
                'to be correct.'
            )
            class_probs[:,old_indices] *= output[:,new_indices]
        return class_probs

class HardTreeSupLossMultiPath(HardTreeSupLoss):

    def forward(self, outputs, targets):
        """
        The supplementary losses are all uniformly down-weighted so that on
        average, each sample incurs half of its loss from standard cross entropy
        and half of its loss from all nodes.

        The code below is structured weirdly to minimize number of tensors
        constructed and moved from CPU to GPU or vice versa. In short,
        all outputs and targets for nodes with 2 children are gathered and
        moved onto GPU at once. Same with those with 3, with 4 etc. On CIFAR10,
        the max is 2. On CIFAR100, the max is 8.
        """
        loss = self.criterion(outputs, targets)
        num_losses = outputs.size(0) * len(self.nodes) / 2.

        outputs_subs = defaultdict(lambda: [])
        targets_subs = defaultdict(lambda: [])
        targets_ints = [int(target) for target in targets.cpu().long()]
        for node in self.nodes:
            if self.max_leaves_supervised > 0 and \
                    node.num_leaves > self.max_leaves_supervised:
                continue

            if self.min_leaves_supervised > 0 and \
                    node.num_leaves < self.min_leaves_supervised:
                continue

            _, outputs_sub, targets_sub = HardTreeSupLossMultiPath.inference(
                node, outputs, targets_ints, self.weighted_average)

            key = node.num_classes
            assert outputs_sub.size(0) == len(targets_sub)
            outputs_subs[key].append(outputs_sub)
            targets_subs[key].extend(targets_sub)

        for key in outputs_subs:
            outputs_sub = torch.cat(outputs_subs[key], dim=0)
            targets_sub = torch.Tensor(targets_subs[key]).long().to(outputs_sub.device).type_as(outputs_sub)
            if not outputs_sub.size(0):
                continue
            fraction = outputs_sub.size(0) / float(num_losses) \
                * self.tree_supervision_weight
            loss += self.tree_criterion(outputs_sub, targets_sub) * fraction
        return loss

    @classmethod
    def inference(cls, node, outputs, targets, weighted_average=False):
        classes = [node.old_to_new_classes[int(t)] for t in targets]
        selector = [bool(cls) for cls in classes]
        # convert to muli label one hot vector
        targets_sub = [convert_to_onehot(cls, node.num_classes) for cls in classes if cls]
        _outputs = outputs[selector]
        if _outputs.size(0) == 0:
            return selector, _outputs[:, :node.num_classes], targets_sub
        outputs_sub = cls.get_output_sub(_outputs, node, weighted_average)
        return selector, outputs_sub, targets_sub

class HardAttentionLoss(HardTreeSupLoss):

    def __init__(self, gcam, path_graph, path_wnids, classes,
                 max_leaves_supervised=-1, min_leaves_supervised=-1,
                 tree_supervision_weight=1., weighted_average=False,
                 criterion=nn.CrossEntropyLoss()):
        super().__init__(path_graph, path_wnids, classes,
                 max_leaves_supervised, min_leaves_supervised,
                 tree_supervision_weight, weighted_average, criterion)
        self.gcam = gcam
        #self.mask_loss = MaskLoss()
        self.mask_loss = nn.KLDivLoss()

    def forward(self, outputs, targets):
        """
        """
        # loss = self.criterion(outputs, targets)
        num_losses = outputs.size(0) * len(self.nodes) / 2.

        outputs_subs = defaultdict(lambda: [])
        targets_subs = defaultdict(lambda: [])
        cams_subs_1 = defaultdict(lambda: [])
        cams_subs_2 = defaultdict(lambda: [])
        targets_ints = [int(target) for target in targets.cpu().long()]
        for node in self.nodes:
            if self.max_leaves_supervised > 0 and \
                    node.num_leaves > self.max_leaves_supervised:
                continue

            if self.min_leaves_supervised > 0 and \
                    node.num_leaves < self.min_leaves_supervised:
                continue

            _, outputs_sub, targets_sub = HardTreeSupLoss.inference(
                node, outputs, targets_ints, self.weighted_average)

            # compute gradcam
            cams = {i: [] for i in range(node.num_classes)}
            probs, ids = self.gcam.forward(outputs_sub)
            output = F.softmax(outputs_sub)

            for i in range(len(node.children)):
                self.gcam.backward(ids=output[:, [i]].long())
                regions = self.gcam.generate(target_layer='module.layer4')
                masks = []
                for j in range(len(outputs_sub)):
                    # Grad-CAM
                    mask = regions[j, 0]
                    if targets_sub[j] == i:
                        cams[1].append(mask.flatten())
                        #cams[1].append(mask)
                    else:
                        cams[0].append(mask.flatten())
                        #cams[0].append(mask)
            if len(cams[0]) == 0 or len(cams[1]) == 0:
                continue
            heatmaps_1 = torch.stack(cams[0])
            heatmaps_2 = torch.stack(cams[1])

            key = node.num_classes
            assert outputs_sub.size(0) == len(targets_sub)
            outputs_subs[key].append(outputs_sub)
            targets_subs[key].extend(targets_sub)
            cams_subs_1[key].append(heatmaps_1)
            cams_subs_2[key].append(heatmaps_2)

        for key in outputs_subs:
            outputs_sub = torch.cat(outputs_subs[key], dim=0)
            targets_sub = torch.Tensor(targets_subs[key]).long().to(outputs_sub.device)
            cams_sub_1 = torch.cat(cams_subs_1[key], dim=0)
            cams_sub_2 = torch.cat(cams_subs_2[key], dim=0)
            if not outputs_sub.size(0):
                continue
            fraction = outputs_sub.size(0) / float(num_losses) \
                       * self.tree_supervision_weight
            loss = self.mask_loss(cams_sub_1, cams_sub_2) * fraction
            print("mask loss ", self.mask_loss(cams_sub_1, cams_sub_2) * fraction * .01)
        return loss

class SoftTreeSupMaskLoss(SoftTreeSupLoss):

    def forward(self, outputs, targets):
        fc, outputs = outputs
        #loss = self.criterion(torch.mm(outputs, torch.transpose(fc, 0, 1)), targets)
        loss = self.criterion(fc(outputs), targets)
        bayesian_outputs = SoftTreeSupMaskLoss.inference(
            self.nodes, (fc, outputs), self.num_classes, self.weighted_average, self.top_k)
        loss += self.criterion(bayesian_outputs, targets) * self.tree_supervision_weight
        return loss

    @classmethod
    def inference(cls, nodes, outputs, num_classes, weighted_average=False, top_k=0.5):
        """
        In theory, the loop over children below could be replaced with just a
        few lines:
            for index_child in range(len(node.children)):
                old_indexes = node.new_to_old_classes[index_child]
                class_probs[:,old_indexes] *= output[:,index_child][:,None]
        However, we collect all indices first, so that only one tensor operation
        is run.
        """
        fc, outputs = outputs
        class_probs = torch.ones((outputs.size(0), num_classes)).to(outputs.device)
        for node in nodes:
            output = cls.get_output_sub(outputs, node, weighted_average, top_k, fc)
            output = F.softmax(output)

            old_indices, new_indices = [], []
            for index_child in range(len(node.children)):
                old = node.new_to_old_classes[index_child]
                old_indices.extend(old)
                new_indices.extend([index_child] * len(old))

            assert len(set(old_indices)) == len(old_indices), (
                'All old indices must be unique in order for this operation '
                'to be correct.'
            )
            class_probs[:,old_indices] *= output[:,new_indices]
        return class_probs


    @staticmethod
    def get_output_sub(_outputs, node, weighted_average, top_k, fc):
        # here, outputs is the feature vector instead of pre-softmax fc outputs
        # get attention mask that is already set
        #attention_mask = node.attention_mask
        # or, calculate attention mask here
        classes_in_node = []
        for i in node.new_to_old_classes:
            classes_in_node.extend(node.new_to_old_classes[i])
        classes_in_node_fc = fc.weight[classes_in_node]
        classes_in_node_var = classes_in_node_fc.var(dim=0)
        _, attention_mask = classes_in_node_var.sort()


        old_indices, new_indices = [], []
        for index_child in range(len(node.children)):
            old = node.new_to_old_classes[index_child]
            old_indices.extend(old)
            new_indices.extend([index_child] * len(old))


        attention_zero_mask = attention_mask[int(top_k*len(attention_mask)):]
        attention_zero_mask = attention_zero_mask.reshape(1, -1).repeat((_outputs.size(0),1)).long()
        _outputs = _outputs.clone()
        _outputs.scatter_(1,attention_zero_mask,0.)
        #_outputs = torch.mm(_outputs, torch.transpose(fc, 0, 1))
        _outputs = fc(_outputs)

        if weighted_average:
            node.move_leaf_weights_to(_outputs.device)

        weights = [
            node.new_to_leaf_weights[new_label] if weighted_average else 1
            for new_label in range(node.num_classes)
        ]

        cls_idxs = [node.new_to_old_classes[new_label] for new_label in range(node.num_classes)]

        return torch.stack([
            (_outputs * weight).T
            [cls_idx].mean(dim=0)
            for cls_idx, weight in zip(cls_idxs, weights)
        ]).T

def CXE(predicted, target):
    return -(target * torch.log(predicted)).sum(dim=1).mean()

class ZSSoftTreeSupLoss(SoftTreeSupLoss):

    def __init__(self, path_graph, path_wnids, classes,
                 max_leaves_supervised=-1, min_leaves_supervised=-1,
                 tree_supervision_weight=1., weighted_average=False,
                 criterion=nn.CrossEntropyLoss()):
        super().__init__(path_graph, path_wnids, classes,
                         max_leaves_supervised, min_leaves_supervised,
                         tree_supervision_weight, weighted_average, criterion)
        self.zs_labels = list(range(64, 84))
        self.pairings = {l: None for l in self.zs_labels}
        self.pair_zs()
        print(self.pairings)

    def pair_zs(self):
        for node in self.nodes:
            children_ids = [node.new_to_old_classes[index_child][0] for index_child in range(len(node.children))]
            intersection = [i for i in self.zs_labels if i in children_ids]
            union = [i for i in children_ids if i not in self.zs_labels]
            for label in intersection:
                if len(union) > 0:
                    print(f"pair {label} with {union}")
                    self.pairings[label] = union[0]

    def forward(self, outputs, targets):
        smoothed_targets = self.smooth_one_hot(targets)
        loss = self.criterion(outputs, targets)
        bayesian_outputs = SoftTreeSupLoss.inference(
            self.nodes, outputs, self.num_classes, self.weighted_average)
        loss += CXE(bayesian_outputs, smoothed_targets) * self.tree_supervision_weight
        return loss

    @classmethod
    def inference(cls, nodes, outputs, num_classes, weighted_average=False):
        """
        In theory, the loop over children below could be replaced with just a
        few lines:

            for index_child in range(len(node.children)):
                old_indexes = node.new_to_old_classes[index_child]
                class_probs[:,old_indexes] *= output[:,index_child][:,None]

        However, we collect all indices first, so that only one tensor operation
        is run.
        """
        class_probs = torch.ones((outputs.size(0), num_classes)).to(outputs.device)
        for node in nodes:
            output = cls.get_output_sub(outputs, node, weighted_average)
            output = F.softmax(output)

            old_indices, new_indices = [], []
            for index_child in range(len(node.children)):
                old = node.new_to_old_classes[index_child]
                old_indices.extend(old)
                new_indices.extend([index_child] * len(old))

            assert len(set(old_indices)) == len(old_indices), (
                'All old indices must be unique in order for this operation '
                'to be correct.'
            )
            class_probs[:,old_indices] *= output[:,new_indices]
        return class_probs

    def smooth_one_hot(self, labels, smoothing=0.1):
        """ create soft labels """
        assert 0 <= smoothing < 1
        label_shape = torch.Size((labels.size(0), self.num_classes))
        confidence = 1.0 - smoothing

        if smoothing == 0 or not self.pairings:
            return torch.zeros_like(label_shape).scatter_(1, labels.data.unsqueeze(1), confidence)

        with torch.no_grad():
            true_dist = torch.zeros(size=label_shape, device=labels.device)
            true_dist.scatter_(1, labels.data.unsqueeze(1), 1)
        for vec in true_dist:
            for zsl, seen in self.pairings.items():
                if seen:
                    if vec[seen] == 1:
                        vec[seen] = confidence
                        vec[zsl] = smoothing
        # for zsl, seen in self.pairings.items():
        #     if seen:
        #         seen_selector = torch.zeros_like(labels.data.unsqueeze(1))
        #         seen_selector[true_dist[:, seen] == 1] = seen
        #         print(seen_selector)
        #         zsl_selector = torch.zeros_like(labels.data.unsqueeze(1))
        #         zsl_selector[true_dist[:, seen] == 1] = zsl
        #         true_dist.scatter_(1, seen_selector, confidence)
        #         true_dist.scatter_(1, zsl_selector, smoothing)
        return true_dist