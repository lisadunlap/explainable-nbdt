from nbdt.graph import get_root, get_wnids, synset_to_name, wnid_to_synset, get_leaves
from nbdt.utils import (
    DEFAULT_CIFAR10_TREE, DEFAULT_CIFAR10_WNIDS, DEFAULT_CIFAR100_TREE,
    DEFAULT_CIFAR100_WNIDS, DEFAULT_TINYIMAGENET200_TREE,
    DEFAULT_TINYIMAGENET200_WNIDS, DEFAULT_IMAGENET1000_TREE,
    DEFAULT_IMAGENET1000_WNIDS, set_np_printoptions
)
from nbdt.loss import HardTreeSupLoss, SoftTreeSupLoss
from nbdt.data.custom import Node
from networkx.readwrite.json_graph import node_link_data, node_link_graph
import torch
import torch.nn as nn
import numpy as np
import csv
import networkx as nx
import os
import json
import wandb
import pandas as pd


__all__ = names = (
    'Noop', 'ConfusionMatrix', 'HardEmbeddedDecisionRules', 'SoftEmbeddedDecisionRules',
    'SingleInference', 'HardFullTreePrior')
keys = ('path_graph', 'path_graph_analysis', 'path_wnids', 'weighted_average', 'trainset', 'testset', 'json_save_path', 'experiment_name')


def add_arguments(parser):
    parser.add_argument('--json-save-path', default=None, type=str,
                    help='Directory to save jsons under for full tree analysis')
    parser.add_argument('--path-graph-analysis', default=None, type=str,
                    help='path for graph for analysis')

class Noop:

    accepts_trainset = lambda trainset, **kwargs: trainset
    accepts_testset = lambda testset, **kwargs: testset

    def __init__(self, trainset, testset, experiment_name, use_wandb=False):
        set_np_printoptions()

        self.trainset = trainset
        self.testset = testset

        self.epoch = None

    def start_epoch(self, epoch):
        self.epoch = epoch

    def start_train(self, epoch):
        assert epoch == self.epoch

    def update_batch(self, outputs, predicted, targets):
        pass

    def end_train(self, epoch):
        assert epoch == self.epoch

    def start_test(self, epoch):
        assert epoch == self.epoch

    def end_test(self, epoch):
        assert epoch == self.epoch

    def end_epoch(self, epoch):
        assert epoch == self.epoch


class ConfusionMatrix(Noop):

    def __init__(self, trainset, testset, experiment_name, use_wandb=False):
        super().__init__(trainset, testset, experiment_name, use_wandb)
        self.k = len(trainset.classes)
        self.m = None

    def start_train(self, epoch):
        super().start_train(epoch)
        raise NotImplementedError()

    def start_test(self, epoch):
        super().start_test(epoch)
        self.m = np.zeros((self.k, self.k))

    def update_batch(self, outputs, predicted, targets):
        super().update_batch(outputs, predicted, targets)
        if len(predicted.shape) == 1:
            predicted = predicted.numpy().ravel()
            targets = targets.numpy().ravel()
            ConfusionMatrix.update(self.m, predicted, targets)

    def end_test(self, epoch):
        super().end_test(epoch)
        recall = self.recall()
        for row, cls in zip(recall, self.trainset.classes):
            print(row, cls)
        print(recall.diagonal(), '(diagonal)')

    @staticmethod
    def update(confusion_matrix, preds, labels):
        preds = tuple(preds)
        labels = tuple(labels)

        for pred, label in zip(preds, labels):
            confusion_matrix[label, pred] += 1

    @staticmethod
    def normalize(confusion_matrix, axis):
        total = confusion_matrix.astype(np.float).sum(axis=axis)
        total = total[:, None] if axis == 1 else total[None]
        return confusion_matrix / total

    def recall(self):
        return ConfusionMatrix.normalize(self.m, 1)

    def precision(self):
        return ConfusionMatrix.normalize(self.m, 0)


class HardEmbeddedDecisionRules(Noop):
    """Evaluation is hard."""

    accepts_path_graph = True
    accepts_path_wnids = True
    accepts_weighted_average = True

    def __init__(self, trainset, testset, experiment_name, path_graph, path_wnids,
            weighted_average=False, use_wandb=False):
        super().__init__(trainset, testset, experiment_name, use_wandb)
        self.nodes = Node.get_nodes(path_graph, path_wnids, trainset.classes)
        self.G = self.nodes[0].G
        self.wnid_to_node = {node.wnid: node for node in self.nodes}

        self.wnids = get_wnids(path_wnids)
        self.classes = trainset.classes
        self.wnid_to_class = {wnid: cls for wnid, cls in zip(self.wnids, self.classes)}

        self.weighted_average = weighted_average
        self.correct = 0
        self.total = 0
        self.class_accuracies = {c:0 for c in self.classes}
        self.class_totals = {c: 0 for c in self.classes}
        self.use_wandb = use_wandb
        if self.use_wandb:
            wandb.init(project=experiment_name, name='Analysis', reinit=True, entity='lisadunlap')

    def update_batch(self, outputs, predicted, targets):
        super().update_batch(outputs, predicted, targets)

        targets_ints = [int(target) for target in targets.cpu().long()]
        wnid_to_pred_selector = {}
        for node in self.nodes:
            selector, outputs_sub, targets_sub = HardTreeSupLoss.inference(
                node, outputs, targets, self.weighted_average)
            if not any(selector):
                continue
            _, preds_sub = torch.max(outputs_sub, dim=1)
            preds_sub = list(map(int, preds_sub.cpu()))
            wnid_to_pred_selector[node.wnid] = (preds_sub, selector)

        n_samples = outputs.size(0)
        predicted = self.traverse_tree(
            predicted, wnid_to_pred_selector, n_samples).to(targets.device)
        self.total += n_samples
        self.correct += (predicted == targets).sum().item()
        for i in range(len(predicted)):
            self.class_accuracies[self.classes[predicted[i]]] += int(predicted[i] == targets[i])
            self.class_totals[self.classes[targets[i]]] += 1
        accuracy = round(self.correct / float(self.total), 4) * 100
        return f'NBDT-Hard: {accuracy}%'

    def traverse_tree(self, _, wnid_to_pred_selector, n_samples):
        wnid_root = get_root(self.G)
        node_root = self.wnid_to_node[wnid_root]
        preds = []
        for index in range(n_samples):
            wnid, node = wnid_root, node_root
            while node is not None:
                if node.wnid not in wnid_to_pred_selector:
                    wnid = node = None
                    break
                pred_sub, selector = wnid_to_pred_selector[node.wnid]
                if not selector[index]:  # we took a wrong turn. wrong.
                    wnid = node = None
                    break
                index_new = sum(selector[:index + 1]) - 1
                index_child = pred_sub[index_new]
                wnid = node.children[index_child]
                node = self.wnid_to_node.get(wnid, None)
            cls = self.wnid_to_class.get(wnid, None)
            pred = -1 if cls is None else self.classes.index(cls)
            preds.append(pred)
        return torch.Tensor(preds).long()

    def end_test(self, epoch):
        super().end_test(epoch)
        accuracy = round(self.correct / self.total * 100., 2)
        print(f'NBDT-Hard Accuracy: {accuracy}%, {self.correct}/{self.total}')
        if self.use_wandb:
            data = [[(self.class_accuracies[k]/self.class_totals[k])*100 for k in self.class_accuracies.keys()]]
            wandb.log({"class accuracies": wandb.Table(data=data, columns=self.classes)})


class SoftEmbeddedDecisionRules(HardEmbeddedDecisionRules):
    """Evaluation is soft."""

    def __init__(self, trainset, testset, experiment_name, path_graph, path_wnids,
            weighted_average=False, use_wandb=False):
        super().__init__(trainset, testset, experiment_name, path_graph, path_wnids, use_wandb)
        self.num_classes = len(trainset.classes)

    def update_batch(self, outputs, predicted, targets):
        bayesian_outputs = SoftTreeSupLoss.inference(
            self.nodes, outputs, self.num_classes, self.weighted_average)
        n_samples = outputs.size(0)
        predicted = bayesian_outputs.max(1)[1].to(targets.device)
        self.total += n_samples
        self.correct += (predicted == targets).sum().item()
        for i in range(len(predicted)):
            self.class_accuracies[self.classes[predicted[i]]] += int(predicted[i] == targets[i])
            self.class_totals[self.classes[targets[i]]] += 1
        accuracy = round(self.correct / float(self.total), 4) * 100
        return f'NBDT-Soft: {accuracy}%'

    def end_test(self, epoch):
        accuracy = round(self.correct / self.total * 100., 2)
        print(f'NBDT-Soft Accuracy: {accuracy}%, {self.correct}/{self.total}')
        if self.use_wandb:
            data = [[(self.class_accuracies[k] / self.class_totals[k]) * 100 for k in self.class_accuracies.keys()]]
            wandb.log({"class accuracies": wandb.Table(data=data, columns=self.classes)})

class SingleInference(HardEmbeddedDecisionRules):
    """Inference on a single image ."""

    def __init__(self, trainset, testset, experiment_name, path_graph, path_wnids,
            weighted_average=False, use_wandb=False):
        super().__init__(trainset, testset, experiment_name, path_graph, path_wnids, use_wandb)
        self.num_classes = len(trainset.classes)

    def single_traversal(self, _, wnid_to_pred_selector, n_samples):
        wnid_root = get_root(self.G)
        node_root = self.wnid_to_node[wnid_root]
        wnid, node = wnid_root, node_root
        path = [wnid]
        while node is not None:
            if node.wnid not in wnid_to_pred_selector:
                wnid = node = None
                break
            pred_sub, selector = wnid_to_pred_selector[node.wnid]
            index_new = sum(selector[:0 + 1]) - 1
            index_child = pred_sub[index_new]
            wnid = node.children[index_child]
            path.append(wnid)
            node = self.wnid_to_node.get(wnid, None)
        return path

    def inf(self, img):
        wnid_to_pred_selector = {}
        for node in self.nodes:
            outputs_sub = HardTreeSupLoss.get_output_sub(
                img, node, self.weighted_average)
            selector = [1 for c in range(node.num_classes)]
            if not any(selector):
                continue
            _, preds_sub = torch.max(outputs_sub, dim=1)
            preds_sub = list(map(int, preds_sub.cpu()))
            wnid_to_pred_selector[node.wnid] = (preds_sub, selector)
        n_samples = 1
        predicted = self.single_traversal(
            [], wnid_to_pred_selector, n_samples)
        wandb.log({"examples": [wandb.Image(img.cpu().numpy(), caption=str(predicted))]})
        print("inference: ", predicted)


class HardFullTreePrior(Noop):
    accepts_path_graph_analysis = True
    accepts_path_wnids = True
    accepts_json_save_path = True
    accepts_weighted_average = True

    """Evaluates model on a decision tree prior. Evaluation is deterministic."""
    """Evaluates on entire tree, tracks all paths."""
    def __init__(self, trainset, testset, experiment_name, path_graph_analysis, path_wnids, json_save_path, csv_save_path=None,
                 weighted_average=False):
        super().__init__(trainset, testset, experiment_name)
        # weird, sometimes self.classes are wnids, and sometimes they are direct classes.
        # just gotta do a check. Its basically CIFAR vs wordnet
        self.nodes = Node.get_nodes(path_graph_analysis, path_wnids, trainset.classes)
        self.G = self.nodes[0].G
        self.wnid_to_node = {node.wnid: node for node in self.nodes}

        self.wnids = get_wnids(path_wnids)
        self.classes = trainset.classes
        self.wnid_to_class = {wnid: cls for wnid, cls in zip(self.wnids, self.classes)}

        self.weighted_average = weighted_average
        self.correct = 0
        self.total = 0

        self.wnid_to_name = {wnid: synset_to_name(wnid_to_synset(wnid)) for wnid in self.wnids}
        self.leaf_counts = {cls:{node:0 for node in get_leaves(self.G)} for cls in self.classes}
        self.node_counts = {cls:{node.wnid:0 for node in self.nodes} for cls in self.classes}
        self.class_counts = {cls:0 for cls in self.classes}  # count how many samples weve seen for each class
        for cls in self.classes:
            self.node_counts[cls].update({wnid:0 for wnid in self.wnids})
        self.csv_save_path = csv_save_path
        self.json_save_path = json_save_path

        self.class_to_wnid = {self.wnid_to_class[wnid]:wnid for wnid in self.wnids}

    def update_batch(self, outputs, predicted, targets):
        wnid_to_pred_selector = {}
        n_samples = outputs.size(0)

        for node in self.nodes:
            outputs_sub = HardTreeSupLoss.get_output_sub(outputs, node, self.weighted_average)
            _, preds_sub = torch.max(outputs_sub, dim=1)
            preds_sub = list(map(int, preds_sub.cpu()))
            wnid_to_pred_selector[node.wnid] = preds_sub

        paths = self.traverse_tree(wnid_to_pred_selector, n_samples, targets)
        for cls, leaf in zip(targets.numpy(), paths):
            self.leaf_counts[self.classes[cls]][leaf] += 1
            self.class_counts[self.classes[cls]] += 1

        predicted = [self.classes.index(self.wnid_to_class[wnid]) for wnid in paths]
        self.correct += np.sum((predicted == targets.numpy()))
        self.total += len(paths)
        accuracy = round(self.correct / self.total, 4) * 100
        return f'TreePrior: {accuracy}%'

    # return leaf node wnids corresponding to each output
    def traverse_tree(self, wnid_to_pred_selector, nsamples, targets):
        leaf_wnids = []
        wnid_root = get_root(self.G)
        node_root = self.wnid_to_node[wnid_root]
        target_classes = targets.numpy()
        for index in range(nsamples):
            wnid, node = wnid_root, node_root
            while node is not None:
                pred_sub = wnid_to_pred_selector[node.wnid]
                index_child = pred_sub[index]
                wnid = node.children[index_child]
                node = self.wnid_to_node.get(wnid, None)
                try:
                    self.node_counts[self.class_to_wnid[self.classes[target_classes[index]]]][wnid] += 1
                except:
                    self.node_counts[self.classes[target_classes[index]]][wnid] += 1
            leaf_wnids.append(wnid)
        return leaf_wnids

    def end_test(self, epoch):
        if self.csv_save_path is not None:
            self.write_to_csv(self.csv_save_path)
        self.write_to_json(self.json_save_path)

    def write_to_csv(self, path):
        columns = {node:[] for node in get_leaves(self.G)}
        for cls in self.classes:
            for node in get_leaves(self.G):
                if node in self.leaf_counts[cls]:
                    columns[node].append(self.leaf_counts[cls][node])
                else:
                    columns[node].append(0)
        new_columns = {}
        for node in get_leaves(self.G):
            new_columns["%s %s" % (synset_to_name(wnid_to_synset(node)), node)] = columns[node]
        try:
            int(self.classes[1:])
            index = [self.wnid_to_name[cls] for cls in self.classes]
        except:
            index = [cls for cls in self.classes]
        df = pd.DataFrame(data=new_columns, index=index)
        df.to_csv(path)
        print("CSV saved to %s" % path)

    def write_to_json(self, path):
        # create separate graph for each node
        if not os.path.exists(path):
            os.makedirs(path)
        for cls in self.classes:
            try:
                int(cls[1:])
                cls = self.class_to_wnid[cls]
            except:
                pass
            G = nx.DiGraph(self.G)
            for node in self.G.nodes():
                G.nodes[node]['weight'] = self.node_counts[cls][node] / self.class_counts[cls]
            G.nodes[get_root(self.G)]['weight'] = 1
            json_data = node_link_data(G)
            try:
                int(cls[1:])
                cls = self.wnid_to_name[cls]
            except:
                pass
            cls_path = path + cls + '.json'
            with open(cls_path, 'w') as f:
                json.dump(json_data, f)
            print("Json saved to %s" % cls_path)
