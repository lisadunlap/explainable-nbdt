from nbdt.graph import get_root, get_roots, get_wnids, synset_to_name, wnid_to_synset, get_leaves, get_path_to_node
from nbdt.utils import (
    DEFAULT_CIFAR10_TREE, DEFAULT_CIFAR10_WNIDS, DEFAULT_CIFAR100_TREE,
    DEFAULT_CIFAR100_WNIDS, DEFAULT_TINYIMAGENET200_TREE,
    DEFAULT_TINYIMAGENET200_WNIDS, DEFAULT_IMAGENET1000_TREE,
    DEFAULT_IMAGENET1000_WNIDS, set_np_printoptions
)
from nbdt.loss import HardTreeSupLoss, SoftTreeSupLoss
from nbdt.data.custom import Node
from generate_vis import generate_vis, build_tree
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
from saliency.RISE.explanations import RISE
from saliency.RISE.utils import get_cam
from saliency.Grad_CAM.gcam import GradCAM
from PIL import Image
import cv2


__all__ = names = (
    'Noop', 'ConfusionMatrix', 'HardEmbeddedDecisionRules', 'SoftEmbeddedDecisionRules',
    'SingleInference', 'HardFullTreePrior', 'HardTrackNodes', 'SoftFullTreePrior', 'SoftTrackDepth', 'SoftFullTreeOODPrior',
    'SingleRISE', 'SingleGradCAM')
keys = ('path_graph', 'path_graph_analysis', 'path_wnids', 'weighted_average',
        'trainset', 'testset', 'json_save_path', 'experiment_name', 'csv_save_path', 'ignore_labels',
        'oodset', 'ood_path_wnids')

def add_arguments(parser):
    parser.add_argument('--json-save-path', default=None, type=str,
                    help='Directory to save jsons under for full tree analysis')
    parser.add_argument('--csv-save-path', default=None, type=str,
                    help='Directory to save jsons under for full tree analysis')
    parser.add_argument('--path-graph-analysis', default=None, type=str,
                    help='path for graph for analysis')
    parser.add_argument('--track-nodes', default=None, type=str, nargs='*',
                    help='node wnids to track')
    parser.add_argument('--ignore-labels', nargs='*', type=int,
                    help='node label indices to ignore for zeroshot')

class Noop:

    accepts_trainset = lambda trainset, **kwargs: trainset
    accepts_testset = lambda testset, **kwargs: testset

    def __init__(self, trainset, testset, experiment_name,
                 use_wandb=False, run_name="Noop"):
        set_np_printoptions()

        self.trainset = trainset
        self.testset = testset
        self.use_wandb = use_wandb
        if self.use_wandb:
            wandb.init(project=experiment_name, name=run_name, reinit=True, entity='lisadunlap')

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

    def write_to_csv(self, path):
        pass


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
    accepts_ignore_labels = True

    def __init__(self, trainset, testset, experiment_name, path_graph, path_wnids, ignore_labels=[],
            weighted_average=False, use_wandb=False, run_name="HardEmbeddedDecisionRules"):
        super().__init__(trainset, testset, experiment_name, use_wandb,
                         run_name=run_name)
        self.nodes = Node.get_nodes(path_graph, path_wnids, trainset.classes)
        self.G = self.nodes[0].G
        self.wnid_to_node = {node.wnid: node for node in self.nodes}
        self.ignore_labels = ignore_labels

        self.wnids = get_wnids(path_wnids)
        self.classes = trainset.classes
        self.wnid_to_class = {wnid: cls for wnid, cls in zip(self.wnids, self.classes)}

        self.weighted_average = weighted_average
        self.correct = 0
        self.total = 0
        self.class_accuracies = {c:0 for c in self.classes}
        self.class_totals = {c: 0 for c in self.classes}


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
        # return f'NBDT-Hard: {accuracy}%'
        return (predicted == targets).sum().item()

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
        # print([(self.class_accuracies[k]/self.class_totals[k])*100 for k in self.class_accuracies.keys()])
        # if self.use_wandb:
        #     wandb.run.summary["NBDT hard accuracy"] = accuracy
        #     data = [[(self.class_accuracies[k]/self.class_totals[k])*100 for k in self.class_accuracies.keys()]]
        #     wandb.log({"class accuracies": wandb.Table(data=data, columns=self.classes)})


class SoftEmbeddedDecisionRules(HardEmbeddedDecisionRules):
    """Evaluation is soft."""

    def __init__(self, trainset, testset, experiment_name, path_graph, path_wnids,
            weighted_average=False, use_wandb=False, run_name="SoftEmbeddedDecisionRules"):
        super().__init__(trainset, testset, experiment_name, path_graph, path_wnids, use_wandb,
                         run_name=run_name)
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
        #return f'NBDT-Soft: {accuracy}%'
        return (predicted == targets).sum().item()

    def end_test(self, epoch):
        accuracy = round(self.correct / self.total * 100., 2)
        print(f'NBDT-Soft Accuracy: {accuracy}%, {self.correct}/{self.total}')
        if self.use_wandb:
            data = [[(self.class_accuracies[k] / self.class_totals[k]) * 100 for k in self.class_accuracies.keys()]]
            wandb.log({"class accuracies": wandb.Table(data=data, columns=self.classes)})

class SingleInference(HardEmbeddedDecisionRules):
    """Inference on a single image ."""

    def __init__(self, trainset, testset, experiment_name, path_graph, path_wnids,
            weighted_average=False, use_wandb=False, run_name="SingleInference"):
        super().__init__(trainset, testset, experiment_name, path_graph, path_wnids, use_wandb,
                         run_name=run_name)
        get_path = lambda wnid: nx.shortest_path(self.G, source=get_root(self.G), target=wnid)
        self.paths = {self.wnid_to_class[wnid]: get_path(wnid) for wnid in self.wnids}
        self.num_classes = len(trainset.classes)

    def single_traversal(self, _, wnid_to_pred_selector):
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

    def inf(self, img, outputs):
        wnid_to_pred_selector = {}
        for node in self.nodes:
            outputs_sub = HardTreeSupLoss.get_output_sub(
                outputs, node, self.weighted_average)
            selector = [1 for c in range(node.num_classes)]
            if not any(selector):
                continue
            _, preds_sub = torch.max(outputs_sub, dim=1)
            preds_sub = list(map(int, preds_sub.cpu()))
            wnid_to_pred_selector[node.wnid] = (preds_sub, selector)
        predicted = self.single_traversal(
            [], wnid_to_pred_selector)
        wandb.log({"examples": [wandb.Image(torch.squeeze(img).cpu().numpy().transpose((1, 2, 0)), caption=str(predicted))]})
        cls = self.wnid_to_class.get(predicted[-1], None)
        pred = -1 if cls is None else self.classes.index(cls)
        print("class: ", pred)
        print("inference: ", predicted)

class HardFullTreePrior(Noop):
    accepts_path_graph = True
    accepts_path_wnids = True
    accepts_json_save_path = True
    accepts_weighted_average = True
    accepts_ignore_labels = True

    """Evaluates model on a decision tree prior. Evaluation is deterministic."""
    """Evaluates on entire tree, tracks all paths."""
    def __init__(self, trainset, testset, experiment_name, path_graph, path_wnids, ignore_labels=[],
                 json_save_path='./out/full_tree_analysis/', csv_save_path='./out/cifar100.csv',
                 weighted_average=False, use_wandb=False, run_name="HardFullTreePrior"):
        super().__init__(trainset, testset, experiment_name, use_wandb, run_name=run_name)
        # weird, sometimes self.classes are wnids, and sometimes they are direct classes.
        # just gotta do a check. Its basically CIFAR vs wordnet
        self.nodes = Node.get_nodes(path_graph, path_wnids, trainset.classes)
        self.G = self.nodes[0].G
        self.wnid_to_node = {node.wnid: node for node in self.nodes}
        self.ignore_labels = ignore_labels

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
        if not os.path.exists(self.json_save_path):
            os.mkdir(self.json_save_path)

        self.class_to_wnid = {self.wnid_to_class[wnid]:wnid for wnid in self.wnids}
        self.class_accuracies = {c: 0 for c in self.classes}
        self.class_totals = {c: 0 for c in self.classes}
        self.ignored_classes = ()

    def update_batch(self, outputs, predicted, targets):
        wnid_to_pred_selector = {}
        n_samples = outputs.size(0)

        for node in self.nodes:
            ignore_classes_pruned = node.prune_ignore_labels(self.ignore_labels)
            outputs_sub = HardTreeSupLoss.get_output_sub(outputs, node, self.weighted_average, ignore_classes_pruned)
            _, preds_sub = torch.max(outputs_sub, dim=1)
            preds_sub = list(map(int, preds_sub.cpu()))
            wnid_to_pred_selector[node.wnid] = preds_sub

        paths = self.traverse_tree(wnid_to_pred_selector, n_samples, targets)
        for cls, leaf in zip(targets.numpy(), paths):
            self.leaf_counts[self.classes[cls]][leaf] += 1
            self.class_counts[self.classes[cls]] += 1

        for i in range(len(predicted)):
            self.class_accuracies[self.classes[predicted[i]]] += int(predicted[i] == targets[i])
            self.class_totals[self.classes[targets[i]]] += 1

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
        if self.csv_save_path is not None or self.use_wandb:
            self.write_to_csv(self.csv_save_path)
        self.write_to_json(self.json_save_path)
        if self.use_wandb:
            for cls in self.class_accuracies:
                label = cls+"-acc"
                wandb.run.summary[label] = self.class_accuracies[cls]
        print(self.class_accuracies)

    def write_to_csv(self, path):
        columns = {node:[] for node in get_leaves(self.G)}
        classes_to_count = self.classes
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
            int(classes_to_count[1:])
            index = [self.wnid_to_name[cls] for cls in classes_to_count]
        except:
            index = [cls for cls in classes_to_count]
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
                ignore=self.class_counts[cls] == 0
                G.nodes[node]['weight'] = self.node_counts[cls][node] / (self.class_counts[cls] + 1e-4)
            G.nodes[get_root(self.G)]['weight'] = 1
            json_data = node_link_data(G)
            try:
                int(cls[1:])
                cls = self.wnid_to_name[cls]
            except:
                pass
            if not ignore:
                cls_path = path + cls + '.json'
                with open(cls_path, 'w') as f:
                    json.dump(json_data, f)
                print("Json saved to %s" % cls_path)
                root = next(get_roots(G))
                tree = build_tree(G, root)
                generate_vis(os.getcwd() + '/vis/tree-weighted-template.html', tree, 'tree', cls, out_dir=path)
                if self.use_wandb:
                    wandb.log({cls + "-path": wandb.Html(open(cls_path.replace('.json', '') + '-tree.html'), inject=False)})
                print("Json saved to %s" % cls_path)


class HardTrackNodes(HardFullTreePrior):
    accepts_path_wnids = True
    accepts_weighted_average = True
    accepts_track_nodes = True

    """Evaluates model on a decision tree prior. Evaluation is deterministic."""
    """Evaluates on entire tree, tracks all paths. Additionally, tracks which images
        go to each node by retaining their index numbers. Stores this into a json.
        Note: only works if dataloader for evaluation is NOT shuffled."""
    def __init__(self, trainset, testset, experiment_name, path_graph, path_wnids, track_nodes,
        json_save_path='./out/hard_track_nodes_analysis/', csv_save_path='./out/hard_track_nodes_analysis.csv', weighted_average=False,
        use_wandb=False, run_name="HardTrackNodes"):
        super().__init__(trainset, testset, experiment_name, path_graph, path_wnids, json_save_path,
                         csv_save_path, weighted_average, use_wandb, run_name)
        self.track_nodes = {wnid:[] for wnid in track_nodes}

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
                if wnid in self.track_nodes:
                    self.track_nodes[wnid].append(self.total + index)
                node = self.wnid_to_node.get(wnid, None)
                try:
                    self.node_counts[self.class_to_wnid[self.classes[target_classes[index]]]][wnid] += 1
                except:
                    self.node_counts[self.classes[target_classes[index]]][wnid] += 1
            leaf_wnids.append(wnid)
        return leaf_wnids

    def write_to_json(self, path):
        # create separate graph for each node
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        for cls in self.classes:
            cls_path = path + cls + '.json'
            with open(cls_path, 'w') as f:
                json.dump(self.track_nodes, f)

                G = nx.DiGraph(self.G)
                for node in self.G.nodes():
                    if self.class_counts[cls] == 0:
                        print(cls)
                    G.nodes[node]['weight'] = self.node_counts[cls][node] / self.class_counts[cls]
                G.nodes[get_root(self.G)]['weight'] = 1

                root=next(get_roots(G))
                tree = build_tree(G, root)
                generate_vis(os.getcwd()+'/vis/tree-weighted-template.html', tree, 'tree', cls, out_dir=path)
                if self.use_wandb:
                    wandb.log({cls+"-path": wandb.Html(open(cls_path.replace('.json', '')+'-tree.html'), inject=False)})
                print("Json saved to %s" % cls_path)

class SoftFullTreePrior(HardFullTreePrior):

    """Evaluates model on a decision tree prior. Evaluation is soft.
     """

    def __init__(self, trainset, testset, experiment_name, path_graph, path_wnids, ignore_labels=[],
                 json_save_path='./out/full_tree_analysis/', csv_save_path='./out/cifar100.csv',
                 weighted_average=False, use_wandb=False, run_name="SoftFullTreePrior"):
        super().__init__(trainset, testset, experiment_name, path_graph, path_wnids, ignore_labels,
                 json_save_path, csv_save_path, weighted_average, use_wandb, run_name)
        self.num_classes = len(trainset.classes)
        get_path = lambda wnid: nx.shortest_path(self.G, source=get_root(self.G), target=wnid)
        self.paths = {self.wnid_to_class[wnid]: get_path(wnid) for wnid in self.wnids}

    def update_batch(self, outputs, predicted, targets):
        bayesian_outputs = SoftTreeSupLoss.inference(
            self.nodes, outputs, self.num_classes, self.weighted_average)
        n_samples = outputs.size(0)
        predicted = bayesian_outputs.max(1)[1].to(targets.device)
        paths = self.traverse_tree(predicted.cpu().numpy(), n_samples, targets)

        for cls, leaf in zip(targets.numpy(), paths):
            self.leaf_counts[self.classes[cls]][leaf] += 1
            self.class_counts[self.classes[cls]] += 1

        for i in range(len(predicted)):
            self.class_accuracies[self.classes[predicted[i]]] += int(predicted[i] == targets[i])
            self.class_totals[self.classes[targets[i]]] += 1

        predicted = [self.classes.index(self.wnid_to_class[wnid]) for wnid in paths]
        self.correct += np.sum((predicted == targets.numpy()))
        self.total += len(paths)
        accuracy = round(self.correct / self.total, 4) * 100
        return f'TreePrior: {accuracy}%'

    # return leaf node wnids corresponding to each output
    def traverse_tree(self, wnid_to_pred_selector, nsamples, targets):
        target_classes = targets.numpy()
        for index in range(nsamples):
            path = self.paths[self.classes[wnid_to_pred_selector[index]]]
            for wnid in path:
                try:
                    self.node_counts[self.class_to_wnid[self.classes[target_classes[index]]]][wnid] += 1
                except:
                    self.node_counts[self.classes[target_classes[index]]][wnid] += 1
        return [self.wnids[i] for i in wnid_to_pred_selector]

class SoftTrackDepth(SoftFullTreePrior):

    """ Track depth metric with SoftFullTreePrior
     """

    def __init__(self, trainset, testset, experiment_name, path_graph, path_wnids, ignore_labels=[],
                 json_save_path='./out/soft_track_depth/', csv_save_path='./out/soft_track_depth/cifar10.csv',
                 weighted_average=False, use_wandb=False, run_name="SoftTrackDepth"):
        super().__init__(trainset, testset, experiment_name, path_graph, path_wnids, ignore_labels,
                 json_save_path, csv_save_path, weighted_average, use_wandb, run_name)

    def calculate_depth_metrics(self):
        self.depth_counts = {cls: {"depth": 0, "total": 0} for cls in self.classes}
        for cls in self.classes:
            cls_counts = self.node_counts[cls]
            cls_wnid = self.class_to_wnid[cls]
            cls_node = self.G.nodes[self.class_to_wnid[cls]]
            true_path_wnids = get_path_to_node(self.G, self.class_to_wnid[cls])
            cls_depth_count, cls_total_count = 0, 0

            for node in true_path_wnids:
                cls_depth_count += cls_counts.get(node, 0)
                cls_total_count += self.class_counts[cls]
            self.depth_counts[cls] = {
                "depth": cls_depth_count,
                "total": cls_total_count,
                "ratio": cls_depth_count / cls_total_count,
            }
        return self.depth_counts

    def end_test(self, epoch):
        self.calculate_depth_metrics()
        print("===> Depth metrics:")
        for cls, depth_dict in self.depth_counts.items():
            print(f"{cls}: {depth_dict['ratio']} ({depth_dict['depth']} / {depth_dict['total']})")
        total_depth_counts = sum(d["depth"] for d in self.depth_counts.values())
        total_counts = sum(d["total"] for d in self.depth_counts.values())
        print(f"Total: {total_depth_counts / total_counts} ({total_depth_counts} / {total_counts})")

class SoftFullTreeOODPrior(SoftFullTreePrior):

    """Evaluates model on a decision tree prior. Evaluation is soft.
     """
    accepts_path_graph = True
    accepts_path_wnids = True
    accepts_json_save_path = True
    accepts_weighted_average = True
    accepts_csv_save_path = True
    accepts_ignore_labels = True
    accepts_oodset = True
    accepts_ood_path_wnids = True

    def __init__(self, trainset, testset, experiment_name, path_graph, path_wnids,
                 oodset, ood_path_wnids, ignore_labels=[],
                 json_save_path='./out/soft_full_tree_analysis/', csv_save_path='./out/cifar100.csv',
                 weighted_average=False, use_wandb=False, run_name="SoftFullTreeOODPrior"):
        self.weighted_average = weighted_average
        self.use_wandb = use_wandb
        self.csv_save_path = csv_save_path
        self.json_save_path = json_save_path
        if not os.path.exists(self.json_save_path):
            os.mkdir(self.json_save_path)

        self.nodes = Node.get_nodes(path_graph, path_wnids, trainset.classes, ood_path_wnids)
        self.G = self.nodes[0].G
        self.wnid_to_node = {node.wnid: node for node in self.nodes}

        self.wnids = get_wnids(path_wnids, ood_path_wnids)
        self.classes = trainset.classes
        self.wnid_to_class = {wnid: cls for wnid, cls in zip(self.wnids, self.classes)}
        self.wnid_to_name = {wnid: synset_to_name(wnid_to_synset(wnid)) for wnid in self.wnids}

        self.ood_classes = oodset.classes
        self.ood_wnids = get_wnids(ood_path_wnids)
        self.wnid_to_class.update({wnid: cls for wnid, cls in zip(self.ood_wnids, self.ood_classes)})
        self.class_to_wnid = {self.wnid_to_class[wnid]:wnid for wnid in self.wnid_to_class.keys()}

        self.leaf_counts = {cls:{node:0 for node in get_leaves(self.G)} for cls in self.ood_classes}
        self.class_counts = {cls:0 for cls in self.ood_classes}
        self.node_counts = {} # count how many samples weve seen for each class

        for cls in self.ood_classes:
            curr_counts = {w: 0 for w in self.wnid_to_class.keys()}
            curr_counts.update({n.wnid: 0 for n in self.nodes})
            self.node_counts[cls] = curr_counts

        self.num_classes = len(trainset.classes)
        get_path = lambda wnid: nx.shortest_path(self.G, source=get_root(self.G), target=wnid)
        self.paths = {self.wnid_to_class[wnid]: get_path(wnid) for wnid in self.wnids}

    def update_batch(self, outputs, predicted, targets):
        bayesian_outputs = SoftTreeSupLoss.inference(
            self.nodes, outputs, self.num_classes, self.weighted_average)
        n_samples = outputs.size(0)
        predicted = bayesian_outputs.max(1)[1].to(targets.device)
        paths = self.traverse_tree(predicted.cpu().numpy(), n_samples, targets)

        for cls, leaf in zip(targets.numpy(), paths):
            self.leaf_counts[self.ood_classes[cls]][leaf] += 1
            self.class_counts[self.ood_classes[cls]] += 1

        accuracy = -1 # cannot evaluate accuracy for OOD samples
        return f'TreePrior: {accuracy}%'

    # return leaf node wnids corresponding to each output
    def traverse_tree(self, wnid_to_pred_selector, nsamples, targets):
        target_classes = targets.numpy()
        for index in range(nsamples):
            path = self.paths[self.classes[wnid_to_pred_selector[index]]]
            for wnid in path:
                try:
                    self.node_counts[self.class_to_wnid[self.ood_classes[target_classes[index]]]][wnid] += 1
                except:
                    self.node_counts[self.ood_classes[target_classes[index]]][wnid] += 1
        return [self.wnids[i] for i in wnid_to_pred_selector]

    def write_to_csv(self, path):
        columns = {node:[] for node in get_leaves(self.G)}
        for cls in self.ood_classes:
            for node in get_leaves(self.G):
                if node in self.leaf_counts[cls]:
                    columns[node].append(self.leaf_counts[cls][node])
                else:
                    columns[node].append(0)
        new_columns = {}
        for node in get_leaves(self.G):
            new_columns["%s %s" % (synset_to_name(wnid_to_synset(node)), node)] = columns[node]
        try:
            int(self.ood_classes[1:])
            index = [self.wnid_to_name[cls] for cls in self.ood_classes]
        except:
            index = [cls for cls in self.ood_classes]
        df = pd.DataFrame(data=new_columns, index=index)
        df.to_csv(path)
        print("CSV saved to %s" % path)

    def write_to_json(self, path):
        # create separate graph for each node
        if not os.path.exists(path):
            os.makedirs(path)
        for cls in self.ood_classes:
            try:
                int(cls[1:])
                cls = self.class_to_wnid[cls]
            except:
                pass
            G = nx.DiGraph(self.G)
            for node in self.G.nodes():
                if self.class_counts[cls] == 0:
                    print(cls)
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

class SingleRISE(SingleInference):
    """Generate RISE saliency map for a single image ."""

    def __init__(self, trainset, testset, experiment_name, path_graph, path_wnids, net,
            weighted_average=False, use_wandb=True, run_name="SingleRISE"):
        super().__init__(trainset, testset, experiment_name, path_graph, path_wnids, use_wandb=use_wandb,
                         run_name=run_name)
        try:
            H,L,W=trainset[0][0].shape
        except:
            H, L, W = trainset[0][0][0].shape
        print("INPUT SIZE: ", (L,W))
        self.net = net
        self.rise = RISE(net, input_size=(L,W))
        self.use_wandb = True

    def inf(self, img, outputs):
        print("=====> starting RISE", img.shape)
        wnid_to_pred_selector = {}
        wnid_to_rise = {}
        examples, sals = [], []
        for node in self.nodes:
            outputs_sub = HardTreeSupLoss.get_output_sub(
                outputs, node, self.weighted_average)
            outputs_sub = nn.functional.softmax(outputs_sub, dim=1)
            selector = [1 for c in range(node.num_classes)]
            if not any(selector):
                continue
            _, preds_sub = torch.max(outputs_sub, dim=1)
            preds_sub = list(map(int, preds_sub.cpu()))
            wnid_to_pred_selector[node.wnid] = (preds_sub, selector)

        predicted = self.single_traversal(
            [], wnid_to_pred_selector)
        for node in self.nodes:
            if node.wnid in predicted:
                print("Generating Rise for ", node.wnid)
                rise_saliency = self.rise.explain_instance(img, node, self.weighted_average)
                wnid_to_rise[node.wnid] = rise_saliency
        if self.use_wandb:
            print("log image")
            wandb.log({"examples": [wandb.Image(torch.squeeze(img).cpu().numpy().transpose((1, 2, 0)),
                                                caption=str(predicted))]})
        cls = self.wnid_to_class.get(predicted[-1], None)
        pred = -1 if cls is None else self.classes.index(cls)
        print("class: ", pred)
        print("inference: ", predicted)

        for wnid, rise_output in wnid_to_rise.items():
            overlay = get_cam(torch.squeeze(img), rise_output.cpu().detach().numpy())
            sals.append(wandb.Image(overlay, caption=f"RISE (wnid={wnid}, idx={predicted.index(wnid)})"))
            if not os.path.exists("./out/RISE/"):
                os.makedirs("./out/RISE/")
            if not cv2.imwrite(f"./out/RISE/RISE_{wnid}.jpg", overlay):
                print("ERROR writing image to file")
        if self.use_wandb:
            print("logging")
            wandb.log({"rise examples": sals})

class SingleGradCAM(SingleInference):
    """Generate RISE saliency map for a single image ."""

    def __init__(self, trainset, testset, experiment_name, path_graph, path_wnids, net,
            weighted_average=False, use_wandb=True, run_name="SingleGradCAM"):
        super().__init__(trainset, testset, experiment_name, path_graph, path_wnids, use_wandb=use_wandb,
                         run_name=run_name)
        try:
            H,L,W=trainset[0][0].shape
        except:
            H, L, W = trainset[0][0][0].shape
        print("INPUT SIZE: ", (L,W))
        self.net = net
        self.gcam = GradCAM(model=net)
        self.use_wandb = True

    def inf(self, img, outputs):
        wnid_to_pred_selector = {}
        wnid_to_rise = {}
        examples, sals = [], []
        for node in self.nodes:
            outputs_sub = HardTreeSupLoss.get_output_sub(
                outputs, node, self.weighted_average)
            outputs_sub = nn.functional.softmax(outputs_sub, dim=1)
            selector = [1 for c in range(node.num_classes)]
            if not any(selector):
                continue
            _, preds_sub = torch.max(outputs_sub, dim=1)
            preds_sub = list(map(int, preds_sub.cpu()))
            wnid_to_pred_selector[node.wnid] = (preds_sub, selector)

        predicted = self.single_traversal(
            [], wnid_to_pred_selector)
        for node in self.nodes:
            if node.wnid in predicted:
                print("Generating GradCAM for ", node.wnid)

                rise_saliency = self.gen_gcam(img, node)
                rise_saliency = self.get_mask(rise_saliency)
                wnid_to_rise[node.wnid] = rise_saliency
        if self.use_wandb:
            print("log image")
            wandb.log({"examples": [wandb.Image(torch.squeeze(img).cpu().numpy().transpose((1, 2, 0)),
                                                caption=str(predicted))]})
        cls = self.wnid_to_class.get(predicted[-1], None)
        pred = -1 if cls is None else self.classes.index(cls)
        print("class: ", pred)
        print("inference: ", predicted)

        for wnid, rise_output in wnid_to_rise.items():
            overlay = get_cam(torch.squeeze(img), rise_output)
            if wnid in predicted:
                sals.append(wandb.Image(overlay, caption=f"GradCAM (idx={predicted.index(wnid)})"))
            else:
                sals.append(wandb.Image(overlay, caption=f"GradCAM (wnid={wnid})"))
            if not os.path.exists("./out/GradCAM/"):
                os.makedirs("./out/GradCAM/")
            if not cv2.imwrite(f"./out/GradCAM/GradCAM_{wnid}.jpg", overlay):
                print("ERROR writing image to file")
        if self.use_wandb:
            print("logging")
            wandb.log({"gcam examples": sals})

    def gen_gcam(self, img, node, target_index=1):
        """
        Visualize model responses given multiple images
        """

        # Get model and forward pass
        probs, ids = self.gcam.forward(img, node)

        for i in range(target_index):
            # Grad-CAM
            self.gcam.backward(ids=ids[:, [i]])
            regions = self.gcam.generate(target_layer='module.layer4')
            masks = []
            for j in range(len(img)):

                # Grad-CAM
                mask = regions[j, 0].cpu().numpy()
                masks += [mask]
        if len(masks) == 1:
            return masks[0]
        self.gcam.remove_hook()
        return masks

    def get_mask(self, mask, sigma=.55, omega=100):
        sigma *= np.max(mask)
        mask = 1/(1+np.exp(-omega*(mask - sigma)))
        return mask
