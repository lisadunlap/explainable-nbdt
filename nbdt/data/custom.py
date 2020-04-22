import torchvision.datasets as datasets
import torch
import numpy as np
import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchsample
import json
from collections import defaultdict
from nbdt.utils import (
    DEFAULT_CIFAR10_TREE, DEFAULT_CIFAR10_WNIDS, DEFAULT_CIFAR100_TREE,
    DEFAULT_CIFAR100_WNIDS, DEFAULT_TINYIMAGENET200_TREE,
    DEFAULT_TINYIMAGENET200_WNIDS, DEFAULT_IMAGENET1000_TREE,
    DEFAULT_IMAGENET1000_WNIDS, DATASET_TO_PATHS
)
from collections import defaultdict
from nbdt.graph import get_wnids, read_graph, get_leaves, get_non_leaves, \
    get_leaf_weights
from . import imagenet
from PIL import Image
import torch.nn as nn
import random


__all__ = names = ('CIFAR10IncludeLabels',
                   'CIFAR100IncludeLabels', 'TinyImagenet200IncludeLabels',
                   'Imagenet1000IncludeLabels','CIFAR10IncludeClasses',
                   'CIFAR100IncludeClasses', 'TinyImagenet200IncludeClasses',
                   'Imagenet1000IncludeClasses', 'CIFAR10ExcludeLabels',
                   'CIFAR100ExcludeLabels', 'TinyImagenet200ExcludeLabels',
                   'Imagenet1000ExcludeLabels', 'CIFAR10ExcludeClasses',
                   'CIFAR100ExcludeClasses', 'TinyImagenet200ExcludeClasses',
                   'Imagenet1000ExcludeClasses','CIFAR10ResampleLabels',
                   'CIFAR100ResampleLabels', 'TinyImagenet200ResampleLabels',
                   'Imagenet1000ResampleLabels', 'CIFAR10CombineClasses',
                   'CIFAR100CombineClasses', 'TinyImagenet200CombineClasses',
                   'Imagenet1000CombineClasses', 'TinyImagenet200GradCAM',)
keys = ('include_labels', 'exclude_labels', 'include_classes', 'probability_labels', 'combine_classes')


def add_arguments(parser):
    parser.add_argument('--probability-labels', nargs='*', type=float)
    parser.add_argument('--include-labels', nargs='*', type=int)
    parser.add_argument('--exclude-labels', nargs='*', type=int)
    parser.add_argument('--include-classes', nargs='*', type=str)
    parser.add_argument('--exclude-classes', nargs='*', type=str)
    parser.add_argument('--combine-classes', nargs='+', type=str, action='append')



def set_default_values(args):
    print(DATASET_TO_PATHS)
    paths = DATASET_TO_PATHS[args.dataset.replace('IncludeLabels', '').replace('IncludeClasses', '').replace('ExcludeLabels', '').replace('ResampleLabels', '').replace('CombineLabels', '').replace('CombineClasses', '')]
    if not args.path_graph:
        args.path_graph = paths['path_graph']
    if not args.path_wnids:
        args.path_wnids = paths['path_wnids']


class Node:

    def __init__(self, wnid, classes,
            path_graph=DEFAULT_CIFAR10_TREE,
            path_wnids=DEFAULT_CIFAR10_WNIDS,
            other_class=False,
            path_wnids_ood=None):
        self.path_graph = path_graph
        self.path_wnids = path_wnids
        self.path_wnids_ood = path_wnids_ood

        self.wnid = wnid
        self.wnids = get_wnids(path_wnids, path_wnids_ood)
        self.G = read_graph(path_graph)

        self.original_classes = classes
        self.num_original_classes = len(self.wnids)

        assert not self.is_leaf(), 'Cannot build dataset for leaf'
        self.has_other = other_class and not (self.is_root() or self.is_leaf())
        self.num_children = len(self.get_children())
        self.num_classes = self.num_children + int(self.has_other)

        self.old_to_new_classes, self.new_to_old_classes = \
            self.build_class_mappings()
        self.classes = self.build_classes()

        assert len(self.classes) == self.num_classes, (
            f'Number of classes {self.num_classes} does not equal number of '
            f'class names found ({len(self.classes)}): {self.classes}'
        )

        self.children = list(self.get_children())
        self.leaves = list(self.get_leaves())
        self.num_leaves = len(self.leaves)

        # I'm sure leaf_weights and output_weights could be recursive/share
        # computation and be more efficient.... buuuuut this is only run once
        # ANYways
        self.leaf_weights = get_leaf_weights(self.G, self.wnid)
        self.new_to_leaf_weights = self.get_new_to_leaf_weights()

        self._probabilities = None
        self._class_weights = None

    def wnid_to_class_index(self, wnid):
        return self.wnids.index(wnid)

    def get_parents(self):
        return self.G.pred[self.wnid]

    def get_children(self):
        return self.G.succ[self.wnid]

    def get_leaves(self):
        return get_leaves(self.G, self.wnid)

    def is_leaf(self):
        return len(self.get_children()) == 0

    def is_root(self):
        return len(self.get_parents()) == 0

    def move_leaf_weights_to(self, device):
        for new_index in self.new_to_leaf_weights:
            self.new_to_leaf_weights[new_index] = self.new_to_leaf_weights[new_index].to(device)

    def get_new_to_leaf_weights(self):
        new_to_leaf_weights = {}
        for new_index, child in enumerate(self.get_children()):
            leaf_weights = [0] * self.num_original_classes
            for leaf, weight in self.leaf_weights.items():
                old_index = self.wnid_to_class_index(leaf)
                leaf_weights[old_index] = weight
            assert abs(sum(leaf_weights) - 1) < 1e-3, \
                'Leaf weights do not sum to 1.'
            new_to_leaf_weights[new_index] = torch.Tensor(leaf_weights)
        return new_to_leaf_weights

    def build_class_mappings(self):
        old_to_new = defaultdict(lambda: [])
        new_to_old = defaultdict(lambda: [])
        for new_index, child in enumerate(self.get_children()):
            for leaf in get_leaves(self.G, child):
                old_index = self.wnid_to_class_index(leaf)
                old_to_new[old_index].append(new_index)
                new_to_old[new_index].append(old_index)
        if not self.has_other:
            return old_to_new, new_to_old

        new_index = self.num_children
        for old in range(self.num_original_classes):
            if old not in old_to_new:
                old_to_new[old].append(new_index)
                new_to_old[new_index].append(old)
        return old_to_new, new_to_old

    def build_classes(self):
        return [
            ','.join([self.original_classes[old] for old in old_indices])
            for new_index, old_indices in sorted(
                self.new_to_old_classes.items(), key=lambda t: t[0])
        ]

    def prune_ignore_labels(self, ignore_labels):
        """ remove labels from ignore_labels that are direct children of this node """
        ignore_labels_pruned = []
        for cls in ignore_labels:
            child_idx = self.old_to_new_classes[cls]
            if len(child_idx) == 0 or self.children[child_idx[0]] in self.wnids:
                continue
            ignore_labels_pruned.append(cls)
        return ignore_labels_pruned

    @property
    def class_counts(self):
        """Number of old classes in each new class"""
        return [len(old_indices) for old_indices in self.new_to_old_classes]

    @property
    def probabilities(self):
        """Calculates probability of training on the ith class.

        If the class contains more than `resample_threshold` samples, the
        probability is lower, as it is likely to cause severe class imbalance
        issues.
        """
        if self._probabilities is None:
            reference = min(self.class_counts)
            self._probabilities = torch.Tensor([
                min(1, reference / len(old_indices))
                for old_indices in self.new_to_old_classes
            ])
        return self._probabilities

    @probabilities.setter
    def probabilities(self, probabilities):
        self._probabilities = probabilities

    @property
    def class_weights(self):
        if self._class_weights is None:
            self._class_weights = self.probabilities
        return self._class_weights

    @class_weights.setter
    def class_weights(self, class_weights):
        self._class_weights = class_weights

    @staticmethod
    def get_wnid_to_node(path_graph, path_wnids, classes, path_wnids_ood=None):
        wnid_to_node = {}
        G = read_graph(path_graph)
        for wnid in get_non_leaves(G):
            wnid_to_node[wnid] = Node(
                wnid, classes, path_graph=path_graph, path_wnids=path_wnids, path_wnids_ood=path_wnids_ood)
        return wnid_to_node

    @staticmethod
    def get_nodes(path_graph, path_wnids, classes, path_wnids_ood=None):
        wnid_to_node = Node.get_wnid_to_node(path_graph, path_wnids, classes, path_wnids_ood)
        wnids = sorted(wnid_to_node)
        nodes = [wnid_to_node[wnid] for wnid in wnids]
        return nodes

    @staticmethod
    def get_root_node_wnid(path_graph):
        raise UserWarning('Root node may have wnid now')
        tree = ET.parse(path_graph)
        for node in tree.iter():
            wnid = node.get('wnid')
            if wnid is not None:
                return wnid
        return None

    @staticmethod
    def dim(nodes):
        return sum([node.num_classes for node in nodes])


class ResampleLabelsDataset(Dataset):
    """
    Dataset that includes only the labels provided, with a limited number of
    samples. Note that labels are integers in [0, k) for a k-class dataset.

    :drop_classes bool: Modifies the dataset so that it is only a m-way
                        classification where m of k classes are kept. Otherwise,
                        the problem is still k-way.
    """

    accepts_probability_labels = True

    def __init__(self, dataset, probability_labels=1, drop_classes=False, seed=0):
        #drop_classes=False
        self.dataset = dataset
        self.classes = dataset.classes
        self.labels = list(range(len(self.classes)))
        self.probability_labels = self.get_probability_labels(dataset, probability_labels)

        self.drop_classes = drop_classes
        if self.drop_classes:
            self.classes, self.labels = self.apply_drop(
                dataset, probability_labels)

        assert self.labels, 'No labels are included in `include_labels`'

        self.new_to_old = self.build_index_mapping(seed=seed)

    def get_probability_labels(self, dataset, ps):
        if not isinstance(ps, (tuple, list)):
            return [ps] * len(dataset.classes)
        if len(ps) == 1:
            return ps * len(dataset.classes)
        assert len(ps) == len(dataset.classes), (
            f'Length of probabilities vector {len(ps)} must equal that of the '
            f'dataset classes {len(dataset.classes)}.'
        )
        return ps

    def apply_drop(self, dataset, ps):
        classes = [
            cls for p, cls in zip(ps, dataset.classes)
            if p > 0
        ]
        labels = [i for p, i in zip(ps, range(len(dataset.classes))) if p > 0]
        return classes, labels

    def build_index_mapping(self, seed=0):
        """Iterates over all samples in dataset.

        Remaps all to-be-included samples to [0, n) where n is the number of
        samples with a class in the whitelist.

        Additionally, the outputted list is truncated to match the number of
        desired samples.
        """
        random.seed(seed)

        new_to_old = []
        for old, (_, label) in enumerate(self.dataset):
            if random.random() < self.probability_labels[label]:
                new_to_old.append(old)
        return new_to_old

    def get_dataset(self):
        return self.dataset

    def __getitem__(self, index_new):
        index_old = self.new_to_old[index_new]
        sample, label_old = self.dataset[index_old]

        label_new = label_old
        if self.drop_classes:
            label_new = self.include_labels.index(label_old)

        return sample, label_new

    def __len__(self):
        return len(self.new_to_old)


class IncludeLabelsDataset(ResampleLabelsDataset):

    accepts_include_labels = True
    accepts_probability_labels = False

    def __init__(self, dataset, include_labels=(0,)):
        super().__init__(dataset, probability_labels=[
            int(cls in include_labels) for cls in range(len(dataset.classes))
        ], drop_classes=True)
        self.include_labels = include_labels


class CombineLabelsDataset(IncludeLabelsDataset):
    # Combines one or more classes into one label

    accepts_combine_labels = True

    def __init__(self, dataset, include_labels=(0,), combine_labels=[]):
        self.old_label_groups = [grp[1:] for grp in combine_labels]
        self.group_heads = [grp[0] for grp in combine_labels]
        super().__init__(dataset, include_labels)
        
        new_label_groups = [[self.include_labels.index(label) for label in grp] for grp in self.old_label_groups]
        heads = {}
        for head, grp in zip(self.group_heads, new_label_groups):
            for label in grp:
                heads[label] = head
        new_labels = {}

        n_labels = 0
        for new_label, old_label in enumerate(self.include_labels):
            if new_label in heads:
                new_labels[new_label] = self.classes.index(heads[new_label])
            else:
                new_labels[new_label] = n_labels
                n_labels += 1

        self.new_labels = new_labels

    def apply_drop(self, dataset, ps):
        # added functionality for dropping classes that are grouped
        old_label_groups_set = set()
        for grp in self.old_label_groups:
            for label in grp:
                old_label_groups_set.add(label)

        classes = [
            cls for p, cls in zip(ps, dataset.classes)
            if p > 0 and dataset.classes.index(cls) not in old_label_groups_set
        ] + self.group_heads

        labels = [i for p, i in zip(ps, range(len(dataset.classes))) if p > 0 and i not in old_label_groups_set]
        labels += [i for i in range(len(labels), len(labels) + len(self.group_heads))]
        return classes, labels

    def __getitem__(self, index_new):
        sample, label_new = super().__getitem__(index_new)
        label_new = self.new_labels[label_new]
        return sample, label_new

class CombineClassesDataset(CombineLabelsDataset):
    """
    Dataset that combines one or more groups of classes into a single class
    """
    accepts_include_classes = True
    accepts_combine_classes = True

    def __init__(self, dataset, include_classes=(0,), combine_classes=[]):
        super().__init__(dataset, include_labels=[
                dataset.classes.index(cls) for cls in include_classes
                ], combine_labels=[[grp[0]] + [dataset.classes.index(cls) for cls in grp[1:]]
                for grp in combine_classes])


class CIFAR10CombineClasses(CombineClassesDataset):

    def __init__(self, *args, root='./data', include_classes=('cat',),
                 combine_classes=[], **kwargs):
        super().__init__(
            dataset=datasets.CIFAR10(*args, root=root, **kwargs),
            include_classes=include_classes, combine_classes=combine_classes)


class CIFAR100CombineClasses(CombineClassesDataset):

    def __init__(self, *args, root='./data', include_classes=('cat',),
                 combine_classes=[], **kwargs):
        super().__init__(
            dataset=datasets.CIFAR100(*args, root=root, **kwargs),
            include_classes=include_classes, combine_classes=combine_classes)


class TinyImagenet200CombineClasses(CombineClassesDataset):

    def __init__(self, *args, root='./data', include_classes=('cat',),
                 combine_classes=[], **kwargs):
        super().__init__(
            dataset=imagenet.TinyImagenet200(*args, root=root, **kwargs),
            include_classes=include_classes, combine_classes=combine_classes)


class Imagenet1000CombineClasses(CombineClassesDataset):

    def __init__(self, *args, root='./data', include_classes=('cat',),
                 combine_classes=[], **kwargs):
        super().__init__(
            dataset=imagenet.Imagenet1000(*args, root=root, **kwargs),
            include_classes=include_classes, combine_classes=combine_classes)


class CIFAR10ResampleLabels(ResampleLabelsDataset):

    def __init__(self, *args, root='./data', probability_labels=1, **kwargs):
        super().__init__(
            dataset=datasets.CIFAR10(*args, root=root, **kwargs),
            probability_labels=probability_labels)


class CIFAR100ResampleLabels(ResampleLabelsDataset):

    def __init__(self, *args, root='./data', probability_labels=1, **kwargs):
        super().__init__(
            dataset=datasets.CIFAR100(*args, root=root, **kwargs),
            probability_labels=probability_labels)


class TinyImagenet200ResampleLabels(ResampleLabelsDataset):

    def __init__(self, *args, root='./data', probability_labels=1, **kwargs):
        super().__init__(
            dataset=imagenet.TinyImagenet200(*args, root=root, **kwargs),
            probability_labels=probability_labels)


class Imagenet1000ResampleLabels(ResampleLabelsDataset):

    def __init__(self, *args, root='./data', probability_labels=1, **kwargs):
        super().__init__(
            dataset=imagenet.Imagenet1000(*args, root=root, **kwargs),
            probability_labels=probability_labels)


class IncludeClassesDataset(IncludeLabelsDataset):
    """
    Dataset that includes only the labels provided, with a limited number of
    samples. Note that classes are strings, like 'cat' or 'dog'.
    """

    accepts_include_labels = False
    accepts_include_classes = True

    def __init__(self, dataset, include_classes=()):
        super().__init__(dataset, include_labels=[
                dataset.classes.index(cls) for cls in include_classes
            ])


class CIFAR10IncludeLabels(IncludeLabelsDataset):

    def __init__(self, *args, root='./data', include_labels=(0,), **kwargs):
        super().__init__(
            dataset=datasets.CIFAR10(*args, root=root, **kwargs),
            include_labels=include_labels)


class CIFAR100IncludeLabels(IncludeLabelsDataset):

    def __init__(self, *args, root='./data', include_labels=(0,), **kwargs):
        super().__init__(
            dataset=datasets.CIFAR100(*args, root=root, **kwargs),
            include_labels=include_labels)


class TinyImagenet200IncludeLabels(IncludeLabelsDataset):

    def __init__(self, *args, root='./data', include_labels=(0,), **kwargs):
        super().__init__(
            dataset=imagenet.TinyImagenet200(*args, root=root, **kwargs),
            include_labels=include_labels)


class Imagenet1000IncludeLabels(IncludeLabelsDataset):

    def __init__(self, *args, root='./data', include_labels=(0,), **kwargs):
        super().__init__(
            dataset=imagenet.Imagenet1000(*args, root=root, **kwargs),
            include_labels=include_labels)

class CIFAR10IncludeClasses(IncludeClassesDataset):

    def __init__(self, *args, root='./data', include_classes=('cat',), **kwargs):
        super().__init__(
            dataset=datasets.CIFAR10(*args, root=root, **kwargs),
            include_classes=include_classes)


class CIFAR100IncludeClasses(IncludeClassesDataset):

    def __init__(self, *args, root='./data', include_classes=('cat',), **kwargs):
        super().__init__(
            dataset=datasets.CIFAR100(*args, root=root, **kwargs),
            include_classes=include_classes)


class TinyImagenet200IncludeClasses(IncludeClassesDataset):

    def __init__(self, *args, root='./data', include_classes=('cat',), **kwargs):
        super().__init__(
            dataset=imagenet.TinyImagenet200(*args, root=root, **kwargs),
            include_classes=include_classes)

    @staticmethod
    def transform_train(input_size=64):
        return transforms.Compose([
            transforms.RandomCrop(input_size, padding=8),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
        ])

    @staticmethod
    def transform_val(input_size=-1):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
        ])


class Imagenet1000IncludeClasses(IncludeClassesDataset):

    def __init__(self, *args, root='./data', include_classes=('cat',), **kwargs):
        super().__init__(
            dataset=imagenet.Imagenet1000(*args, root=root, **kwargs),
            include_classes=include_classes)


class ExcludeLabelsDataset(IncludeLabelsDataset):

    accepts_include_labels = False
    accepts_exclude_labels = True

    def __init__(self, dataset, exclude_labels=(0,)):
        k = len(dataset.classes)
        include_labels = list(set(range(k)) - set(exclude_labels))
        super().__init__(
            dataset=dataset,
            include_labels=include_labels)
        self.include_labels = include_labels


class CIFAR10ExcludeLabels(ExcludeLabelsDataset):

    def __init__(self, *args, root='./data', exclude_labels=(0,), **kwargs):
        super().__init__(
            dataset=datasets.CIFAR10(*args, root=root, **kwargs),
            exclude_labels=exclude_labels)


class CIFAR100ExcludeLabels(ExcludeLabelsDataset):

    def __init__(self, *args, root='./data', exclude_labels=(0,), **kwargs):
        super().__init__(
            dataset=datasets.CIFAR100(*args, root=root, **kwargs),
            exclude_labels=exclude_labels)


class TinyImagenet200ExcludeLabels(ExcludeLabelsDataset):

    def __init__(self, *args, root='./data', exclude_labels=(0,), **kwargs):
        super().__init__(
            dataset=imagenet.TinyImagenet200(*args, root=root, **kwargs),
            exclude_labels=exclude_labels)


class Imagenet1000ExcludeLabels(ExcludeLabelsDataset):

    def __init__(self, *args, root='./data', exclude_labels=(0,), **kwargs):
        super().__init__(
            dataset=imagenet.Imagenet1000(*args, root=root, **kwargs),
            exclude_labels=exclude_labels)

class ExcludeClassesDataset(ExcludeLabelsDataset):
    """
    Dataset that includes only the labels provided, with a limited number of
    samples. Note that classes are strings, like 'cat' or 'dog'.
    """

    accepts_exclude_labels = False
    accepts_exclude_classes = True

    def __init__(self, dataset, exclude_classes=()):
        super().__init__(dataset, exclude_labels=[
                dataset.classes.index(cls) for cls in include_classes
            ])


class CIFAR10ExcludeLabels(ExcludeLabelsDataset):

    def __init__(self, *args, root='./data', include_labels=(0,), **kwargs):
        super().__init__(
            dataset=datasets.CIFAR10(*args, root=root, **kwargs),
            exclude_labels=include_labels)


class CIFAR100ExcludeLabels(ExcludeLabelsDataset):

    def __init__(self, *args, root='./data', include_labels=(0,), **kwargs):
        super().__init__(
            dataset=datasets.CIFAR100(*args, root=root, **kwargs),
            exclude_labels=include_labels)


class TinyImagenet200ExcludeLabels(ExcludeLabelsDataset):

    def __init__(self, *args, root='./data', include_labels=(0,), **kwargs):
        super().__init__(
            dataset=imagenet.TinyImagenet200(*args, root=root, **kwargs),
            exclude_labels=include_labels)


class Imagenet1000ExcludeLabels(ExcludeLabelsDataset):

    def __init__(self, *args, root='./data', include_labels=(0,), **kwargs):
        super().__init__(
            dataset=imagenet.Imagenet1000(*args, root=root, **kwargs),
            exclude_labels=include_labels)

class CIFAR10ExcludeClasses(ExcludeLabelsDataset):

    def __init__(self, *args, root='./data', include_classes=('cat',), **kwargs):
        super().__init__(
            dataset=datasets.CIFAR10(*args, root=root, **kwargs),
            exclude_labels=include_classes)


class CIFAR100ExcludeClasses(ExcludeLabelsDataset):

    def __init__(self, *args, root='./data', include_classes=('cat',), **kwargs):
        super().__init__(
            dataset=datasets.CIFAR100(*args, root=root, **kwargs),
            exclude_labels=include_classes)


class TinyImagenet200ExcludeClasses(ExcludeLabelsDataset):

    def __init__(self, *args, root='./data', include_classes=('cat',), **kwargs):
        super().__init__(
            dataset=imagenet.TinyImagenet200(*args, root=root, **kwargs),
            exclude_labels=include_classes)

    @staticmethod
    def transform_train(input_size=64):
        return transforms.Compose([
            transforms.RandomCrop(input_size, padding=8),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
        ])

    @staticmethod
    def transform_val(input_size=-1):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
        ])


class Imagenet1000ExcludeClasses(ExcludeLabelsDataset):

    def __init__(self, *args, root='./data', include_classes=('cat',), **kwargs):
        super().__init__(
            dataset=imagenet.Imagenet1000(*args, root=root, **kwargs),
            exclude_labels=include_classes)

class TinyImagenet200GradCAM(TinyImagenet200IncludeClasses):
    def __init__(self, root='./data',
                 *args, model, include_classes=('cat',), target_layer='layer4', cam_threshold=-1, **kwargs):
        super().__init__(root=root, include_classes=include_classes)
        self.model = model
        self.target_layer = target_layer
        self.cam_threshold = cam_threshold

    def __getitem__(self, i):
        curr_img, target = super().__getitem__(i)
        transf = imagenet.TinyImagenet200.transform_val()
        cam_mask = gen_gcam_target(
            imgs=[curr_img],
            model=self.model,
            target_layer=self.target_layer,
            target_index=[target],
            transf=transf
        )
        print(curr_img, cam_mask)
        masked_img = curr_img[cam_mask > self.cam_threshold]
        return curr_img, masked_img
