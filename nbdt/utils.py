'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math
import numpy as np
from numpy import linalg as LA

import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision.transforms as transforms
from gensim.models import Word2Vec

from pathlib import Path

# tree-generation consntants
METHODS = ('prune', 'wordnet', 'random', 'image', 'induced', 'clustered', 'extra_paths', 'weighted', 'replace_node', 'insert_node')
DATASETS = ('CIFAR10', 'CIFAR100', 'TinyImagenet200', 'TinyImagenet200IncludeClasses', 'Imagenet1000',
            'TinyImagenet200CombineClasses', 'MiniPlaces', 'AnimalsWithAttributes2', 'CUB2011')

DATASET_TO_FOLDER_NAME = {
    'CIFAR10': 'CIFAR10',
    'CIFAR10ExcludeLabels': 'CIFAR10-zeroshot',
    'CIFAR10ExcludeClasses': 'CIFAR10',
    'CIFAR100': 'CIFAR100',
    'TinyImagenet200': 'tiny-imagenet-200',
    'TinyImagenet200IncludeClasses': 'tiny-imagenet-200-custom',
    'Imagenet1000' : 'imagenet-1000',
    'TinyImagenet200CombineClasses': 'tiny-imagenet-200-custom-combined',
    'MiniPlaces': 'miniplaces',
    'AnimalsWithAttributes2': 'awa2',
    'CUB2011': 'CUB_200_2011'
}

# main script constants
CIFAR10PATHSANITY = 'CIFAR10PathSanity'

DEFAULT_CIFAR10_TREE = './data/CIFAR10/graph-wordnet-single.json'
DEFAULT_CIFAR10_WNIDS = './data/CIFAR10/wnids.txt'
DEFAULT_CIFAR100_TREE = './data/CIFAR100/graph-wordnet-single.json'
DEFAULT_CIFAR100_WNIDS = './data/CIFAR100/wnids.txt'
DEFAULT_TINYIMAGENET200_TREE = './data/tiny-imagenet-200/graph-wordnet-single.json'
DEFAULT_TINYIMAGENET200_WNIDS = './data/tiny-imagenet-200/wnids.txt'
DEFAULT_IMAGENET1000_TREE = './data/imagenet-1000/graph-wordnet-single.json'
DEFAULT_IMAGENET1000_WNIDS = './data/imagenet-1000/wnids.txt'
DEFAULT_MINIPLACES_TREE = '/data/miniplaces/graph-default.json'
DEFAULT_MINIPLACES_WNID = './data/miniplaces/wnids.txt'
DEFAULT_AWA2_TREE = '/data/awa2/graph-default.json'
DEFAULT_AWA2_WNID = './data/awa2/wnids.txt'
DEFAULT_CUB_TREE = '/data/CUB_200_2011/graph-default.json'
DEFAULT_CUB_WNID = './data/CUB_200_2011/wnids.txt'


DATASET_TO_PATHS = {
    'CIFAR10': {
        'path_graph': DEFAULT_CIFAR10_TREE,
        'path_wnids': DEFAULT_CIFAR10_WNIDS
    },
    'CIFAR100': {
        'path_graph': DEFAULT_CIFAR100_TREE,
        'path_wnids': DEFAULT_CIFAR100_WNIDS
    },
    'TinyImagenet200': {
        'path_graph': DEFAULT_TINYIMAGENET200_TREE,
        'path_wnids': DEFAULT_TINYIMAGENET200_WNIDS
    },
    'Imagenet1000': {
        'path_graph': DEFAULT_IMAGENET1000_TREE,
        'path_wnids': DEFAULT_IMAGENET1000_WNIDS
    },
    'MiniPlaces': {
        'path_graph': DEFAULT_MINIPLACES_TREE,
        'path_wnids': DEFAULT_MINIPLACES_WNID
    },
    'AnimalsWithAttributes2': {
        'path_graph': DEFAULT_AWA2_TREE,
        'path_wnids': DEFAULT_AWA2_WNID
    },
    'CUB2011': {
        'path_graph': DEFAULT_CUB_TREE,
        'path_wnids': DEFAULT_CUB_WNID
    }
}

WORD2VEC_NAMES_TO_MODEL = {
    'wiki': {
        'name': 'glove-wiki-gigaword-300',
        'dim': 300
    },
    'wiki-300': {
        'name': 'glove-wiki-gigaword-300',
        'dim': 300
    },
    'wiki-200': {
        'name': 'glove-wiki-gigaword-200',
        'dim': 200
    },
    'wiki-100': {
        'name': 'glove-wiki-gigaword-100',
        'dim': 100
    },
    'wiki-50': {
        'name': 'glove-wiki-gigaword-50',
        'dim': 50
    },

    'twitter': {
        'name': 'glove-twitter-200',
        'dim': 200
    }
}

def populate_kwargs(args, kwargs, object, name='Dataset', keys=(), globals={}):
    for key in keys:
        accepts_key = getattr(object, f'accepts_{key}', False)
        if not accepts_key:
            continue
        assert key in args or callable(accepts_key), key

        value = getattr(args, key, None)
        if callable(accepts_key):
            kwargs[key] = accepts_key(**globals)
            Colors.cyan(f'{key}:\t(callable)')
        elif accepts_key and value:
            kwargs[key] = value
            Colors.cyan(f'{key}:\t{value}')
        elif value:
            Colors.red(
                f'Warning: {name} does not support custom '
                f'{key}: {value}')


def get_transform_from_name(dataset_name, dataset, input_size):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # , 'TinyImagenet200IncludeClasses'
    if dataset_name in ('TinyImagenet200', 'Imagenet1000', 'CUB2011'):
        default_input_size = 64 if 'TinyImagenet200' in dataset_name else 224
        input_size = input_size or default_input_size
        transform_train = dataset.transform_train(input_size)
        transform_test = dataset.transform_val(input_size)

    if dataset_name in ('MiniPlaces'):
        transform_train = dataset.transform_train()
        transform_test = dataset.transform_test()

    return transform_train, transform_test


class Colors:
    RED = '\x1b[31m'
    GREEN = '\x1b[32m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    CYAN = '\x1b[36m'

    @classmethod
    def red(cls, *args):
        print(cls.RED + args[0], *args[1:], cls.ENDC)

    @classmethod
    def green(cls, *args):
        print(cls.GREEN + args[0], *args[1:], cls.ENDC)

    @classmethod
    def cyan(cls, *args):
        print(cls.CYAN + args[0], *args[1:], cls.ENDC)

    @classmethod
    def bold(cls, *args):
        print(cls.BOLD + args[0], *args[1:], cls.ENDC)


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


try:
    _, term_width = os.popen('stty size', 'r').read().split()
    term_width = int(term_width)
except Exception as e:
    print(e)
    term_width = 50

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def set_np_printoptions():
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})


def generate_fname(dataset, model, path_graph, wnid=None, name='',
        trainset=None, include_labels=(), exclude_labels=(),
        include_classes=(), num_samples=0, max_leaves_supervised=-1,
        min_leaves_supervised=-1, tree_supervision_weight=0.5,
        weighted_average=False, fine_tune=False,
        loss='CrossEntropyLoss', word2vec=False, **kwargs):
    fname = 'ckpt'
    fname += '-' + dataset
    fname += '-' + model
    if name:
        fname += '-' + name
    if path_graph:
        path = Path(path_graph)
        fname += '-' + path.stem.replace('graph-', '', 1)
    if include_labels:
        labels = ",".join(map(str, include_labels))
        fname += f'-incl{labels}'
    if exclude_labels:
        labels = ",".join(map(str, exclude_labels))
        fname += f'-excl{labels}'
    if include_classes:
        labels = ",".join(map(str, include_classes))
        fname += f'-incc{labels}'
    if num_samples != 0 and num_samples is not None:
        fname += f'-samples{num_samples}'
    if loss != 'CrossEntropyLoss':
        fname += f'-{loss}'
        if max_leaves_supervised > 0:
            fname += f'-mxls{max_leaves_supervised}'
        if min_leaves_supervised > 0:
            fname += f'-mnls{min_leaves_supervised}'
        if tree_supervision_weight is not None and tree_supervision_weight != 1:
            fname += f'-tsw{tree_supervision_weight}'
        if weighted_average:
            fname += '-weighted'
    if word2vec:
        fname += '-word2vec'
    return fname

def get_saved_word2vec(path, dimension, projection_matrix):
    word_vec = np.load(path)
    word_vec = np.asarray(word_vec).reshape(1, dimension)
    word_vec = np.matmul(word_vec, projection_matrix)[0]
    return np.array(word_vec / LA.norm(word_vec), dtype=float)

def get_word_embedding(cls, trainset, dataset_name='CIFAR10'):
    word2vec_path = os.path.join(os.path.join(trainset.root, DATASET_TO_FOLDER_NAME[dataset_name]), "word2vec/")
    word_vec = np.load(word2vec_path + cls + '.npy')
    return word_vec/LA.norm(word_vec)


def word2vec_model(net, trainset, dataset_name='CIFAR10', exclude_classes=None, dimension=300):
    """ Sets FC layer weights to word2vec embeddings, freezing them unless
    exclude classes is given, in which case those specific rows are frozen in
    the backward call"""

    print('==> Adding in word2vec embeddings...')
    word2vec_path = os.path.join(os.path.join('./data',DATASET_TO_FOLDER_NAME[dataset_name]), "word2vec/")
    if not os.path.exists(word2vec_path):
        raise Exception("No saved word2vec embeddings, run generate_word2vec.py")
    fc_weights = []

    for i, cls in enumerate(trainset.classes):
        word_vec = np.load(word2vec_path+cls+'.npy')
        word_vec /= LA.norm(word_vec)
        fc_weights = np.append(fc_weights, word_vec)
    fc_weights = fc_weights.reshape((len(trainset.classes), 512))
    net.module.linear = nn.Linear(fc_weights.shape[1], len(trainset.classes)).to("cuda")
    net.module.linear.weight = nn.Parameter(torch.from_numpy(fc_weights).float().to("cuda"))
    # test_word2vec(net, trainset, dataset_name, exclude_classes, dimension)
    # Colors.cyan("All word2vec checks passed!")

    # freeze layer
    net.module.linear.weight.requires_grad = False
    net.module.linear.bias.requires_grad = False
    net.module.linear.requires_grad = False
    Colors.cyan("Freezing FC weights..")
    return net

def test_word2vec(net, trainset, dataset_name='CIFAR10', exclude_classes=None, dimension=300):
    """ Check that word2vec weights are frozen in ZS rows """
    word2vec_path = os.path.join(os.path.join('./data', DATASET_TO_FOLDER_NAME[dataset_name]), "word2vec/")
    if not os.path.exists(word2vec_path):
        raise Exception("No saved word2vec embeddings, run generate_word2vec.py")

    net.eval()

    # get FC weights
    fc_weights = net.module.linear.weight.detach().cpu().numpy()

    # if no exclude classes, all FC rows should be word2vec embeddings
    if not exclude_classes:
        for i, cls in enumerate(trainset.classes):
            word_vec = word_vec = np.load(word2vec_path+cls+'.npy')
            assert all(fc_weights[i] == word_vec)
    else:
        for i, cls in enumerate(exclude_classes):
            word_vec = word_vec = np.load(word2vec_path+cls+'.npy')
            assert all(fc_weights[i] == word_vec)
    Colors.cyan("Freezing certain FC rows check passed!")

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

def smooth_one_hot(labels, classes, seen_to_zsl_cls, smoothing=0.0):
    """ smoothing == 0: one-hot; 0 < smoothing < 1: smoothing 

    How to use:
    Loss = CrossEntropyLoss(NonSparse=True, ...)
    . . .
    data = ...
    labels = ...

    outputs = model(data)

    smooth_label = smooth_one_hot(labels, ...)
    loss = (outputs, smooth_label) """
    if smoothing == 0 or not seen_to_zsl_cls:
        return labels

    assert 0 <= smoothing < 1
    num_classes = len(classes)
    label_shape = torch.Size((labels.size(0), num_classes))
    confidence = 1.0 - smoothing

    with torch.no_grad():
        true_dist = torch.zeros(size=label_shape, device=labels.device)
        true_dist.scatter_(1, labels.data.unsqueeze(1), 1)
    for seen, zsl in seen_to_zsl_cls.items():
        zsl_idx, seen_idx = classes.index(zsl), classes.index(seen)
        seen_selector = torch.zeros_like(labels.data.unsqueeze(1))
        seen_selector[true_dist[:, seen_idx] == 1] = seen_idx
        zsl_selector = torch.zeros_like(labels.data.unsqueeze(1))
        zsl_selector[true_dist[:, seen_idx] == 1] = zsl_idx
        true_dist.scatter_(1, seen_selector, confidence)
        true_dist.scatter_(1, zsl_selector, smoothing)
    return true_dist
