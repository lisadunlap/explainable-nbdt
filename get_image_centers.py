'''Visualize tsne on samples that pass through specific nodes.'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from nbdt import data

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import models

from PIL import Image, ImageOps
from PIL.ImageColor import getcolor
from numpy import linalg as LA
from scipy.spatial.distance import cosine

from nbdt.utils import (
    generate_fname, populate_kwargs, Colors, get_saved_word2vec, DATASET_TO_FOLDER_NAME, get_word_embedding, get_transform_from_name
)
from nbdt.graph import wnid_to_synset

datasets = ('CIFAR10', 'CIFAR100') + data.imagenet.names + data.custom.names + data.awa2.names

parser = argparse.ArgumentParser(description='T-SNE vis generation')
parser.add_argument('--batch-size', default=512, type=int,
                    help='Batch size used for training')
parser.add_argument('--dataset', default='CIFAR10', choices=datasets)
parser.add_argument('--model', default='ResNet18', choices=list(models.get_model_choices()))
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')

# extra general options for main script
parser.add_argument('--checkpoint-fname', default='',
                    help='Fname to save new model to')
parser.add_argument('--path-resume', default='',
                    help='Overrides checkpoint path generation')
parser.add_argument('--name', default='',
                    help='Name of experiment. Used for checkpoint filename')
parser.add_argument('--pretrained', action='store_true',
                    help='Download pretrained model. Not all models support this.')
parser.add_argument('--new-classes', nargs='*',
                    help='New class names used for zero-shot.')
parser.add_argument('--new-labels', nargs='*', type=int,
                    help='New class indices used for zero-shot.')
parser.add_argument('--input-size', type=int,
                    help='Set transform train and val. Samples are resized to '
                    'input-size + 32.')

parser.add_argument('--experiment-name', type=str, help='name of experiment in wandb')
parser.add_argument('--wandb', action='store_true', help='log using wandb')
parser.add_argument('--word2vec', action='store_true')
parser.add_argument('--dimension', type=int, default=300, help='dimension of word2vec embeddings')
parser.add_argument('--num-samples', type=int, default=1)
parser.add_argument('--replace', action='store_true', help='replace the fc rows')


data.custom.add_arguments(parser)

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')


dataset = getattr(data, args.dataset)

transform_train, transform_test = get_transform_from_name(args.dataset, dataset, args.input_size)

dataset_kwargs = {}
populate_kwargs(args, dataset_kwargs, dataset, name=f'Dataset {args.dataset}',
                keys=data.custom.keys, globals=globals())

if args.dataset == 'MiniImagenet':
    trainset = dataset(**dataset_kwargs, root='./data',
                       train=False, download=True, transform=transform_test)
    testset = dataset(**dataset_kwargs, root='./data',
                      zeroshot=True, train=False, download=True, transform=transform_test)
else:
    trainset = dataset(**dataset_kwargs, root='./data', train=True, download=True, transform=transform_train)
    testset = dataset(**dataset_kwargs, root='./data', train=False, download=True, transform=transform_test)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=min(args.batch_size, 100), shuffle=False, num_workers=0)
testloader = torch.utils.data.DataLoader(testset, batch_size=min(args.batch_size, 100), shuffle=True, num_workers=0)

# Model
print('==> Building model..')
model = getattr(models, args.model)
Colors.cyan(f'Testing with dataset {args.dataset} and {len(testset.classes)} classes')
if args.replace:
    model_kwargs = {'num_classes': len(testset.classes)}
else:
    if args.dataset == 'MiniImagenet':
        n_new_classes = 20
    elif args.new_classes is not None:
        n_new_classes = len(args.new_classes)
    else:
        n_new_classes = len(args.new_labels)
    model_kwargs = {'num_classes': len(testset.classes) - n_new_classes}

if args.pretrained:
    try:
        print('==> Loading pretrained model..')
        # net = model(pretrained=True, **model_kwargs)
        net = model(pretrained=True)
        # TODO: this is hardcoded
        if int(args.model[6:]) <= 34:
            net.fc = nn.Linear(512, model_kwargs['num_classes'])
        else:
            net.fc = nn.Linear(512*4, model_kwargs['num_classes'])
    except Exception as e:
        Colors.red(f'Fatal error: {e}')
        exit()
else:
    net = model(**model_kwargs)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

net = net.to(device)

checkpoint_fname = args.checkpoint_fname
resume_path = args.path_resume or './checkpoint/{}.pth'.format(checkpoint_fname)
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    if not os.path.exists(resume_path):
        print('==> No checkpoint found. Skipping...')
    else:
        checkpoint = torch.load(resume_path)

        if 'net' in checkpoint:
            net.load_state_dict(checkpoint['net'])
            best_acc = checkpoint['acc']
            start_epoch = checkpoint['epoch']
            Colors.cyan(f'==> Checkpoint found for epoch {start_epoch} with accuracy '
                        f'{best_acc} at {resume_path}')
        else:
            net.load_state_dict(checkpoint)
            Colors.cyan(f'==> Checkpoint found at {resume_path}')

# get one sample of each zeroshot class, and get its output at linear layer
# cls_to_vec = {trainset.classes[cls]:[] for i, cls in enumerate(list(range(64, 84)))}
#
# current_weights = {testset.classes[cls]:[] for i, cls in enumerate(list(range(64)))}

zs_labels = (args.new_labels) if args.new_labels else set(range(64, 84))
current_weights = {trainset.classes[cls]:[] for i, cls in enumerate(list(set(range(50)).difference(zs_labels)))}

cls_to_vec = {testset.classes[cls]:[] for i, cls in enumerate(list(zs_labels))}

hooked_inputs = None

def testhook(self, input, output):
    global hooked_inputs
    hooked_inputs = input[0].cpu().numpy()

keys = ['fc', 'linear']
for key in keys:
    fc = getattr(net.module, key, None)
    if fc is not None:
        break
fc.register_forward_hook(testhook)

net.eval()

num_samples = 0
with torch.no_grad():
    for i, (inputs, labels) in enumerate(trainloader):
        if args.dataset in ("AnimalsWithAttributes2"):
            inputs, predicates = inputs
        net(inputs)

        for vec, label in zip(hooked_inputs, labels):
            cls_name = trainset.classes[label]
            if cls_name in current_weights:
                current_weights[cls_name].append(vec)

    for cls in current_weights:
        print(f"{cls} with length {len(current_weights[cls])}")
        current_weights[cls] = np.average(np.array(current_weights[cls]), axis=0)
        current_weights[cls] /= LA.norm(current_weights[cls])

def get_most_similar(c, cls_to_vec, weights):
    dist = {cls: cosine(cls_to_vec[c], weights[cls]) for cls in weights}
    dist_total = sum(dist.values())
    dist = {cls: dist[cls] / dist_total for cls in dist}
    #dist = {cls: LA.norm(cls_to_vec[c]-weights[cls]) for cls in weights}
    print(dist)
    return min(dist, key=dist.get)

with torch.no_grad():
    for i, (inputs, labels) in enumerate(testloader):
        if args.dataset in ("AnimalsWithAttributes2"):
            inputs, predicates = inputs
        net(inputs)

        for vec, label in zip(hooked_inputs, labels):
            num_samples = min([len(cls_to_vec[c]) for c in cls_to_vec])
            if num_samples >= args.num_samples:
                print("found and breaking")
                break
            cls_name = trainset.classes[label]
            if cls_name in cls_to_vec and len(cls_to_vec[cls_name]) < args.num_samples:
                cls_to_vec[cls_name].append(vec)
                num_samples = min([len(cls_to_vec[c]) for c in cls_to_vec])

    for cls in cls_to_vec:
        print(f"{cls} with length {len(cls_to_vec[cls])}")
        cls_to_vec[cls] = np.average(np.array(cls_to_vec[cls]), axis=0)
        cls_to_vec[cls] /= LA.norm(cls_to_vec[cls])

    fc_weights = fc.weight.cpu().numpy()

    cls_to_vec = {cls: cls_to_vec[cls] / sum(cls_to_vec[cls]) for cls in cls_to_vec}
    current_weights = {cls: current_weights[cls] / sum(current_weights[cls]) for cls in current_weights}
    pairings = {c: None for c in cls_to_vec}
    for cls in cls_to_vec:
        pairings[cls] = get_most_similar(cls, cls_to_vec, current_weights)
        print(f"{cls}: {pairings[cls]}")
        # print(f"{wnid_to_synset(cls).name().split('.')[0]}: {wnid_to_synset(pairings[cls]).name().split('.')[0]}")
    print(pairings)
    print({wnid_to_synset(cls).name().split('.')[0]: wnid_to_synset(pairings[cls]).name().split('.')[0] for cls in cls_to_vec})
#
#
# new_classes = [testset.classes[i] for i in list(range(64, 84))]
# pairings = {c: None for c in new_classes}
# for c in new_classes:
#     pairings[c] = get_most_similar(c, cls_to_vec, new_classes)
#     print(f"{wnid_to_synset(c).name().split('.')[0]}: {wnid_to_synset(pairings[c]).name().split('.')[0]}")
#
# print(pairings)
