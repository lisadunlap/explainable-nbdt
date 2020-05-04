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

from nbdt.utils import (
    generate_fname, populate_kwargs, Colors, get_saved_word2vec, DATASET_TO_FOLDER_NAME, get_word_embedding, get_transform_from_name
)

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

trainset = dataset(**dataset_kwargs, root='./data', train=True, download=True, transform=transform_test)
testset = dataset(**dataset_kwargs, root='./data', train=False, download=True, transform=transform_test)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=min(args.batch_size, 100), shuffle=False, num_workers=0)
testloader = torch.utils.data.DataLoader(testset, batch_size=min(args.batch_size, 100), shuffle=False, num_workers=0)

# Model
print('==> Building model..')
model = getattr(models, args.model)
Colors.cyan(f'Testing with dataset {args.dataset} and {len(testset.classes)} classes')
if args.replace:
    model_kwargs = {'num_classes': len(testset.classes)}
else:
    if args.new_classes is not None:
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
if args.new_classes is None:
    cls_to_vec = {cls: [] for i, cls in enumerate(trainset.classes) if i in args.new_labels}
else:
    cls_to_vec = {cls: [] for cls in args.new_classes}
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

# load projection matrix
if args.word2vec:
    word2vec_path = os.path.join(os.path.join(trainset.root, DATASET_TO_FOLDER_NAME[args.dataset]), "word2vec/")

num_samples = 0
with torch.no_grad():
    for i, (inputs, labels) in enumerate(trainloader):
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
                if args.word2vec:
                    word_vec = get_saved_word2vec(word2vec_path + cls_name + '.npy', args.dimension)
                    cls_to_vec[cls_name] = word_vec
                else:
                    cls_to_vec[cls_name].append(vec)
                num_samples = min([len(cls_to_vec[c]) for c in cls_to_vec])

    for cls in cls_to_vec:
        cls_to_vec[cls] = np.average(np.array(cls_to_vec[cls]), axis=0)
        cls_to_vec[cls] -= np.mean(cls_to_vec[cls])
        cls_to_vec[cls] /= LA.norm(cls_to_vec[cls])

    # insert vectors into linear layer for model

    fc_weights = fc.weight.cpu().numpy()

    for i, cls in enumerate(trainset.classes):
        if cls in cls_to_vec:
            if args.replace:
                fc_weights[i] = cls_to_vec[cls]
            else:
                fc_weights = np.insert(fc_weights, i, cls_to_vec[cls], axis=0)
        else:
            fc_weights[i] -= np.mean(fc_weights[i])
            fc_weights[i] /= LA.norm(fc_weights[i])
        # else:
        #     assert all(fc_weights[i] == get_word_embedding(cls, trainset, args.dataset))
    setattr(net.module, key, nn.Linear(fc_weights.shape[1], len(trainset.classes)))
    setattr(getattr(net.module, key), "weight", nn.Parameter(torch.from_numpy(fc_weights)))

    # save model
    state = {
        'net': net.state_dict(),
        'acc': best_acc,
        'epoch': start_epoch,
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')

    print(f'Saving to {checkpoint_fname}..')
    torch.save(state, f'./checkpoint/{checkpoint_fname}.pth')