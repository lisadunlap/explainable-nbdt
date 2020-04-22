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

from nbdt.utils import (
    generate_fname, populate_kwargs, Colors
)

datasets = ('CIFAR10', 'CIFAR100') + data.imagenet.names + data.custom.names

parser = argparse.ArgumentParser(description='T-SNE vis generation')
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
                    help='New classes used for zero-shot.')

parser.add_argument('--experiment-name', type=str, help='name of experiment in wandb')
parser.add_argument('--wandb', action='store_true', help='log using wandb')
parser.add_argument('--word2vec', action='store_true')

data.custom.add_arguments(parser)

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

dataset = getattr(data, args.dataset)

# , 'TinyImagenet200IncludeClasses'
if args.dataset in ('TinyImagenet200', 'Imagenet1000'):
    default_input_size = 64 if 'TinyImagenet200' in args.dataset else 224
    input_size = args.input_size or default_input_size
    transform_train = dataset.transform_train(input_size)
    transform_test = dataset.transform_val(input_size)

dataset_kwargs = {}
populate_kwargs(args, dataset_kwargs, dataset, name=f'Dataset {args.dataset}',
                keys=data.custom.keys, globals=globals())

trainset = dataset(**dataset_kwargs, root='./data', train=True, download=True, transform=transform_test)
testset = dataset(**dataset_kwargs, root='./data', train=False, download=True, transform=transform_test)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=False, num_workers=0)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)

# Model
print('==> Building model..')
model = getattr(models, args.model)
model_kwargs = {'num_classes': len(testset.classes) - len(args.new_classes)}

if args.pretrained:
    try:
        print('==> Loading pretrained model..')
        net = model(pretrained=True, **model_kwargs)
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
cls_to_vec = {cls: None for cls in args.new_classes}

hooked_inputs = None

from gensim.models import Word2Vec

try:
    model = Word2Vec.load("./data/wiki.en.word2vec.model")
except:
    print("Word2Vec model not found")


def testhook(self, input, output):
    global hooked_inputs
    hooked_inputs = input[0].cpu().numpy()


net.module.linear.register_forward_hook(testhook)

net.eval()

with torch.no_grad():
    for _, (inputs, labels) in enumerate(trainloader):
        net(inputs)

        done = True
        for vec, label in zip(hooked_inputs, labels):
            cls_name = trainset.classes[label]
            if cls_name in cls_to_vec and cls_to_vec[cls_name] is None:
                done = False
                if args.word2vec:
                    cls_to_vec[cls_name] = model.wv[cls_name]
                else:
                    cls_to_vec[cls_name] = vec[0]
        if done:
            break

    # insert vectors into linear layer for model
    fc_weights = net.module.linear.weight.cpu().numpy()
    for i, cls in enumerate(trainset.classes):
        if cls in cls_to_vec:
            fc_weights = np.insert(fc_weights, i, cls_to_vec[cls], axis=0)
    net.module.linear = nn.Linear(fc_weights.shape[1], len(trainset.classes))
    net.module.linear.weight = nn.Parameter(torch.from_numpy(fc_weights))

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