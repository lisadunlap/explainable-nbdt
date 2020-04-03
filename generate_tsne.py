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
parser.add_argument('--vis-json-path', type=str, help='json path where node specific information is stored')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--perplexity', default=20, type=int, help='perplexity for tsne')
parser.add_argument('--angle', default=0.2, type=float, help='angle for tsne')

# extra general options for main script
parser.add_argument('--checkpoint-fname', default='',
                    help='Overrides checkpoint name generation')
parser.add_argument('--path-resume', default='',
                    help='Overrides checkpoint path generation')
parser.add_argument('--name', default='',
                    help='Name of experiment. Used for checkpoint filename')
parser.add_argument('--pretrained', action='store_true',
                    help='Download pretrained model. Not all models support this.')

data.custom.add_arguments(parser)

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
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

testset = dataset(**dataset_kwargs, root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)

# Model
print('==> Building model..')
model = getattr(models, args.model)
model_kwargs = {'num_classes': len(testset.classes) }

if args.pretrained:
    try:
        print('==> Loading pretrained model..')
        net = model(pretrained=True, **model_kwargs)
    except Exception as e:
        Colors.red(f'Fatal error: {e}')
        exit()
else:
    net = model(**model_kwargs)

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

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

net.module.linear = nn.Flatten()

with open(args.vis_json_path, 'r') as f:
    samples_in_node = json.load(f)

# note: samples are indices of the dataset
samples_to_node = {}
samples_idx = []  # maps row indexes from sample_vectors to actual index in dataset
sample_vectors = [] # FC inputs
for node in samples_in_node:
    for sample in samples_in_node[node]:
        samples_to_node[sample] = node

net.eval()
total = 0
with torch.no_grad():
    for batch_idx, (inputs, _) in enumerate(testloader):
        inputs = inputs.to(device)
        outputs = net(inputs)

        total += outputs.size(0)

        if device == 'cuda':
            outputs = outputs.cpu()
        outputs = outputs.numpy()

        for idx, output in zip(range(total, total+100), outputs):
            if idx in samples_to_node:
                samples_idx.append(idx)
                sample_vectors.append(output)

sample_vectors = np.array(sample_vectors)

from sklearn.manifold import TSNE

# get raw images
testset = dataset(**dataset_kwargs, root='./data', train=False, download=True)
images = [testset[i][0] for i in samples_idx]

tsne = TSNE(n_components=2, learning_rate=150, perplexity=args.perplexity, angle=args.angle, verbose=2).fit_transform(sample_vectors)
tx, ty = tsne[:,0], tsne[:,1]
tx = (tx-np.min(tx)) / (np.max(tx) - np.min(tx))
ty = (ty-np.min(ty)) / (np.max(ty) - np.min(ty))


width = 1000
height = 500
max_dim = 64

handles = []
colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange']
COLOR_MAPS = [(color, getcolor(color, mode='RGB')) for color in colors]
node_to_color = {}

for node, (color_plt, color_pil) in zip(samples_in_node, COLOR_MAPS):
    patch = mpatches.Patch(color=color_plt, label=node)
    handles.append(patch)
    node_to_color[node] = color_pil

full_image = Image.new('RGBA', (width, height))
for img, x, y, sample_idx in zip(images, tx, ty, samples_idx):
    #tile = Image.open(img)
    node = samples_to_node[sample_idx]
    img = ImageOps.expand(img, border=5, fill=node_to_color[node])
    tile = img
    rs = max(1, tile.width/max_dim, tile.height/max_dim)
    tile = tile.resize((int(tile.width/rs), int(tile.height/rs)), Image.ANTIALIAS)
    full_image.paste(tile, (int((width-max_dim)*x), int((height-max_dim)*y)), mask=tile.convert('RGBA'))

plt.figure(figsize = (16,12))
plt.imshow(full_image)
plt.legend(handles=handles)
plt.show()
