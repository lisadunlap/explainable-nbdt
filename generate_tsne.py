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
from numpy import linalg as LA
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

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
parser.add_argument('--vis-json-path', type=str,
                    help='json path where node specific information is stored.')
parser.add_argument('--plot-all', action='store_true', help='plot all feature vectors')
parser.add_argument('--plot-fc', action='store_true', help='plot rows of FC layers')
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

samples_to_node = {}
samples_idx = []  # maps row indexes from sample_vectors to actual index in dataset
sample_vectors = [] # vectors we plot

if not args.plot_all:  
    with open(args.vis_json_path, 'r') as f:
        samples_in_node = json.load(f)

    # note: samples are indices of the dataset
    for node in samples_in_node:
        for sample in samples_in_node[node]:
            samples_to_node[sample] = node

else:
    samples_in_node = [i for i, _ in enumerate(testset.classes)]
    samples_to_node = {i:i for i in samples_in_node}

hooked_inputs = None

def testhook(self, input, output):
    global hooked_inputs
    hooked_inputs = input[0].cpu().numpy()

net.module.linear.register_forward_hook(testhook)

net.eval()
with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs = inputs.to(device)
        net(inputs)

        for idx, output in enumerate(hooked_inputs):
            if args.plot_all:
                samples_idx.append(targets[idx])
                sample_vectors.append(output / LA.norm(output))
            elif idx in samples_to_node:
                samples_idx.append(idx)
                sample_vectors.append(output)

    if args.plot_fc:
        fc_weights = net.module.linear.weight.cpu().numpy()
        for i, row in enumerate(fc_weights):
            sample_vectors.append(row)
            samples_idx.append(i)

sample_vectors = np.array(sample_vectors)
print("Samples collected shape: ", sample_vectors.shape)

from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, learning_rate=150, perplexity=args.perplexity, angle=args.angle, verbose=2).fit_transform(sample_vectors)
tx, ty = tsne[:,0], tsne[:,1]
tx = (tx-np.min(tx)) / (np.max(tx) - np.min(tx))
ty = (ty-np.min(ty)) / (np.max(ty) - np.min(ty))


width = 1000
height = 500
max_dim = 64

handles = []
colors = ['red', 'blue', 'green', 'brown', 'purple', 'orange', 'olive', 'cyan', 'darkkhaki', 'lavender', 'pink',]
markers = ['_', 'v', '^', '<', '>', 'D', 's', 'p', '*', '+',]
COLOR_MAPS = [(color, getcolor(color, mode='RGB')) for color in colors]
node_to_color = {}

for node, (color_plt, color_pil) in zip(samples_in_node, COLOR_MAPS):
    if args.plot_all:
        patch = mlines.Line2D([], [], color=color_plt, marker=markers[node], label=testset.classes[node], linestyle='None')
    else:
        patch = mpatches.Patch(color=color_plt, label=node)
    handles.append(patch)
    node_to_color[node] = color_pil

if args.plot_fc:
    for i, color_plt in zip(range(len(testset.classes)), colors):
        patch = mlines.Line2D([], [], color='black', marker=markers[i], label=f"{testset.classes[i]} FC row", linestyle='None')
        handles.append(patch)

plt.figure(figsize = (12,9))

if args.plot_all:
    for i, (x, y, sample_idx) in enumerate(zip(tx, ty, samples_idx)):
        if args.plot_fc and i >= len(samples_idx) - 10:
            print(x,y)
            plt.scatter([x], [y], s=200, color='black', marker=markers[sample_idx])
        else:
            plt.scatter([x], [y], marker=markers[sample_idx], color=colors[sample_idx])
else:
    # get raw images
    testset = dataset(**dataset_kwargs, root='./data', train=False, download=True)
    images = [testset[i][0] for i in samples_idx]
    full_image = Image.new('RGBA', (width, height))
    for img, x, y, sample_idx in zip(images, tx, ty, samples_idx):
        node = samples_to_node[sample_idx]
        img = ImageOps.expand(img, border=5, fill=node_to_color[node])
        tile = img
        rs = max(1, tile.width/max_dim, tile.height/max_dim)
        tile = tile.resize((int(tile.width/rs), int(tile.height/rs)), Image.ANTIALIAS)
        full_image.paste(tile, (int((width-max_dim)*x), int((height-max_dim)*y)), mask=tile.convert('RGBA'))
    plt.imshow(full_image)


plt.legend(handles=handles, bbox_to_anchor=(0.9,1.1), loc="upper left")
plt.show()
