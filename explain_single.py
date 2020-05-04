'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from nbdt import data, analysis, loss
from PIL import Image

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import wandb

import models
from nbdt.utils import (
    progress_bar, generate_fname, DATASET_TO_PATHS, populate_kwargs, Colors, get_transform_from_name
)

datasets = ('CIFAR10', 'CIFAR100') + data.imagenet.names + data.custom.names + data.awa2.names + data.cub.names + data.miniplaces.names

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--batch-size', default=512, type=int,
                    help='Batch size used for training')
parser.add_argument('--epochs', '-e', default=200, type=int,
                    help='By default, lr schedule is scaled accordingly')
parser.add_argument('--dataset', default='CIFAR10', choices=datasets)
parser.add_argument('--model', default='ResNet18', choices=list(models.get_model_choices()))
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')

# extra general options for main script
parser.add_argument('--checkpoint-fname', default='',
                    help='Overrides checkpoint name generation')
parser.add_argument('--path-resume', default='',
                    help='Overrides checkpoint path generation')
parser.add_argument('--name', default='',
                    help='Name of experiment. Used for checkpoint filename')
parser.add_argument('--pretrained', action='store_true',
                    help='Download pretrained model. Not all models support this.')
parser.add_argument('--freeze-conv', action='store_true')
parser.add_argument('--eval', help='eval only', action='store_true')

# options specific to this project and its dataloaders
parser.add_argument('--loss', choices=loss.names, default='CrossEntropyLoss')
parser.add_argument('--analysis', choices=analysis.names, default='SingleInference',
                    help='Run analysis after each epoch')
parser.add_argument('--input-size', type=int,
                    help='Set transform train and val. Samples are resized to '
                    'input-size + 32.')
parser.add_argument('--image-path', default='./data/samples/bcats.jpg',
                    help='path to image')
parser.add_argument('--experiment-name', type=str, help='name of experiment in wandb')
parser.add_argument('--wandb', action='store_true', help='log using wandb')

data.custom.add_arguments(parser)
loss.add_arguments(parser)
analysis.add_arguments(parser)

args = parser.parse_args()

data.custom.set_default_values(args)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

experiment_name = args.experiment_name if args.experiment_name else '{}-{}-{}'.format(args.model, args.dataset, args.analysis)
if args.wandb:
    wandb.init(project=experiment_name, name='main', entity='lisadunlap')
    wandb.config.update({
        k: v for k, v in vars(args).items() if (isinstance(v, str) or isinstance(v, int) or isinstance(v, float))
    })

# Data
print('==> Preparing data..')

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

dataset = getattr(data, args.dataset)

if args.input_size is not None:
     input_size = args.input_size
else:
    input_size = 224

transform_train, transform_test = get_transform_from_name(args.dataset, dataset, args.input_size)

dataset_kwargs = {}
populate_kwargs(args, dataset_kwargs, dataset, name=f'Dataset {args.dataset}',
    keys=data.custom.keys, globals=globals())

testset = dataset(**dataset_kwargs, root='./data', train=False, download=True, transform=transform_test)
trainset=testset

testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# load image
img = transform_test(Image.open(args.image_path).resize((input_size, input_size))).unsqueeze(0)
print("image size: ", img.shape)

# Model
print('==> Building model..')
model = getattr(models, args.model)
model_kwargs = {'num_classes': len(trainset.classes) }

if args.pretrained:
    try:
        print('==> Loading pretrained model..')
        # net = model(pretrained=True, **model_kwargs)
        net = model(pretrained=True)
        # TODO: this is hardcoded
        if args.freeze_conv:
            for param in net.parameters():
                param.requires_grad = False
            net.fc.requires_grad = True
        if int(args.model[6:]) <= 34:
            net.fc = nn.Linear(512, model_kwargs['num_classes'])
        else:
            net.fc = nn.Linear(512*4, model_kwargs['num_classes'])
    except Exception as e:
        Colors.red(f'Fatal error: {e}')
        exit()
else:
    net = model(**model_kwargs)

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

checkpoint_fname = args.checkpoint_fname or generate_fname(**vars(args))
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

elif args.path_resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    if not os.path.exists(resume_path):
        print('==> No checkpoint found. Skipping...')
    else:
        checkpoint = torch.load(resume_path)
        net.load_state_dict(checkpoint['net'])
        Colors.cyan(f'==> Checkpoint found at {resume_path}')

analyzer_kwargs = {}
class_analysis = getattr(analysis, args.analysis or 'Noop')
populate_kwargs(args, analyzer_kwargs, class_analysis,
    name=f'Analyzer {args.analysis}', keys=analysis.keys, globals=globals())
net.eval()

if args.analysis in ("SingleRISE", "SingleGradCAM"):
    analyzer = class_analysis(**analyzer_kwargs, net=net, experiment_name=experiment_name, use_wandb=True)
else:
    analyzer = class_analysis(**analyzer_kwargs, experiment_name=experiment_name, use_wandb=args.wandb)

# run inference
outputs = net(img.to(device))
analyzer.inf(img.to(device), outputs)