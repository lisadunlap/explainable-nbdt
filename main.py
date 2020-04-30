'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from nbdt import data, analysis, loss

import torchvision

import os
import argparse
import wandb

import models
from nbdt.utils import (
    progress_bar, generate_fname, DATASET_TO_PATHS, populate_kwargs, Colors, word2vec_model,
    get_transform_from_name, test_word2vec
)

datasets = ('CIFAR10', 'CIFAR100') + data.imagenet.names + data.custom.names + data.cub.names + data.awa2.names


parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--batch-size', default=512, type=int,
                    help='Batch size used for training')
parser.add_argument('--epochs', '-e', default=200, type=int,
                    help='By default, lr schedule is scaled accordingly')
parser.add_argument('--dataset', default='CIFAR10', choices=datasets)
parser.add_argument('--model', default='ResNet10', choices=list(models.get_model_choices()))
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--ood-dataset', choices=datasets, help='dataset to use for out of distribution images')
parser.add_argument('--ood-classes', nargs='*', type=str, help='classes to include for ood-dataset')
parser.add_argument('--ood-path-wnids', type=str, help='path to wnids.txt for ood-dataset')
# extra general options for main script
parser.add_argument('--checkpoint-fname', default='',
                    help='Overrides checkpoint name generation')
parser.add_argument('--path-resume', default='',
                    help='Overrides checkpoint path generation')
parser.add_argument('--name', default='',
                    help='Name of experiment. Used for checkpoint filename')
parser.add_argument('--pretrained', action='store_true',
                    help='Download pretrained model. Not all models support this.')
parser.add_argument('--eval', help='eval only', action='store_true')

# options specific to this project and its dataloaders
parser.add_argument('--loss', choices=loss.names, default='CrossEntropyLoss')
parser.add_argument('--analysis', choices=analysis.names, help='Run analysis after each epoch')
parser.add_argument('--input-size', type=int,
                    help='Set transform train and val. Samples are resized to '
                    'input-size + 32.')
parser.add_argument('--experiment-name', type=str, help='name of experiment in wandb')
parser.add_argument('--wandb', action='store_true', help='log using wandb')
parser.add_argument('--word2vec', action='store_true')
parser.add_argument("--track_nodes", nargs="*", type=str, help="nodes to keep track of")
parser.add_argument("--train-word2vec", action='store_true', help="fit model to pretrained weights")
parser.add_argument('--freeze-classes',  nargs="*", type=str, help="classes whose FC weights should freeze")
parser.add_argument('--weight-decay', type=float, default=5e-4)
parser.add_argument('--label-smoothing', type=float, default=0.0, help='Smoothing factor for label smoothing (0 < s <= 1)')
parser.add_argument('--seen-to-zsl', nargs="*", type=str, help="list strings in pairs, mapping seen class to zeroshot class (e.g. 'dog' 'cat')")

data.custom.add_arguments(parser)
loss.add_arguments(parser)
analysis.add_arguments(parser)

args = parser.parse_args()

data.custom.set_default_values(args)

experiment_name = args.experiment_name if args.experiment_name \
    else '{}-{}-{}-{}'.format(args.model, args.dataset, args.loss, args.analysis)

if args.wandb:
    wandb.init(project=experiment_name, name='main')
    wandb.config.update({
        k: v for k, v in vars(args).items() if (isinstance(v, str) or isinstance(v, int) or isinstance(v, float))
    })
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

trainset = dataset(**dataset_kwargs, root='./data', train=True, download=True, transform=transform_train)
testset = dataset(**dataset_kwargs, root='./data', train=False, download=True, transform=transform_test)

assert trainset.classes == testset.classes, (trainset.classes, testset.classes)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

Colors.cyan(f'Training with dataset {args.dataset} and {len(trainset.classes)} classes')

seen_to_zsl_cls = {}
if args.seen_to_zsl:
    assert len(args.seen_to_zsl) % 2 == 0, "Classes must be given in pairs"
    for i in range(0, len(args.seen_to_zsl), 2):
        seen, zero = args.seen_to_zsl[i:i+2]
        seen_to_zsl_cls[seen] = zero
args.seen_to_zsl_cls = seen_to_zsl_cls

# Model
print('==> Building model..')
model = getattr(models, args.model)
model_kwargs = {'num_classes': len(trainset.classes) }

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


if args.word2vec:
    net = word2vec_model(net, trainset, exclude_classes=args.exclude_classes, dataset_name=args.dataset)

loss_kwargs = {}
class_criterion = getattr(loss, args.loss)
populate_kwargs(args, loss_kwargs, class_criterion, name=f'Loss {args.loss}',
    keys=loss.keys, globals=globals())
criterion = class_criterion(**loss_kwargs)

optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

def adjust_learning_rate(epoch, lr):
    if epoch <= 150 / 350. * args.epochs:  # 32k iterations
      return lr
    elif epoch <= 250 / 350. * args.epochs:  # 48k iterations
      return lr/10
    elif epoch <= 350 / 500. * args.epochs:  # 48k iterations
        return lr / 100
    else:
      return lr/1000

def exp_lr_scheduler(epoch, init_lr=0.0001, lr_decay_epoch=30, weight=0.1):
    lr = init_lr * (weight ** (epoch // lr_decay_epoch))
    return lr

# Training
def train(epoch, analyzer):
    analyzer.start_train(epoch)
    if args.dataset in ("MiniPlaces",):
        lr = exp_lr_scheduler(epoch)
        optimizer = optim.RMSprop(net.parameters(), lr=lr)
    else:
        lr = adjust_learning_rate(epoch, args.lr)
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        # Freeze rows of the last FC layer that correspond with the ZS classes
        # TODO: optimize this with backward hook
        # if args.exclude_classes:
        #     for cls in args.exclude_classes:
        #         net.module.linear.weight.grad[trainset.classes.index(cls)] = 0
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        stat = analyzer.update_batch(outputs, predicted, targets)
        extra = f'| {stat}' if stat else ''

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) %s'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total, extra))

    analyzer.end_train(epoch)

def test(epoch, analyzer, checkpoint=True, ood_loader=None):
    analyzer.start_test(epoch)
    global testloader
    if ood_loader:
        testloader = ood_loader
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if device == 'cuda':
                predicted = predicted.cpu()
                targets = targets.cpu()

            stat = analyzer.update_batch(outputs, predicted, targets)
            extra = f'| {stat}' if stat else ''

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) %s'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total, extra))

    if args.wandb:
        if args.eval:
            wandb.run.summary["best_accuracy"] = 100.*correct/total
            wandb.run.summary["best_loss"] = test_loss/(batch_idx+1)
        else:
            wandb.log({
                'loss':  test_loss/(batch_idx+1),
                'accuracy': 100.*correct/total
            }, step=epoch)

    # Save checkpoint.
    acc = 100.*correct/total
    print("Accuracy: {}, {}/{}".format(acc, correct, total))
    if acc > best_acc and checkpoint:
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')

        print(f'Saving to {checkpoint_fname} ({acc})..')
        torch.save(state, f'./checkpoint/{checkpoint_fname}.pth')
        best_acc = acc

    analyzer.end_test(epoch)

if args.ood_dataset:
    ood_dataset = getattr(data, args.ood_dataset)
    ood_dataset_kwargs = {}
    populate_kwargs(args, ood_dataset_kwargs, ood_dataset, name=f'Dataset {args.ood_dataset}',
        keys=data.custom.keys, globals=globals())
    ood_dataset_kwargs['include_classes'] = args.ood_classes # manual override
    ood_set = dataset(**ood_dataset_kwargs, root='./data', train=True, download=True, transform=transform_train)
    ood_loader = torch.utils.data.DataLoader(ood_set, batch_size=args.batch_size, shuffle=True, num_workers=2)

analyzer_kwargs = {}
class_analysis = getattr(analysis, args.analysis or 'Noop')
if args.ood_dataset:
    args.oodset = ood_set
populate_kwargs(args, analyzer_kwargs, class_analysis,
    name=f'Analyzer {args.analysis}', keys=analysis.keys, globals=globals())
analyzer = class_analysis(**analyzer_kwargs, experiment_name=experiment_name, use_wandb=args.wandb)


if args.eval:
    if not args.resume and not args.pretrained:
        Colors.red(' * Warning: Model is not loaded from checkpoint. '
        'Use --resume or --pretrained (if supported)')
    net.eval()
    analyzer.start_epoch(0)
    if args.ood_dataset:
        test(0, analyzer, checkpoint=False, ood_loader=ood_loader)
    else:
        test(0, analyzer, checkpoint=False)
    exit()

for epoch in range(start_epoch, args.epochs):
    analyzer.start_epoch(epoch)
    train(epoch, analyzer)
    test(epoch, analyzer)
    analyzer.end_epoch(epoch)

if args.epochs == 0:
    analyzer.start_epoch(0)
    test(0, analyzer)
    analyzer.end_epoch(0)
print(f'Best accuracy: {best_acc} // Checkpoint name: {checkpoint_fname}')
