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
    get_transform_from_name, test_word2vec, DATASET_TO_FOLDER_NAME
)
import numpy as np
from numpy import linalg as LA

def get_word2vec(trainset, dimension=300, save_path='./data'):
    """ Gets Word2Vec embeddings from pretrained models and saves
    them and the projection matrix"""

    print('==> Generate in word2vec embeddings...')
    path = save_path+'/word2vec/'
    if not os.path.exists(path):
        os.makedirs(path)
    # load pretrained model
    import gensim.downloader as api
    model = api.load('conceptnet-numberbatch-17-06-300')
    try:
        projection_matrix = np.load('./data/projection.npy')
    except:
        projection_matrix = np.random.rand(dimension, 512)
        np.save('./data/projection.npy', projection_matrix)

    for i, cls in enumerate(trainset.classes):
        word_vec = model.wv[f"/c/en/{cls}"]
        word_vec = np.asarray(word_vec).reshape(1, dimension)
        word_vec = np.matmul(word_vec, projection_matrix)[0]
        word_vec = np.array(word_vec/LA.norm(word_vec), dtype=float)
        np.save(path+cls+'.npy', word_vec)
    Colors.cyan(f"Word2Vec embeddings save to {path}")

if __name__ == '__main__':
    datasets = ('CIFAR10', 'CIFAR100') + data.imagenet.names + data.custom.names

    parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
    parser.add_argument('--dataset', default='CIFAR10', choices=datasets)
    parser.add_argument('--input-size', type=int,
                        help='Set transform train and val. Samples are resized to '
                             'input-size + 32.')
    parser.add_argument('--save-path', type=str, help="override save path")

    data.custom.add_arguments(parser)
    loss.add_arguments(parser)
    analysis.add_arguments(parser)
    args = parser.parse_args()

    data.custom.set_default_values(args)

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

    if args.save_path:
        get_word2vec(trainset, save_path=args.save_path)
    else:
        get_word2vec(trainset, save_path='./data/' + DATASET_TO_FOLDER_NAME[args.dataset])