"""Generates wnids using class names for torchvision dataset"""

import argparse
import torchvision
from nltk.corpus import wordnet as wn
from nbdt.graph import synset_to_wnid
from pathlib import Path
from nbdt.utils import Colors
import os


parser = argparse.ArgumentParser()
parser.add_argument('--dataset',
    choices=('CIFAR100', 'CIFAR10'),
    default='CIFAR10')
parser.add_argument('--root', default='./data')
args = parser.parse_args()

dataset = getattr(torchvision.datasets, args.dataset)(root='./data', download=True)

path = Path(os.path.join(args.root, args.dataset, 'wnids.txt'))
os.makedirs(path.parent, exist_ok=True)
failures = []

hardcoded_mapping = {
    'aquarium_fish': wn.synsets('fingerling', pos=wn.NOUN)[0],
    'beaver': wn.synsets('beaver', pos=wn.NOUN)[-1],
    'castle': wn.synsets('castle', pos=wn.NOUN)[1],
    'flatfish': wn.synsets('flatfish', pos=wn.NOUN)[1],
    'leopard': wn.synsets('leopard', pos=wn.NOUN)[1],
    'lobster': wn.synsets('lobster', pos=wn.NOUN)[1],
    'maple_tree': wn.synsets('maple', pos=wn.NOUN)[1],
    'otter': wn.synsets('otter', pos=wn.NOUN)[1],
    'plate': wn.synsets('plate', pos=wn.NOUN)[3],
    'raccoon': wn.synsets('raccoon', pos=wn.NOUN)[1],
    'ray': wn.synsets('ray', pos=wn.NOUN)[-1],
    'seal': wn.synsets('seal', pos=wn.NOUN)[-1],
    'shrew': wn.synsets('shrew', pos=wn.NOUN)[1],
    'skunk': wn.synsets('skunk', pos=wn.NOUN)[1],
    'tiger': wn.synsets('tiger', pos=wn.NOUN)[1],
    'table': wn.synsets('table', pos=wn.NOUN)[1],
    'turtle': wn.synsets('turtle', pos=wn.NOUN)[1],
    'whale': wn.synsets('whale', pos=wn.NOUN)[1],
}

wnids = []
for cls in dataset.classes:
    if cls in hardcoded_mapping:
        synset = hardcoded_mapping[cls]
    else:
        synsets = wn.synsets(cls, pos=wn.NOUN)
        if not synsets:
            Colors.red(f'==> Failed to find synset for {cls}')
            failures.append(cls)
            continue
        synset = synsets[0]
    wnid = synset_to_wnid(synset)
    print(f'{wnid}: ({cls}) {synset.definition()}')
    wnids.append(wnid)

with open(str(path), 'w') as f:
    f.write('\n'.join(wnids))

if failures:
    Colors.red(f'==> Warning: failed to find wordnet IDs for {failures}')
Colors.green(f'==> Wrote to {path}')
