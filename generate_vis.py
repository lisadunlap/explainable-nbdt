import json
import argparse
import torchvision
import os
import numpy as np

from pathlib import Path
from nbdt.utils import Colors, METHODS, DATASET_TO_FOLDER_NAME
from nbdt.graph import generate_fname, get_parser, read_graph, get_roots, \
    get_wnids_from_dataset, get_directory, get_graph_path_from_args
from networkx.readwrite.json_graph import adjacency_data
from nbdt import data


def build_tree(G, root, parent='null'):
    weight = 2 * G.nodes[root].get('weight', float('inf')) - 1
    return {
        'name': root,
        'label': G.nodes[root].get('label', ''),
        'weight': G.nodes[root].get('weight', ''),
        'parent': parent,
        'children': [build_tree(G, child, root) for child in G.succ[root]],
        'fill': 24 * (np.tanh(weight) + 1)
    }

def build_graph(G):
    return {
        'nodes': [{
            'name': wnid,
            'label': G.nodes[wnid].get('label', ''),
            'id': wnid
        } for wnid in G.nodes],
        'links': [{
            'source': u,
            'target': v
        } for u, v in G.edges]
    }


def generate_vis(path_template, data, name, fname, out_dir='out/'):
    with open(path_template) as f:
        html = f.read().replace(
            "'TREE_DATA_CONSTANT_TO_BE_REPLACED'",
            json.dumps(data))

    os.makedirs('out', exist_ok=True)
    path_html = f'{out_dir}{fname}-{name}.html'
    with open(path_html, 'w') as f:
        f.write(html)

    Colors.green('==> Wrote HTML to {}'.format(path_html))


def main():
    parser = get_parser()
    args = parser.parse_args()

    path = args.json_path

    if not os.path.isfile(path):  # is directory
        for f in os.listdir(path):
            if f.split('.')[-1] != 'json':
                continue
            setup_vis(args, os.path.join(path, f), out_dir=f'out/{path.split("/")[-2]}/')
    else:
        setup_vis(args, path)

def setup_vis(args, path, out_dir='out/'):
    print('==> Reading from {}'.format(path))

    G = read_graph(path)

    roots = list(get_roots(G))
    num_roots = len(roots)
    root = next(get_roots(G))
    tree = build_tree(G, root)
    graph = build_graph(G)

    if num_roots > 1:
        Colors.red(f'Found {num_roots} roots! Should be only 1: {roots}')
    else:
        print(f'Found just {num_roots} root.')

    #fname = generate_fname(**vars(args)).replace('graph-', '', 1)
    fname = path.split('/')[-1].replace('.pth', '').replace('.json', '')
    print(fname)
    if args.method == 'weighted':
        generate_vis('vis/tree-weighted-template.html', tree, 'tree', fname, out_dir=out_dir)
    else:
        generate_vis('vis/tree-template.html', tree, 'tree', fname)
        generate_vis('vis/graph-template.html', graph, 'graph', fname)


if __name__ == '__main__':
    main()
