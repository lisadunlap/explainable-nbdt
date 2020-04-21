from nbdt.utils import DATASETS, METHODS, DATASET_TO_FOLDER_NAME, Colors
from nbdt.graph import get_parser, get_wnids_from_dataset, read_graph, \
    get_leaves, generate_fname, get_directory, get_graph_path_from_args, \
    get_roots, add_paths, write_graph, augment_graph, condense_leaves, delete_paths
from pathlib import Path
import argparse
import os

def main():

    parser = get_parser()
    args = parser.parse_args()

    wnids = get_wnids_from_dataset(args.dataset)
    path = get_graph_path_from_args(args)
    if args.json_path:
        path = args.json_path
    print('==> Reading from {}'.format(path))

    G = read_graph(path)

    if args.method == 'clustered':
        G = condense_leaves(G)
        write_path = path.replace('.json', '-clustered.json')
        write_graph(G, write_path)

    if args.parents:
        G = add_paths(G, args.parents, args.children)
        write_path = path.replace('.json', '-modified.json')
        write_graph(G, write_path)

    Colors.green('==> Wrote tree to {}'.format(write_path))

if __name__ == '__main__':
    main()
