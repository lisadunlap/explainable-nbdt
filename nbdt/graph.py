import networkx as nx
import json
import random
import numpy as np
from nltk.corpus import wordnet as wn
from nbdt.utils import DATASETS, METHODS, DATASET_TO_FOLDER_NAME, Colors
from networkx.readwrite.json_graph import node_link_data, node_link_graph
from sklearn.cluster import AgglomerativeClustering
from pathlib import Path
import torch
import argparse
import os


def get_parser():
    import models

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        help='Must be a folder data/{dataset} containing a wnids.txt',
        choices=DATASETS,
        default='CIFAR10')
    parser.add_argument(
        '--extra',
        type=int,
        default=0,
        help='Percent extra nodes to add to the tree. If 100, the number of '
        'nodes in tree are doubled. Note this is an integral percent.')
    parser.add_argument(
        '--single-path',
        action='store_true',
        help='Ensure every leaf only has one path from the root.')
    parser.add_argument('--no-prune', action='store_true', help='Do not prune.')
    parser.add_argument('--fname', type=str,
        help='Override all settings and just provide a path to a graph')
    parser.add_argument('--method', choices=METHODS,
        help='structure_released.xml apparently is missing many CIFAR100 classes. '
        'As a result, pruning does not work for CIFAR100. Random will randomly '
        'join clusters together, iteratively, to make a roughly-binary tree.',
        default='wordnet')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--branching-factor', type=int, default=2)
    parser.add_argument('--induced-checkpoint', type=str,
        help='(induced graph) Checkpoint to load into model. The fc weights '
        'are used for clustering.')
    parser.add_argument('--induced-linkage', type=str, default='ward',
        help='(induced graph) Linkage type used for agglomerative clustering')
    parser.add_argument('--induced-affinity', type=str, default='euclidean',
        help='(induced graph) Metric used for computing similarity')
    parser.add_argument('--json-path', type=str, default='',
                        help='Path to json file of graph')
    parser.add_argument("--parents", nargs='*', type=str,
                        help="extra paths to add, this should be a list of parents")
    parser.add_argument("--children", nargs='*', type=str,
                        help="extra paths to add, this should be a list of \
                        children corresponding to the parent nodes")
    parser.add_argument('--ignore-labels', nargs='*', type=int, default=[],
                        help='labels to ignore when clustering, will be added at the end')
    parser.add_argument("--include-classes", nargs='*', type=str, help='classes to include')
    parser.add_argument('--ood-path-wnids', type=str, help='path to wnids.txt for ood-dataset')
    return parser


def generate_fname(method, seed=0, branching_factor=2, extra=0,
                   no_prune=False, fname='', single_path=False,
                   induced_linkage='ward', induced_affinity='euclidean',
                   induced_checkpoint=None, **kwargs):
    if fname:
        return fname

    fname = f'graph-{method}'
    if method == 'random':
        if seed != 0:
            fname += f'-seed{seed}'
    if method == 'induced':
        assert induced_checkpoint is not None, \
            'Cannot build induced graph without a checkpoint'
        if induced_linkage != 'ward' and induced_linkage is not None:
            fname += f'-linkage{induced_linkage}'
        if induced_affinity != 'euclidean' and induced_affinity is not None:
            fname += f'-affinity{induced_affinity}'
        checkpoint_stem = Path(induced_checkpoint).stem
        checkpoint_suffix = '-'.join(checkpoint_stem.split('-')[2:])
        checkpoint_fname = checkpoint_suffix.replace('-induced', '')
        fname += f'-{checkpoint_fname}'
    if method in ('random', 'induced'):
        if branching_factor != 2:
            fname += f'-branch{branching_factor}'
    if extra > 0:
        fname += f'-extra{extra}'
    if no_prune:
        fname += '-noprune'
    if single_path:
        fname += '-single'
    return fname


def get_directory(dataset, root='./data'):
    folder = DATASET_TO_FOLDER_NAME[dataset]
    return os.path.join(root, folder)


def get_wnids_from_dataset(dataset, root='./data', path_wnids_ood=None):
    directory = get_directory(dataset, root)
    return get_wnids(os.path.join(directory, 'wnids.txt'), path_wnids_ood)


def get_wnids(path_wnids, path_wnids_ood=None):
    with open(path_wnids) as f:
        wnids = set([wnid.strip() for wnid in f.readlines()])
    if path_wnids_ood:
        with open(path_wnids_ood) as f:
            wnids_ood = set([wnid.strip() for wnid in f.readlines()])
            wnids = wnids - wnids_ood
    return list(wnids)


def get_graph_path_from_args(args):
    fname = generate_fname(**vars(args))
    directory = get_directory(args.dataset)
    path = os.path.join(directory, f'{fname}.json')
    return path


def synset_to_wnid(synset):
    return f'{synset.pos()}{synset.offset():08d}'


def wnid_to_synset(wnid):
    offset = int(wnid[1:])
    pos = wnid[0]

    try:
        return wn.synset_from_pos_and_offset(wnid[0], offset)
    except:
        try:
            return non_wn_synset_from_wnid(wnid)
        except:
            return FakeSynset(wnid)


def non_wn_synset_from_wnid(wnid):
    with open('./data/fake_wnid_to_class.json', 'r') as f:
        wnid_to_class = json.load(f)
    return FakeSynset(wnid, wnid_to_class[wnid])


def synset_to_name(synset):
    return synset.name().split('.')[0]


def is_leaf(G, node):
    return len(G.succ[node]) == 0


def get_leaves(G, root=None):
    nodes = G.nodes if root is None else nx.descendants(G, root) | {root}
    for node in nodes:
        if is_leaf(G, node):
            yield node


def get_non_leaves(G):
    for node in G.nodes:
        if len(G.succ[node]) > 0:
            yield node


def get_roots(G):
    for node in G.nodes:
        if len(G.pred[node]) == 0:
            yield node


def get_root(G):
    roots = list(get_roots(G))
    assert len(roots) == 1, f'Multiple ({len(roots)}) found'
    return roots[0]

def get_node_depth(G, node):
    if not G.succ[node]:
        return 1
    return max([_get_depth(child) for child in G.succ[node]]) + 1


def get_depth(G):
    return max([get_node_depth(G, root) for root in get_roots(G)])


def get_path_nodes(G, source, target=None):
    """ Returns list of nodes on the shortest path from source to target.
        If target is None or ommitted, return a list of lists, with item i
        being the list of nodes on the shortest path from source to node index i. """
    return nx.shortest_path(G, source=source, target=target)


def get_leaf_weights(G, node, weight=1):
    """
    This is rather specific to our needs. Basically, a node with k children
    splits 'weight' 1/k to each child. This continutes recursively until the
    leaves. A tree with L different leaves may not distribute 1/L weight to
    each class.
    """
    if is_leaf(G, node):
        return {node: weight}
    num_children = len(G.succ[node])
    weight_per_child = weight / float(num_children)

    weights = {}
    for child in G.succ[node]:
        for wnid, weight in get_leaf_weights(G, child, weight_per_child).items():
            weights[wnid] = weights.get(wnid, 0) + weight
    return weights


def set_node_label(G, synset):
    nx.set_node_attributes(G, {
        synset_to_wnid(synset): synset_to_name(synset)
    }, 'label')


def set_random_node_label(G, i):
    nx.set_node_attributes(G, {i: ''}, 'label')


def build_minimal_wordnet_graph(wnids, single_path=False):
    G = nx.DiGraph()

    for wnid in wnids:
        G.add_node(wnid)
        synset = wnid_to_synset(wnid)
        set_node_label(G, synset)

        if wnid == 'n10129825':  # hardcode 'girl' to not be child of 'woman'
            if single_path:
                G.add_edge('n09624168', 'n10129825')  # child of 'male' (sibling to 'male_child')
            else:
                G.add_edge('n09619168', 'n10129825')  # child of 'female'
            G.add_edge('n09619168', 'n10129825')  # child of 'female'
            continue

        hypernyms = [synset]
        while hypernyms:
            current = hypernyms.pop(0)
            set_node_label(G, current)
            for hypernym in current.hypernyms():
                G.add_edge(synset_to_wnid(hypernym), synset_to_wnid(current))
                hypernyms.append(hypernym)

                if single_path:
                    break

        children = [(key, wnid_to_synset(key).name()) for key in G.succ[wnid]]
        assert len(children) == 0, \
            f'Node {wnid} ({synset.name()}) is not a leaf. Children: {children}'
    return G


def build_random_graph(wnids, seed=0, branching_factor=2):
    random.seed(seed)

    G = nx.DiGraph()

    if seed >= 0:
        random.shuffle(wnids)
    current = None
    remaining = wnids

    # Build the graph from the leaves up
    while len(remaining) > 1:
        current, remaining = remaining, []
        while current:
            nodes, current = current[:branching_factor], current[branching_factor:]
            remaining.append(nodes)

    # Construct networkx graph from root down
    G.add_node('0')
    set_random_node_label(G, '0')
    next = [(remaining[0], '0')]
    i = 1
    while next:
        group, parent = next.pop(0)
        if len(group) == 1:
            if isinstance(group[0], str):
                G.add_node(group[0])
                synset = wnid_to_synset(group[0])
                set_node_label(G, synset)
                G.add_edge(parent, group[0])
            else:
                next.append((group[0], parent))
            continue

        for candidate in group:
            is_leaf = not isinstance(candidate, list)
            wnid = candidate if is_leaf else str(i)
            G.add_node(wnid)
            if is_leaf:
                synset = wnid_to_synset(wnid)
                set_node_label(G, synset)
            else:
                set_random_node_label(G, wnid)
            G.add_edge(parent, wnid)
            i += 1

            if not is_leaf:
                next.append((candidate, wnid))
    return G


def prune_single_successor_nodes(G):
    for node in G.nodes:
        if len(G.succ[node]) == 1:
            succ = list(G.succ[node])[0]
            G = nx.contracted_nodes(G, succ, node, self_loops=False)
    return G


def write_graph(G, path):
    with open(path, 'w') as f:
        json.dump(node_link_data(G), f)


def read_graph(path):
    with open(path) as f:
        return node_link_graph(json.load(f))


################
# INDUCED TREE #
################


def build_induced_graph(wnids, checkpoint, linkage='ward', affinity='euclidean',
        branching_factor=2, ignore_labels=[]):
    centers_all = get_centers(checkpoint)
    wnids_all = wnids
    use_labels = [label for label in range(centers_all.size(0)) if label not in ignore_labels]
    wnids = [wnid for label, wnid in enumerate(wnids) if label not in ignore_labels]
    centers = centers_all[use_labels,:]
    n_classes = centers.size(0)

    G = nx.DiGraph()

    # add leaves
    center_to_wnid = {}
    for i, wnid in enumerate(wnids):
        center_to_wnid[i] = wnid
        G.add_node(wnid)
        set_node_label(G, wnid_to_synset(wnid))

    # add rest of tree
    clustering = AgglomerativeClustering(
        linkage=linkage,
        n_clusters=branching_factor,
        affinity=affinity,
    ).fit(centers)
    children = clustering.children_
    index_to_wnid = {}

    for index, pair in enumerate(map(tuple, children)):
        parent = FakeSynset.create_from_offset(len(G.nodes))
        G.add_node(parent.wnid)
        index_to_wnid[index] = parent.wnid

        for child in pair:
            if child < n_classes:
                child_wnid = wnids[child]
            else:
                child_wnid = index_to_wnid[child - n_classes]
            G.add_edge(parent.wnid, child_wnid)

    # add originally ignored labels
    
    for label in ignore_labels:
        wnid_new = wnids_all[label]
        G.add_node(wnid_new)
        set_node_label(G, wnid_to_synset(wnid_new))

        # find center in tree that most closely matches
        label_vec = centers_all[label]
        _, closest = torch.max(torch.matmul(centers,label_vec), dim=0)
        closest = int(closest.cpu().numpy())
        
        wnid = center_to_wnid[closest]

        parent = FakeSynset.create_from_offset(len(G.nodes))
        G.add_node(parent.wnid)

        in_node = list(G.in_edges(wnid))[0][0]
        G.remove_edge(in_node, wnid)
        G.add_edge(in_node, parent.wnid)
        G.add_edge(parent.wnid, wnid)
        G.add_edge(parent.wnid, wnid_new)

    assert len(list(get_roots(G))) == 1, list(get_roots(G))
    return G


def get_centers(checkpoint):
    data = torch.load(checkpoint, map_location=torch.device('cpu'))
    try:
        net = data['net']
    except:
        net = data

    keys = ('fc.weight', 'linear.weight', 'module.linear.weight',
            'module.net.linear.weight', 'output.weight', 'module.output.weight',
            'output.fc.weight', 'module.output.fc.weight')
    fc = None
    for key in keys:
        if key in net:
            fc = net[key]
            break
    assert fc is not None, (
        f'Could not find FC weights in {checkpoint} with keys: {net.keys()}')
    return fc.detach()


####################
# AUGMENTING GRAPH #
####################


class FakeSynset:
    def __init__(self, wnid, name='(generated)'):
        self.wnid = wnid
        self.name_ = name

        assert isinstance(wnid, str)

    @staticmethod
    def create_from_offset(offset):
        return FakeSynset('f{:08d}'.format(offset))

    def offset(self):
        return int(self.wnid[1:])

    def pos(self):
        return 'f'

    def name(self):
        return self.name_


def augment_graph(G, extra, allow_imaginary=False, seed=0, max_retries=10000):
    """Augment graph G with extra% more nodes.

    e.g., If G has 100 nodes and extra = 0.5, the final graph will have 150
    nodes.
    """
    n = len(G.nodes)
    n_extra = int(extra / 100. * n)
    random.seed(seed)

    n_imaginary = 0
    for i in range(n_extra):
        candidate, is_imaginary_synset, children = get_new_node(G)
        if not is_imaginary_synset or \
                (is_imaginary_synset and allow_imaginary):
            add_node_to_graph(G, candidate, children)
            n_imaginary += is_imaginary_synset
            continue

        # now, must be imaginary synset AND not allowed
        if n_imaginary > 0:  # hit max retries before, not likely to find real
            return G, i, n_imaginary

        retries, is_imaginary_synset = 0, True
        while is_imaginary_synset:
            candidate, is_imaginary_synset, children = get_new_node(G)
            if retries > max_retries:
                print(f'Exceeded max retries ({max_retries})')
                return G, i, n_imaginary
        add_node_to_graph(G, candidate, children)

    return G, n_extra, n_imaginary


def get_new_node(G):
    """Get new candidate node for the graph"""
    root = get_root(G)
    nodes = list(filter(lambda node: node is not root and not node.startswith('f'), G.nodes))

    children = get_new_adjacency(G, nodes)
    synsets = [wnid_to_synset(wnid) for wnid in children]
    common_hypernyms = get_common_hypernyms(synsets)

    assert len(common_hypernyms) > 0, [synset.name() for synset in synsets]

    candidate = pick_unseen_hypernym(G, common_hypernyms)
    if candidate is None:
        return FakeSynset.create_from_offset(len(G.nodes)), True, children
    return candidate, False, children


def add_node_to_graph(G, candidate, children):
    root = get_root(G)

    wnid = synset_to_wnid(candidate)
    G.add_node(wnid)
    set_node_label(G, candidate)

    for child in children:
        G.add_edge(wnid, child)
    G.add_edge(root, wnid)


def get_new_adjacency(G, nodes):
    adjacency = set(tuple(adj) for adj in G.adj.values())
    children = next(iter(adjacency))

    while children in adjacency:
        k = random.randint(2, 4)
        children = tuple(random.sample(nodes, k=k))
    return children


def get_common_hypernyms(synsets):
    common_hypernyms = set(synsets[0].common_hypernyms(synsets[1]))
    for synset in synsets[2:]:
        common_hypernyms &= set(synsets[0].common_hypernyms(synset))
    return common_hypernyms


def deepest_synset(synsets):
    return max(synsets, key=lambda synset: synset.max_depth())


def pick_unseen_hypernym(G, common_hypernyms):
    candidate = deepest_synset(common_hypernyms)
    wnid = synset_to_wnid(candidate)

    while common_hypernyms and wnid in G.nodes:
        common_hypernyms -= {candidate}
        if not common_hypernyms:
            return None

        candidate = deepest_synset(common_hypernyms)
        wnid = synset_to_wnid(candidate)
    return candidate

def find_leaf_parents(G):
    parents = [n for n in G.nodes if 'f' in n]
    return [G.neighbors(n) for n in parents]

def condense_leaves(G):
    """ For nodes who have children that are leaves and children
    that are other nodes, moves the leaves to be a child of one of
    its neighbor nodes (this only works well for small graphs)
    """
    for node in G.nodes():
        parents = [n for n in G.neighbors(node) if 'f' in n]
        leaves = [n for n in G.neighbors(node) if 'n' in n]
        if len(list(G.neighbors(node))) > 0 and (len(parents) > 0 and len(leaves) > 0):
            #randomly put the leaves with the a parent on the same level
            for leaf in leaves:
                G.add_edge(np.random.choice(parents), leaf)
                G.remove_edge(node, leaf)

    return prune_single_successor_nodes(G)

def add_paths(G, parents, children):
    """ Randomly adds paths in the tree so that each class has
    NUM_PATHS paths
    """
    for (parent, node) in zip(parents, children):
        G.add_edge(parent, node)
        Colors.green('==> added path from {} to {}'.format(parent, node))
    return prune_single_successor_nodes(balance_tree(G))

def delete_paths(G, parents, children):
    for (parent, node) in zip(parents, children):
        G.remove_edge(parent, node)
        Colors.green('==> removed path from {} to {}'.format(parent, node))
    return prune_single_successor_nodes(balance_tree(G))

def balance_tree(G):
    """ when there are more than two leaves in a parent node,
    condense them into one parent (for balancing)"""
    def check_balance(G):
        for node in G.nodes():
            try:
                parents = [n for n in G.neighbors(node) if 'f' in n]
                leaves = [n for n in G.neighbors(node) if 'n' in n]
            except:
                return None, None
            if len(leaves) > 1 and len(parents) > 0:
                return node, leaves
        return None, None

    node, leaves = check_balance(G)
    while node != None:
        new_parent = FakeSynset.create_from_offset(len(G.nodes))
        G.add_node(new_parent.wnid)
        for leaf in leaves:
            G.add_edge(new_parent.wnid, leaf)
            G.remove_edge(node, leaf)
        G.add_edge(node, new_parent.wnid)
        node, leaves = check_balance(G)
        Colors.green('==> combined children {} to node {}'.format(leaves, new_parent.wnid))
    return G
