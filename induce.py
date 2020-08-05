#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  2 21:37:34 2020

@author: Weifan Chen , Diala @ Boston Univeristy

A fast demo on the guided induction decision tree.

similarity matrix meaning, for m_ij
is the similarity score between class i and j (so the matrix is symmetric)
if i and j are the mutual  best match, then m_ij should be the largest entry among all the
entries in the cross, centered at i,j. (not counting i,i and j,j the self similarity
score which is the largest)

Result:
    for the CIFAR10 dataset, the pretrained WideResNet28 model (without any interference
    by NBDT, just a pure pretrained model on CIFAR10), the guided induction algorithm
    would give the same structure as WordNet gives, with the sole exception on
    label 2 (bird) and label 6 (frog).

"""
from operator import itemgetter
import numpy as np
import nbdt
import networkx as nx

'''
Node represents a node in the NBDT
Attribute:
	index: as identifier of each class, only on the leaf nodes layer, the index rep the real class label. 
		   for intermediate node, the index should also give the correct weights by matrix[index,:]
	depth: leaf node has depth==0, increment by 1 for each induced layer. Caveat: there exists node increases depth 
			but has no pair (go to next layer by itself, without induction with other node)
	weight: it's associate weight, the number of entry should equal to the number of output nodes from the backbone DNN
			this should equal to matrix[index,:]
	children: the previous layer sub node derived from this node
	parent: node
	matrix: the weight matrix that serves a similar role as the starting big matrix. Semantically this is quite different from
			a weight matrix in dnn		
	best_match: tuple consisting of node index and similarity value between this node and its best match
	(new) terminals: list of Node, to calculate the similarity between two trees, each intermediate node needs
		the information of all the leaves (terminals) branched from it (the root of the substree)
	(new) isLeaf: bool : 
	(new) isTransit: bool: if a node has only one child, then this node is Transit. Leaf node is also transit, since we dont' want them 
		to participate the calculation for tree similarity
Method:
	calc_weight (static) : this could be used by a parent to get its weights by passing its children to this method
'''


# TODO: consider wnids instead of index
# TODO: then we can use wnid_to_node to create Node objects
# TODO: num_classes needed in TreeSupLoss
# TODO: test passing the Graph to TreeSupLoss class / SoftNBDT

class Node():
    def __init__(self, index, depth, weight, *children, name=None, formal_name=None, parent=None, matrix=None,
                 matches=None, best_match=None, wnid=None):
        self.index = index
        self.depth = depth
        self.weight = weight
        self.matrix = matrix
        self.children = children
        self.name = name
        self.parent = parent
        self.matches = matches
        self.best_match = best_match
        self.terminals = []  ## leaf hold itself as an entry, so that for parents, they can get theirs by concatenate all their children's terminal
        self.isLeaf = False
        self.isTransit = False
        self.wnid = wnid
        self.formal_name = None

    def __len__(self):
        return len(self.children)

    def __eq__(self, other):
        return self.name == other.name

    def __str__(self):
        return self.name

    def __hash__(self):
        return hash(str(self))

    def set_isLeaf_true(self):
        self.isLeaf = True

    def set_isTransit_true(self):
        self.isTransit = True

    @staticmethod
    def calc_weight(nodes, method='average'):
        weights = np.array([node.weight for node in nodes])
        if method == 'average':
            return np.average(weights, axis=0)


'''
Method for building the leaf node of NBDT

Params:
	matrix: the very last weight matrix of the pretrained model
	depth: should always be zero, since this method is only used for building leaf
	classes: list of names for each label
'''


def build_leaf_nodes(matrix, metric, depth=0, classes=None, wnids=None, verbose=1):
    # get the similarity matrix
    if verbose > 0: print('build_leaf_nodes called')
    simi_mat = simi_matrix(matrix, metric)
    if verbose > 0: print('simi mat generate done')
    # create nodes
    nodes = []
    for i in range(0, matrix.shape[0]):
        if verbose > 0: print('start create node ', i)
        node = Node(i, depth, matrix[i, :].numpy().copy(), None, matrix=matrix, matches=simi_mat[i])
        if classes is not None:
            node.name = classes[i]
            node.formal_name = classes[i]
        if wnids is not None: node.wnid = wnids[i]
        node.matches = set()
        nodes.append(node)

    # compute matches for each node
    for i in range(len(simi_mat)):
        if verbose > 0: print('compute match, ', i)
        mask = np.zeros(simi_mat[i].size, dtype=bool)
        mask[i] = True
        masked = np.ma.array(simi_mat[i], mask=mask)
        j = masked.argmax()
        value = masked.max()
        nodes[i].matches.add((j, value))
        nodes[j].matches.add((i, value))
    [node.set_isLeaf_true() for node in nodes]
    [node.set_isTransit_true() for node in nodes]
    [node.terminals.append(node) for node in nodes]
    return nodes


"""
*** Method for user ***
build the entire hierarchy

Param: 
	model: the pretrained model
	metric (callable) to calculate the similarity between two class 
		TODO: extend the metric for multi class
	classes: list of names for each class
	method: how to calculate the node weight
		avail args: 'average' : use the average node weight from all its children
	policy: how to deal with nodes that do not have mutual best match
		avail args: 'wait' : unpaired node would become parent of itself, nothing changes
"""


def build_full(w, b, metric, classes=None, wnids=None, method='average', policy='wait', verbose=0):
    # try:
    #     w = model['fc.weight'].numpy()  ## TODO: this part could udpate according to alvin's implementation
    #     b = model['fc.bias'].numpy()
    # except:  # if model has been loaded from checkpoint
    #     w = model['module.fc.weight'].cpu().numpy()
    #     b = model['module.fc.bias'].cpu().numpy()
    print('building starts...')
    print('shape of the output layer: ', w.shape)
    G = nx.DiGraph()
    nodes = build_leaf_nodes(w, metric, classes=classes, wnids=wnids, verbose=verbose)
    if verbose > 0: print('build leaf done.')
    for n in nodes:
        G.add_node(n)
    nodes_temp = nodes
    n_nodes = len(nodes)
    while True:
        if verbose > 0: print('call init_next_layer')
        nodes, mat, G = init_next_layer(G, nodes_temp, metric, method=method, policy=policy)
        if len(nodes) == 1 or len(nodes) == n_nodes:
            print('induction done with ', len(nodes), ' root nodes.')
            return G, nodes
        n_nodes = len(nodes)
        nodes_temp = nodes


'''
Output a list of induced node and the weight matrix for the newly generated layers.
The output could be the new input for the method, until reducing to only one node or 
model does not converge anymore.

Params:
	nodes: list of nodes to be induced on
	metric (callable) : rule to compute similarity

Return:
	new_nodes: list of nodes for induction layer
	new_matrix: a finished big matrix used by induction nodes
'''


def init_next_layer(G, nodes, metric, method='average', policy='wait', verbose=0):
    # step 1 : get variables needed for build induction node
    cur_depth = nodes[0].depth
    matrix = nodes[0].matrix
    pairs, singles = get_pairs(matrix, metric)
    assert len(pairs) * 2 + len(singles) == matrix.shape[0], 'this should add up'
    if policy == 'wait':
        class_n = matrix.shape[0] - len(
            pairs)  ## implicitly , two nodes induce one , or a node leftover .total - 2 * n_pair + n_pair
        new_matrix = np.zeros((class_n, matrix.shape[1]))  ## each new parent should hold a reference of this matrix
    elif policy == 'wait-match':
        from math import ceil
        new_matrix = np.zeros(
            (ceil(matrix.shape[0] / 2), matrix.shape[1]))  ## this is the ceil of division by 2 , ceil(5/2) = 3
    """
    BUG: the bug should be here somehow, since right now this calculation does not hold true
    since singles are somehow also paired, so the matrix is in wrong dimentiosn at axis=0
    """
    index = 0
    new_nodes = []

    # step 2 : for those have mutual best match nodes, instantiate their parent
    for pair in pairs:
        children = _pair_to_nodes(nodes, pair)
        parent = Node(index, cur_depth + 1, None, *children, matrix=new_matrix)
        [parent.terminals.extend(child.terminals) for child in children]  ## add terminal info to parent
        parent.wnid = ''.join([child.wnid for child in parent.children])
        parent.weight = Node.calc_weight(parent.children, method)
        parent_name = ''
        for child in children:
            child.parent = parent
            parent_name += child.name + '-'
        parent.formal_name = '(' + ','.join([child.formal_name for child in children]) + ')'
        parent.name = parent_name
        new_nodes.append(parent)
        G.add_node(parent)
        for i in children: G.add_edge(parent, i)  # this is a directed edge
        index += 1

    # step 3 : for those unpaired, based on policy, to decide how to connect them
    if verbose > 0:
        print('at depth ', cur_depth, ', ', len(singles), ' unpaired nodes.')

    if policy == 'wait':
        # Not creating new node, simply pass the old node to the new layer, otherwise G would be disconnected
        for n in singles:
            child = nodes[n]
            child.index = index
            child.depth = cur_depth + 1
            child.matrix = new_matrix
            new_nodes.append(child)
            index += 1

    # add a new policy here that based on the fact that each submatrix contains at least one mutual best match albeit this is not proved
    elif policy == 'wait-match':
        # step 1: use all the singles weight to create the sub matrix
        # step 2: use get_pairs to form a merge
        # step 3: build parents as necessary
        # step 4: repeat until there is nothing left or only one remains    
        while (len(singles) > 1):
            weights = [nodes[n].weight for n in singles]
            mat = np.stack(weights, axis=0)
            ## reindex the nodes here 
            t_node = [nodes[n] for n in singles]
            nodes = t_node
            pairs, singles = get_pairs(mat, metric)
            assert len(pairs) * 2 + len(singles) == mat.shape[0], 'this should add up'
            ## same code from the starting init_next_layer, with modification, since we need to concat the existing new_matrix
            for pair in pairs:
                children = _pair_to_nodes(nodes, pair)
                parent = Node(index, cur_depth + 1, None, *children, matrix=new_matrix)
                [parent.terminals.extend(child.terminals) for child in children]  ## add terminal info to parent
                parent.wnid = ''.join([child.wnid for child in parent.children])
                parent.weight = Node.calc_weight(parent.children, method)
                parent_name = ''
                for child in children:
                    child.parent = parent
                    parent_name += child.name + '-'
                parent.formal_name = '(' + ','.join([child.formal_name for child in children]) + ')'
                parent.name = parent_name
                new_nodes.append(parent)
                G.add_node(parent)
                for i in children: G.add_edge(parent, i)  # this is a directed edge
                index += 1
        if len(singles) == 1:
            child = nodes[singles[0]]
            child.index = index
            child.depth = cur_depth + 1
            child.matrix = new_matrix
            new_nodes.append(child)
            index += 1
    elif policy == 'match':
        # new_mat = nodes[singles[0]].matrix
        if len(singles) > 1:
            children = [nodes[i] for i in singles]
            # print(simi_mat)
            # get best match for each single
            for n in singles:
                ## WF: if all singles are at the same depth, their .matrix should be the same, so this can move outside the for loop
                simi_mat = simi_matrix(nodes[n].matrix, metric)
                mask = np.zeros(len(simi_mat[n]), dtype=bool)
                mask[n] = True
                masked = np.ma.array(simi_mat[n], mask=mask)
                j = masked.argmax()
                value = masked.max()
                nodes[n].best_match = (j, value)
                nodes[n].matches.remove(nodes[n].best_match)
                print(nodes[n].matches)
                # now update similarities of the other matching classes
                for cls, val in nodes[n].matches:
                    excluded = nodes[cls].matches - {(cls, val)}
                    print(excluded)
                    cls_simi = [(cls1, min(val, val1)) for cls1, val1 in excluded]
                    # print("matching %d with %s" % (cls, cls_simi))
                    nodes[cls].matches = {max(cls_simi, key=itemgetter(1))}
                    print(nodes[cls].__str__(), nodes[cls].best_match[0])
                    parent = Node(index, cur_depth + 1, None, [nodes[cls], nodes[n]], matrix=new_matrix)
                    parent.weight = Node.calc_weight(parent.children, method)
                    parent_name = ''
                    for child in children:
                        child.parent = parent
                        parent_name += child.name + '-'
                    parent.name = parent_name
                    G.add_node(parent)
                    new_nodes.append(parent)
                    for i in children:
                        G.add_edge(parent, i)

    # step 4: use parents' weight to fill up the bigger matrix
    for i, parent in enumerate(new_nodes):
        new_matrix[i, :] = parent.weight.reshape(1, new_matrix.shape[1])
    return new_nodes, new_matrix, G


def _pair_to_nodes(nodes, pair):
    return [nodes[index] for index in pair]


'''
parameter: 
    w: np.array for the weight matrix, 
    clas1, clas2 are index of the class
    metric (callable) the function map to the weight pairs
        e.g. input a_i conects class1 and class2 by two weights, the 
             metric would output a number based on this two value,
             the returned value should be a indication of the similarity 
             between the two classes from the perspective of input a_i
return: similarity score between cls1 and cls2
'''


def simi(w, cls1, cls2, metric):
    """it metric as a method only used here, so we can savely make it a string and use accordingly"""
    v1, v2 = w[cls1,:], w[cls2,:]
    if metric=='naive':
        metric = naive
        return sum(map(metric, list(zip(w[cls1, :], w[cls2, :]))))
    elif metric=='l1':
        return np.average(abs(v1-v2))
    elif metric=='l2':
        return np.average((v1-v2)**2)
    elif metric=='cos':
        return np.average(v1*v2)
    elif metric=='euc':
        return np.sqrt(sum((v1-v2)**2))
    else:
        raise AttributeError('unknown metric ', metric)


'''
The callable for metric in simi
if two weights are all positive, then we take the smaller one
if two weights are different sign, we take the negative of the absolution of their difference
otherwise, return 0

TODO: try more metric callable to see which has better meaning

'''

def naive(pair):
    x, y = pair
    if x >= 0 and y >= 0:
        return min(x, y)
    elif x * y <= 0:
        return -np.abs(x - y)
    else:
        return 0


'''
Param:
    w: np.array of the weight matrix
    metric: the callable
Return:
    the similarity weight matrix (np.array)
    the value m_ij is the similarity score between class i and j
    the matrix is symmetric by definition 
    the diagonal entry should be the largest along its row and column

TODO:
    1. normalization?
'''


def simi_matrix(w, metric):
    # print('simi matrix called')
    n = w.shape[0]
    # print('n: ', n)
    mat = np.zeros((n, n))
    # print('mat shape: ', mat.shape)
    for i in range(0, n):
        for j in range(0, n):
            mat[i, j] = simi(w, i, j, metric)
    return mat


def resmat(mat):
    mat_cpy = mat.copy()
    col_mat = np.zeros(mat.shape)
    row_mat = np.zeros(mat.shape)
    np.fill_diagonal(mat_cpy, np.NINF)
    indice_col = np.argmax(mat_cpy, axis=0)
    indice_row = np.argmax(mat_cpy, axis=1)
    for i in range(0, col_mat.shape[0]):
        col_mat[indice_col[i], i] = 1
        row_mat[i, indice_row[i]] = 1
    return np.logical_and(col_mat, row_mat), col_mat, row_mat


def induce_pairs(hot_mat):
    x, y = np.where(np.tril(hot_mat) == 1)
    pairs = [pair for pair in zip(x, y)]
    assert len(pairs) == hot_mat.sum() / 2, 'unmatched pair'
    indice = [index for pair in pairs for index in pair]
    singles = []
    for i in range(0, hot_mat.shape[0]):
        if i not in indice:
            singles.append(i)
    return pairs, singles


'''
    This method is a pipline of the previous three, given the big matrix and metric,
    output the best match pairs indice, and index of node that are not paired
'''


def get_pairs(w, metric):
    hot_mat, _, _ = resmat(simi_matrix(w, metric))
    return induce_pairs(hot_mat)


def build_Gw_from_Gn(
        Gn):  ## build G_wnid from G_node, wnid is only for leaf node and other edge tracking. The label would be replaced by Node.formal_name
    Gw = nx.DiGraph()
    d = {}
    for node in list(Gn.nodes):
        d[node.wnid] = node.formal_name
    [Gw.add_edge(list(edge)[0].wnid, list(edge)[1].wnid) for edge in list(Gn.edges)]
    nx.set_node_attributes(Gw, d, 'label')
    return Gw


def display_tree(start_node, verbose=0):
    visited = []
    queue = []
    root = start_node[0]
    visited.append(root)
    queue.append(root)
    while queue:
        s = queue.pop(0)
        assert s
        print(s.depth, s.index, s.name)
        if verbose > 0:
            print(s.formal_name)
            print(s.wnid, '\n')
        if s.children[0] is None:  ## reaching the leaf, leaf.children = (None,)
            continue
        for i in s.children:
            if i not in visited:
                visited.append(i)
                queue.append(i)
