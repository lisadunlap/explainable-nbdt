from networkx.readwrite.json_graph import node_link_data, node_link_graph
import networkx as nx
import argparse

from nbdt.graph import read_graph, get_roots, write_graph

parser = argparse.ArgumentParser()
parser.add_argument('--path-graph', type=str, help='path to json file of graph to edit')
parser.add_argument('--path-save', type=str, help='path to save json file of graph')
parser.add_argument('--max-depth', type=int, help='nodes past this depth are grouped together')

args = parser.parse_args()


def prune_height(G, node, depth, max_depth):
	if depth < max_depth:
		for node_ in G.neighbors(node):
			prune_height(G, node_, depth + 1, max_depth)
	else:
		child_nodes, leaf_nodes = get_child_nodes_and_leaves(G, node)
		for child in child_nodes:
			G.remove_node(child)
		for leaf in leaf_nodes:
			if not G.has_edge(node, leaf):
				G.add_edge(node, leaf)

# get all descendant nodes of this node, inserting into child_nodes and leaf_nodes
def get_child_nodes_and_leaves(G, node):
	child_nodes = []
	leaf_nodes = []
	for node_ in G.neighbors(node):
		get_child_nodes_and_leaves_helper(G, node_, child_nodes, leaf_nodes)

	return child_nodes, leaf_nodes

def get_child_nodes_and_leaves_helper(G, node, child_nodes, leaf_nodes):
	if len(list(G.neighbors(node))) == 0:
		leaf_nodes.append(node)
	else:
		child_nodes.append(node)

	for node_ in G.neighbors(node):
		get_child_nodes_and_leaves_helper(G, node_, child_nodes, leaf_nodes)

path = args.path_graph
G = read_graph(path)
roots = list(get_roots(G))

assert len(roots) == 1, "expected one root"

prune_height(G, roots[0], 0, args.max_depth)
write_graph(G, args.path_save)
