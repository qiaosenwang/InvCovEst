import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def make_relation(precision_list):
    p = precision_list[0].shape[0]
    relation = np.zeros_like(precision_list[0])
    for i in range(1, p):
        for j in range(0, i):
            for precision in precision_list:
                if precision[i, j] != 0:
                    relation[i, j] += 1
    return relation + relation.T


def network(label, precision_list):
    p = len(label)
    n = len(precision_list)
    G = nx.Graph()
    e_sure = []
    e_suspect = []
    relation = make_relation(precision_list)

    for name in label:
        G.add_node(name)
    for i in range(1, p):
        for j in range(0, i):
            if relation[i, j] == n:
                G.add_edge(label[i], label[j])
                e_sure.append(tuple((label[i], label[j])))
            elif relation[i, j] > 0:
                G.add_edge(label[i], label[j])
                e_suspect.append(tuple((label[i], label[j])))

    pos = nx.circular_layout(G)
    fig, ax = plt.subplots()

    nx.draw_networkx_edges(G, pos, edgelist=e_sure,
                           width=1, edge_color='darkorange', nodelist=label, node_size=300)
    nx.draw_networkx_edges(G, pos, edgelist=e_suspect,
                           width=1, alpha=0.7, edge_color='firebrick', style='dashed')
    nx.draw_networkx_nodes(G, pos, node_color='royalblue', alpha=0.9)
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')


    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.savefig('./generated_image.png', dpi=600)
    plt.show()



