import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from .sequential import Sequential


def visualize_nn(model: Sequential, title="Multi Layer Perceptron"):
    """Visualize weights & biases of network."""

    def _centered_layout(n):
        """1->1/2; 2->1/3, 2/3; 3->1/4, 2/4, 3/4; 4->1/5, 2/5, 3/5, 4/5 etc."""
        return [i / (n + 1) for i in range(1, n + 1)]

    def _justified_layout(n):
        """1->1/2; 2->0, 1; 3->0, 1/2, 1; 4->0, 1/3, 2/3, 1"""
        if n == 1:
            return [1 / 2]
        if n == 2:
            return [0, 1]

        return [0] + _centered_layout(n - 2) + [1]


    graph = nx.DiGraph()

    num_layers = len(model) + 1
    num_neurons = [model.parameters[0].in_features] + [layer.out_features for layer in model.parameters]

    # add neurons to the graph
    for layer in range(num_layers):
        if layer == 0 or layer == num_layers - 1:
            positions = _centered_layout(num_neurons[layer])
            color = 'green'
        else:
            positions = _justified_layout(num_neurons[layer])
            color = 'lightblue'

        for neuron in range(num_neurons[layer]):
            graph.add_node((layer, neuron), pos=(layer, positions[neuron]), color=color)

    # add weights & biases to the graph
    for layer in range(num_layers-1):
        for neuron1 in range(num_neurons[layer]):
            for neuron2 in range(num_neurons[layer + 1]):
                # weights
                graph.add_edge(
                    (layer, neuron1),
                    (layer + 1, neuron2),
                    weight=round(model.parameters[layer].weights[neuron1, neuron2], 2)
                )
                # biases
                graph.nodes[(layer + 1, neuron2)]['bias'] = round(model.parameters[layer].biases[neuron2], 2)

    # get graph attributes
    pos = nx.get_node_attributes(graph, 'pos')
    weights = [graph[u][v]['weight'] for u, v in graph.edges]
    # normalize
    max_weight = max(np.abs(weights))
    weights = [abs(w / max_weight) for w in weights]

    colors = [graph.nodes[n]['color'] for n in graph.nodes()]
    labels = nx.get_node_attributes(graph, 'bias')

    nx.draw(
        graph,
        pos,
        labels=labels,
        with_labels=True,
        node_color=colors,
        edge_color='grey',
        width=weights,
        node_size=800,
        font_size=7,
    )

    # plot weights on edges
    edge_labels = nx.get_edge_attributes(graph, 'weight')
    nx.draw_networkx_edge_labels(
        graph,
        pos,
        edge_labels=edge_labels,
        font_size=7,
    )

    plt.title(title)
    plt.show()
