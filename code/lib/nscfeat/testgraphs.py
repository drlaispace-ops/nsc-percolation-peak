"""Пари графів, нерозрізненних 1-WL, для перевірок виразності (прогноз П4.B2)."""
import networkx as nx


def two_triangles():
    return nx.disjoint_union(nx.cycle_graph(3), nx.cycle_graph(3))


def decalin():
    """Два 6-цикли зі спільним ребром (0–5)."""
    G = nx.cycle_graph(6)
    nx.add_path(G, [5, 6, 7, 8, 9, 0])
    return G


def bicyclopentyl():
    """Два 5-цикли, з'єднані ребром 0–5."""
    G = nx.disjoint_union(nx.cycle_graph(5), nx.cycle_graph(5))
    G.add_edge(0, 5)
    return G


def shrikhande():
    """Граф Келі на Z4×Z4 з твірними ±(1,0), ±(0,1), ±(1,1); srg(16,6,2,2)."""
    G = nx.Graph()
    gens = [(1, 0), (3, 0), (0, 1), (0, 3), (1, 1), (3, 3)]
    for a in range(4):
        for b in range(4):
            for da, db in gens:
                G.add_edge(4 * a + b, 4 * ((a + da) % 4) + (b + db) % 4)
    return G


def rook4():
    """Граф тури 4×4 = K4 □ K4; srg(16,6,2,2)."""
    return nx.convert_node_labels_to_integers(nx.cartesian_product(nx.complete_graph(4), nx.complete_graph(4)))


PAIRS = {
    "C6 / 2×C3": (lambda: nx.cycle_graph(6), two_triangles),
    "декалін / біциклопентил": (decalin, bicyclopentyl),
    "Шрікханде / тура 4×4": (shrikhande, rook4),
}
