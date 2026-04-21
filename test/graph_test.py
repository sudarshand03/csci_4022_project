from graph_generator import generate_erdos_renyi

G = generate_erdos_renyi(100, 0.1, seed=42)
print(G.number_of_nodes(), G.number_of_edges())
