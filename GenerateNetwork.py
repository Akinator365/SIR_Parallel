import os
import json
import networkx as nx
import numpy as np

def Generate_Graph(g_type, network_para, scope):
    num_nodes = network_para['n']
    num_min = num_nodes - scope
    num_max = num_nodes + scope
    num_nodes = np.random.randint(num_max - num_min + 1) + num_min

    if g_type == 'erdos_renyi':
        g = nx.erdos_renyi_graph(n=num_nodes, p=0.02)
    elif g_type == 'small-world':
        g = nx.connected_watts_strogatz_graph(n=num_nodes, k=4, p=0.1)
    elif g_type == 'barabasi_albert':
        g = nx.barabasi_albert_graph(n=num_nodes, m=network_para['m'])
    elif g_type == 'powerlaw':
        g = nx.powerlaw_cluster_graph(n=num_nodes, m=3, p=0.05)

    g.remove_nodes_from(list(nx.isolates(g)))
    g.remove_edges_from(nx.selfloop_edges(g))
    num_nodes = len(g.nodes)

    return g, num_nodes

def GenerateNetwork(train_dataset_path, id, network, network_para, scope):

    graph_type = network_para['type']
    graph_name = network + f'_{id}.txt'
    print(f'Generating No.{id} training {graph_type} graphs')

    data_path = os.path.join(train_dataset_path, graph_type+'_graph', network)
    os.makedirs(data_path, exist_ok=True)

    # 查看文件是否存在，如果存在则跳过
    if os.path.exists(os.path.join(data_path, graph_name)):
        print(f"File {graph_name} already exists, skipping...")
        return

    if graph_type == 'ER':
        g_type = 'erdos_renyi'
    elif graph_type == 'WS':
        g_type = 'small-world'
    elif graph_type == 'BA':
        g_type = 'barabasi_albert'
    elif graph_type == 'PLC':
        g_type = 'powerlaw'

    # Generate Graph
    g, num_nodes = Generate_Graph(g_type, network_para, scope)

    # 保存图为txt文件
    # 将边写入文件
    with open(os.path.join(data_path,graph_name), 'w') as f:
        for edge in g.edges():
            f.write(f"{edge[0]} {edge[1]}\n")

    print(f"Edges saved to {data_path}\\{graph_name}")

if __name__ == '__main__':
    DATASET_PATH = os.path.join(os.getcwd(), 'data', 'networks')

    # 从文件中读取参数
    with open("Network_Parameters.json", "r") as f:
        network_params = json.load(f)
    # 图的节点数量浮动范围
    scope = 100
    for network in network_params:
        num_graph = network_params[network]['num']
        for id in range(num_graph):
            GenerateNetwork(DATASET_PATH, id, network, network_params[network], scope)
