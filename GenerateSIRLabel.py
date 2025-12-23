import random
import networkx as nx
import numpy as np
import os
import time
import json
import warnings
from multiprocessing import Pool  # 导入并行计算池
from tqdm import tqdm
from functools import partial


# --- 计时器 ---
def start_timer():
    return time.time()

def stop_timer(start_time):
    return time.time() - start_time

# --- 传播阈值计算 ---
def calculate_beta_c(G):
    """
    根据度分布的均场近似 (DBMF) 理论计算流行阈值 beta_c。
    beta_c = <k> / (<k^2> - <k>)

    :param G: networkx 图对象
    :return: (beta_c, <k>, <k^2>) 元组，如果分母为0则返回 (None, <k>, <k^2>)
    """

    num_nodes = G.number_of_nodes()
    if num_nodes == 0:
        print("图为空，无法计算。")
        return None, 0, 0

    # 1. 获取所有节点的度
    degrees = [d for n, d in G.degree()]

    # 2. 计算 <k> (平均度)
    k_avg = np.mean(degrees)

    # 3. 计算 <k^2> (度分布的二阶矩)
    k2_avg = np.mean([d ** 2 for d in degrees])

    # 4. 计算分母 (<k^2> - <k>)
    denominator = k2_avg - k_avg

    if denominator == 0:
        # 这种情况很少见，但可能发生在所有节点度都为1的图上
        warnings.warn(f"计算 beta_c 失败：分母 (<k^2> - <k>) 为 0。 (k_avg={k_avg}, k2_avg={k2_avg})")
        return None, k_avg, k2_avg

    # 5. 计算 beta_c
    beta_c = k_avg / denominator

    return beta_c, k_avg, k2_avg

def read_edges(file_path):
    adj_list = {}

    # 打开并读取文件
    with open(file_path, 'r') as f:
        for line in f:
            # 每行是一个边，格式为 "node1 node2"
            node1, node2 = map(int, line.strip().split())

            # 如果节点1不在字典中，初始化它
            if node1 not in adj_list:
                adj_list[node1] = set()
            # 如果节点2不在字典中，初始化它
            if node2 not in adj_list:
                adj_list[node2] = set()

            # 将每条边加到两个节点的邻接表中
            adj_list[node1].add(node2)
            adj_list[node2].add(node1)

    return adj_list


# --- 模拟核心步骤 (SIR 和 IC) ---

def sir_step(S, I, R, adj_list, beta, gamma):
    new_infected = set()
    new_recovered = set()

    # 传播感染
    for i in I:
        for neighbor in adj_list[i]:
            if neighbor in S and random.random() < beta:
                new_infected.add(neighbor)

    # 康复
    for i in I:
        if random.random() < gamma:
            new_recovered.add(i)

    # 更新状态
    S -= new_infected
    I |= new_infected
    I -= new_recovered
    R |= new_recovered

    return S, I, R

def ic_step(S, I, R, adj_list, beta):
    new_infected = set()

    # 传播感染
    for i in I:
        for neighbor in adj_list[i]:
            if neighbor in S and random.random() < beta:
                new_infected.add(neighbor)

    S -= new_infected  # 易感者变为感染者
    R |= I  # 将本轮的感染者加入已传播者集合
    I = new_infected  # 新的感染者集合由本轮传播的感染者构成

    return S, I, R

# --- 核心工作函数 (被 Single 和 Multiple 共享) ---

def simulate_node(node, adj_list, beta, gamma, simulations):
    """
    对单个起始节点进行多次模拟（IC 或 SIR）
    """
    count = 0
    n = len(adj_list)
    nodes = adj_list.keys()  # 获取所有实际节点

    for _ in range(simulations):
        # 初始化SIR状态
        S = set(nodes)  # 易感者
        I = {node}  # 选择种子节点作为感染者
        R = set()  # 康复者

        # 移动种子节点到已传播者集合中，因为它是第一个感染者
        S.remove(node)  # 将种子节点从易感者集合中移除

        # 进行模拟
        # 复制 I 集合用于迭代，避免在循环中修改
        current_I = I

        # --- 关键逻辑：根据 gamma 决定模型 ---
        if gamma == 1.0:
            while len(current_I) > 0:
                S, current_I, R = ic_step(S, current_I, R, adj_list, beta)
        else:
            while len(current_I) > 0:
                S, current_I, R = sir_step(S, current_I, R, adj_list, beta, gamma)

        # 循环结束时, I 集合为空，最终分数是 R 集合的大小
        inf = len(R) / n
        count += inf

    ave_inf = count / simulations
    return node, ave_inf  # 返回节点和它的影响力

# --- 单进程模拟函数 ---

def SIR_Single(graph_path, labels_path, network_params):
    """
    (单进程) 计算所有节点的影响力。
    功能与 SIR_Multiple 一致，用于调试。
    """
    print("---- start creating labels (SIR_Single) ----")

    adj_list = read_edges(graph_path)
    graph = nx.read_edgelist(graph_path, nodetype=int)
    node_list = sorted(list(graph.nodes()))  # 排序以保证处理顺序

    n = len(node_list)
    if n == 0:
        print(f"图 {graph_path} 为空，跳过。")
        return

    # 从参数文件加载参数
    beta = network_params['beta']
    gamma = network_params['gamma']
    simulations = network_params['simulations']

    # --- 计算并打印理论阈值 ---
    beta_c, k_avg, k2_avg = calculate_beta_c(graph)
    print(f"  [Info] 理论阈值 beta_c: {beta_c:.6f} (<k>={k_avg:.4f}, <k^2>={k2_avg:.4f})")
    print(f"  [Info] 正在使用文件中的 Beta: {beta:.6f} | Gamma: {gamma}")
    if gamma == 1.0:
        print("  [Info] Gamma=1.0, 运行 IC 模型。")
    else:
        print(f"  [Info] Gamma={gamma}, 运行 SIR 模型。")
    # ---

    influence = {}

    print(f"  [Info] 使用单进程开始串行计算...")

    # --- 核心区别：使用 For 循环代替 Pool ---
    for i, node in enumerate(node_list):
        if (i + 1) % 10 == 0 or i == 0:  # 打印进度
            print(f"    Processing node {node} ({i + 1}/{n})...")

        # 调用与 Multiple 完全相同的核心工作函数
        node_id, ave_inf = simulate_node(node, adj_list, beta, gamma, simulations)
        influence[node_id] = ave_inf
    # --- 循环结束 ---

    print("  [Info] ...单进程计算完成。")

    # 创建并打开文件，写入影响力数据
    txt_filename = labels_path + ".txt"
    with open(txt_filename, "w") as f:
        for node in influence:
            f.write(f"{node}\t{influence[node]:.8f}\n")

    print(f"Influence values saved to {txt_filename}")
    print("---- end creating labels ----")


def SIR_Multiple(graph_path, labels_path, network_params):
    """
    (多进程) 并行计算所有节点的影响力
    """
    print("---- start creating labels (SIR_Multiple) ----")

    adj_list = read_edges(graph_path)
    graph = nx.read_edgelist(graph_path)
    node_list = list(graph.nodes())
    # 使用 numpy 的 array 函数和 astype 方法将元素转换为整数
    int_node_list = np.array(node_list).astype(int)

    # 初始化网络参数
    beta = network_params['beta']
    gamma = network_params['gamma']
    simulations = network_params['simulations']

    # 计算传播阈值
    beta_c_result, k_avg, k2_avg = calculate_beta_c(graph)
    print(f"  [Info] 模拟次数 simulations: {simulations}")
    print(f"  [Info] 理论阈值 beta_c: {beta_c_result:.6f} (<k>={k_avg:.4f}, <k^2>={k2_avg:.4f})")
    print(f"  [Info] 正在使用文件中的 Beta: {beta:.6f} | Gamma: {gamma}")
    if gamma == 1.0:
        print("  [Info] Gamma=1.0, 运行 IC 模型。")
    else:
        print(f"  [Info] Gamma={gamma}, 运行 SIR 模型。")

    # 使用字典存储每个节点的影响力
    influence = {}

    # 建议：使用 Pool() 自动适配核心数，或 Pool(os.cpu_count() - 2)
    num_processes = 6  # (您设置的固定值)

    print(f"  [Info] 使用 {num_processes} 个进程开始并行计算...")

    # 使用 multiprocessing.Pool 并行计算每个节点的影响力
    with Pool(processes=num_processes) as pool:
        # 提交任务给进程池
        results = pool.starmap(
            simulate_node,
            [(node, adj_list, beta, gamma, simulations) for node in int_node_list]
        )
        # 处理并更新影响力字典
        for node, ave_inf in results:
            influence[node] = ave_inf

    # 打印每个节点的影响力
    #for node in influence:
    #    print(f"Node {node}: Influence {influence[node]:.8f}")

    # 创建并打开文件，写入影响力数据
    txt_filename = labels_path + ".txt"
    with open(txt_filename, "w") as f:
        for node in influence:
            f.write(f"{node}\t{influence[node]:.8f}\n")

    print(f"Influence values saved to {txt_filename}")
    print("---- end creating labels ----")

def Conver_to_Array(labels_path):
    # 读取label，转换为array
    with open(labels_path + '.txt', "r") as f:
        lines = f.readlines()
        labels = np.array([float(line.strip().split("\t")[1]) for line in lines])
    #print(labels)
    np.save(labels_path + '.npy', labels)

# --- 优化原有的 SIR_Multiple ---
def SIR_Multiple_Dynamic(graph_path, labels_path, network_params):
    """
    (多进程) ：保持节点并行，使用动态负载均衡 (imap_unordered)
    """
    print("---- start creating labels (SIR_Multiple_Dynamic) ----")

    adj_list = read_edges(graph_path)
    graph = nx.read_edgelist(graph_path)
    node_list = list(graph.nodes())
    int_node_list = np.array(node_list).astype(int)
    n = len(int_node_list)  # <-- 确保 n 被定义

    beta = network_params['beta']
    gamma = network_params['gamma']
    simulations = network_params['simulations']

    # ... (打印 beta_c 等信息的代码不变) ...
    beta_c_result, k_avg, k2_avg = calculate_beta_c(graph)
    print(f"  [Info] 模拟次数 simulations: {simulations}")
    print(f"  [Info] 理论传播阈值 beta_c: {beta_c_result:.6f} (<k>={k_avg:.4f}, <k^2>={k2_avg:.4f})")
    print(f"  [Info] 正在使用文件中的 Beta: {beta:.6f} | Gamma: {gamma}")
    if gamma == 1.0:
        print("  [Info] Gamma=1.0, 运行 IC 模型。")
    else:
        print(f"  [Info] Gamma={gamma}, 运行 SIR 模型。")

    influence = {}
    num_processes = 12  # (您设置的固定值)
    print(f"  [Info] 使用 {num_processes} 个进程开始并行计算 (动态负载均衡)...")

    with Pool(processes=num_processes) as pool:

        # 1. 使用 partial 创建一个新函数
        #    我们把所有 *不变* 的参数 (adj_list, beta, etc.) 固定住
        #    这样, sim_task 变成一个只需要 *一个* 参数 (node) 的新函数
        sim_task_partial = partial(simulate_node,
                                   adj_list=adj_list,
                                   beta=beta,
                                   gamma=gamma,
                                   simulations=simulations)

        # 2. 准备任务列表 (现在 *只* 需要变化的 'node' 列表)
        tasks_node_list = int_node_list
        n_tasks = len(tasks_node_list)

        # 3. pool.imap_unordered 现在可以正确工作了
        #    它会迭代 tasks_node_list, 每次取出一个 node
        #    然后调用 sim_task_partial(node)
        results_iterator = pool.imap_unordered(
            sim_task_partial,  # <--- 调用我们的新函数
            tasks_node_list  # <--- 迭代节点列表
        )

        # --- 修复结束 ---

        print(f"  [Info] 提交 {n_tasks} 个节点任务到池中...")

        # 4. 收集结果 (这里不需要改变)
        #    tqdm 会正确地显示进度
        for (node, ave_inf) in tqdm(results_iterator, total=n_tasks):
            influence[node] = ave_inf

    print("  [Info] ...动态计算完成。")

    # ... (后续保存文件的代码不变) ...
    txt_filename = labels_path + ".txt"
    with open(txt_filename, "w") as f:
        # 建议保存时也排序，确保 .txt 和 .npy 顺序一致
        for node in sorted(influence.keys()):
            f.write(f"{node}\t{influence[node]:.8f}\n")
    print(f"Influence values saved to {txt_filename}")
    print("---- end creating labels ----")

def GenerateSIRLabel(DATASET_PATH, LABELS_PATH, network_params):
    """
    主协调函数，用于生成标签
    """
    def GetLabel(graph_path, labels_path, name, params):
        txt_filepath = labels_path + ".txt"
        if os.path.exists(txt_filepath):
            print(f"File {txt_filepath} already exists, skipping...")
            return

        print(f"Processing {name}")
        start_time = start_timer()
        # --- 在这里选择要运行的函数 ---
        # 0. 运行多进程优化版本 (默认)
        SIR_Multiple_Dynamic(graph_path, labels_path, params)
        # 1. 运行多进程版本
        # SIR_Multiple(graph_path, labels_path, params)
        # 2. 运行单进程版本 (用于调试)
        # SIR_Single(graph_path, labels_path, params)
        # ---
        elapsed_time = stop_timer(start_time)

        Conver_to_Array(labels_path)
        print(f"Total time taken: {elapsed_time:.2f} seconds")

    for network in network_params:
        params = network_params[network]
        network_type = params['type']
        print(f'\nProcessing {network} graphs...')

        entries = []
        if network_type == 'realworld':
            # Realworld 类型路径构造
            graph_path = os.path.join(DATASET_PATH, f"{network}.txt")
            labels_path = os.path.join(LABELS_PATH, f"{network}_labels")
            entries.append((graph_path, labels_path, network))
        else:
            # 合成数据集路径构造
            base_dir = f"{network_type}_graph"
            for id in range(params['num']):
                network_name = f"{network}_{id}"
                graph_path = os.path.join(DATASET_PATH, base_dir, network, f"{network_name}.txt")
                labels_path = os.path.join(LABELS_PATH, base_dir, network, f"{network_name}_labels")
                entries.append((graph_path, labels_path, network_name))

        for graph_path, labels_path, name in entries:
            os.makedirs(os.path.dirname(labels_path), exist_ok=True)
            GetLabel(graph_path, labels_path, name, params)


if __name__ == '__main__':
    DATASET_PATH = os.path.join(os.getcwd(), 'data', 'networks')
    LABELS_PATH = os.path.join(os.getcwd(), 'data', 'labels')

    # 从文件中读取参数
    with open("Network_Parameters.json", "r") as f:
        network_params = json.load(f)

    GenerateSIRLabel(DATASET_PATH, LABELS_PATH, network_params)
