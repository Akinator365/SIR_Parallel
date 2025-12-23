import numpy as np
import networkx as nx
import time
import os
import json
import cupy as cp
import warnings
import cupyx.scipy.sparse as csp
from tqdm import tqdm
import gc

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

# --- GPU 模拟器类 ---
class GPU_SIR_Simulator:
    def __init__(self, graph_path, beta, gamma):
        self.beta = beta
        self.gamma = gamma

        # 1. 读取图并转换为 CSR 稀疏矩阵 (移至 GPU)
        # NetworkX 读取比较慢，大图建议直接读 EdgeList 为 numpy 数组再转 cupy
        nx_graph = nx.read_edgelist(graph_path, nodetype=int)
        self.nodes = sorted(list(nx_graph.nodes()))
        self.num_nodes = len(self.nodes)
        self.node_map = {node: i for i, node in enumerate(self.nodes)}  # 真实ID -> 矩阵索引
        self.rev_node_map = {i: node for i, node in enumerate(self.nodes)}  # 矩阵索引 -> 真实ID

        # 构建邻接矩阵
        adj = nx.adjacency_matrix(nx_graph, nodelist=self.nodes)
        self.adj_gpu = csp.csr_matrix(adj, dtype=cp.float32)  # 移至 GPU 显存

        # 预计算 (1 - beta) 的对数，用于快速计算概率
        # P = 1 - (1-beta)^k  =>  log(1-P) = k * log(1-beta)
        # 但直接用幂运算在 GPU 上也很快: 1 - (1-beta) ** k
        self.not_beta = 1.0 - self.beta

    def run_batch_simulation(self, seeds, num_sims_per_seed):
        """
        批量运行模拟
        :param seeds: 种子节点列表 (真实ID)
        :param num_sims_per_seed: 每个种子跑多少次模拟 (原代码中的 simulations)
        :return: 字典 {node_id: average_influence}
        """
        # 总共需要并行的列数 = 种子数 * 每个种子的模拟次数
        total_cols = len(seeds) * num_sims_per_seed

        # 显存保护：如果 total_cols 太大 (比如 > 100,000)，需要切分 batch
        # 假设我们一次最大处理 50000 列 (根据显存大小调整)
        gpu_total_bytes = cp.cuda.Device(0).mem_info[1]
        gpu_total_mb = gpu_total_bytes / (1024 * 1024)
        BATCH_LIMIT = get_optimal_batch_limit(self.num_nodes, gpu_total_mb)

        results = {}

        # 将 seeds 扩充，例如 seeds=[1, 2], sims=2 => expanded_seeds=[1, 1, 2, 2]
        expanded_seeds_indices = []
        for s in seeds:
            s_idx = self.node_map[s]
            expanded_seeds_indices.extend([s_idx] * num_sims_per_seed)

        expanded_seeds_indices = np.array(expanded_seeds_indices)

        # 分块处理
        for i in range(0, total_cols, BATCH_LIMIT):
            batch_seed_indices = expanded_seeds_indices[i: i + BATCH_LIMIT]
            current_batch_size = len(batch_seed_indices)

            # --- 初始化状态矩阵 ---
            # 0: Susceptible, 1: Infected, 2: Recovered
            # 形状: (Num_Nodes, Batch_Size)
            status = cp.zeros((self.num_nodes, current_batch_size), dtype=cp.int8)

            # 设置种子节点为 Infected (1)
            # 行索引是种子节点的索引，列索引是 0 到 batch_size
            row_idx = cp.array(batch_seed_indices)
            col_idx = cp.arange(current_batch_size)
            status[row_idx, col_idx] = 1

            # 记录活跃的列（还有节点处于 Infected 状态的列）
            active_cols = cp.ones(current_batch_size, dtype=bool)

            # --- 模拟循环 ---
            while cp.any(active_cols):
                # 1. 找到所有 Infected 节点的位置
                # 创建一个只包含 Infected 的掩码矩阵 (float 用于矩阵乘法)
                I_mask = (status == 1).astype(cp.float32)

                # 2. 计算每个节点连接到的 Infected 邻居数量 (k)
                # 矩阵乘法: (N, N) * (N, Batch) -> (N, Batch)
                infected_neighbor_counts = self.adj_gpu.dot(I_mask)

                # 3. 找到 Susceptible 节点
                S_mask = (status == 0)

                # 4. 计算感染概率 P = 1 - (1 - beta)^k
                # 只有 S 节点且 k > 0 的地方才需要计算
                # 使用掩码只计算必要部分 (为了代码简洁，这里全量计算后掩码过滤，GPU 并行很快)
                infection_prob = 1.0 - (self.not_beta ** infected_neighbor_counts)

                # 5. 生成随机数并判定感染
                rand_vals = cp.random.random(infection_prob.shape, dtype=cp.float32)
                new_infections = (rand_vals < infection_prob) & S_mask

                # 6. SIR/IC 更新逻辑
                # IC (Gamma=1): I -> R, New_I -> I
                # SIR (Gamma<1): I -> R (prob gamma), New_I -> I

                # 处理康复 (I -> R)
                current_infected = (status == 1)

                if self.gamma >= 1.0:
                    # IC模型: 所有当前 Infected 变为 Recovered
                    status[current_infected] = 2
                else:
                    # SIR模型: 概率康复
                    recovery_prob = cp.random.random(status.shape, dtype=cp.float32)
                    recovering = current_infected & (recovery_prob < self.gamma)
                    status[recovering] = 2
                    # 注意：未康复的 Infected 下一轮继续保持 Infected 并具备传染力

                # 应用新感染 (S -> I)
                status[new_infections] = 1

                # 更新活跃列状态 (如果某列没有 Infected 节点了，该次模拟结束)
                # axis=0 检查每一列
                has_infected = cp.any(status == 1, axis=0)
                active_cols = active_cols & has_infected

            # --- 统计结果 ---
            # 最终影响力 = Recovered + Infected (理论上结束时I为0，但在SIR中可能强行截断)
            # 也就是非 0 的节点数
            final_influence = cp.sum(status != 0, axis=0)  # (Batch_Size, )

            # 将结果转回 CPU
            final_influence_cpu = final_influence.get()

            # 聚合结果到 results 字典
            # batch_seed_indices 是原始 numpy 数组
            for idx, seed_matrix_idx in enumerate(batch_seed_indices):
                node_real_id = self.rev_node_map[seed_matrix_idx]
                inf_val = final_influence_cpu[idx] / self.num_nodes  # 归一化

                if node_real_id not in results:
                    results[node_real_id] = []
                results[node_real_id].append(inf_val)

        # --- 关键修改 1: 一个 Batch 跑完后，立刻释放 GPU 临时显存 ---
        # 虽然是局部变量，但手动释放池可以防止碎片化堆积
        del status, I_mask, infected_neighbor_counts, infection_prob, rand_vals
        cp.get_default_memory_pool().free_all_blocks()
        # 计算平均值
        final_output = {k: np.mean(v) for k, v in results.items()}
        return final_output


# --- 替换原有函数的 GPU 版本 ---

def SIR_GPU_Driver(graph_path, labels_path, network_params):
    print("---- start creating labels (GPU Accelerated) ----")

    start_time = time.time()

    graph = nx.read_edgelist(graph_path)
    beta_c_result, k_avg, k2_avg = calculate_beta_c(graph)

    # beta = network_params['beta']
    beta = beta_c_result * 3
    gamma = network_params['gamma']
    simulations = network_params['simulations']

    print(f"  [Info] Initialization CuPy/GPU resources...")
    print(f"  [Info] 理论阈值 beta_c: {beta_c_result:.6f} (<k>={k_avg:.4f}, <k^2>={k2_avg:.4f})")
    print(f"  [Info] Beta: {beta} | Gamma: {gamma} | Sims: {simulations}")

    # 初始化模拟器 (加载图到 GPU)
    simulator = GPU_SIR_Simulator(graph_path, beta, gamma)

    node_list = simulator.nodes  # 获取所有节点ID
    num_nodes = len(node_list)

    print(f"  [Info] Graph Loaded. Nodes: {num_nodes}. Starting simulation...")

    # 策略：为了充分利用 GPU，我们一次性把所有节点放入 batch 计算
    # 如果图很大 (比如 >1万节点) 且 simulations 很大，可能显存不够
    # 我们可以分批传入种子节点

    influence = {}

    # 定义每次处理多少个种子节点 (根据显存大小调整，可以设大一点)
    # 比如一次处理 100 个种子，每个种子跑 simulations 次 -> 矩阵宽 100 * sims
    SEED_BATCH_SIZE = 512

    for i in tqdm(range(0, num_nodes, SEED_BATCH_SIZE), desc="GPU Batches"):
        batch_nodes = node_list[i: i + SEED_BATCH_SIZE]

        # 运行模拟
        batch_results = simulator.run_batch_simulation(batch_nodes, simulations)
        influence.update(batch_results)

    print(f"  [Info] GPU Computation Finished.")

    # 保存文件
    txt_filename = labels_path + ".txt"
    with open(txt_filename, "w") as f:
        for node in sorted(influence.keys()):
            f.write(f"{node}\t{influence[node]:.8f}\n")

    print(f"Influence values saved to {txt_filename}")
    elapsed = time.time() - start_time
    print(f"Total time taken: {elapsed:.2f} seconds")
    # --- 关键修改 2: 彻底销毁模拟器对象，并强制垃圾回收 ---
    del simulator  # 删除 Python 对象引用
    gc.collect()  # 强制 Python 进行垃圾回收
    cp.get_default_memory_pool().free_all_blocks()  # 强制归还显存给操作系统
    print(f"  [Info] Memory Cleaned.")

    print("---- end creating labels ----")


def get_optimal_batch_limit(num_nodes, total_memory_mb, safety_margin=0.95):
    """
    根据显存大小和节点数，自动计算最佳 BATCH_LIMIT。

    :param num_nodes: 图的节点数量
    :param total_memory_mb: 显卡总显存 (例如 32768 对于 32GB)
    :param safety_margin: 安全系数，默认使用 95% 的显存，预留 5% 给系统
    :return: 建议的 BATCH_LIMIT (int)
    """
    # 1. 计算可用显存 (Bytes)
    # 减去约 2GB 的固定开销 (CUDA context, PyCharm, System, Graph Storage)
    # 32GB 显卡建议保留 1GB 给系统，剩下的用于计算
    fixed_overhead_mb = 1024
    available_mem_bytes = (total_memory_mb - fixed_overhead_mb) * 1024 * 1024 * safety_margin

    if available_mem_bytes <= 0:
        print("[Warning] 显存预估不足，使用默认最小 Batch")
        return 10000

    # 2. 核心系数：每个 (Node * Batch) 单元占用的字节数
    # 根据你的实测数据推算约为 32-36 Bytes。
    bytes_per_cell = 32.0

    # 3. 计算 Limit
    # Formula: Limit = Available_Mem / (Nodes * Bytes_Per_Cell)
    batch_limit = int(available_mem_bytes / (num_nodes * bytes_per_cell))

    # 4. 上下限截断
    # 设个上限防止太小图时数字过大导致溢出 int 范围或 Python 循环过慢
    max_limit = 5000000  # 最大允许 500万
    min_limit = 10000  # 最小允许 1万

    batch_limit = max(min_limit, min(batch_limit, max_limit))

    # 向上取整到 1000 的倍数好看点
    return (batch_limit // 1000) * 1000

# --- 集成到你的主流程 ---

def GenerateSIRLabel(DATASET_PATH, LABELS_PATH, network_params):
    # 在 GetLabel 内部调用 GPU 版本
    def GetLabel(graph_path, labels_path, name, params):
        txt_filepath = labels_path + ".txt"
        if os.path.exists(txt_filepath):
            print(f"File {txt_filepath} already exists, skipping...")
            return

        print(f"Processing {name}")

        # === 调用 GPU 版本 ===
        try:
            SIR_GPU_Driver(graph_path, labels_path, params)
        except Exception as e:
            print(f"GPU Error: {e}. Falling back to CPU...")
            # 如果 GPU 显存炸了或者没环境，可以回退到 CPU
            # SIR_Multiple_Dynamic(graph_path, labels_path, params)

        # 转换为 numpy 数组
        # 注意：需要单独处理转换，因为 SIR_GPU_Driver 已经写了 txt
        if os.path.exists(txt_filepath):
            # 读取label，转换为array (你可以直接用你原来的 Conver_to_Array)
            with open(txt_filepath, "r") as f:
                lines = f.readlines()
                # 确保排序正确
                labels = []
                for line in lines:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        labels.append(float(parts[1]))
                np.save(labels_path + '.npy', np.array(labels))

    # ... (后续的循环遍历逻辑不变) ...
    for network in network_params:
        # ... Copy your original loop code here ...
        params = network_params[network]
        network_type = params['type']
        # ... (Same loop logic as your original code) ...
        entries = []
        if network_type == 'realworld':
            graph_path = os.path.join(DATASET_PATH, f"{network}.txt")
            labels_path = os.path.join(LABELS_PATH, f"{network}_labels")
            entries.append((graph_path, labels_path, network))
        else:
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
