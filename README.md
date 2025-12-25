# SIR_Parallel

## 目录

- [工具简介](#工具简介)

- [文件说明](#文件说明)

- [环境依赖项](#环境依赖项)

- [使用步骤](#使用步骤)

    - [1. 生成网络](#1-生成网络)

    - [2. 运行SIR模拟](#2-运行sir模拟)

- [参数配置详解](#参数配置详解)

- [性能优化建议](#性能优化建议)

## 工具简介

本工具提供了网络生成和SIR/IC模型传播模拟的完整流程，支持CPU并行计算和GPU加速两种模式。可用于生成不同类型的网络（如BA无标度网络、ER随机网络等），并通过模拟病毒/信息传播过程计算每个节点的影响力。

## 文件说明

|文件名|功能描述|
|---|---|
|`GenerateNetwork.py`|生成各类网络（BA、ER、WS、PLC等）并保存为边列表文件|
|`GenerateSIRLabel.py`|CPU版本的SIR/IC模拟工具，支持多进程并行计算|
|`GenerateSIRLabel_GPU.py`|GPU加速版本的SIR/IC模拟工具，适合大规模计算|
|`Network_Parameters.json`|网络生成和模拟的参数配置文件|
## 环境依赖项

- 基础依赖：

    - Python 3.8+

    - networkx

    - numpy

    - tqdm

- CPU版本额外依赖：

    - multiprocessing（Python标准库）

- GPU版本额外依赖：

    - cupy（需配合CUDA环境）

    - cupyx

安装命令示例：

```Bash
pip install networkx numpy tqdm cupy-cuda12x  # 根据CUDA版本选择合适的cupy包
```

## 使用步骤

### 1. 生成网络

**功能**：根据`Network_Parameters.json`配置生成指定类型和数量的网络

**运行命令**：

```Bash
python GenerateNetwork.py
```

**工作流程**：

1. 程序读取`Network_Parameters.json`中的配置

2. 按参数生成对应类型的网络（BA、ER、WS或PLC）

3. 网络以边列表格式保存到`data/networks/{网络类型}_graph/`目录下

### 2. 运行SIR模拟

#### CPU版本（多进程并行）

**运行命令**：

```Bash
python GenerateSIRLabel.py
```

#### GPU版本（加速计算）

**运行命令**：

```Bash
python GenerateSIRLabel_GPU.py
```

**工作流程**：

1. 读取已生成的网络文件

2. 根据配置参数进行SIR/IC传播模拟

3. 计算每个节点的平均影响力

4. 结果保存为`.txt`格式到`data/labels/`目录下

## 参数配置详解

所有参数通过`Network_Parameters.json`文件配置，每个网络类型的配置项说明如下：

```JSON
{
    "网络名称": {
        "num": 8,           // 生成该类型网络的数量
        "type": "BA",       // 网络类型：BA/ER/WS/PLC/realworld
        "n": 500,           // 节点数量（生成网络时使用）
        "m": 3,             // BA网络参数：每次添加新节点时连接的边数
        "beta": 0.3,        // 感染概率
        "gamma": 1.0,       // 康复概率（gamma=1.0时为IC模型）
        "simulations": 10000 // 每个节点的模拟次数
    }
}
```

### 关键参数说明：

1. **网络类型（type）**：

    - `BA`：Barabási-Albert无标度网络

    - `ER`：Erdős-Rényi随机网络

    - `WS`：Watts-Strogatz小世界网络

    - `PLC`：Powerlaw集群网络

    - `realworld`：真实世界网络（需自行准备数据）

2. **传播参数**：

    - `beta`：感染概率（SIR模型中，易感节点被感染的概率）

    - `gamma`：康复概率（SIR模型中，感染节点康复的概率）

    - 当`gamma=1.0`时，自动切换为IC（独立级联）模型

3. **模拟参数**：

    - `simulations`：每个节点作为初始感染源的模拟次数，次数越多结果越稳定，但计算时间越长

## 性能优化建议

1. **GPU版本使用建议**：

    - 对于节点数>1000的网络，优先使用GPU版本

    - 显存不足时，可减小`SEED_BATCH_SIZE`参数（在`GenerateSIRLabel_GPU.py`中）

    - 大型网络建议将`simulations`分多次运行后取平均

2. **CPU版本使用建议**：

    - 可在`SIR_Multiple_Dynamic`函数中调整`num_processes`参数以匹配CPU核心数

    - 小规模网络（节点数<500）适合使用CPU多进程版本

3. **参数调整策略**：

    - 网络规模较大时，可适当降低`simulations`数值

    - 若只需定性结果，`simulations`可设为1000-5000；如需高精度结果，建议设为10000以上

4. **内存管理**：

    - GPU版本会自动管理显存，大型任务结束后会自动释放资源

    - 若遇到内存溢出，可减少每次处理的节点批次大小

可以灵活生成不同类型的网络并模拟传播过程，获取节点影响力数据用于后续分析。