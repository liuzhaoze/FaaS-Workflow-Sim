import numpy as np
import rustworkx as rx


def get_standard_cp_length(
    standard_execution_times: np.ndarray, dag: rx.PyDiGraph, data_transfer_speed: int
) -> np.ndarray:
    """计算一个工作流中，从每个函数节点出发的标准关键路径长度

    Args:
        standard_execution_times (np.ndarray): 工作流中所有函数的标准执行时间
        dag (rx.PyDiGraph): 工作流的 DAG 结构，边的权重为需要传输的数据量 (单位：字节)
        data_transfer_speed (int): 数据传输速度 (单位：bytes/second)

    Returns:
        np.ndarray: 每个节点的标准关键路径长度
    """
    topological_order = list(rx.topological_sort(dag))
    result = np.zeros_like(standard_execution_times)

    # 按照拓扑排序的逆序遍历节点
    for node in reversed(topological_order):
        # 获取所有后继节点
        successors = dag.successor_indices(node)

        # 后继节点最大关键路径长度
        max_successor_length = 0.0

        if successors:
            max_successor_length = max(
                result[succ] + dag.get_edge_data(node, succ) / data_transfer_speed for succ in successors
            )

        result[node] = standard_execution_times[node] + max_successor_length

    return result


def get_standard_data_transfer_time(dag: rx.PyDiGraph, data_transfer_speed: int) -> np.ndarray:
    """计算工作流中每个函数的标准数据传输时间（无数据传输时为 1）

    Args:
        dag (rx.PyDiGraph): 工作流的 DAG 结构，边的权重为需要传输的数据量 (单位：字节)
        data_transfer_speed (int): 数据传输速度 (单位：bytes/second)

    Returns:
        np.ndarray: 每个函数的标准数据传输时间
    """
    result = np.ones(dag.num_nodes())

    for node in dag.node_indices():
        total_data_size = sum(dag.get_edge_data(pred, node) for pred in dag.predecessor_indices(node))
        if total_data_size <= 0:
            continue

        result[node] = total_data_size / data_transfer_speed

    return result
