import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

import time

import numpy as np
from utils import get_standard_cp_length, get_standard_data_transfer_time, log_squash

from serverless_workflow_arena import ClusterConfig, RawEnv, WorkflowTemplate
from serverless_workflow_arena.location import Location

# 数据和配置文件路径
dax_files = tuple((PROJECT_ROOT / "tests" / "data").glob("*.dax"))
config_file = PROJECT_ROOT / "tests" / "data" / "cluster_config.yaml"

# 加载集群配置
cluster_config = ClusterConfig.from_yaml(str(config_file))

# 生成到达时间和工作流模板
arrival_times = [float(i) for i in range(5)]
workflow_templates = [WorkflowTemplate(str(f), cluster_config.single_core_speed) for f in dax_files]

# 内存大小选项
memory_options = sorted([128, 256, 512, 1024])

# NUMA 节点选项
numa_options: list[tuple[str, int, int]] = []
for server_config in cluster_config.servers:
    for server_id in range(server_config.count):
        for numa_node_id in range(server_config.numa_nodes.count):
            numa_options.append((server_config.name, server_id, numa_node_id))

env = RawEnv(arrival_times, workflow_templates, cluster_config)
env.reset()

# 归一化所需变量
MAX_CPU = max(srv.numa_nodes.cpu for srv in cluster_config.servers)
MAX_MEM_OPT = max(memory_options)
MAX_QUEUED_FUNCTIONS = 1  # 在工作负载执行过程中，提交队列中函数的最大数量
MAX_ACTIVE_WORKFLOWS = 1  # 在工作负载执行过程中，已到达但尚未完成的工作流的最大数量
HOUR_IN_SECONDS = 3600
CFG_DICT = {srv.name: (srv.numa_nodes.cpu, srv.numa_nodes.memory) for srv in cluster_config.servers}

# 预先计算状态/观测空间需要的变量
# 工作流的到达时间
arrival_times = np.array([wf.arrival_time for wf in env.workload])
# 函数的标准执行时间
standard_execution_times = np.array(
    [[fn.computation / (fn.parallelism * cluster_config.single_core_speed) for fn in wf] for wf in env.workload]
)
norm_standard_execution_times = standard_execution_times / standard_execution_times.max()
# 函数的并行度
parallelisms = np.array([[fn.parallelism for fn in wf] for wf in env.workload])
norm_parallelisms = parallelisms / MAX_CPU
# 函数的内存需求
memory_reqs = np.array([[fn.memory_req for fn in wf] for wf in env.workload])
norm_memory_reqs = memory_reqs / MAX_MEM_OPT
# 满足函数内存需求的最低内存选项索引
lowest_feasible_memory_tiers = np.searchsorted(memory_options, memory_reqs, side="left")
norm_lowest_feasible_memory_tiers = (lowest_feasible_memory_tiers / (len(memory_options) - 1)).clip(0, 1)
# 从当前函数节点出发的标准关键路径长度
standard_critical_path_lengths = np.vstack(
    [
        get_standard_cp_length(standard_execution_times[i], wf.dag, cluster_config.network_bandwidth)
        for i, wf in enumerate(env.workload)
    ]
)
# 工作流的标准完工时间
standard_makespans = standard_critical_path_lengths.max(axis=1, keepdims=True)
norm_standard_critical_path_lengths = standard_critical_path_lengths / standard_makespans
# 函数的标准数据传输时间（无数据传输时为 1）
standard_data_transfer_times = np.vstack(
    [get_standard_data_transfer_time(wf.dag, cluster_config.network_bandwidth) for wf in env.workload]
)


step_count = 0

tic = time.time()

while True:
    if env.current_function is None:
        raise RuntimeError("No function to schedule. The environment may have terminated.")

    submission_time, wf_id, fn_id = env.current_function

    # 工作流和函数相关特征
    workflow_elapsed_time = submission_time - arrival_times[wf_id]
    print(
        "当前函数所属工作流的归一化已执行时间：",
        workflow_elapsed_time / (workflow_elapsed_time + standard_makespans[wf_id, 0]),
    )
    print("函数的归一化标准执行时间：", norm_standard_execution_times[wf_id, fn_id])
    print("函数的归一化并行度：", norm_parallelisms[wf_id, fn_id])
    print("函数的归一化内存需求：", min(norm_memory_reqs[wf_id, fn_id], 1.0))
    print("函数的内存需求是否超过最大内存选项：", norm_memory_reqs[wf_id, fn_id] > 1.0)
    print("满足函数内存需求的归一化最低内存选项索引：", norm_lowest_feasible_memory_tiers[wf_id, fn_id])
    print("从当前函数节点出发的归一化标准关键路径长度：", norm_standard_critical_path_lengths[wf_id, fn_id])

    # 工作负载相关特征
    print("归一化提交队列长度：", env.submit_queue_length / MAX_QUEUED_FUNCTIONS)
    print("归一化已到达但尚未完成的工作流数量：", env.active_workflow_count / MAX_ACTIVE_WORKFLOWS)

    # 数据传输相关特征
    data_distribution = env.get_data_distribution(wf_id, fn_id)
    norm_data_transfer_times = {srv.name: np.zeros((srv.count, srv.numa_nodes.count)) for srv in cluster_config.servers}
    for name, ndarray in norm_data_transfer_times.items():
        for idx in np.ndindex(ndarray.shape):
            ndarray[idx] = sum(
                (size / env.cluster.get_data_transfer_speed(loc, Location(name, *idx)))
                for loc, size in data_distribution
            )
        ndarray /= standard_data_transfer_times[wf_id, fn_id]
    print("将当前函数分配到每个NUMA节点的预计归一化数据传输时间：", norm_data_transfer_times)

    cluster_status = env.cluster.get_status_at(submission_time)

    # 服务器级特征
    server_features = {
        name: np.array(
            [
                [
                    server_status.remaining_lease_time / HOUR_IN_SECONDS,
                    log_squash(server_status.time_to_first_completion),
                    log_squash(server_status.time_to_last_completion),
                ]
                for server_status in server_statuses
            ]
        )
        for name, server_statuses in cluster_status.items()
    }
    print("服务器级特征（剩余租期，下一个函数完成需要的时间，所有函数完成需要的时间）：", server_features)

    # NUMA 节点级特征
    numa_node_features = {
        name: np.array(
            [
                [nn_status.cpu, nn_status.memory, nn_status.load]
                for server_status in server_statuses
                for nn_status in server_status.numa_node_statuses
            ]
        )
        for name, server_statuses in cluster_status.items()
    }
    print("NUMA节点级特征（CPU利用率，内存利用率，负载水平）：", numa_node_features)

    """[多智能体] 全局状态空间需要包括以上所有特征"""

    """[多智能体] 内存分配智能体的观测空间包括以下特征"""
    # 函数的归一化内存需求
    # 函数的内存需求是否超过最大内存选项
    # 满足函数内存需求的归一化最低内存选项索引
    memory_utilization = {
        name: np.array(
            [nn_status.memory for server_status in server_statuses for nn_status in server_status.numa_node_statuses]
        )
        for name, server_statuses in cluster_status.items()
    }
    print(
        "各类型服务器的内存利用率统计信息：",
        {
            name: np.array([ndarray.min(), ndarray.max(), ndarray.mean()])
            for name, ndarray in memory_utilization.items()
        },
    )
    free_memory = {
        name: np.array(
            [
                nn_status.free_memory
                for server_status in server_statuses
                for nn_status in server_status.numa_node_statuses
            ]
        )
        for name, server_statuses in cluster_status.items()
    }
    print(
        "各类型的所有NUMA节点空闲内存满足各内存选项的比例：",
        {name: (ndarray[:, None] >= memory_options).mean(axis=0) for name, ndarray in free_memory.items()},
    )
    print(
        "集群所有NUMA节点空闲内存满足各内存选项的比例：",
        (np.concatenate(list(free_memory.values()))[:, None] >= memory_options).mean(axis=0),
    )

    """[多智能体] 内存分配智能体根据观测做出内存分配决策"""
    memory = memory_options[step_count % len(memory_options)]

    """[多智能体] 服务器类型智能体的观测空间包括以下特征"""
    # 当前函数所属工作流的归一化已执行时间
    # 函数的归一化标准执行时间
    # 从当前函数节点出发的归一化标准关键路径长度
    # 归一化已到达但尚未完成的工作流数量
    remaining_lease_times = {
        name: np.array([server_status.remaining_lease_time for server_status in server_statuses])
        for name, server_statuses in cluster_status.items()
    }
    print(
        "各类型已经租用的服务器数量占比：",
        {name: (ndarray > 0).mean() for name, ndarray in remaining_lease_times.items()},
    )
    print(
        "各类型已租用服务器的归一化平均剩余租期：",
        {
            name: ndarray[ndarray > 0].mean() / HOUR_IN_SECONDS if (ndarray > 0).any() else 0.0
            for name, ndarray in remaining_lease_times.items()
        },
    )
    total_parallelism = {
        name: np.array(
            [
                nn_status.total_parallelism
                for server_status in server_statuses
                for nn_status in server_status.numa_node_statuses
            ]
        )
        for name, server_statuses in cluster_status.items()
    }
    print(
        "各类型中不会出现CPU资源竞争的NUMA节点数量占比：",
        {
            name: ((CFG_DICT[name][0] - parallelisms[wf_id, fn_id] - ndarray) >= 0).mean()
            for name, ndarray in total_parallelism.items()
        },
    )
    print(
        "各类型中不会出现内存资源竞争的NUMA节点数量占比：",
        {name: ((ndarray - memory) >= 0).mean() for name, ndarray in free_memory.items()},
    )
    total_data_size = 0
    server_type_data_distribution = {srv.name: 0 for srv in cluster_config.servers}
    for loc, size in data_distribution:
        total_data_size += size
        server_type_data_distribution[loc.server_name] += size
    print(
        "数据在不同类型服务器上的分布：",
        (
            np.zeros(len(server_type_data_distribution))
            if total_data_size == 0
            else (np.array(list(server_type_data_distribution.values())) / total_data_size)
        ),
    )

    """[多智能体] 服务器类型智能体根据观测选择合适的服务器类型"""
    server_name, server_id, numa_node_id = numa_options[step_count % len(numa_options)]

    """[多智能体] NUMA节点智能体的观测空间包括以下特征"""
    # 当前函数所属工作流的归一化已执行时间
    # 函数的归一化标准执行时间
    # 从当前函数节点出发的归一化标准关键路径长度
    print(
        "CPU利用率：",
        np.array(
            [
                nn_status.cpu
                for server_status in cluster_status[server_name]
                for nn_status in server_status.numa_node_statuses
            ]
        ),
    )
    print(
        "各NUMA节点的CPU资源是否足够：",
        (CFG_DICT[server_name][0] - parallelisms[wf_id, fn_id] - total_parallelism[server_name])
        / CFG_DICT[server_name][0],
    )
    print("各NUMA节点的内存资源是否足够：", (free_memory[server_name] - memory) / CFG_DICT[server_name][1])
    print(
        "各NUMA节点（下一个函数完成需要的时间，所有函数完成需要的时间）：",
        np.array(
            [
                [log_squash(nn_status.time_to_first_completion), log_squash(nn_status.time_to_last_completion)]
                for server_status in cluster_status[server_name]
                for nn_status in server_status.numa_node_statuses
            ]
        ),
    )
    print("NUMA节点所在的服务器的剩余租期：", remaining_lease_times[server_name] / HOUR_IN_SECONDS)
    print("将当前函数分配到该类型的NUMA节点的预计归一化数据传输时间：", norm_data_transfer_times[server_name])

    """[单智能体] 状态空间包括以下特征"""
    # 工作流、函数、工作负载总共 9 个特征
    print("固定类型的NUMA节点特征（每个服务器只有一个NUMA节点）：", numa_node_features[server_name])
    # 各NUMA节点（下一个函数完成需要的时间，所有函数完成需要的时间）
    # 各NUMA节点的CPU资源是否足够
    print(
        "固定类型的NUMA节点空闲内存能满足的最大内存选项归一化索引：",
        (
            (np.searchsorted(memory_options, free_memory[server_name], side="right") - 1) / (len(memory_options) - 1)
        ).clip(0, 1),
    )

    env_log = env.step(server_name, server_id, numa_node_id, memory)

    print(
        "Step {}: Allocate({}, {}) -> ({}, {}, {}); Finished: [{}]".format(
            step_count,
            env_log.wf_id,
            env_log.fn_id,
            env_log.server_name,
            env_log.server_id,
            env_log.numa_node_id,
            ", ".join(f"({r.wf_id}, {r.fn_id})" for r in env_log.finished_function_records),
        )
    )

    step_count += 1
    if env_log.terminated:
        break

print(f"Environment finished in {time.time() - tic:.2f} seconds")
