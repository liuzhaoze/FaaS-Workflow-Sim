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

    # 使用 Round-Robin 为函数分配资源
    server_name, server_id, numa_node_id = numa_options[step_count % len(numa_options)]
    memory = memory_options[step_count % len(memory_options)]

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
