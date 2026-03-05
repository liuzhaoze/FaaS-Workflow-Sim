import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

import time

from serverless_workflow_arena import ClusterConfig, RawEnv, WorkflowTemplate

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
numa_options: list[tuple[str, int, int]] = [
    (srv.name, srv_id, nn_id)
    for srv in cluster_config.servers
    for srv_id in range(srv.count)
    for nn_id in range(srv.numa_nodes.count)
]

env = RawEnv(arrival_times, workflow_templates, cluster_config)
env.reset()

step_count = 0

tic = time.time()

while True:
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
