import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

import time

from faas_workflow_sim import ClusterConfig, RawEnv, WorkflowTemplate

# 数据和配置文件路径
dax_files = tuple((PROJECT_ROOT / "tests" / "data").glob("*.dax"))
config_file = PROJECT_ROOT / "tests" / "data" / "cluster_config.yaml"

# 加载集群配置
cluster_config = ClusterConfig.from_yaml(str(config_file))

# 生成到达时间和工作流模板
arrival_times = [float(i) for i in range(5)]
workflow_templates = [WorkflowTemplate(str(f), cluster_config.single_core_speed) for f in dax_files]

# 合法的资源分配策略
valid_strategies: list[tuple[str, int, int]] = []
for server_config in cluster_config.servers:
    for server_id in range(server_config.count):
        for numa_node_id in range(server_config.numa_nodes.count):
            valid_strategies.append((server_config.name, server_id, numa_node_id))

env = RawEnv(arrival_times, workflow_templates, cluster_config)
env.reset()

step_count = 0

tic = time.time()

while True:
    # 使用 Round-Robin 为函数分配资源
    server_name, server_id, numa_node_id = valid_strategies[step_count % len(valid_strategies)]
    allocated_memory = 256

    env_log = env.step(server_name, server_id, numa_node_id, allocated_memory)

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
