"""raw_env.py

原始环境
"""

from .cluster import Cluster
from .config import ClusterConfig
from .container import Container
from .env_log import EnvLog, FunctionExecutionRecord
from .location import Location
from .workflow_template import WorkflowTemplate
from .workload import SubmittedFunc, Workload


class RawEnv:
    """Serverless 资源分配环境模型

    Args:
        arrival_times (list[float]): 各工作流到达时间
        workflow_templates (list[WorkflowTemplate]): 可用于生成工作流的工作流模板列表
        cluster_config (ClusterConfig): 集群配置
    """

    def __init__(
        self, arrival_times: list[float], workflow_templates: list[WorkflowTemplate], cluster_config: ClusterConfig
    ):
        self.workload = Workload(arrival_times, workflow_templates)
        self.cluster = Cluster(cluster_config)

        self.current_function: SubmittedFunc | None = None
        self._location_records: dict[tuple[int, int], Location] = {}

    def reset(self):
        """重置无服务器计算环境"""
        self.workload.reset()
        self.cluster.reset()

        self.current_function = self.workload.get()
        self._location_records.clear()

    def step(self, server_name: str, server_id: int, numa_node_id: int, allocated_memory: int) -> EnvLog:
        """无服务器计算状态转移逻辑

        Args:
            server_name (str): 云函数分配到的服务器类型名称
            server_id (int): 云函数分配到的服务器ID
            numa_node_id (int): 云函数分配到的NUMA节点ID
            allocated_memory (int): 云函数分配到的内存大小（MB）

        Returns:
            EnvLog: 本次状态转移的环境日志
        """

        if self.current_function is None:
            raise RuntimeError("No function to schedule. The environment may have terminated.")

        submission_time, wf_id, fn_id = self.current_function
        self._location_records[(wf_id, fn_id)] = Location(server_name, server_id, numa_node_id)
        self.workload[wf_id][fn_id].allocate_memory(allocated_memory)

        # 冷启动时间
        cold_start_time = self.cluster.start_server(server_name, server_id, submission_time)

        # 数据传输时间
        data_transfer_time = 0.0
        for pred_fn_id, _, data_size in self.workload[wf_id].get_predecessor_functions(fn_id):
            data_transfer_time += data_size / self.cluster.get_data_transfer_speed(
                src=self._location_records[(wf_id, pred_fn_id)],
                dst=self._location_records[(wf_id, fn_id)],
            )

        # 服务器冷启动及数据传输完成后，创建容器
        container = Container(
            wf_id=wf_id,
            fn_id=fn_id,
            memory_req=self.workload[wf_id][fn_id].memory_req,
            memory_alloc=allocated_memory,
            computation=self.workload[wf_id][fn_id].computation,
            parallelism=self.workload[wf_id][fn_id].parallelism,
            submission_time=submission_time,
            data_transfer_time=data_transfer_time,
        )

        # 租金
        rent = 0.0

        # 在集群中的指定 NUMA 节点上创建容器
        rent += self.cluster.on_container_creation(server_name, server_id, numa_node_id, container)

        # 获取下一个函数
        function_execution_records: list[FunctionExecutionRecord] = []

        while (not self.workload.completed) and (
            self.workload.peek() is None or self.cluster.earliest_finished_time <= self.workload.peek().submission_time  # type: ignore
        ):
            r, c = self.cluster.on_container_completion()
            rent += r
            self.workload.run_function(c.wf_id, c.fn_id, c.start_time)
            self.workload.finish_function(c.wf_id, c.fn_id, c.completion_time)
            function_execution_records.append(
                FunctionExecutionRecord(c.wf_id, c.fn_id, c.start_time, c.completion_time)
            )

        self.current_function = self.workload.get()

        # 返回本次步骤的环境日志
        return EnvLog(
            wf_id=wf_id,
            fn_id=fn_id,
            required_memory=self.workload[wf_id][fn_id].memory_req,
            allocated_memory=allocated_memory,
            server_name=server_name,
            server_id=server_id,
            numa_node_id=numa_node_id,
            submission_time=submission_time,
            creation_time=container.creation_time,
            cold_start_time=cold_start_time,
            data_transfer_time=data_transfer_time,
            finished_function_records=function_execution_records,
            rent=rent,
            terminated=self.workload.completed,
        )
