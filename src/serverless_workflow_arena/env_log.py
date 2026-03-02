"""log.py

环境日志记录定义
"""

from typing import NamedTuple


class FunctionExecutionRecord(NamedTuple):
    """记录函数容器执行的开始时间和完成时间

    Attributes:
        wf_id (int): 工作流ID
        fn_id (int): 函数ID
        start_time (float): 函数容器开始执行的时间
        completion_time (float): 函数容器执行完成的时间
    """

    wf_id: int
    fn_id: int
    start_time: float
    completion_time: float


class EnvLog(NamedTuple):
    """环境日志记录

    Attributes:
        wf_id (int): 工作流ID
        fn_id (int): 函数ID
        required_memory (int): 函数所需内存大小 (MB)
        allocated_memory (int): 函数分配到的内存大小 (MB)
        server_name (str): 函数容器执行的服务器类型名称
        server_id (int): 函数容器执行的服务器ID
        numa_node_id (int): 函数容器执行的NUMA节点ID
        submission_time (float): 函数提交的时间
        creation_time (float): 函数容器创建的时间
        cold_start_time (float): 目标服务器的冷启动时间
        data_transfer_time (float): 函数容器执行前的数据传输时间
        finished_function_records (list[FunctionExecutionRecord]): 执行完毕的函数的开始和完成时间记录列表
        rent (float): 函数容器在集群中执行所产生的租金
        terminated (bool): 环境是否结束
    """

    wf_id: int
    fn_id: int
    required_memory: int
    allocated_memory: int
    server_name: str
    server_id: int
    numa_node_id: int
    submission_time: float
    creation_time: float
    cold_start_time: float
    data_transfer_time: float
    finished_function_records: list[FunctionExecutionRecord]
    rent: float
    terminated: bool
