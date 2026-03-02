"""container.py

容器建模
"""

from dataclasses import dataclass, field


@dataclass(slots=True)
class Container:
    """容器模型

    Args:
        wf_id (int): 容器中执行的函数所属的工作流ID
        fn_id (int): 容器中执行的函数在工作流中的函数ID
        memory_req (int): 容器中执行的函数所需的内存大小 (MB)
        memory_alloc (int): 实际分配给容器的内存大小 (MB)
        computation (int): 执行函数所需的 CPU 计算操作数量
        parallelism (int): 函数并行度
        submission_time (float): 该容器对应的函数的提交时间
        data_transfer_time (float): 传输函数执行所需数据的时间
    """

    wf_id: int
    fn_id: int
    memory_req: int
    memory_alloc: int
    computation: int
    parallelism: int
    submission_time: float
    data_transfer_time: float

    remaining_computation: int = field(init=False)
    creation_time: float = field(init=False)
    start_time: float = field(init=False)
    completion_time: float = field(init=False)

    def __post_init__(self):
        self.remaining_computation = self.computation

    def create(self, time: float):
        """创建容器

        Args:
            time (float): 创建容器的时间
        """

        if time < self.submission_time + self.data_transfer_time:
            raise ValueError(f"Creation time {time} cannot be earlier than submission time + data transfer time")

        self.creation_time = time

    def run(self, time: float):
        """运行容器

        Args:
            time (float): 运行容器的时间
        """
        if time < self.creation_time:
            raise ValueError(f"Start time {time} cannot be earlier than creation time {self.creation_time}")

        self.start_time = time

    def finish(self, time: float):
        """容器运行结束

        Args:
            time (float): 容器运行结束的时间
        """
        if time < self.start_time:
            raise ValueError(f"Finish time {time} cannot be earlier than start time {self.start_time}")

        self.completion_time = time
