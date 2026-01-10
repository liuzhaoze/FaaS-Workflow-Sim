"""function.py

云函数实例建模
"""

from enum import IntEnum
from typing import Optional


class FuncStat(IntEnum):
    """函数状态

    - PENDING: 初始状态
    - SUBMITTED: 函数已经提交到调度器队列中
    - RUNNING: 函数正在执行
    - FINISHED: 函数执行完成
    """

    PENDING = 0
    SUBMITTED = 1
    RUNNING = 2
    FINISHED = 3


class Function:
    """云函数实例模型

    Args:
        wf_id (int): 云函数所属的工作流ID
        fn_id (int): 云函数ID
        computation (int): 执行该函数所需的 CPU 计算操作数量
        memory_req (int): 执行该函数所需的内存大小 (MB)
        parallelism (int): 函数的并行度
    """

    __slots__ = (
        "wf_id",
        "fn_id",
        "computation",
        "memory_req",
        "parallelism",
        "state",
        "memory_alloc",
        "submission_time",
        "start_time",
        "completion_time",
    )

    ValidStateTransitions: dict[FuncStat, set[FuncStat]] = {
        FuncStat.PENDING: {FuncStat.SUBMITTED},
        FuncStat.SUBMITTED: {FuncStat.RUNNING},
        FuncStat.RUNNING: {FuncStat.FINISHED},
        FuncStat.FINISHED: set(),
    }

    def __init__(self, wf_id: int, fn_id: int, computation: int, memory_req: int, parallelism: int):

        self.wf_id: int = wf_id
        self.fn_id: int = fn_id
        self.computation: int = computation
        self.memory_req: int = memory_req
        self.parallelism: int = parallelism

        self.state: FuncStat = FuncStat.PENDING  # 云函数实例的状态
        self.memory_alloc: Optional[int] = None  # 分配到该函数实例的内存大小 (MB)
        self.submission_time: Optional[float] = None
        self.start_time: Optional[float] = None
        self.completion_time: Optional[float] = None

    def reset(self):
        """重置函数状态"""
        self.state = FuncStat.PENDING
        self.memory_alloc = None
        self.submission_time = None
        self.start_time = None
        self.completion_time = None

    def _validate_state_transition(self, new_state: FuncStat):
        """验证函数的状态转换是否合法"""
        if new_state not in self.ValidStateTransitions[self.state]:
            raise ValueError(f"Invalid status transition: {self.state.name} -> {new_state.name}")

    def submit(self, time: float):
        """提交函数到调度器队列 (函数为工作流源点时，提交时间为工作流的提交时间；否则为最后一个前驱函数的完成时间)"""
        if time < 0:
            raise ValueError(f"Submission time ({time}) must be non-negative")

        self._validate_state_transition(FuncStat.SUBMITTED)
        self.submission_time = time
        self.state = FuncStat.SUBMITTED

    def run(self, time: float):
        """开始执行函数 (函数开始执行之前需要等待服务器冷启动和前驱函数数据传输的时间)"""
        if self.submission_time is None:
            raise ValueError("Function must be submitted before running")
        if time < self.submission_time:
            raise ValueError(f"Start time ({time}) cannot be earlier than submission time ({self.submission_time})")

        self._validate_state_transition(FuncStat.RUNNING)
        self.start_time = time
        self.state = FuncStat.RUNNING

    def finish(self, time: float):
        """函数执行结束"""
        if self.start_time is None:
            raise ValueError("Function must be running before finishing")
        if time < self.start_time:
            raise ValueError(f"Finish time ({time}) cannot be earlier than start time ({self.start_time})")

        self._validate_state_transition(FuncStat.FINISHED)
        self.completion_time = time
        self.state = FuncStat.FINISHED

    def allocate_memory(self, memory_size: int):
        """为函数分配内存资源"""
        if memory_size <= 0:
            raise ValueError("Memory size must be positive")
        self.memory_alloc = memory_size

    def standard_execution_time(self, single_core_speed: int) -> float:
        """云函数的标准执行时间

        当云函数在满足其并行度要求的计算资源上运行时，其执行时间为标准执行时间。
        当然，不包括冷启动和数据传输时间。

        Args:
            single_core_speed (int): 单核计算速度 (单个 CPU 核心每秒可执行的计算操作数量)

        Returns:
            float: 标准执行时间 (秒)
        """
        if single_core_speed <= 0:
            raise ValueError("Single core speed must be positive")

        return self.computation / (self.parallelism * single_core_speed)
