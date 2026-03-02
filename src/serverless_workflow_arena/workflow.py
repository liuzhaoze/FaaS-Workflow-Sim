"""workflow.py

serverless工作流建模
"""

from typing import NamedTuple

import rustworkx as rx
from rustworkx import WeightedEdgeList

from .function import Function


class SubmittableFunc(NamedTuple):
    """可提交的函数

    Attributes:
        fn_id (int): 函数ID
        submission_time (float): 提交时间
    """

    fn_id: int
    submission_time: float


class Workflow:
    """serverless工作流模型

    Args:
        wf_id (int): 工作流ID
        arrival_time (float): 工作流到达时间
        computations (tuple[int, ...]): 各函数所需的 CPU 计算操作数量
        memory_reqs (tuple[int, ...]): 各函数所需的内存大小 (MB)
        parallelisms (tuple[int, ...]): 各函数的并行度
        edges (tuple[tuple[int, int, int], ...]): 函数间的数据传输依赖关系，格式为 (源函数ID, 目标函数ID, 传输数据大小 bytes)
    """

    __slots__ = (
        "wf_id",
        "arrival_time",
        "_functions",
        "_dag",
        "source_functions",
        "pending_functions",
        "submitted_functions",
        "running_functions",
        "finished_functions",
    )

    def __init__(
        self,
        wf_id: int,
        arrival_time: float,
        computations: tuple[int, ...],
        memory_reqs: tuple[int, ...],
        parallelisms: tuple[int, ...],
        edges: tuple[tuple[int, int, int], ...],
    ):
        if arrival_time < 0:
            raise ValueError("Arrival time must be non-negative")
        if not (len(computations) == len(memory_reqs) == len(parallelisms)):
            raise ValueError("Length of computations, memory_reqs, and parallelisms must be equal")
        if not all(len(edge) == 3 for edge in edges):
            raise ValueError(
                "Each edge must be a tuple of (source_function_id, target_function_id, data_transfer_size)"
            )

        self.wf_id: int = wf_id
        self.arrival_time: float = arrival_time

        self._functions: list[Function] = [
            Function(wf_id, fn_id, comp, mem, par)
            for fn_id, (comp, mem, par) in enumerate(zip(computations, memory_reqs, parallelisms))
        ]

        self._dag: rx.PyDiGraph = rx.PyDiGraph()
        n_nodes = len(self._functions)
        self._dag.add_nodes_from(range(n_nodes))
        self._dag.add_edges_from(edges)

        self.source_functions = set(rx.topological_generations(self._dag)[0])

        # 处于不同状态的函数的集合
        self.pending_functions: set[int] = set()
        self.submitted_functions: set[int] = set()
        self.running_functions: set[int] = set()
        self.finished_functions: set[int] = set()

    def __len__(self) -> int:
        return len(self._functions)

    def __getitem__(self, fn_id: int) -> Function:
        return self._functions[fn_id]

    def __iter__(self):
        return iter(self._functions)

    def reset(self):
        """重置工作流中所有函数的状态，然后初始化各状态集合"""
        for fn in self._functions:
            fn.reset()

        self.pending_functions = set(self._dag.node_indices())
        self.submitted_functions.clear()
        self.running_functions.clear()
        self.finished_functions.clear()

    def submit_function(self, fn_id: int, time: float):
        # 更新函数状态
        self._functions[fn_id].submit(time)

        # 更新状态集合
        self.pending_functions.remove(fn_id)
        self.submitted_functions.add(fn_id)

    def run_function(self, fn_id: int, time: float):
        # 更新函数状态
        self._functions[fn_id].run(time)

        # 更新状态集合
        self.submitted_functions.remove(fn_id)
        self.running_functions.add(fn_id)

    def finish_function(self, fn_id: int, time: float):
        # 更新函数状态
        self._functions[fn_id].finish(time)

        # 更新状态集合
        self.running_functions.remove(fn_id)
        self.finished_functions.add(fn_id)

    def get_submittable_functions(self, fn_id: int) -> list[SubmittableFunc]:
        """从指定函数的后继函数中获取可提交的函数列表

        该方法只在 `.workload.Workload.finish_function` 中调用。
        因为只有在一个函数完成后，才有可能使其后继函数变为可提交状态。

        Args:
            fn_id (int): 由运行状态变为完成状态的函数ID
        """

        result: list[SubmittableFunc] = []
        successors = set(self._dag.successor_indices(fn_id))

        # 如果一个函数刚刚完成，那么它的后继函数一定是 PENDING 状态
        for succ in successors:
            # 获取 fn_id 后继函数 succ 的所有前驱函数
            predecessors = set(self._dag.predecessor_indices(succ))
            # 检查所有前驱函数是否都已完成
            if not predecessors.issubset(self.finished_functions):
                continue
            # 计算 succ 的提交时间
            submission_time = max(self._functions[pred].completion_time for pred in predecessors)  # type: ignore

            result.append(SubmittableFunc(succ, submission_time))

        return result

    def get_predecessor_functions(self, fn_id: int) -> WeightedEdgeList:
        """获取指定函数的所有前驱函数以及两者之间的数据传输大小

        Args:
            fn_id (int): 指定函数ID

        Returns:
            WeightedEdgeList: 元组列表，格式为 (源函数ID, 目标函数ID, 传输数据大小)
        """

        return self._dag.in_edges(fn_id)

    @property
    def completed(self) -> bool:
        """工作流是否完成"""
        return len(self.finished_functions) == len(self._functions)
