"""workload.py

工作负载管理模块
"""

import random
from queue import PriorityQueue
from typing import NamedTuple

from .workflow import Workflow
from .workflow_template import WorkflowTemplate


class SubmittedFunc(NamedTuple):
    """已提交的函数

    Attributes:
        submission_time (float): 提交时间
        wf_id (int): 工作流ID
        fn_id (int): 函数ID
    """

    submission_time: float
    wf_id: int
    fn_id: int


class Workload:
    """工作负载模型

    Args:
        arrival_times (list[float]): 各工作流到达时间
        workflow_templates (list[WorkflowTemplate]): 可用于生成工作流的工作流模板列表
    """

    __slots__ = (
        "_workflows",
        "_submit_queue",
    )

    def __init__(self, arrival_times: list[float], workflow_templates: list[WorkflowTemplate]):
        if not arrival_times:
            raise ValueError("Arrival times list cannot be empty")
        if not workflow_templates:
            raise ValueError("Workflow templates list cannot be empty")
        if not all(at >= 0 for at in arrival_times):
            raise ValueError("All arrival times must be non-negative")

        # 随机确定每个工作流使用的模板
        workflow_template_indices = [random.randint(0, len(workflow_templates) - 1) for _ in arrival_times]

        self._workflows: list[Workflow] = [
            Workflow(
                wf_id=i,
                arrival_time=a,
                computations=workflow_templates[t].computations,
                memory_reqs=workflow_templates[t].memory_reqs,
                parallelisms=workflow_templates[t].parallelisms,
                edges=workflow_templates[t].edges,
            )
            for i, (a, t) in enumerate(zip(arrival_times, workflow_template_indices))
        ]

        self._submit_queue: PriorityQueue[SubmittedFunc] = PriorityQueue()

    def __len__(self) -> int:
        return len(self._workflows)

    def __getitem__(self, wf_id: int) -> Workflow:
        return self._workflows[wf_id]

    def __iter__(self):
        return iter(self._workflows)

    def reset(self):
        """重置工作负载中所有工作流的状态，并初始化提交队列"""
        for wf in self._workflows:
            wf.reset()

        # 清空提交队列
        self._submit_queue.queue.clear()

        # 提交所有工作流的源点函数
        for wf in self._workflows:
            source_fns = wf.source_functions
            for fn_id in source_fns:
                submission_time = wf.arrival_time
                wf.submit_function(fn_id, submission_time)
                self._submit_queue.put(SubmittedFunc(submission_time, wf.wf_id, fn_id))

    def peek(self) -> SubmittedFunc | None:
        """查看下一个函数，但不从队列中移除"""
        with self._submit_queue.mutex:
            if self._submit_queue.queue:
                return self._submit_queue.queue[0]
            else:
                return None

    def get(self) -> SubmittedFunc | None:
        """获取下一个函数，并从队列中移除"""
        if not self._submit_queue.empty():
            return self._submit_queue.get()
        else:
            return None

    def run_function(self, wf_id: int, fn_id: int, time: float):
        """将指定工作流中的函数标记为运行状态"""
        self._workflows[wf_id].run_function(fn_id, time)

    def finish_function(self, wf_id: int, fn_id: int, time: float):
        """将指定工作流中的函数标记为完成状态"""
        self._workflows[wf_id].finish_function(fn_id, time)

        # 已完成的函数的后继函数有提交的可能
        submittable_fns = self._workflows[wf_id].get_submittable_functions(fn_id)

        # 提交可提交的函数
        for f in submittable_fns:
            self._workflows[wf_id].submit_function(f.fn_id, f.submission_time)
            self._submit_queue.put(SubmittedFunc(f.submission_time, wf_id, f.fn_id))

    @property
    def completed(self) -> bool:
        """工作负载是否完成"""
        return all(wf.completed for wf in self._workflows)
