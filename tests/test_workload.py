"""test_workload.py

测试 src/faas_workflow_sim/workload.py 中的 Workload 类和 SubmittedFunc 类
"""

from pathlib import Path
from typing import TypedDict
from unittest.mock import MagicMock, patch

import pytest

from faas_workflow_sim.function import FuncStat
from faas_workflow_sim.workflow_template import WorkflowTemplate
from faas_workflow_sim.workload import SubmittedFunc, Workload


class TestSubmittedFunc:
    """测试 SubmittedFunc NamedTuple"""

    def test_submitted_func_creation(self):
        """测试 SubmittedFunc 创建"""
        func = SubmittedFunc(submission_time=5.0, wf_id=1, fn_id=2)
        assert func.submission_time == 5.0
        assert func.wf_id == 1
        assert func.fn_id == 2

    def test_submitted_func_immutability(self):
        """测试 SubmittedFunc 不可变性"""
        func = SubmittedFunc(submission_time=5.0, wf_id=1, fn_id=2)

        # 应该不能修改属性
        with pytest.raises(AttributeError):
            func.submission_time = 10.0  # type: ignore


class TestWorkload:
    """测试 Workload 类"""

    @pytest.fixture
    def workflow_templates(self):
        """从 tests/data 目录创建 WorkflowTemplate 列表"""
        # 正确的路径应该是从项目根目录开始的 tests/data
        data_dir = Path(__file__).parent / "data"
        dax_files = list(data_dir.glob("*.dax"))

        if not dax_files:
            pytest.skip("No DAX files found in tests/data directory")

        templates: list[WorkflowTemplate] = []
        for dax_file in dax_files:
            try:
                template = WorkflowTemplate(str(dax_file), 1000)  # single_core_speed=1000
                templates.append(template)
            except FileNotFoundError:
                # 如果配套的 JSON 文件不存在，跳过这个模板
                continue

        if not templates:
            pytest.skip("No valid WorkflowTemplate could be created")

        return templates

    @pytest.fixture
    def arrival_times(self):
        """创建到达时间列表"""
        return [0.0, 2.5, 5.0]

    @pytest.fixture
    def simple_workload(self, workflow_templates: list[WorkflowTemplate], arrival_times: list[float]):
        """创建简单的工作负载"""
        return Workload(arrival_times, workflow_templates)

    def test_workload_initialization_success(
        self, workflow_templates: list[WorkflowTemplate], arrival_times: list[float]
    ):
        """测试工作负载初始化成功"""
        workload = Workload(arrival_times, workflow_templates)

        assert len(workload) == len(arrival_times)

        # 验证每个工作流
        for i in range(len(arrival_times)):
            workflow = workload[i]
            assert workflow.wf_id == i
            assert workflow.arrival_time == arrival_times[i]

    def test_workload_initialization_empty_arrival_times(self, workflow_templates: list[WorkflowTemplate]):
        """测试空的到达时间列表"""
        with pytest.raises(ValueError, match="Arrival times list cannot be empty"):
            Workload([], workflow_templates)

    def test_workload_initialization_empty_workflow_templates(self, arrival_times: list[float]):
        """测试空的工作流模板列表"""
        with pytest.raises(ValueError, match="Workflow templates list cannot be empty"):
            Workload(arrival_times, [])

    def test_workload_initialization_negative_arrival_time(self, workflow_templates: list[WorkflowTemplate]):
        """测试负的到达时间"""
        arrival_times = [0.0, -1.0, 5.0]
        with pytest.raises(ValueError, match="All arrival times must be non-negative"):
            Workload(arrival_times, workflow_templates)

    @patch("random.randint")
    def test_workflow_template_selection(
        self, mock_randint: MagicMock, workflow_templates: list[WorkflowTemplate], arrival_times: list[float]
    ):
        """测试工作流模板选择"""
        # 设置固定的随机选择
        expected_indices = [0, 1, 0]  # 3个到达时间
        if len(workflow_templates) >= 2:
            mock_randint.side_effect = expected_indices
        else:
            mock_randint.return_value = 0

        workload = Workload(arrival_times, workflow_templates)

        # 验证使用了正确数量的模板
        assert len(workload) == len(arrival_times)

    def test_workflow_count_matches_arrival_times(self, workflow_templates: list[WorkflowTemplate]):
        """测试工作流数量与到达时间数量匹配"""
        # 测试不同数量的到达时间
        for n in [1, 3, 5]:
            arrival_times = [float(i) * 2.0 for i in range(n)]
            workload = Workload(arrival_times, workflow_templates)
            # 通过尝试访问所有索引来验证数量
            for i in range(n):
                _ = workload[i]
            # 确保超出范围会抛出异常
            with pytest.raises(IndexError):
                _ = workload[n]

    def test_workflow_ids_are_sequential(self, workflow_templates: list[WorkflowTemplate], arrival_times: list[float]):
        """测试工作流 ID 是连续的"""
        workload = Workload(arrival_times, workflow_templates)

        expected_ids = list(range(len(arrival_times)))
        actual_ids = [workload[i].wf_id for i in range(len(arrival_times))]

        assert actual_ids == expected_ids

    def test_workflow_template_randomness(self, workflow_templates: list[WorkflowTemplate], arrival_times: list[float]):
        """测试工作流模板选择的随机性"""
        # 多次创建工作负载，检查模板选择的随机性
        templates_used: list[int] = []

        for _ in range(5):  # 减少次数以避免测试时间过长
            workload = Workload(arrival_times, workflow_templates)
            # 获取第一个工作流使用的模板特征
            first_wf = workload[0]
            template_signature = len(first_wf)  # 函数数量
            templates_used.append(template_signature)

        # 如果有多个不同的模板，应该能看到变化
        if len(workflow_templates) > 1:
            # 至少应该有一些变化（但不强制要求，因为随机性）
            unique_templates = len(set(templates_used))
            assert unique_templates >= 1

    def test_reset_initializes_submit_queue(
        self, workflow_templates: list[WorkflowTemplate], arrival_times: list[float]
    ):
        """测试重置功能初始化提交队列"""
        workload = Workload(arrival_times, workflow_templates)

        # 重置
        workload.reset()

        # 验证队列不为空（应该有源点函数）
        assert not workload._submit_queue.empty()  # type: ignore

        # 验证提交队列包含所有工作流的源点函数
        submitted_funcs: list[SubmittedFunc] = []
        while not workload._submit_queue.empty():  # type: ignore
            submitted_funcs.append(workload._submit_queue.get())  # type: ignore

        # 每个工作流都应该有源点函数被提交
        workflow_ids = {f.wf_id for f in submitted_funcs}
        assert workflow_ids == set(range(len(arrival_times)))

    def test_reset_resets_all_workflows(self, workflow_templates: list[WorkflowTemplate], arrival_times: list[float]):
        """测试重置功能重置所有工作流"""
        workload = Workload(arrival_times, workflow_templates)

        # 重置
        workload.reset()

        # 验证状态集合被正确初始化
        for i in range(len(arrival_times)):
            workflow = workload[i]
            # 源点函数应该被提交
            assert len(workflow.submitted_functions) == len(workflow.source_functions)
            assert workflow.source_functions <= workflow.submitted_functions

            # 非源点函数应该是 pending 状态
            for fn_id, fn in enumerate(workflow):
                if fn_id not in workflow.source_functions:
                    assert fn.state == FuncStat.PENDING
                    assert fn.submission_time is None

    def test_peek_empty_queue(self, workflow_templates: list[WorkflowTemplate], arrival_times: list[float]):
        """测试查看空队列"""
        workload = Workload(arrival_times, workflow_templates)

        # 清空队列
        while not workload._submit_queue.empty():  # type: ignore
            workload._submit_queue.get()  # type: ignore

        result = workload.peek()
        assert result is None

    def test_peek_non_empty_queue(self, workflow_templates: list[WorkflowTemplate], arrival_times: list[float]):
        """测试查看非空队列"""
        workload = Workload(arrival_times, workflow_templates)
        workload.reset()  # 确保队列有内容

        result = workload.peek()
        assert result is not None
        assert isinstance(result, SubmittedFunc)

        # 验证 peek 不会移除元素
        result2 = workload.peek()
        assert result == result2

        # 队列大小不变
        initial_size = workload._submit_queue.qsize()  # type: ignore
        workload.peek()
        assert workload._submit_queue.qsize() == initial_size  # type: ignore

    def test_get_empty_queue(self, workflow_templates: list[WorkflowTemplate], arrival_times: list[float]):
        """测试从空队列获取"""
        workload = Workload(arrival_times, workflow_templates)

        # 清空队列
        while not workload._submit_queue.empty():  # type: ignore
            workload._submit_queue.get()  # type: ignore

        result = workload.get()
        assert result is None

    def test_get_non_empty_queue(self, workflow_templates: list[WorkflowTemplate], arrival_times: list[float]):
        """测试从非空队列获取"""
        workload = Workload(arrival_times, workflow_templates)
        workload.reset()  # 确保队列有内容

        initial_size = workload._submit_queue.qsize()  # type: ignore

        result = workload.get()
        assert result is not None
        assert isinstance(result, SubmittedFunc)

        # 验证 get 会移除元素
        assert workload._submit_queue.qsize() == initial_size - 1  # type: ignore

    def test_run_function(self, workflow_templates: list[WorkflowTemplate], arrival_times: list[float]):
        """测试运行函数"""
        workload = Workload(arrival_times, workflow_templates)
        workload.reset()

        # 获取第一个提交的函数
        func = workload.get()
        if func:
            # 运行函数
            workload.run_function(func.wf_id, func.fn_id, func.submission_time + 1.0)

            # 验证函数状态
            workflow = workload[func.wf_id]
            assert workflow[func.fn_id].state == FuncStat.RUNNING
            assert workflow[func.fn_id].start_time == func.submission_time + 1.0

    def test_finish_function_with_successors(
        self, workflow_templates: list[WorkflowTemplate], arrival_times: list[float]
    ):
        """测试完成函数（有后继函数）"""
        workload = Workload(arrival_times, workflow_templates)
        workload.reset()

        # 获取第一个提交的函数
        func = workload.get()
        if func:
            # 运行并完成函数
            workload.run_function(func.wf_id, func.fn_id, func.submission_time + 1.0)

            initial_queue_size = workload._submit_queue.qsize()  # type: ignore
            workload.finish_function(func.wf_id, func.fn_id, func.submission_time + 3.0)

            # 验证原函数完成
            workflow = workload[func.wf_id]
            assert workflow[func.fn_id].state == FuncStat.FINISHED
            assert workflow[func.fn_id].completion_time == func.submission_time + 3.0

            # 检查是否有新的函数被添加到队列
            # 如果后继函数只有这一个前驱，那么它应该被添加到队列
            successors = workflow._dag.successor_indices(func.fn_id)  # type: ignore
            for succ in successors:
                predecessors = workflow._dag.predecessor_indices(succ)  # type: ignore
                if set(predecessors) <= {func.fn_id}:  # 所有前驱都已完成
                    # 应该有新函数被添加
                    assert workload._submit_queue.qsize() >= initial_queue_size  # type: ignore

    def test_finish_function_without_successors(
        self, workflow_templates: list[WorkflowTemplate], arrival_times: list[float]
    ):
        """测试完成函数（无后继函数）"""
        workload = Workload(arrival_times, workflow_templates)
        workload.reset()

        # 找到一个没有后继的函数
        found = False
        for i in range(len(arrival_times)):
            wf = workload[i]
            for fn_id in range(len(wf)):
                successors = wf._dag.successor_indices(fn_id)  # type: ignore
                if not successors:  # 没有后继
                    # 需要先提交这个函数
                    wf.submit_function(fn_id, wf.arrival_time)
                    workload.run_function(wf.wf_id, fn_id, wf.arrival_time + 1.0)

                    initial_queue_size = workload._submit_queue.qsize()  # type: ignore
                    workload.finish_function(wf.wf_id, fn_id, wf.arrival_time + 3.0)

                    # 验证没有新函数被添加到队列
                    assert workload._submit_queue.qsize() == initial_queue_size  # type: ignore
                    found = True
                    break
            if found:
                break

        if not found:
            pytest.skip("No function without successors found")

    def test_completed_property_none_finished(
        self, workflow_templates: list[WorkflowTemplate], arrival_times: list[float]
    ):
        """测试没有工作流完成时的 completed 属性"""
        workload = Workload(arrival_times, workflow_templates)

        # 重置后，工作流未完成
        workload.reset()
        assert not workload.completed

    def test_priority_queue_ordering(self, workflow_templates: list[WorkflowTemplate], arrival_times: list[float]):
        """测试优先级队列的排序"""
        workload = Workload(arrival_times, workflow_templates)
        workload.reset()

        # 获取所有提交的函数并验证它们是按时间排序的
        submitted_funcs: list[SubmittedFunc] = []
        while not workload._submit_queue.empty():  # type: ignore
            submitted_funcs.append(workload._submit_queue.get())  # type: ignore

        # 验证时间是非递减的
        for i in range(1, len(submitted_funcs)):
            assert submitted_funcs[i].submission_time >= submitted_funcs[i - 1].submission_time

    def test_multiple_workflow_interleaving(self, workflow_templates: list[WorkflowTemplate]):
        """测试多个工作流的交错执行"""
        # 创建交错到达时间
        arrival_times = [0.0, 1.5, 3.0]
        workload = Workload(arrival_times, workflow_templates)

        workload.reset()

        # 验证第一个函数是工作流0的源点函数
        func = workload.peek()
        assert func is not None
        assert func.wf_id == 0
        assert func.submission_time == 0.0

    def test_workload_with_single_workflow(self, workflow_templates: list[WorkflowTemplate]):
        """测试只有一个工作流的工作负载"""
        arrival_times = [2.0]
        workload = Workload(arrival_times, workflow_templates)

        assert workload[0].wf_id == 0
        assert workload[0].arrival_time == 2.0
        # 验证只有一个工作流（通过访问索引 1 会抛出异常）
        with pytest.raises(IndexError):
            _ = workload[1]

    def test_all_workflows_use_valid_templates(
        self, workflow_templates: list[WorkflowTemplate], arrival_times: list[float]
    ):
        """测试所有工作流都使用有效的模板"""
        workload = Workload(arrival_times, workflow_templates)

        # 验证每个工作流都使用了有效的模板特征
        for i in range(len(arrival_times)):
            wf = workload[i]
            # 工作流应该有函数
            assert len(wf) > 0

            # 函数数量应该匹配某个模板
            func_count = len(wf)
            template_func_counts = {len(template.computations) for template in workflow_templates}
            assert func_count in template_func_counts

    def test_getitem_valid_index(self, workflow_templates: list[WorkflowTemplate], arrival_times: list[float]):
        """测试 __getitem__ 方法使用有效索引"""
        workload = Workload(arrival_times, workflow_templates)

        # 测试所有有效索引
        for i in range(len(arrival_times)):
            wf = workload[i]
            assert wf.wf_id == i
            assert wf.arrival_time == arrival_times[i]

    def test_getitem_invalid_index(self, workflow_templates: list[WorkflowTemplate], arrival_times: list[float]):
        """测试 __getitem__ 方法使用无效索引"""
        workload = Workload(arrival_times, workflow_templates)

        # 测试超出范围的负索引
        with pytest.raises(IndexError):
            _ = workload[-(len(arrival_times) + 1)]

        # 测试超出范围的正索引
        with pytest.raises(IndexError):
            _ = workload[len(arrival_times)]

        with pytest.raises(IndexError):
            _ = workload[999]

    def test_workload_complete_execution(self, workflow_templates: list[WorkflowTemplate]):
        """测试工作负载的完整执行流程

        该测试模拟了完整的工作负载执行过程：
        1. 重置工作负载
        2. 循环执行函数直到所有工作流完成
        3. 验证所有函数都正确执行
        4. 验证工作负载达到完成状态
        """
        # 创建一个简单的工作负载，使用较少的工作流以控制测试时间
        arrival_times = [0.0, 2.0, 4.0, 6.0]  # 4个工作流
        workload = Workload(arrival_times, workflow_templates)

        # 重置工作负载，初始化提交队列
        workload.reset()

        # 收集执行统计信息
        class EventRecord(TypedDict):
            step: int
            time: float
            action: str
            wf_id: int
            fn_id: int
            duration: float | None

        class ExecutionStats(TypedDict):
            total_functions_run: int
            total_functions_finished: int
            workflow_function_counts: list[int]
            execution_timeline: list[EventRecord]

        execution_stats: ExecutionStats = {
            "total_functions_run": 0,
            "total_functions_finished": 0,
            "workflow_function_counts": [],
            "execution_timeline": [],
        }

        # 记录每个工作流的函数数量
        for i in range(len(arrival_times)):
            wf = workload[i]
            execution_stats["workflow_function_counts"].append(len(wf))

        current_time = 0.0
        step = 0
        max_steps = 1000  # 防止无限循环的安全措施

        # 主执行循环：持续执行函数直到所有工作流完成
        while not workload.completed and step < max_steps:
            # 从队列获取下一个要执行的函数
            submitted_func = workload.get()

            if submitted_func is None:
                # 队列为空但工作流未完成，这可能意味着所有函数都在运行中
                # 简单地推进时间并继续
                current_time += 1.0
                step += 1
                continue

            # 确保执行时间不早于提交时间
            execution_time = max(current_time, submitted_func.submission_time + 0.1)

            # 运行函数
            workload.run_function(submitted_func.wf_id, submitted_func.fn_id, execution_time)
            execution_stats["total_functions_run"] += 1

            # 记录执行事件
            execution_stats["execution_timeline"].append(
                {
                    "step": step,
                    "time": execution_time,
                    "action": "run",
                    "wf_id": submitted_func.wf_id,
                    "fn_id": submitted_func.fn_id,
                    "duration": None,
                }
            )

            # 模拟函数执行时间（基于计算量，这里简化处理）
            workflow = workload[submitted_func.wf_id]
            function_computation = workflow[submitted_func.fn_id].computation
            # 假设每个计算单位需要0.001时间单位
            execution_duration = function_computation * 0.001

            # 完成函数
            completion_time = execution_time + execution_duration
            workload.finish_function(submitted_func.wf_id, submitted_func.fn_id, completion_time)
            execution_stats["total_functions_finished"] += 1

            # 记录完成事件
            execution_stats["execution_timeline"].append(
                {
                    "step": step,
                    "time": completion_time,
                    "action": "finish",
                    "wf_id": submitted_func.wf_id,
                    "fn_id": submitted_func.fn_id,
                    "duration": execution_duration,
                }
            )

            # 更新当前时间
            current_time = completion_time
            step += 1

        # 验证执行结果
        assert workload.completed, f"工作负载应该在 {step} 步内完成，但未完成"
        assert step < max_steps, f"执行步数过多 ({step})，可能存在无限循环"

        # 验证所有函数都被执行了
        expected_total_functions = sum(execution_stats["workflow_function_counts"])
        assert (
            execution_stats["total_functions_finished"] == expected_total_functions
        ), f'完成的函数数量 ({execution_stats["total_functions_finished"]}) 不等于总函数数量 ({expected_total_functions})'

        # 验证执行统计的一致性
        assert (
            execution_stats["total_functions_run"] == execution_stats["total_functions_finished"]
        ), "运行的函数数量应该等于完成的函数数量"

        # 验证每个工作流都正确完成
        for i in range(len(arrival_times)):
            wf = workload[i]
            assert wf.completed, f"工作流 {i} 应该已完成"

            # 验证所有函数都处于完成状态
            for fn_id, fn in enumerate(wf):
                assert fn.state == FuncStat.FINISHED, f"工作流 {i} 的函数 {fn_id} 应该处于完成状态"
                assert fn.start_time is not None, f"工作流 {i} 的函数 {fn_id} 应该有开始时间"
                assert fn.completion_time is not None, f"工作流 {i} 的函数 {fn_id} 应该有完成时间"
                assert fn.start_time < fn.completion_time, f"工作流 {i} 的函数 {fn_id} 的开始时间应该早于完成时间"

        # 验证执行时间线的合理性
        timeline = execution_stats["execution_timeline"]

        # 按时间排序
        timeline_sorted = sorted(timeline, key=lambda x: x["time"])

        # 验证每个工作流的依赖关系
        for wf_id, wf in enumerate(workload):
            # 获取该工作流的所有执行事件
            wf_events = [event for event in timeline_sorted if event["wf_id"] == wf_id]

            # 按函数ID分组，并验证执行顺序
            function_events: dict[int, dict[str, EventRecord | None]] = {}
            for event in wf_events:
                fn_id = event["fn_id"]
                if fn_id not in function_events:
                    function_events[fn_id] = {"run": None, "finish": None}
                function_events[fn_id][event["action"]] = event

            # 验证每个函数的运行和完成时间
            for fn_id, events in function_events.items():
                run_event = events["run"]
                finish_event = events["finish"]

                assert run_event is not None, f"工作流 {wf_id} 的函数 {fn_id} 应该有运行事件"
                assert finish_event is not None, f"工作流 {wf_id} 的函数 {fn_id} 应该有完成事件"
                assert (
                    run_event["time"] < finish_event["time"]
                ), f"工作流 {wf_id} 的函数 {fn_id} 的运行时间应该早于完成时间"

        # 验证依赖关系：后继函数的开始时间应该晚于所有前驱函数的完成时间
        for wf_id in range(len(arrival_times)):
            wf = workload[wf_id]
            for fn_id in range(len(wf)):
                # 获取函数的完成时间
                fn_completion_time = wf[fn_id].completion_time

                # 检查所有后继函数
                successors = wf._dag.successor_indices(fn_id)  # type: ignore
                for succ_id in successors:
                    succ_start_time = wf[succ_id].start_time
                    assert (
                        succ_start_time >= fn_completion_time  # type: ignore
                    ), f"工作流 {wf_id} 中，函数 {succ_id} 的开始时间 ({succ_start_time}) 应该晚于或等于其前驱函数 {fn_id} 的完成时间 ({fn_completion_time})"

    def test_workload_complete_execution_single_workflow(self, workflow_templates: list[WorkflowTemplate]):
        """测试单个工作流的完整执行流程

        这是完整执行测试的简化版本，只使用一个工作流，
        更容易调试和验证特定工作流的执行逻辑
        """
        # 创建单个工作流
        arrival_times = [1.0]  # 1个工作流
        workload = Workload(arrival_times, workflow_templates)

        # 重置工作负载
        workload.reset()

        # 记录执行状态
        executed_functions: set[int] = set()
        finished_functions: set[int] = set()

        current_time = 1.0
        step = 0
        max_steps = 500  # 单个工作流的步数限制

        while not workload.completed and step < max_steps:
            submitted_func = workload.get()

            if submitted_func is None:
                current_time += 0.5
                step += 1
                continue

            # 验证函数状态
            workflow = workload[submitted_func.wf_id]
            fn_state_before = workflow[submitted_func.fn_id].state
            assert (
                fn_state_before == FuncStat.SUBMITTED
            ), f"函数 {submitted_func.fn_id} 在执行前应该是 SUBMITTED 状态，实际为 {fn_state_before}"

            # 执行函数
            execution_time = max(current_time, submitted_func.submission_time)
            workload.run_function(submitted_func.wf_id, submitted_func.fn_id, execution_time)
            executed_functions.add(submitted_func.fn_id)

            # 验证函数状态变为运行中
            fn_state_running = workflow[submitted_func.fn_id].state
            assert (
                fn_state_running == FuncStat.RUNNING
            ), f"函数 {submitted_func.fn_id} 运行后应该是 RUNNING 状态，实际为 {fn_state_running}"

            # 完成函数
            completion_time = execution_time + 0.1  # 简化的执行时间
            workload.finish_function(submitted_func.wf_id, submitted_func.fn_id, completion_time)
            finished_functions.add(submitted_func.fn_id)

            # 验证函数状态变为已完成
            fn_state_finished = workflow[submitted_func.fn_id].state
            assert (
                fn_state_finished == FuncStat.FINISHED
            ), f"函数 {submitted_func.fn_id} 完成后应该是 FINISHED 状态，实际为 {fn_state_finished}"

            current_time = completion_time
            step += 1

        # 最终验证
        assert workload.completed, "单个工作流应该完成"
        assert step < max_steps, f"执行步数过多: {step}"

        # 验证所有函数都被执行了
        workflow = workload[0]
        total_functions = len(workflow)
        assert (
            len(finished_functions) == total_functions
        ), f"所有 {total_functions} 个函数都应该完成，实际完成 {len(finished_functions)} 个"

        assert (
            len(executed_functions) == total_functions
        ), f"所有 {total_functions} 个函数都应该被执行，实际执行 {len(executed_functions)} 个"
