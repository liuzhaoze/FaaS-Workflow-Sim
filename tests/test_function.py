"""test_function.py

测试 src/faas_workflow_sim/function.py 中的 Function 类和 FuncStat 枚举
"""

import pytest

from faas_workflow_sim.function import FuncStat, Function


class TestFuncStat:
    """测试 FuncStat 枚举"""

    def test_funcstat_values(self):
        """测试枚举值"""
        assert FuncStat.PENDING == 0
        assert FuncStat.SUBMITTED == 1
        assert FuncStat.RUNNING == 2
        assert FuncStat.FINISHED == 3

    def test_funcstat_names(self):
        """测试枚举名称"""
        assert FuncStat.PENDING.name == "PENDING"
        assert FuncStat.SUBMITTED.name == "SUBMITTED"
        assert FuncStat.RUNNING.name == "RUNNING"
        assert FuncStat.FINISHED.name == "FINISHED"


class TestFunction:
    """测试 Function 类"""

    @pytest.fixture
    def sample_function(self) -> Function:
        """创建示例函数"""
        return Function(wf_id=1, fn_id=2, computation=1000, memory_req=512, parallelism=4)

    def test_function_initialization(self, sample_function: Function):
        """测试函数初始化"""
        assert sample_function.wf_id == 1
        assert sample_function.fn_id == 2
        assert sample_function.computation == 1000
        assert sample_function.memory_req == 512
        assert sample_function.parallelism == 4
        assert sample_function.state == FuncStat.PENDING
        assert sample_function.memory_alloc is None
        assert sample_function.submission_time is None
        assert sample_function.start_time is None
        assert sample_function.completion_time is None

    def test_function_slots(self, sample_function: Function):
        """测试 __slots__ 属性"""
        # 测试所有 slot 都存在
        expected_slots = {
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
        }
        actual_slots = set(Function.__slots__)
        assert actual_slots == expected_slots

        # 测试不能添加新属性
        with pytest.raises(AttributeError):
            sample_function.new_attribute = "test"  # type: ignore

    def test_reset(self, sample_function: Function):
        """测试重置功能"""
        # 修改函数状态
        sample_function.state = FuncStat.RUNNING
        sample_function.memory_alloc = 1024
        sample_function.submission_time = 10.0
        sample_function.start_time = 12.0
        sample_function.completion_time = 15.0

        # 重置
        sample_function.reset()

        # 验证重置后状态
        assert sample_function.state == FuncStat.PENDING
        assert sample_function.memory_alloc is None
        assert sample_function.submission_time is None
        assert sample_function.start_time is None
        assert sample_function.completion_time is None

    def test_submit_success(self, sample_function: Function):
        """测试成功提交函数"""
        submission_time = 5.0
        sample_function.submit(submission_time)

        assert sample_function.state == FuncStat.SUBMITTED
        assert sample_function.submission_time == submission_time

    def test_submit_negative_time(self, sample_function: Function):
        """测试提交时间为负数"""
        with pytest.raises(ValueError, match="Submission time \\(-1.0\\) must be non-negative"):
            sample_function.submit(-1.0)

    def test_submit_invalid_state_transition(self, sample_function: Function):
        """测试无效的状态转换"""
        # 先设置为 RUNNING 状态
        sample_function.state = FuncStat.RUNNING

        # 尝试从 RUNNING 直接 submit 会失败
        with pytest.raises(ValueError, match="Invalid status transition: RUNNING -> SUBMITTED"):
            sample_function.submit(5.0)

    def test_run_success(self, sample_function: Function):
        """测试成功运行函数"""
        # 先提交
        sample_function.submit(5.0)

        # 然后运行
        start_time = 10.0
        sample_function.run(start_time)

        assert sample_function.state == FuncStat.RUNNING
        assert sample_function.start_time == start_time

    def test_run_without_submit(self, sample_function: Function):
        """测试未提交就运行"""
        with pytest.raises(ValueError, match="Function must be submitted before running"):
            sample_function.run(10.0)

    def test_run_earlier_than_submit(self, sample_function: Function):
        """测试开始时间早于提交时间"""
        sample_function.submit(10.0)

        with pytest.raises(ValueError, match="Start time \\(5.0\\) cannot be earlier than submission time \\(10.0\\)"):
            sample_function.run(5.0)

    def test_run_invalid_state_transition(self, sample_function: Function):
        """测试运行时的无效状态转换"""
        # 先提交函数以满足 run 对 submission_time 验证
        sample_function.submit(5.0)
        # 直接修改 state 属性到 FINISHED
        object.__setattr__(sample_function, "state", FuncStat.FINISHED)

        # 尝试从 FINISHED 直接 run 会失败
        with pytest.raises(ValueError, match="Invalid status transition: FINISHED -> RUNNING"):
            sample_function.run(10.0)

    def test_finish_success(self, sample_function: Function):
        """测试成功完成函数"""
        # 先提交和运行
        sample_function.submit(5.0)
        sample_function.run(10.0)

        # 然后完成
        completion_time = 15.0
        sample_function.finish(completion_time)

        assert sample_function.state == FuncStat.FINISHED
        assert sample_function.completion_time == completion_time

    def test_finish_without_run(self, sample_function: Function):
        """测试未运行就完成"""
        with pytest.raises(ValueError, match="Function must be running before finishing"):
            sample_function.finish(15.0)

    def test_finish_earlier_than_start(self, sample_function: Function):
        """测试完成时间早于开始时间"""
        sample_function.submit(5.0)
        sample_function.run(10.0)

        with pytest.raises(ValueError, match="Finish time \\(8.0\\) cannot be earlier than start time \\(10.0\\)"):
            sample_function.finish(8.0)

    def test_finish_invalid_state_transition(self, sample_function: Function):
        """测试完成时的无效状态转换"""
        # 先提交和运行函数以满足 finish 对 start_time 验证
        sample_function.submit(5.0)
        sample_function.run(10.0)
        # 直接修改 state 属性到 PENDING
        object.__setattr__(sample_function, "state", FuncStat.PENDING)

        # 尝试从 PENDING 直接 finish 会失败
        with pytest.raises(ValueError, match="Invalid status transition: PENDING -> FINISHED"):
            sample_function.finish(15.0)

    def test_allocate_memory_success(self, sample_function: Function):
        """测试成功分配内存"""
        memory_size = 1024
        sample_function.allocate_memory(memory_size)

        assert sample_function.memory_alloc == memory_size

    def test_allocate_memory_zero_or_negative(self, sample_function: Function):
        """测试分配零或负数内存"""
        with pytest.raises(ValueError, match="Memory size must be positive"):
            sample_function.allocate_memory(0)

        with pytest.raises(ValueError, match="Memory size must be positive"):
            sample_function.allocate_memory(-512)

    def test_standard_execution_time(self, sample_function: Function):
        """测试标准执行时间计算"""
        single_core_speed = 1000
        expected_time = sample_function.computation / (sample_function.parallelism * single_core_speed)

        actual_time = sample_function.standard_execution_time(single_core_speed)

        assert actual_time == expected_time

    def test_standard_execution_time_zero_speed(self, sample_function: Function):
        """测试零计算速度"""
        with pytest.raises(ValueError, match="Single core speed must be positive"):
            sample_function.standard_execution_time(0)

    def test_standard_execution_time_negative_speed(self, sample_function: Function):
        """测试负计算速度"""
        with pytest.raises(ValueError, match="Single core speed must be positive"):
            sample_function.standard_execution_time(-1000)

    def test_complete_lifecycle(self, sample_function: Function):
        """测试完整的函数生命周期"""
        # 初始状态
        assert sample_function.state == FuncStat.PENDING

        # 提交
        sample_function.submit(1.0)
        assert sample_function.state == FuncStat.SUBMITTED
        assert sample_function.submission_time == 1.0

        # 分配内存
        sample_function.allocate_memory(2048)
        assert sample_function.memory_alloc == 2048

        # 运行
        sample_function.run(3.0)
        assert sample_function.state == FuncStat.RUNNING
        assert sample_function.start_time == 3.0

        # 完成
        sample_function.finish(8.0)
        assert sample_function.state == FuncStat.FINISHED
        assert sample_function.completion_time == 8.0

        # 验证执行时间
        exec_time = sample_function.standard_execution_time(1000)
        expected_time = 1000 / (4 * 1000)  # computation / (parallelism * single_core_speed)
        assert exec_time == expected_time

    def test_valid_state_transitions(self):
        """测试所有有效的状态转换"""
        function = Function(1, 1, 100, 256, 2)

        # PENDING -> SUBMITTED
        function.submit(1.0)
        assert function.state == FuncStat.SUBMITTED

        # SUBMITTED -> RUNNING
        function.run(2.0)
        assert function.state == FuncStat.RUNNING

        # RUNNING -> FINISHED
        function.finish(5.0)
        assert function.state == FuncStat.FINISHED

    def test_invalid_state_transitions(self):
        """测试所有无效的状态转换"""
        function1 = Function(1, 1, 100, 256, 2)

        # 先提交函数以满足 submission_time 验证，然后手动设置状态回 PENDING
        function1.submit(0.5)
        object.__setattr__(function1, "state", FuncStat.PENDING)

        # 不能从 PENDING 直接到 RUNNING
        with pytest.raises(ValueError, match="Invalid status transition"):
            function1.run(1.0)

        # 创建新函数测试 PENDING -> FINISHED 转换
        function2 = Function(1, 1, 100, 256, 2)
        # 先提交和运行函数以满足 start_time 验证，然后手动设置状态回 PENDING
        function2.submit(1.0)
        function2.run(2.0)
        object.__setattr__(function2, "state", FuncStat.PENDING)

        # 不能从 PENDING 直接到 FINISHED
        with pytest.raises(ValueError, match="Invalid status transition"):
            function2.finish(3.0)

        # 正常流程到 SUBMITTED
        function1.submit(1.0)

        # 先运行函数以满足 start_time 验证，然后手动设置状态回 SUBMITTED
        function1.run(2.0)
        object.__setattr__(function1, "state", FuncStat.SUBMITTED)

        # 不能从 SUBMITTED 直接到 FINISHED
        with pytest.raises(ValueError, match="Invalid status transition"):
            function1.finish(3.0)

        # 不能从 SUBMITTED 再次提交
        with pytest.raises(ValueError, match="Invalid status transition"):
            function1.submit(2.0)

        # 正常流程到 RUNNING
        function1.run(2.0)

        # 不能从 RUNNING 再次运行
        with pytest.raises(ValueError, match="Invalid status transition"):
            function1.run(3.0)

        # 不能从 RUNNING 再次提交
        with pytest.raises(ValueError, match="Invalid status transition"):
            function1.submit(3.0)
