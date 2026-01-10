"""test_numa_node.py

测试 src/faas_workflow_sim/numa_node.py 中的 NumaNode、ResourceUtilizationManager、ExecutionSpeedManager 和相关辅助类
"""

import pytest

from faas_workflow_sim.container import Container
from faas_workflow_sim.numa_node import (
    ExecutionSpeedManager,
    NumaNode,
    ResourceUtilizationManager,
    SpeedRecord,
    UtilizationRecord,
)


class TestUtilizationRecord:
    """测试 UtilizationRecord NamedTuple"""

    def test_utilization_record_creation(self):
        """测试利用率记录创建"""
        record = UtilizationRecord(
            timestamp=1.0,
            cpu=0.8,
            memory=0.6,
            load=0.75,
            total_parallelism=6,
            free_memory=2048,
        )
        assert record.timestamp == 1.0
        assert record.cpu == 0.8
        assert record.memory == 0.6
        assert record.load == 0.75
        assert record.total_parallelism == 6
        assert record.free_memory == 2048


class TestSpeedRecord:
    """测试 SpeedRecord NamedTuple"""

    def test_speed_record_creation(self):
        """测试速度记录创建"""
        record = SpeedRecord(timestamp=1.0, speed=1000)
        assert record.timestamp == 1.0
        assert record.speed == 1000


class TestResourceUtilizationManager:
    """测试 ResourceUtilizationManager 类"""

    @pytest.fixture
    def rum(self) -> ResourceUtilizationManager:
        """创建资源利用率管理器"""
        return ResourceUtilizationManager()

    def test_rum_initialization(self, rum: ResourceUtilizationManager):
        """测试资源利用率管理器初始化"""
        assert rum.records == []

    def test_rum_reset(self, rum: ResourceUtilizationManager):
        """测试重置资源利用率记录"""
        # 添加一些记录
        rum.records.append(UtilizationRecord(0.0, 0.0, 0.0, 0.0, 0, 8192))
        rum.records.append(UtilizationRecord(1.0, 0.5, 0.3, 0.4, 2, 4096))

        # 重置
        rum.reset(2.0, 0.8, 0.6, 0.75, 6, 2048)

        assert len(rum.records) == 1
        assert rum.records[0] == UtilizationRecord(2.0, 0.8, 0.6, 0.75, 6, 2048)

    def test_rum_clear_after(self, rum: ResourceUtilizationManager):
        """测试清除指定时间点之后的记录"""
        # 添加一些记录
        rum.records = [
            UtilizationRecord(0.0, 0.0, 0.0, 0.0, 0, 8192),
            UtilizationRecord(1.0, 0.5, 0.3, 0.4, 2, 4096),
            UtilizationRecord(2.0, 0.8, 0.6, 0.75, 6, 2048),
            UtilizationRecord(3.0, 0.6, 0.4, 0.5, 4, 6144),
        ]

        # 清除 1.5 之后的记录
        rum.clear_after(1.5)

        assert len(rum.records) == 2
        assert rum.records[0].timestamp == 0.0
        assert rum.records[1].timestamp == 1.0

    def test_rum_add_record_success(self, rum: ResourceUtilizationManager):
        """测试添加记录成功"""
        # 添加初始记录
        rum.reset(0.0, 0.0, 0.0, 0.0, 0, 8192)

        # 添加新记录
        new_record = UtilizationRecord(1.0, 0.5, 0.3, 0.4, 2, 4096)
        rum.add_record(new_record)

        assert len(rum.records) == 2
        assert rum.records[1] == new_record

    def test_rum_add_record_invalid_timestamp(self, rum: ResourceUtilizationManager):
        """测试添加记录时间戳无效"""
        # 添加初始记录
        rum.reset(1.0, 0.0, 0.0, 0.0, 0, 8192)

        # 尝试添加时间戳较早的记录
        with pytest.raises(ValueError, match="New record timestamp must be greater than the last record timestamp"):
            rum.add_record(UtilizationRecord(0.5, 0.5, 0.3, 0.4, 2, 4096))

    def test_rum_get_record_at_existing_time(self, rum: ResourceUtilizationManager):
        """测试获取指定时间点的记录（时间点存在）"""
        # 添加一些记录
        rum.records = [
            UtilizationRecord(0.0, 0.0, 0.0, 0.0, 0, 8192),
            UtilizationRecord(1.0, 0.5, 0.3, 0.4, 2, 4096),
            UtilizationRecord(2.0, 0.8, 0.6, 0.75, 6, 2048),
        ]

        # 获取时间点 1.5 的记录（应该返回时间戳为 1.0 的记录）
        record = rum.get_record_at(1.5)
        assert record == UtilizationRecord(1.0, 0.5, 0.3, 0.4, 2, 4096)

    def test_rum_get_record_at_exact_time(self, rum: ResourceUtilizationManager):
        """测试获取指定时间点的记录（时间点完全匹配）"""
        # 添加一些记录
        rum.records = [
            UtilizationRecord(0.0, 0.0, 0.0, 0.0, 0, 8192),
            UtilizationRecord(1.0, 0.5, 0.3, 0.4, 2, 4096),
        ]

        # 获取时间点 1.0 的记录
        record = rum.get_record_at(1.0)
        assert record == UtilizationRecord(1.0, 0.5, 0.3, 0.4, 2, 4096)

    def test_rum_get_record_at_early_time(self, rum: ResourceUtilizationManager):
        """测试获取指定时间点的记录（时间点早于所有记录）"""
        # 添加一些记录
        rum.records = [
            UtilizationRecord(1.0, 0.5, 0.3, 0.4, 2, 4096),
            UtilizationRecord(2.0, 0.8, 0.6, 0.75, 6, 2048),
        ]

        # 尝试获取时间点 0.5 的记录
        with pytest.raises(ValueError, match="No utilization record available before the specified time"):
            rum.get_record_at(0.5)


class TestExecutionSpeedManager:
    """测试 ExecutionSpeedManager 类"""

    @pytest.fixture
    def esm(self) -> ExecutionSpeedManager:
        """创建执行速度管理器"""
        return ExecutionSpeedManager()

    def test_esm_initialization(self, esm: ExecutionSpeedManager):
        """测试执行速度管理器初始化"""
        assert esm.records == {}

    def test_esm_reset(self, esm: ExecutionSpeedManager):
        """测试重置执行速度记录"""
        # 添加一些记录
        esm.records[(1, 1)] = [SpeedRecord(0.0, 1000)]
        esm.records[(1, 2)] = [SpeedRecord(1.0, 2000)]

        # 重置
        esm.reset()

        assert esm.records == {}

    def test_esm_add_record_new_function(self, esm: ExecutionSpeedManager):
        """测试为新函数添加记录"""
        record = SpeedRecord(timestamp=1.0, speed=1000)
        esm.add_record(1, 2, record)

        assert len(esm.records) == 1
        assert (1, 2) in esm.records
        assert len(esm.records[(1, 2)]) == 1
        assert esm.records[(1, 2)][0] == record

    def test_esm_add_record_existing_function(self, esm: ExecutionSpeedManager):
        """测试为已存在的函数添加记录"""
        # 添加初始记录
        esm.add_record(1, 2, SpeedRecord(timestamp=1.0, speed=1000))

        # 添加新记录
        new_record = SpeedRecord(timestamp=2.0, speed=1500)
        esm.add_record(1, 2, new_record)

        assert len(esm.records[(1, 2)]) == 2
        assert esm.records[(1, 2)][1] == new_record

    def test_esm_add_record_invalid_timestamp(self, esm: ExecutionSpeedManager):
        """测试添加记录时间戳无效"""
        # 添加初始记录
        esm.add_record(1, 2, SpeedRecord(timestamp=1.0, speed=1000))

        # 尝试添加时间戳较早的记录
        with pytest.raises(
            ValueError, match="New speed record timestamp must be greater than the last record timestamp"
        ):
            esm.add_record(1, 2, SpeedRecord(timestamp=0.5, speed=800))

    def test_esm_get_computation_until_single_period(self, esm: ExecutionSpeedManager):
        """测试计算到指定时间点的完成计算量（单时间段）"""
        # 添加记录
        esm.add_record(1, 2, SpeedRecord(timestamp=1.0, speed=1000))

        # 计算到时间点 2.0 的完成计算量
        completed = esm.get_computation_until(1, 2, 2.0)
        assert completed == 1000  # 1000 * (2.0 - 1.0)

    def test_esm_get_computation_until_multiple_periods(self, esm: ExecutionSpeedManager):
        """测试计算到指定时间点的完成计算量（多时间段）"""
        # 添加多条记录
        esm.add_record(1, 2, SpeedRecord(timestamp=1.0, speed=1000))
        esm.add_record(1, 2, SpeedRecord(timestamp=2.0, speed=1500))
        esm.add_record(1, 2, SpeedRecord(timestamp=3.0, speed=2000))

        # 计算到时间点 2.5 的完成计算量
        completed = esm.get_computation_until(1, 2, 2.5)
        # 1000 * (2.0 - 1.0) + 1500 * (2.5 - 2.0) = 1000 + 750 = 1750
        assert completed == 1750

    def test_esm_get_computation_until_early_time(self, esm: ExecutionSpeedManager):
        """测试计算到指定时间点的完成计算量（时间点早于记录）"""
        # 添加记录
        esm.add_record(1, 2, SpeedRecord(timestamp=1.0, speed=1000))

        # 尝试计算到时间点 0.5 的完成计算量
        with pytest.raises(ValueError, match="No speed record available before the specified time"):
            esm.get_computation_until(1, 2, 0.5)

    def test_esm_get_computation_until_nonexistent_function(self, esm: ExecutionSpeedManager):
        """测试计算不存在函数的完成计算量"""
        with pytest.raises(KeyError):
            esm.get_computation_until(1, 2, 2.0)


class TestNumaNode:
    """测试 NumaNode 类"""

    @pytest.fixture
    def sample_numa_node(self) -> NumaNode:
        """创建示例 NUMA 节点"""
        return NumaNode(node_id=1, cpu=4, memory=8192, single_core_speed=1000)

    def test_numa_node_initialization(self, sample_numa_node: NumaNode):
        """测试 NUMA 节点初始化"""
        assert sample_numa_node.node_id == 1
        assert sample_numa_node.cpu == 4
        assert sample_numa_node.memory == 8192
        assert sample_numa_node.single_core_speed == 1000
        assert sample_numa_node.computing_capacity == 4000  # 4 * 1000
        assert sample_numa_node._current_time == 0.0  # type: ignore
        assert sample_numa_node._waiting_containers.empty()  # type: ignore
        assert sample_numa_node._running_containers == []  # type: ignore

    def test_numa_node_reset(self, sample_numa_node: NumaNode):
        """测试重置 NUMA 节点"""
        # 先重置以初始化资源记录
        sample_numa_node.reset()

        # 创建并完成一个容器以改变状态
        c1 = Container(
            wf_id=1,
            fn_id=1,
            memory_req=512,
            memory_alloc=512,
            computation=1000,
            parallelism=1,
            submission_time=0.6,
            data_transfer_time=0.4,
        )
        sample_numa_node.on_container_creation(c1)

        # 完成容器以推进时间
        sample_numa_node.on_container_completion()

        # 验证状态已改变
        assert sample_numa_node._current_time > 0.0  # type: ignore

        # 重置
        sample_numa_node.reset()

        # 验证所有状态已重置
        assert sample_numa_node._current_time == 0.0  # type: ignore
        assert sample_numa_node._waiting_containers.empty()  # type: ignore
        assert sample_numa_node._running_containers == []  # type: ignore

    def test_numa_node_container_creation_sufficient_memory(self, sample_numa_node: NumaNode):
        """测试容器创建（内存充足）"""
        # 先重置节点以初始化资源利用率记录
        sample_numa_node.reset()

        c = Container(
            wf_id=1,
            fn_id=1,
            memory_req=512,
            memory_alloc=1024,
            computation=10000,
            parallelism=2,
            submission_time=0.6,
            data_transfer_time=0.4,
        )
        sample_numa_node.on_container_creation(c)

        assert sample_numa_node._waiting_containers.empty()  # type: ignore
        assert len(sample_numa_node._running_containers) == 1  # type: ignore
        assert sample_numa_node._running_containers[0].wf_id == 1  # type: ignore
        assert sample_numa_node._running_containers[0].fn_id == 1  # type: ignore
        assert sample_numa_node._running_containers[0].start_time == 1.0  # type: ignore

    def test_numa_node_container_creation_insufficient_memory(self, sample_numa_node: NumaNode):
        """测试容器创建（内存不足）"""
        # 先重置节点以初始化资源利用率记录
        sample_numa_node.reset()

        # 创建占用大量内存的容器
        c1 = Container(
            wf_id=1,
            fn_id=1,
            memory_req=4096,
            memory_alloc=4096,
            computation=10000,
            parallelism=2,
            submission_time=0.6,
            data_transfer_time=0.4,
        )
        sample_numa_node.on_container_creation(c1)

        # 尝试创建另一个内存不足的容器
        c2 = Container(
            wf_id=2,
            fn_id=1,
            memory_req=8192,
            memory_alloc=8192,
            computation=10000,
            parallelism=2,
            submission_time=0.6,
            data_transfer_time=0.4,
        )
        sample_numa_node.on_container_creation(c2)

        assert len(sample_numa_node._running_containers) == 1  # type: ignore
        assert not sample_numa_node._waiting_containers.empty()  # type: ignore

    def test_numa_node_container_creation_invalid_creation_time(self, sample_numa_node: NumaNode):
        """测试容器创建（提交时间晚于当前时间的情况）"""
        # 先重置节点以初始化资源利用率记录
        sample_numa_node.reset()

        # 先推进当前时间
        sample_numa_node._current_time = 5.0  # type: ignore

        # 提交一个 submission_time 晚于当前时间的容器
        # creation_time = max(5.0, 6.0) + 0.4 = 6.4
        c = Container(
            wf_id=1,
            fn_id=1,
            memory_req=512,
            memory_alloc=1024,
            computation=10000,
            parallelism=2,
            submission_time=6.0,  # 提交时间晚于当前时间
            data_transfer_time=0.4,
        )
        # 容器应该成功创建，creation_time 为 6.4
        sample_numa_node.on_container_creation(c)
        assert len(sample_numa_node._running_containers) == 1  # type: ignore
        assert sample_numa_node._running_containers[0].creation_time == 6.4  # type: ignore

    def test_numa_node_container_completion_no_running(self, sample_numa_node: NumaNode):
        """测试容器完成（没有运行中的容器）"""
        with pytest.raises(RuntimeError, match="No running containers to complete"):
            sample_numa_node.on_container_completion()

    def test_numa_node_get_earliest_finished_container_empty(self, sample_numa_node: NumaNode):
        """测试获取最早完成容器（没有运行中的容器）"""
        # 该方法在没有运行容器时应该返回 (None, float("inf"))
        index, completion_time = sample_numa_node.get_earliest_finished()
        assert index is None
        assert completion_time == float("inf")

    def test_numa_node_get_earliest_finished_container_success(self, sample_numa_node: NumaNode):
        """测试获取最早完成容器（成功）"""
        # 先重置节点以初始化资源利用率记录
        sample_numa_node.reset()

        # 创建三个计算量不同的容器，验证最早的完成时间被正确识别
        # 容器1: 大计算量，完成时间最晚
        c1 = Container(
            wf_id=1,
            fn_id=1,
            memory_req=512,
            memory_alloc=1024,
            computation=4000,
            parallelism=1,
            submission_time=0.6,
            data_transfer_time=0.4,
        )
        # 容器2: 小计算量，完成时间最早
        c2 = Container(
            wf_id=2,
            fn_id=1,
            memory_req=512,
            memory_alloc=1024,
            computation=1000,
            parallelism=1,
            submission_time=0.6,
            data_transfer_time=0.4,
        )
        # 容器3: 中等计算量，完成时间居中
        c3 = Container(
            wf_id=3,
            fn_id=1,
            memory_req=512,
            memory_alloc=1024,
            computation=2500,
            parallelism=1,
            submission_time=0.6,
            data_transfer_time=0.4,
        )

        sample_numa_node.on_container_creation(c1)
        sample_numa_node.on_container_creation(c2)
        sample_numa_node.on_container_creation(c3)

        # 获取最早完成的容器
        index, completion_time = sample_numa_node.get_earliest_finished()

        # 容器2应该最早完成（计算量最小）
        # 计算量1000，在单核速度1000下需要1秒，完成时间 = 1.0 + 1.0 = 2.0
        assert index == 1  # c2 的索引
        assert completion_time == pytest.approx(2.0, rel=1e-9)  # type: ignore

    def test_numa_node_workflow_simple(self, sample_numa_node: NumaNode):
        """测试完整工作流程（简单场景）"""
        # 先重置节点以初始化资源利用率记录
        sample_numa_node.reset()

        # 创建并运行一个容器
        c = Container(
            wf_id=1,
            fn_id=1,
            memory_req=512,
            memory_alloc=1024,
            computation=4000,  # 在速度1000下需要4秒
            parallelism=1,
            submission_time=0.6,
            data_transfer_time=0.4,
        )
        sample_numa_node.on_container_creation(c)

        assert len(sample_numa_node._running_containers) == 1  # type: ignore

        # 完成容器
        completed_container = sample_numa_node.on_container_completion()

        assert completed_container.wf_id == 1
        assert completed_container.fn_id == 1
        assert sample_numa_node._current_time == pytest.approx(5.0, rel=1e-9)  # 1.0 + 4.0  # type: ignore
        assert len(sample_numa_node._running_containers) == 0  # type: ignore

    def test_numa_node_workflow_memory_waiting(self, sample_numa_node: NumaNode):
        """测试工作流程（内存等待场景）"""
        # 先重置节点以初始化资源利用率记录
        sample_numa_node.reset()

        # 创建占用大部分内存的容器
        c1 = Container(
            wf_id=1,
            fn_id=1,
            memory_req=4096,
            memory_alloc=4096,
            computation=2000,
            parallelism=2,
            submission_time=0.6,
            data_transfer_time=0.4,
        )
        sample_numa_node.on_container_creation(c1)

        # 尝试创建内存不足的容器（会进入等待队列）
        c2 = Container(
            wf_id=2,
            fn_id=1,
            memory_req=8192,
            memory_alloc=8192,
            computation=2000,
            parallelism=2,
            submission_time=0.6,
            data_transfer_time=0.4,
        )
        sample_numa_node.on_container_creation(c2)

        assert len(sample_numa_node._running_containers) == 1  # type: ignore
        assert not sample_numa_node._waiting_containers.empty()  # type: ignore

        # 完成第一个容器，应该会自动启动等待中的容器
        completed_container = sample_numa_node.on_container_completion()

        assert completed_container.wf_id == 1
        assert len(sample_numa_node._running_containers) == 1  # 等待中的容器应该启动 # type: ignore
        assert sample_numa_node._running_containers[0].wf_id == 2  # type: ignore
        assert sample_numa_node._waiting_containers.empty()  # type: ignore

    @pytest.mark.parametrize(
        "cpu,memory,single_core_speed",
        [
            (2, 4096, 500),
            (8, 16384, 2000),
            (16, 32768, 1500),
        ],
    )
    def test_numa_node_different_configurations(self, cpu: int, memory: int, single_core_speed: int):
        """测试不同配置的 NUMA 节点"""
        node = NumaNode(node_id=1, cpu=cpu, memory=memory, single_core_speed=single_core_speed)

        assert node.node_id == 1
        assert node.cpu == cpu
        assert node.memory == memory
        assert node.single_core_speed == single_core_speed
        assert node.computing_capacity == cpu * single_core_speed

    def test_numa_node_parallelism_computation(self, sample_numa_node: NumaNode):
        """测试并行度计算影响"""
        # 先重置节点以初始化资源利用率记录
        sample_numa_node.reset()

        # 创建两个容器，总并行度超过 CPU 核心数
        c1 = Container(
            wf_id=1,
            fn_id=1,
            memory_req=512,
            memory_alloc=1024,
            computation=4000,
            parallelism=3,  # 并行度3
            submission_time=0.6,
            data_transfer_time=0.4,
        )
        sample_numa_node.on_container_creation(c1)

        # 两个容器同时提交
        c2 = Container(
            wf_id=2,
            fn_id=1,
            memory_req=512,
            memory_alloc=1024,
            computation=2000,
            parallelism=3,  # 并行度3，总并行度6 > CPU核心数4
            submission_time=0.6,
            data_transfer_time=0.4,
        )
        sample_numa_node.on_container_creation(c2)

        # 当总并行度 > CPU核心数时，每个并行度的速度应该下降
        # 总并行度 = 3 + 3 = 6
        # computing_capacity = 4 * 1000 = 4000
        # single_parallelism_speed = floor(4000 / 6) = 666
        # 每个容器的实际速度 = 666 * 3 = 1998
        # memory_bottleneck = min(1.0, 1024 / 512) = 1.0

        assert len(sample_numa_node._running_containers) == 2  # type: ignore

        # 获取最早完成的容器
        earliest_index, earliest_completion_time = sample_numa_node.get_earliest_finished()

        # 验证最早完成的是 c2（索引为1），因为它的计算量更小
        assert earliest_index == 1

        # c2 的计算量 2000，速度 1998
        # 预期完成时间 = 1.0 + 2000 / 1998 ≈ 2.001
        assert earliest_completion_time == pytest.approx(1.0 + 2000 / 1998, rel=1e-9)  # type: ignore

        # 另一个容器 c1 的索引应该是 1 - earliest_index = 0
        other_index = 1 - earliest_index
        assert other_index == 0

        # c1 的完成时间计算：
        # 阶段1（t=1.0 到 t≈2.001）：两个容器同时运行
        #   - c1 速度 1998，运行时间 ≈1.001秒
        #   - 已完成计算 = 1998 * 1.001 ≈ 2000
        #   - 剩余计算 = 4000 - 2000 = 2000
        # 阶段2（t≈2.001 到结束）：只剩 c1，速度提升
        #   - 总并行度 = 3 < CPU核心数 4
        #   - single_parallelism_speed = 1000
        #   - c1 新速度 = 1000 * 3 = 3000
        #   - 剩余时间 = 2000 / 3000 ≈ 0.667
        #   - 总完成时间 = 2.001 + 0.667 ≈ 2.668
        container1 = sample_numa_node._running_containers[other_index]  # type: ignore
        expected_c1_finish = 1.0 + 2000 / 1998 + 2000 / 3000
        assert container1.completion_time == pytest.approx(expected_c1_finish, rel=1e-9)  # type: ignore

        # 验证初始执行速度记录（在 c2 完成前）
        # c1 的速度记录
        c1_speed_records = sample_numa_node._esm.records[(1, 1)]  # type: ignore
        # 第一条记录是在 _current_time=0.0 时为 future_containers 初始化的记录，速度为0
        assert c1_speed_records[0].timestamp == 0.0
        assert c1_speed_records[0].speed == 0
        # 第二条记录是在容器开始运行时（time=1.0）的记录
        assert c1_speed_records[1].timestamp == 1.0
        assert c1_speed_records[1].speed == 1998

        # c2 的速度记录
        c2_speed_records = sample_numa_node._esm.records[(2, 1)]  # type: ignore
        # 第一条记录是在 _current_time=0.0 时为 future_containers 初始化的记录，速度为0
        assert c2_speed_records[0].timestamp == 0.0
        assert c2_speed_records[0].speed == 0
        # 第二条记录是在容器开始运行时（time=1.0）的记录
        assert c2_speed_records[1].timestamp == 1.0
        assert c2_speed_records[1].speed == 1998

    def test_numa_node_memory_bottleneck(self, sample_numa_node: NumaNode):
        """测试内存瓶颈影响"""
        # 先重置节点以初始化资源利用率记录
        sample_numa_node.reset()

        # 创建内存分配不足的容器
        # memory_bottleneck = memory_alloc / memory_req = 1024 / 2048 = 0.5
        c = Container(
            wf_id=1,
            fn_id=1,
            memory_req=2048,  # 需要2048MB
            memory_alloc=1024,  # 只分配1024MB，内存不足
            computation=2000,
            parallelism=2,
            submission_time=0.6,
            data_transfer_time=0.4,
        )
        sample_numa_node.on_container_creation(c)

        assert len(sample_numa_node._running_containers) == 1  # type: ignore

        # 由于内存瓶颈，执行速度应该下降
        # memory_bottleneck = 1024 / 2048 = 0.5
        # 理论速度 = single_core_speed * parallelism = 1000 * 2 = 2000
        # 实际速度 = 2000 * 0.5 = 1000
        # 预期执行时间 = 2000 / 1000 = 2.0 秒
        # 预期完成时间 = 1.0 + 2.0 = 3.0
        container = sample_numa_node._running_containers[0]  # type: ignore

        expected_completion_time = 1.0 + 2000 / 1000
        assert container.completion_time == pytest.approx(expected_completion_time, rel=1e-9)  # type: ignore

    def test_numa_node_complete_execution_simulation(self, sample_numa_node: NumaNode):
        """测试完整容器执行模拟（基于 serverless/README.md 中的执行速度模型例子）"""
        # 先重置节点以初始化资源利用率记录
        sample_numa_node.reset()

        # 创建一系列容器，模拟 serverless/README.md 中的执行速度模型
        container_to_submit = [
            Container(1, 1, 512, 512, 2585, 1, 0.6, 0.4),  # 容器A - 小计算量，低并行度 - creation_time=1.0
            Container(2, 1, 1024, 1536, 7171, 2, 0.6, 0.4),  # 容器B - 中等计算量，中等并行度 - creation_time=1.0
            Container(3, 1, 1024, 1024, 1142, 4, 1.6, 0.4),  # 容器C - 小计算量，高并行度 - creation_time=2.0
            Container(4, 1, 2048, 1536, 4200, 2, 2.6, 0.4),  # 容器D - 大计算量，内存不足 - creation_time=3.0
        ]

        completed_containers: list[Container] = []

        # 提交容器并完成提交前的所有容器
        while container_to_submit:
            next_container = container_to_submit[0]
            _, completion_time = sample_numa_node.get_earliest_finished()
            # 计算下一个容器的创建时间
            next_creation_time = (
                max(sample_numa_node._current_time, next_container.submission_time) + next_container.data_transfer_time  # type: ignore
            )
            if completion_time <= next_creation_time:
                # 有容器在下一个提交时间之前完成
                completed_container = sample_numa_node.on_container_completion()
                completed_containers.append(completed_container)
                continue
            c = container_to_submit.pop(0)
            sample_numa_node.on_container_creation(c)

        # 完成所有剩余的容器
        while True:
            index, _ = sample_numa_node.get_earliest_finished()
            if index is None:
                break
            completed_container = sample_numa_node.on_container_completion()
            completed_containers.append(completed_container)

        # 验证执行结果
        assert len(completed_containers) == 4, f"应该完成4个容器，实际完成{len(completed_containers)}个"

        # 按wf_id排序以便验证
        completed_containers.sort(key=lambda c: c.wf_id)

        # 验证每个容器的完成时间
        expected_completion_times = {1: 4.0, 2: 5.0, 3: 2.5, 4: 6.0}
        for container in completed_containers:
            expected_time = expected_completion_times[container.wf_id]
            assert container.completion_time == pytest.approx(  # type: ignore
                expected_time, rel=1e-9
            ), f"容器{container.wf_id}的完成时间应该为{expected_time}，实际为{container.completion_time}"

        # 验证资源利用率记录
        rum_records = sample_numa_node._rum.records  # type: ignore
        assert len(rum_records) == 8, f"应该有8条资源利用率记录，实际有{len(rum_records)}条"

        # 验证初始状态记录
        initial_record = rum_records[0]
        assert initial_record.timestamp == 0.0
        assert initial_record.cpu == 0.0
        assert initial_record.memory == 0.0
        assert initial_record.load == 0.0
        assert initial_record.total_parallelism == 0
        assert initial_record.free_memory == 8192

        # 验证最终状态记录（所有容器完成后的状态）
        final_record = rum_records[-1]
        assert final_record.timestamp == pytest.approx(6.0, rel=1e-9)  # type: ignore
        assert final_record.cpu == 0.0  # 所有容器完成，CPU利用率为0
        assert final_record.memory == 0.0  # 所有容器完成，内存利用率为0
        assert final_record.load == 0.0  # 所有容器完成，负载为0
        assert final_record.total_parallelism == 0  # 所有容器完成，并行度为0
        assert final_record.free_memory == 8192  # 所有容器完成，内存全部释放

        # 验证关键时间点的资源利用率记录
        # 时间点1.0：容器A和B启动
        record_at_1_0 = rum_records[1]
        assert record_at_1_0.timestamp == 1.0
        assert record_at_1_0.total_parallelism == 3  # 容器A并行度1 + 容器B并行度2
        assert record_at_1_0.free_memory == 6144  # 8192 - 512 - 1536 = 6144

        # 时间点2.0：容器C启动
        record_at_2_0 = rum_records[2]
        assert record_at_2_0.timestamp == 2.0
        assert record_at_2_0.total_parallelism == 7  # 1+2+4
        assert record_at_2_0.free_memory == 5120  # 8192 - 512 - 1536 - 1024 = 5120

        # 时间点2.5：容器C完成
        record_at_2_5 = rum_records[3]
        assert record_at_2_5.timestamp == 2.5
        assert record_at_2_5.total_parallelism == 3  # 容器C完成，剩下容器A和B，并行度1+2=3
        assert record_at_2_5.free_memory == 6144  # 容器C释放1024MB内存，8192 - 512 - 1536 = 6144

        # 时间点3.0：容器D启动
        record_at_3_0 = rum_records[4]
        assert record_at_3_0.timestamp == 3.0
        assert record_at_3_0.total_parallelism == 5  # 容器D启动，总并行度1+2+2=5
        assert record_at_3_0.free_memory == 4608  # 容器D分配1536MB，8192 - 512 - 1536 - 1536 = 4608

        # 时间点4.0：容器A完成
        record_at_4_0 = rum_records[5]
        assert record_at_4_0.timestamp == 4.0
        assert record_at_4_0.total_parallelism == 4  # 容器A完成，剩下容器B和D，并行度2+2=4
        assert record_at_4_0.free_memory == 5120  # 容器A释放512MB，8192 - 1536 - 1536 = 5120

        # 时间点5.0：容器B完成
        record_at_5_0 = rum_records[6]
        assert record_at_5_0.timestamp == 5.0
        assert record_at_5_0.total_parallelism == 2  # 容器B完成，只剩下容器D，并行度2
        assert record_at_5_0.free_memory == 6656  # 容器B释放1536MB，8192 - 1536 = 6656

        # 时间点6.0：容器D完成
        record_at_6_0 = rum_records[7]
        assert record_at_6_0.timestamp == 6.0
        assert record_at_6_0.total_parallelism == 0  # 无容器运行
        assert record_at_6_0.free_memory == 8192  # 所有内存释放
