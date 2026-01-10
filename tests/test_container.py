"""test_container.py

测试 src/faas_workflow_sim/container.py 中的 Container 类
"""

import pytest

from faas_workflow_sim.container import Container


class TestContainer:
    """测试 Container 类"""

    @pytest.fixture
    def sample_container(self) -> Container:
        """创建示例容器"""
        return Container(
            wf_id=1,
            fn_id=2,
            memory_req=512,
            memory_alloc=1024,
            computation=10000,
            parallelism=4,
            submission_time=0.0,
            data_transfer_time=0.5,
        )

    def test_container_initialization(self, sample_container: Container):
        """测试容器初始化"""
        assert sample_container.wf_id == 1
        assert sample_container.fn_id == 2
        assert sample_container.memory_req == 512
        assert sample_container.memory_alloc == 1024
        assert sample_container.computation == 10000
        assert sample_container.parallelism == 4
        assert sample_container.submission_time == 0.0
        assert sample_container.data_transfer_time == 0.5
        assert sample_container.remaining_computation == 10000  # 应该等于 computation

    def test_container_create_success(self, sample_container: Container):
        """测试容器创建成功"""
        sample_container.create(1.0)
        assert sample_container.creation_time == 1.0

    def test_container_create_invalid_time(self, sample_container: Container):
        """测试容器创建时间无效（早于提交时间+数据传输时间）"""
        with pytest.raises(ValueError, match="Creation time .* cannot be earlier than submission time"):
            sample_container.create(0.3)  # 早于 submission_time + data_transfer_time (0.5)

    def test_container_run_success(self, sample_container: Container):
        """测试容器运行成功"""
        sample_container.create(1.0)
        sample_container.run(2.0)
        assert sample_container.start_time == 2.0

    def test_container_run_invalid_time(self, sample_container: Container):
        """测试容器运行时间无效（早于创建时间）"""
        sample_container.create(1.0)
        with pytest.raises(ValueError, match="Start time .* cannot be earlier than creation time"):
            sample_container.run(0.5)  # 早于 creation_time (1.0)

    def test_container_finish_success(self, sample_container: Container):
        """测试容器完成成功"""
        sample_container.create(1.0)
        sample_container.run(2.0)
        sample_container.finish(3.0)
        assert sample_container.start_time == 2.0
        assert sample_container.completion_time == 3.0

    def test_container_finish_without_run(self, sample_container: Container):
        """测试容器未运行就完成"""
        sample_container.create(1.0)
        with pytest.raises(AttributeError, match="'Container' object has no attribute 'start_time'"):
            sample_container.finish(2.0)

    def test_container_finish_invalid_time(self, sample_container: Container):
        """测试容器完成时间无效（早于运行时间）"""
        sample_container.create(1.0)
        sample_container.run(2.0)
        with pytest.raises(ValueError, match="Finish time .* cannot be earlier than start time"):
            sample_container.finish(1.5)  # 早于 start_time (2.0)
