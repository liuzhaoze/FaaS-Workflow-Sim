"""test_server.py

测试 src/faas_workflow_sim/server.py 中的 Server 类
"""

import math

import pytest

from faas_workflow_sim.container import Container
from faas_workflow_sim.server import HOUR, Server


class TestServerInit:
    """测试 Server 类的初始化"""

    def test_init_basic(self):
        """测试基本初始化"""
        server = Server(
            server_id=0,
            hourly_rate=1.5,
            cold_start_latency=5.0,
            numa_node_count=2,
            numa_node_cpu=4,
            numa_node_memory=8192,
            single_core_speed=1000,
        )

        assert server.server_id == 0
        assert server.hourly_rate == 1.5
        assert server.cold_start_latency == 5.0
        assert len(server) == 2
        assert server.expiration_time == 0.0
        assert server.earliest_finished_nn_id is None
        assert server.earliest_finished_time == float("inf")
        assert server.latest_finished_nn_id is None
        assert server.latest_finished_time == float("-inf")

    def test_init_single_numa_node(self):
        """测试只有一个 NUMA 节点的服务器"""
        server = Server(
            server_id=1,
            hourly_rate=2.0,
            cold_start_latency=3.0,
            numa_node_count=1,
            numa_node_cpu=8,
            numa_node_memory=16384,
            single_core_speed=2000,
        )

        assert len(server) == 1
        assert server[0].node_id == 0
        assert server[0].cpu == 8
        assert server[0].memory == 16384
        assert server[0].single_core_speed == 2000

    def test_init_multiple_numa_nodes(self):
        """测试有多个 NUMA 节点的服务器"""
        server = Server(
            server_id=2,
            hourly_rate=3.0,
            cold_start_latency=10.0,
            numa_node_count=4,
            numa_node_cpu=16,
            numa_node_memory=32768,
            single_core_speed=3000,
        )

        assert len(server) == 4
        for i, nn in enumerate(server):
            assert nn.node_id == i
            assert nn.cpu == 16
            assert nn.memory == 32768
            assert nn.single_core_speed == 3000


class TestServerReset:
    """测试 Server 类的 reset 方法"""

    def test_reset_clears_server_state(self):
        """测试 reset 清除服务器状态"""
        server = Server(
            server_id=0,
            hourly_rate=1.5,
            cold_start_latency=5.0,
            numa_node_count=2,
            numa_node_cpu=4,
            numa_node_memory=8192,
            single_core_speed=1000,
        )

        # 修改服务器状态
        server.expiration_time = 100.0
        server.earliest_finished_nn_id = 0
        server.earliest_finished_time = 10.0
        server.latest_finished_nn_id = 1
        server.latest_finished_time = 20.0

        # 重置
        server.reset()

        # 验证服务器状态被重置
        assert server.expiration_time == 0.0
        assert server.earliest_finished_nn_id is None
        assert server.earliest_finished_time == float("inf")
        assert server.latest_finished_nn_id is None
        assert server.latest_finished_time == float("-inf")

    def test_reset_clears_numa_nodes(self):
        """测试 reset 清除所有 NUMA 节点状态"""
        server = Server(
            server_id=0,
            hourly_rate=1.5,
            cold_start_latency=5.0,
            numa_node_count=2,
            numa_node_cpu=4,
            numa_node_memory=8192,
            single_core_speed=1000,
        )

        # 先重置 NUMA 节点
        server[0].reset()

        # 在 NUMA 节点上创建容器
        container = Container(
            wf_id=0,
            fn_id=0,
            memory_req=512,
            memory_alloc=512,
            computation=1000,
            parallelism=1,
            submission_time=0.0,
            data_transfer_time=0.0,
        )
        server[0].on_container_creation(container)

        # 重置
        server.reset()

        # 验证 NUMA 节点状态被重置
        for nn in server:
            assert nn._current_time == 0.0  # type: ignore
            assert len(nn._running_containers) == 0  # type: ignore
            assert nn._waiting_containers.empty()  # type: ignore


class TestServerStartupAt:
    """测试 Server 类的 startup_at 方法"""

    def test_startup_before_expiration(self):
        """测试在租期内启动服务器，无冷启动时间"""
        server = Server(
            server_id=0,
            hourly_rate=1.5,
            cold_start_latency=5.0,
            numa_node_count=2,
            numa_node_cpu=4,
            numa_node_memory=8192,
            single_core_speed=1000,
        )

        # 设置服务器到期时间
        server.expiration_time = 100.0

        # 在租期内启动服务器
        cold_start_time = server.startup_at(50.0)
        assert cold_start_time == 0.0

    def test_startup_after_expiration(self):
        """测试在租期外启动服务器，有冷启动时间"""
        server = Server(
            server_id=0,
            hourly_rate=1.5,
            cold_start_latency=5.0,
            numa_node_count=2,
            numa_node_cpu=4,
            numa_node_memory=8192,
            single_core_speed=1000,
        )

        # 设置服务器到期时间
        server.expiration_time = 100.0

        # 在租期外启动服务器
        cold_start_time = server.startup_at(150.0)
        assert cold_start_time == 5.0

    def test_startup_at_expiration_boundary(self):
        """测试在到期时间边界启动服务器"""
        server = Server(
            server_id=0,
            hourly_rate=1.5,
            cold_start_latency=5.0,
            numa_node_count=2,
            numa_node_cpu=4,
            numa_node_memory=8192,
            single_core_speed=1000,
        )

        # 设置服务器到期时间
        server.expiration_time = 100.0

        # 在到期时间之前启动服务器（无冷启动）
        cold_start_time = server.startup_at(99.9)
        assert cold_start_time == 0.0

        # 在到期时间启动服务器（有冷启动）
        cold_start_time = server.startup_at(100.0)
        assert cold_start_time == 5.0

    def test_startup_initial_state(self):
        """测试初始状态下启动服务器（expiration_time=0）"""
        server = Server(
            server_id=0,
            hourly_rate=1.5,
            cold_start_latency=5.0,
            numa_node_count=2,
            numa_node_cpu=4,
            numa_node_memory=8192,
            single_core_speed=1000,
        )

        # 初始状态下启动服务器（有冷启动）
        cold_start_time = server.startup_at(0.0)
        assert cold_start_time == 5.0

    def test_startup_updates_numa_node_current_time(self):
        """测试服务器启动时会更新所有 NUMA 节点的 _current_time"""
        server = Server(
            server_id=0,
            hourly_rate=1.5,
            cold_start_latency=10.0,
            numa_node_count=3,
            numa_node_cpu=4,
            numa_node_memory=8192,
            single_core_speed=1000,
        )

        # 初始状态下，所有 NUMA 节点的 _current_time 应该是 0.0
        for nn in server:
            assert nn._current_time == 0.0  # type: ignore

        # 在租期外启动服务器（触发冷启动）
        startup_time = 50.0
        cold_start_latency = server.startup_at(startup_time)

        assert cold_start_latency == 10.0

        # 验证所有 NUMA 节点的 _current_time 被更新到 startup_time + cold_start_latency
        expected_current_time = startup_time + 10.0
        for nn in server:
            assert nn._current_time == expected_current_time  # type: ignore

    def test_startup_within_lease_does_not_update_current_time(self):
        """测试在租期内启动服务器（无冷启动）时，_current_time 不会被更新"""
        server = Server(
            server_id=0,
            hourly_rate=1.5,
            cold_start_latency=5.0,
            numa_node_count=2,
            numa_node_cpu=4,
            numa_node_memory=8192,
            single_core_speed=1000,
        )

        # 设置服务器到期时间
        server.expiration_time = 100.0

        # 先手动设置一个 _current_time
        server[0]._current_time = 50.0  # type: ignore
        server[1]._current_time = 50.0  # type: ignore

        # 在租期内启动服务器（无冷启动）
        cold_start_time = server.startup_at(60.0)
        assert cold_start_time == 0.0

        # 验证 _current_time 没有被修改
        assert server[0]._current_time == 50.0  # type: ignore
        assert server[1]._current_time == 50.0  # type: ignore

    def test_startup_after_lease_updates_current_time(self):
        """测试在租期外启动服务器时，_current_time 被正确更新"""
        server = Server(
            server_id=0,
            hourly_rate=1.5,
            cold_start_latency=8.0,
            numa_node_count=2,
            numa_node_cpu=4,
            numa_node_memory=8192,
            single_core_speed=1000,
        )

        # 设置服务器到期时间为较早的时间
        server.expiration_time = 100.0

        # 先设置一个 _current_time
        server[0]._current_time = 50.0  # type: ignore
        server[1]._current_time = 50.0  # type: ignore

        # 在租期外启动服务器（有冷启动）
        startup_time = 150.0
        cold_start_latency = server.startup_at(startup_time)
        assert cold_start_latency == 8.0

        # 验证 _current_time 被更新到 startup_time + cold_start_latency
        expected_current_time = startup_time + 8.0
        assert server[0]._current_time == expected_current_time  # type: ignore
        assert server[1]._current_time == expected_current_time  # type: ignore


class TestServerOnContainerCreation:
    """测试 Server 类的 on_container_creation 方法"""

    def test_create_container_basic(self):
        """测试基本的容器创建"""
        server = Server(
            server_id=0,
            hourly_rate=1.5,
            cold_start_latency=5.0,
            numa_node_count=2,
            numa_node_cpu=4,
            numa_node_memory=8192,
            single_core_speed=1000,
        )

        # 先重置服务器以初始化 NUMA 节点
        server.reset()

        container = Container(
            wf_id=0,
            fn_id=0,
            memory_req=512,
            memory_alloc=512,
            computation=1000,
            parallelism=1,
            submission_time=0.0,
            data_transfer_time=0.0,
        )

        rent = server.on_container_creation(numa_node_id=0, c=container)

        # 验证租金计算：容器在 1 秒完成，需要续租 1 小时
        assert rent == pytest.approx(1.5)  # type: ignore
        # 验证服务器到期时间被更新
        assert server.expiration_time == pytest.approx(HOUR)  # type: ignore

    def test_create_container_updates_earliest_latest(self):
        """测试容器创建后更新最早和最晚完成时间"""
        server = Server(
            server_id=0,
            hourly_rate=1.5,
            cold_start_latency=5.0,
            numa_node_count=2,
            numa_node_cpu=4,
            numa_node_memory=8192,
            single_core_speed=1000,
        )

        # 先重置服务器
        server.reset()

        container = Container(
            wf_id=0,
            fn_id=0,
            memory_req=512,
            memory_alloc=512,
            computation=1000,
            parallelism=1,
            submission_time=0.0,
            data_transfer_time=0.0,
        )

        rent = server.on_container_creation(numa_node_id=0, c=container)

        # 验证租金：容器执行 1000/1000=1.0 秒，需要续租 ceil(1.0/3600)=1 小时
        assert rent == pytest.approx(1.5)  # type: ignore
        assert server.expiration_time == pytest.approx(3600.0)  # type: ignore

        # 验证最早和最晚完成时间
        # 单个容器，完成时间 = 1000/1000 = 1.0 秒
        expected_completion_time = 1.0
        assert server.earliest_finished_nn_id == 0
        assert server.latest_finished_nn_id == 0
        assert server.earliest_finished_time == pytest.approx(expected_completion_time)  # type: ignore
        assert server.latest_finished_time == pytest.approx(expected_completion_time)  # type: ignore

    def test_create_container_on_different_numa_nodes(self):
        """测试在不同 NUMA 节点上创建容器"""
        server = Server(
            server_id=0,
            hourly_rate=1.5,
            cold_start_latency=5.0,
            numa_node_count=2,
            numa_node_cpu=4,
            numa_node_memory=8192,
            single_core_speed=1000,
        )

        # 先重置服务器
        server.reset()

        # 在 NUMA 节点 0 上创建容器
        container1 = Container(
            wf_id=0,
            fn_id=0,
            memory_req=512,
            memory_alloc=512,
            computation=1000,
            parallelism=1,
            submission_time=0.0,
            data_transfer_time=0.0,
        )
        rent1 = server.on_container_creation(numa_node_id=0, c=container1)

        # 验证第一次创建的租金：容器1 完成 1000/1000=1.0 秒，需要续租 1 小时
        assert rent1 == pytest.approx(1.5)  # type: ignore
        assert server.expiration_time == pytest.approx(3600.0)  # type: ignore

        # 在 NUMA 节点 1 上创建容器
        container2 = Container(
            wf_id=0,
            fn_id=1,
            memory_req=512,
            memory_alloc=512,
            computation=2000,
            parallelism=1,
            submission_time=0.0,
            data_transfer_time=0.0,
        )
        rent2 = server.on_container_creation(numa_node_id=1, c=container2)

        # 验证第二次创建的租金：容器2 完成 2000/1000=2.0 秒，但小于到期时间 3600，无需续租
        assert rent2 == 0.0
        assert server.expiration_time == pytest.approx(3600.0)  # type: ignore

        # 验证两个 NUMA 节点都有运行中的容器
        assert len(server[0]._running_containers) == 1  # type: ignore
        assert len(server[1]._running_containers) == 1  # type: ignore

        # 验证最早和最晚完成时间
        # 容器1完成时间 = 1000/1000 = 1.0 秒
        # 容器2完成时间 = 2000/1000 = 2.0 秒
        assert server.earliest_finished_nn_id == 0
        assert server.latest_finished_nn_id == 1
        assert server.earliest_finished_time == pytest.approx(1.0)  # type: ignore
        assert server.latest_finished_time == pytest.approx(2.0)  # type: ignore

    def test_create_container_after_lease_expiration(self):
        """测试在服务器租期到期后创建容器，验证重新租赁"""
        server = Server(
            server_id=0,
            hourly_rate=1.5,
            cold_start_latency=5.0,
            numa_node_count=2,
            numa_node_cpu=4,
            numa_node_memory=8192,
            single_core_speed=1000,
        )

        # 先重置服务器
        server.reset()

        # 第一个容器：函数在 0.0 秒提交
        container1 = Container(
            wf_id=0,
            fn_id=0,
            memory_req=512,
            memory_alloc=512,
            computation=1000,
            parallelism=1,
            submission_time=0.0,
            data_transfer_time=0.0,
        )
        rent1 = server.on_container_creation(numa_node_id=0, c=container1)

        # 验证第一次租赁：容器1 完成 1000/1000=1.0 秒，需要续租 1 小时
        assert rent1 == pytest.approx(1.5)  # type: ignore
        assert server.expiration_time == pytest.approx(3600.0)  # type: ignore

        # 第二个容器：函数在 3600 秒后提交（正好是第一个租期到期时间）
        container2 = Container(
            wf_id=0,
            fn_id=1,
            memory_req=512,
            memory_alloc=512,
            computation=2000,
            parallelism=1,
            submission_time=3650.0,
            data_transfer_time=0.0,
        )
        rent2 = server.on_container_creation(numa_node_id=1, c=container2)

        # 验证第二次租赁：容器2 完成 2000/1000=2.0 秒
        # 因为提交时间 3650 > 到期时间 3600，所以从提交时间开始续租
        # 续租时长 = ceil(3652.0 - 3650.0) / 3600) = 1 小时
        assert rent2 == pytest.approx(1.5)  # type: ignore
        assert server.expiration_time == pytest.approx(7250.0)  # type: ignore

        # 验证两个 NUMA 节点都有运行中的容器
        assert len(server[0]._running_containers) == 1  # type: ignore
        assert len(server[1]._running_containers) == 1  # type: ignore

        # 验证最早和最晚完成时间
        # 容器1完成时间 = 0.0 + 1.0 = 1.0 秒
        # 容器2完成时间 = 3650.0 + 2.0 = 3652.0 秒
        assert server.earliest_finished_nn_id == 0
        assert server.latest_finished_nn_id == 1
        assert server.earliest_finished_time == pytest.approx(1.0)  # type: ignore
        assert server.latest_finished_time == pytest.approx(3652.0)  # type: ignore

    def test_container_waiting_queue_with_lease_renewal(self):
        """测试容器在等待队列中，完成后触发续租"""
        server = Server(
            server_id=0,
            hourly_rate=1.5,
            cold_start_latency=5.0,
            numa_node_count=2,
            numa_node_cpu=4,
            numa_node_memory=8192,
            single_core_speed=1000,
        )

        # 先重置服务器
        server.reset()

        # 第一个容器：占用所有内存，在 0.0 秒提交，执行很长时间到接近租期到期
        container1 = Container(
            wf_id=0,
            fn_id=0,
            memory_req=8192,
            memory_alloc=8192,
            computation=3590000,
            parallelism=1,
            submission_time=0.0,
            data_transfer_time=0.0,
        )
        rent1 = server.on_container_creation(numa_node_id=0, c=container1)

        # 验证第一次创建：容器1 完成 3590000/1000=3590 秒，需要续租 1 小时
        assert rent1 == pytest.approx(1.5)  # type: ignore
        assert server.expiration_time == pytest.approx(3600.0)  # type: ignore

        # 第二个容器：内存不足，进入等待队列，在 0.0 秒提交
        container2 = Container(
            wf_id=0,
            fn_id=1,
            memory_req=512,
            memory_alloc=512,
            computation=15000,
            parallelism=1,
            submission_time=0.0,
            data_transfer_time=0.0,
        )
        rent2 = server.on_container_creation(numa_node_id=0, c=container2)

        # 验证第二次创建：内存不足，进入等待队列，无租金（因为容器1的完成时间3590已经在租期内）
        assert rent2 == 0.0
        assert server.expiration_time == pytest.approx(3600.0)  # type: ignore
        # 验证容器在等待队列中
        assert not server[0]._waiting_containers.empty()  # type: ignore

        # 完成第一个容器
        completion_rent1, completed1 = server.on_container_completion()

        # 验证第一个容器完成
        assert completed1.wf_id == 0
        assert completed1.fn_id == 0
        assert completed1.completion_time == pytest.approx(3590.0)  # type: ignore

        # 验证第二次续租：容器2 从等待队列取出，在 3590.0 秒开始执行
        # 执行时间 = 15000/1000 = 15 秒，完成时间 = 3590.0 + 15.0 = 3605.0 秒
        # 3605.0 > 3600.0，需要续租到 7200.0
        assert completion_rent1 == pytest.approx(1.5)  # type: ignore
        assert server.expiration_time == pytest.approx(7200.0)  # type: ignore

        # 完成第二个容器
        completion_rent2, completed2 = server.on_container_completion()

        # 验证第二个容器完成
        assert completed2.wf_id == 0
        assert completed2.fn_id == 1
        assert completed2.completion_time == pytest.approx(3605.0)  # type: ignore

        # 所有容器已完成，无需续租，验证租金和到期时间
        assert completion_rent2 == 0.0
        assert server.expiration_time == pytest.approx(7200.0)  # type: ignore

        # 验证没有容器了
        assert server.earliest_finished_nn_id is None
        assert server.latest_finished_nn_id is None


class TestServerOnContainerCompletion:
    """测试 Server 类的 on_container_completion 方法"""

    def test_complete_container_basic(self):
        """测试基本的容器完成"""
        server = Server(
            server_id=0,
            hourly_rate=1.5,
            cold_start_latency=5.0,
            numa_node_count=2,
            numa_node_cpu=4,
            numa_node_memory=8192,
            single_core_speed=1000,
        )

        # 先重置服务器
        server.reset()

        # 创建容器
        container = Container(
            wf_id=0,
            fn_id=0,
            memory_req=512,
            memory_alloc=512,
            computation=1000,
            parallelism=1,
            submission_time=0.0,
            data_transfer_time=0.0,
        )
        rent = server.on_container_creation(numa_node_id=0, c=container)

        # 验证创建时的租金：容器完成 1000/1000=1.0 秒，需要续租 1 小时
        assert rent == pytest.approx(1.5)  # type: ignore
        assert server.expiration_time == pytest.approx(3600.0)  # type: ignore

        # 完成容器
        completion_rent, completed_container = server.on_container_completion()

        # 验证返回的容器
        assert completed_container.wf_id == 0
        assert completed_container.fn_id == 0
        # 验证完成时间 = 1000/1000 = 1.0 秒
        assert completed_container.completion_time == pytest.approx(1.0)  # type: ignore

        # 验证完成时的租金（因为只有一个容器且已完成，不需要续租）
        assert completion_rent == 0.0

    def test_complete_container_when_no_containers(self):
        """测试没有运行中的容器时完成容器"""
        server = Server(
            server_id=0,
            hourly_rate=1.5,
            cold_start_latency=5.0,
            numa_node_count=2,
            numa_node_cpu=4,
            numa_node_memory=8192,
            single_core_speed=1000,
        )

        # 尝试完成容器（没有运行中的容器）
        with pytest.raises(RuntimeError, match="No running containers on the server"):
            server.on_container_completion()

    def test_complete_container_updates_earliest_latest(self):
        """测试容器完成后更新最早和最晚完成时间"""
        server = Server(
            server_id=0,
            hourly_rate=1.5,
            cold_start_latency=5.0,
            numa_node_count=2,
            numa_node_cpu=4,
            numa_node_memory=8192,
            single_core_speed=1000,
        )

        # 先重置服务器
        server.reset()

        # 创建容器
        container = Container(
            wf_id=0,
            fn_id=0,
            memory_req=512,
            memory_alloc=512,
            computation=1000,
            parallelism=1,
            submission_time=0.0,
            data_transfer_time=0.0,
        )
        server.on_container_creation(numa_node_id=0, c=container)

        # 完成容器
        server.on_container_completion()

        # 验证最早和最晚完成时间被更新（因为没有容器了，应该为 None）
        assert server.earliest_finished_nn_id is None
        assert server.latest_finished_nn_id is None


class TestServerRenewLease:
    """测试 Server 类的租期续租逻辑"""

    def test_renew_lease_when_covered(self):
        """测试租期已覆盖所有容器执行，无需续租"""
        server = Server(
            server_id=0,
            hourly_rate=1.5,
            cold_start_latency=5.0,
            numa_node_count=2,
            numa_node_cpu=4,
            numa_node_memory=8192,
            single_core_speed=1000,
        )

        # 设置服务器到期时间为较晚的时间
        server.expiration_time = 1000.0
        server.latest_finished_time = 100.0

        # 租期已覆盖，无需续租
        rent = server._renew_lease_at(50.0)  # type: ignore
        assert rent == 0.0
        assert server.expiration_time == 1000.0

    def test_renew_lease_when_not_covered(self):
        """测试租期未覆盖，需要续租"""
        server = Server(
            server_id=0,
            hourly_rate=1.5,
            cold_start_latency=5.0,
            numa_node_count=2,
            numa_node_cpu=4,
            numa_node_memory=8192,
            single_core_speed=1000,
        )

        # 设置服务器到期时间较早，最晚完成时间较晚
        server.expiration_time = 100.0
        server.latest_finished_time = 5000.0

        # 需要续租
        rent = server._renew_lease_at(50.0)  # type: ignore
        expected_rent_hours = math.ceil((5000.0 - 100.0) / HOUR)
        expected_rent = expected_rent_hours * 1.5

        assert rent == pytest.approx(expected_rent)  # type: ignore
        assert server.expiration_time == pytest.approx(100.0 + expected_rent_hours * HOUR)  # type: ignore

    def test_renew_lease_from_submission_time(self):
        """测试从函数提交时间开始续租"""
        server = Server(
            server_id=0,
            hourly_rate=1.5,
            cold_start_latency=5.0,
            numa_node_count=2,
            numa_node_cpu=4,
            numa_node_memory=8192,
            single_core_speed=1000,
        )

        # 服务器已过期，函数提交时间晚于到期时间
        server.expiration_time = 100.0
        server.latest_finished_time = 4000.0

        # 从函数提交时间 200.0 开始续租
        rent = server._renew_lease_at(200.0)  # type: ignore
        expected_rent_hours = math.ceil((4000.0 - 200.0) / HOUR)
        expected_rent = expected_rent_hours * 1.5

        assert rent == pytest.approx(expected_rent)  # type: ignore
        assert server.expiration_time == pytest.approx(200.0 + expected_rent_hours * HOUR)  # type: ignore

    def test_renew_lease_rounding(self):
        """测试续租时间的向上取整"""
        server = Server(
            server_id=0,
            hourly_rate=1.0,
            cold_start_latency=5.0,
            numa_node_count=2,
            numa_node_cpu=4,
            numa_node_memory=8192,
            single_core_speed=1000,
        )

        # 需要续租 1.1 小时 -> 向上取整为 2 小时
        server.expiration_time = 0.0
        server.latest_finished_time = 1.1 * HOUR

        rent = server._renew_lease_at(0.0)  # type: ignore
        expected_rent = 2.0 * 1.0  # 2 小时 * 1.0/小时

        assert rent == pytest.approx(expected_rent)  # type: ignore
        assert server.expiration_time == pytest.approx(2.0 * HOUR)  # type: ignore


class TestServerEarliestLatestUpdate:
    """测试 Server 类的最早/最晚完成时间更新逻辑"""

    def test_update_earliest_latest_basic_comparison(self):
        """测试基本的比较更新逻辑（不触发完整遍历）"""
        server = Server(
            server_id=0,
            hourly_rate=1.5,
            cold_start_latency=5.0,
            numa_node_count=3,
            numa_node_cpu=4,
            numa_node_memory=8192,
            single_core_speed=1000,
        )

        # 先重置服务器
        server.reset()

        # 在节点 0 上创建容器
        container1 = Container(
            wf_id=0,
            fn_id=0,
            memory_req=512,
            memory_alloc=512,
            computation=1000,
            parallelism=1,
            submission_time=0.0,
            data_transfer_time=0.0,
        )
        server.on_container_creation(numa_node_id=0, c=container1)

        # 验证最早和最晚都是节点 0
        assert server.earliest_finished_nn_id == 0
        assert server.latest_finished_nn_id == 0
        initial_earliest_time = server.earliest_finished_time
        initial_latest_time = server.latest_finished_time

        # 在节点 1 上创建更早完成的容器（计算量更小）
        container2 = Container(
            wf_id=0,
            fn_id=1,
            memory_req=512,
            memory_alloc=512,
            computation=500,
            parallelism=1,
            submission_time=0.0,
            data_transfer_time=0.0,
        )
        server.on_container_creation(numa_node_id=1, c=container2)

        # 验证最早完成时间被更新为节点 1（比较更新）
        assert server.earliest_finished_nn_id == 1
        assert server.earliest_finished_time < initial_earliest_time
        # 最晚完成时间应该仍然是节点 0
        assert server.latest_finished_nn_id == 0
        assert server.latest_finished_time == initial_latest_time

    def test_update_earliest_latest_triggers_scan(self):
        """测试触发完整遍历的场景"""
        server = Server(
            server_id=0,
            hourly_rate=1.5,
            cold_start_latency=5.0,
            numa_node_count=2,
            numa_node_cpu=4,
            numa_node_memory=8192,
            single_core_speed=1000,
        )

        # 先重置服务器
        server.reset()

        # 在节点 0 上创建小计算量容器（会最早完成）
        container1 = Container(
            wf_id=0,
            fn_id=0,
            memory_req=512,
            memory_alloc=512,
            computation=500,
            parallelism=1,
            submission_time=0.0,
            data_transfer_time=0.0,
        )
        server.on_container_creation(numa_node_id=0, c=container1)

        # 在节点 1 上创建大计算量容器（会最晚完成）
        container2 = Container(
            wf_id=0,
            fn_id=1,
            memory_req=512,
            memory_alloc=512,
            computation=2000,
            parallelism=1,
            submission_time=0.0,
            data_transfer_time=0.0,
        )
        server.on_container_creation(numa_node_id=1, c=container2)

        # 验证初始状态：节点 0 最早，节点 1 最晚
        assert server.earliest_finished_nn_id == 0
        assert server.latest_finished_nn_id == 1

        # 完成节点 0 的容器
        server.on_container_completion()

        # 验证：现在应该只有节点 1 的容器了
        assert server.earliest_finished_nn_id == 1
        assert server.latest_finished_nn_id == 1
        assert server.earliest_finished_time == server.latest_finished_time

    def test_update_earliest_latest_multiple_nodes(self):
        """测试多个 NUMA 节点的完整工作流"""
        server = Server(
            server_id=0,
            hourly_rate=1.5,
            cold_start_latency=5.0,
            numa_node_count=3,
            numa_node_cpu=4,
            numa_node_memory=8192,
            single_core_speed=1000,
        )

        # 先重置服务器
        server.reset()

        # 在三个节点上创建不同计算量的容器
        containers = [
            Container(
                wf_id=0,
                fn_id=i,
                memory_req=512,
                memory_alloc=512,
                computation=1000 * (i + 1),
                parallelism=1,
                submission_time=0.0,
                data_transfer_time=0.0,
            )
            for i in range(3)
        ]

        # 创建容器：节点 0 (1000), 节点 1 (2000), 节点 2 (3000)
        for i in range(3):
            server.on_container_creation(numa_node_id=i, c=containers[i])

        # 验证最早是节点 0，最晚是节点 2
        assert server.earliest_finished_nn_id == 0
        assert server.latest_finished_nn_id == 2
        assert server.earliest_finished_time < server.latest_finished_time

        # 完成最早的容器（节点 0，在 1.0 秒完成）
        _, completed = server.on_container_completion()
        assert completed.fn_id == 0

        # 验证：现在最早是节点 1（在 2.0 秒完成），最晚仍是节点 2（在 3.0 秒完成）
        assert server.earliest_finished_nn_id == 1
        assert server.latest_finished_nn_id == 2

        # 完成次早的容器（节点 1，在 2.0 秒完成）
        _, completed = server.on_container_completion()
        assert completed.fn_id == 1

        # 验证：现在只有节点 2 的容器
        assert server.earliest_finished_nn_id == 2
        assert server.latest_finished_nn_id == 2

        # 完成最后的容器（节点 2）
        _, completed = server.on_container_completion()
        assert completed.fn_id == 2

        # 验证：所有容器都已完成
        assert server.earliest_finished_nn_id is None
        assert server.latest_finished_nn_id is None


class TestServerIntegration:
    """集成测试：测试多个容器创建和完成的完整流程"""

    def test_multiple_containers_workflow(self):
        """测试多个容器的完整工作流程"""
        server = Server(
            server_id=0,
            hourly_rate=1.5,
            cold_start_latency=5.0,
            numa_node_count=2,
            numa_node_cpu=4,
            numa_node_memory=8192,
            single_core_speed=1000,
        )

        # 先重置服务器
        server.reset()

        # 创建第一个容器
        container1 = Container(
            wf_id=0,
            fn_id=0,
            memory_req=512,
            memory_alloc=512,
            computation=1000,
            parallelism=1,
            submission_time=0.0,
            data_transfer_time=0.0,
        )
        rent1 = server.on_container_creation(numa_node_id=0, c=container1)
        assert rent1 > 0

        # 创建第二个容器
        container2 = Container(
            wf_id=0,
            fn_id=1,
            memory_req=512,
            memory_alloc=512,
            computation=2000,
            parallelism=1,
            submission_time=0.0,
            data_transfer_time=0.0,
        )
        rent2 = server.on_container_creation(numa_node_id=1, c=container2)
        # 第二个容器不需要额外租金（因为第一个容器提交时已经租用了足够长的时间）
        assert rent2 == 0

        # 完成第一个容器
        _, completed1 = server.on_container_completion()
        assert completed1.wf_id == 0
        assert completed1.fn_id == 0

        # 完成第二个容器
        _, completed2 = server.on_container_completion()
        assert completed2.wf_id == 0
        assert completed2.fn_id == 1

        # 验证没有容器了
        assert server.earliest_finished_nn_id is None
        assert server.latest_finished_nn_id is None
