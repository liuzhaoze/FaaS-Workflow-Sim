"""test_cluster.py

测试 src/faas_workflow_sim/cluster.py 中的 Cluster 类
"""

from pathlib import Path

import pytest
import yaml

from faas_workflow_sim.cluster import Cluster
from faas_workflow_sim.config import ClusterConfig
from faas_workflow_sim.container import Container


@pytest.fixture
def cluster_config() -> ClusterConfig:
    """从 YAML 文件加载集群配置"""
    config_path = Path(__file__).parent / "data" / "cluster_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)
    return ClusterConfig(**config_data)


@pytest.fixture
def cluster(cluster_config: ClusterConfig) -> Cluster:
    """创建集群实例"""
    return Cluster(cluster_config)


@pytest.fixture
def sample_container() -> Container:
    """创建示例容器"""
    return Container(
        wf_id=0,
        fn_id=0,
        memory_req=1024,
        memory_alloc=2048,
        computation=1000000000,  # 1e9 次计算操作
        parallelism=1,
        submission_time=0.0,
        data_transfer_time=0.0,
    )


class TestClusterInitialization:
    """测试集群初始化"""

    def test_initialization(self, cluster: Cluster, cluster_config: ClusterConfig):
        """测试集群初始化是否正确设置配置参数"""
        assert cluster.memory_bandwidth == cluster_config.memory_bandwidth
        assert cluster.numa_bandwidth == cluster_config.numa_bandwidth
        assert cluster.network_bandwidth == cluster_config.network_bandwidth
        assert cluster.single_core_speed == cluster_config.single_core_speed

    def test_servers_created(self, cluster: Cluster, cluster_config: ClusterConfig):
        """测试是否正确创建了所有服务器"""
        # 配置文件中定义了 6 种服务器类型，每种 1 台
        assert len(cluster) == len(cluster_config.servers)

        for server_config in cluster_config.servers:
            assert server_config.name in cluster
            servers = cluster[server_config.name]
            assert len(servers) == server_config.count

            # 验证每台服务器的属性
            for server in servers:
                assert server.server_id >= 0
                assert server.hourly_rate == server_config.hourly_rate
                assert server.cold_start_latency == server_config.cold_start_latency

    def test_initial_state(self, cluster: Cluster):
        """测试集群初始状态变量"""
        assert cluster.earliest_finished_srv_name is None
        assert cluster.earliest_finished_srv_id is None
        assert cluster.earliest_finished_time == float("inf")


class TestClusterReset:
    """测试集群重置功能"""

    def test_reset_clears_cluster_state(self, cluster: Cluster, sample_container: Container):
        """测试 reset 是否正确清除集群状态变量"""
        # 先初始化集群
        cluster.reset()

        # 进行一些操作来改变状态
        cluster.on_container_creation("c5.large", 0, 0, sample_container)

        # 重置
        cluster.reset()

        # 验证状态变量被重置
        assert cluster.earliest_finished_srv_name is None
        assert cluster.earliest_finished_srv_id is None
        assert cluster.earliest_finished_time == float("inf")

    def test_reset_resets_all_servers(self, cluster: Cluster, sample_container: Container):
        """测试 reset 是否重置所有服务器状态"""
        # 先初始化集群
        cluster.reset()

        # 在某台服务器上创建容器
        cluster.on_container_creation("c5.large", 0, 0, sample_container)

        # 获取服务器状态
        server = cluster["c5.large"][0]
        assert server.expiration_time > 0.0

        # 重置
        cluster.reset()

        # 验证服务器被重置
        assert server.expiration_time == 0.0
        assert server.earliest_finished_nn_id is None
        assert server.earliest_finished_time == float("inf")


class TestClusterContainerCreation:
    """测试容器创建功能"""

    def test_on_container_creation_basic(self, cluster: Cluster, sample_container: Container):
        """测试基本容器创建"""
        cluster.reset()
        rent = cluster.on_container_creation("c5.large", 0, 0, sample_container)

        # 验证返回租金
        assert rent > 0  # 首次创建需要租用服务器

        # 验证集群状态更新
        server = cluster["c5.large"][0]
        assert cluster.earliest_finished_srv_name == "c5.large"
        assert cluster.earliest_finished_srv_id == 0
        assert cluster.earliest_finished_time == server.earliest_finished_time

    def test_on_container_creation_multiple_servers(self, cluster: Cluster):
        """测试在多台服务器上创建容器"""
        cluster.reset()
        containers = [
            Container(
                wf_id=i,
                fn_id=i,
                memory_req=1024,
                memory_alloc=2048,
                computation=1000000000,
                parallelism=1,
                submission_time=float(i),
                data_transfer_time=0.0,
            )
            for i in range(3)
        ]

        # 在三台不同的服务器上创建容器
        cluster.on_container_creation("c5.large", 0, 0, containers[0])
        cluster.on_container_creation("c5.xlarge", 0, 0, containers[1])
        cluster.on_container_creation("c5.2xlarge", 0, 0, containers[2])

        # 验证最早完成时间
        assert cluster.earliest_finished_srv_name == "c5.large"

    def test_on_container_creation_same_server_different_numa(self, cluster: Cluster):
        """测试在同一服务器的不同 NUMA 节点上创建容器"""
        cluster.reset()
        c5_2xlarge_servers = cluster["c5.2xlarge"]
        server = c5_2xlarge_servers[0]

        # 验证该服务器有 2 个 NUMA 节点
        assert len(server) == 2

        containers = [
            Container(
                wf_id=i,
                fn_id=i,
                memory_req=1024,
                memory_alloc=2048,
                computation=1000000000,
                parallelism=1,
                submission_time=float(i) * 0.1,
                data_transfer_time=0.0,
            )
            for i in range(2)
        ]

        # 在两个 NUMA 节点上创建容器
        cluster.on_container_creation("c5.2xlarge", 0, 0, containers[0])
        cluster.on_container_creation("c5.2xlarge", 0, 1, containers[1])

        # 验证最早完成时间已更新
        assert cluster.earliest_finished_srv_name == "c5.2xlarge"

    def test_on_container_creation_updates_earliest_time(self, cluster: Cluster):
        """测试容器创建更新最早完成时间"""
        cluster.reset()
        containers = [
            Container(
                wf_id=i,
                fn_id=i,
                memory_req=1024,
                memory_alloc=2048,
                computation=500000000 * (i + 1),  # 不同的计算量
                parallelism=1,
                submission_time=0.0,
                data_transfer_time=0.0,
            )
            for i in range(2)
        ]

        # 创建第一个容器（计算量小，完成早）
        cluster.on_container_creation("c5.large", 0, 0, containers[0])
        earliest_time_1 = cluster.earliest_finished_time

        # 创建第二个容器（计算量大，完成晚）
        cluster.on_container_creation("c5.xlarge", 0, 0, containers[1])
        earliest_time_2 = cluster.earliest_finished_time

        # 最早完成时间应该不变（仍然是第一个容器）
        assert earliest_time_1 == earliest_time_2


class TestClusterContainerCompletion:
    """测试容器完成功能"""

    def test_on_container_completion_basic(self, cluster: Cluster, sample_container: Container):
        """测试基本容器完成"""
        cluster.reset()
        # 创建容器
        cluster.on_container_creation("c5.large", 0, 0, sample_container)

        # 完成容器
        rent, completed_container = cluster.on_container_completion()

        # 验证返回值
        assert isinstance(rent, float)
        assert isinstance(completed_container, Container)
        assert completed_container == sample_container

    def test_on_container_completion_no_running_containers(self, cluster: Cluster):
        """测试没有运行容器时完成操作抛出异常"""
        with pytest.raises(RuntimeError, match="No running containers in the cluster"):
            cluster.on_container_completion()

    def test_on_container_completion_updates_state(self, cluster: Cluster, sample_container: Container):
        """测试容器完成更新集群状态"""
        cluster.reset()
        # 创建容器
        cluster.on_container_creation("c5.large", 0, 0, sample_container)

        # 完成容器
        cluster.on_container_completion()

        # 验证状态更新：完成后所有容器都已运行完成，最早完成时间应为 inf
        assert cluster.earliest_finished_time == float("inf")

    def test_on_container_completion_multiple_containers(self, cluster: Cluster):
        """测试完成多个容器"""
        cluster.reset()
        containers = [
            Container(
                wf_id=i,
                fn_id=i,
                memory_req=1024,
                memory_alloc=2048,
                computation=1000000000,
                parallelism=1,
                submission_time=0.0,
                data_transfer_time=0.0,
            )
            for i in range(3)
        ]

        # 在三台服务器上创建容器
        cluster.on_container_creation("c5.large", 0, 0, containers[0])
        cluster.on_container_creation("c5.xlarge", 0, 0, containers[1])
        cluster.on_container_creation("c5.2xlarge", 0, 0, containers[2])

        # 完成最早的容器
        rent1, c1 = cluster.on_container_completion()
        assert c1 in containers
        assert rent1 >= 0

        # 完成第二个容器
        rent2, c2 = cluster.on_container_completion()
        assert c2 in containers
        assert c2 != c1
        assert rent2 >= 0


class TestClusterUpdateMethods:
    """测试更新最早完成时间的相关方法"""

    def test_update_earliest_compare(self, cluster: Cluster, sample_container: Container):
        """测试 _update_earliest_compare 方法"""
        cluster.reset()
        # 初始最早完成时间应为 inf
        assert cluster.earliest_finished_time == float("inf")

        # 创建第一个容器
        cluster.on_container_creation("c5.large", 0, 0, sample_container)
        first_earliest_time = cluster.earliest_finished_time

        # 创建第二个容器（完成时间更晚）
        container2 = Container(
            wf_id=1,
            fn_id=1,
            memory_req=1024,
            memory_alloc=2048,
            computation=2000000000,  # 更大的计算量
            parallelism=1,
            submission_time=0.0,
            data_transfer_time=0.0,
        )
        cluster.on_container_creation("c5.xlarge", 0, 0, container2)
        second_earliest_time = cluster.earliest_finished_time

        # 最早完成时间应该不变
        assert first_earliest_time == second_earliest_time

    def test_update_earliest_scan(self, cluster: Cluster, sample_container: Container):
        """测试 _update_earliest_scan 方法（遍历所有服务器）"""
        cluster.reset()
        # 创建一个容器并记录其完成时间
        cluster.on_container_creation("c5.large", 0, 0, sample_container)
        original_earliest_time = cluster.earliest_finished_time

        # 创建另一个容器
        container2 = Container(
            wf_id=1,
            fn_id=1,
            memory_req=1024,
            memory_alloc=2048,
            computation=2000000000,
            parallelism=1,
            submission_time=0.0,
            data_transfer_time=0.0,
        )
        cluster.on_container_creation("c5.xlarge", 0, 0, container2)

        # 最早完成时间应该不变（仍然是第一个容器）
        assert cluster.earliest_finished_time == original_earliest_time


class TestClusterEdgeCases:
    """测试集群边缘情况"""

    def test_multiple_servers_same_type(self, cluster_config: ClusterConfig):
        """测试同一服务器类型有多个实例"""
        # 修改配置创建多个同类型服务器
        cluster_config.servers[0].count = 3
        cluster = Cluster(cluster_config)
        cluster.reset()

        # 验证创建了多台服务器
        assert len(cluster["c5.large"]) == 3

        # 在每台服务器上创建容器
        containers = [
            Container(
                wf_id=i,
                fn_id=i,
                memory_req=1024,
                memory_alloc=2048,
                computation=1000000000,
                parallelism=1,
                submission_time=float(i),
                data_transfer_time=0.0,
            )
            for i in range(3)
        ]

        for i in range(3):
            cluster.on_container_creation("c5.large", i, 0, containers[i])

        # 验证最早完成时间在第一台服务器上
        assert cluster.earliest_finished_srv_name == "c5.large"
        assert cluster.earliest_finished_srv_id == 0

    def test_container_with_high_parallelism(self, cluster: Cluster):
        """测试高并行度容器"""
        cluster.reset()
        container = Container(
            wf_id=0,
            fn_id=0,
            memory_req=1024,
            memory_alloc=2048,
            computation=1000000000,
            parallelism=4,  # 高并行度
            submission_time=0.0,
            data_transfer_time=0.0,
        )

        # 在有 4 个 NUMA 节点的服务器上创建
        cluster.on_container_creation("c5.4xlarge", 0, 0, container)

        # 验证集群状态更新
        assert cluster.earliest_finished_srv_name == "c5.4xlarge"

    def test_rapid_container_creation_and_completion(self, cluster: Cluster):
        """测试快速连续创建和完成容器"""
        cluster.reset()
        containers = [
            Container(
                wf_id=i,
                fn_id=i,
                memory_req=1024,
                memory_alloc=2048,
                computation=100000000,
                parallelism=1,
                submission_time=0.0,
                data_transfer_time=0.0,  # 所有容器在同一时间创建
            )
            for i in range(10)
        ]

        # 快速创建多个容器
        for container in containers:
            cluster.on_container_creation("c5.large", 0, 0, container)

        # 快速完成所有容器
        completed: list[Container] = []
        for _ in range(len(containers)):
            _, c = cluster.on_container_completion()
            completed.append(c)

        # 验证所有容器都被完成
        assert len(completed) == len(containers)

        # 验证所有容器的 wf_id 都不相同
        completed_wf_ids = [c.wf_id for c in completed]
        assert len(set(completed_wf_ids)) == len(containers)  # 所有容器都不相同


class TestClusterIntegration:
    """集成测试：测试完整的容器生命周期"""

    def test_container_lifecycle(self, cluster: Cluster):
        """测试容器的完整生命周期：创建 -> 运行 -> 完成"""
        cluster.reset()
        container = Container(
            wf_id=0,
            fn_id=0,
            memory_req=1024,
            memory_alloc=2048,
            computation=1000000000,
            parallelism=1,
            submission_time=0.0,
            data_transfer_time=0.0,
        )

        # 创建容器
        creation_rent = cluster.on_container_creation("c5.large", 0, 0, container)
        assert creation_rent >= 0

        # 完成容器
        completion_rent, completed_container = cluster.on_container_completion()
        assert isinstance(completion_rent, float)
        assert completed_container == container

    def test_multiple_containers_different_servers(self, cluster: Cluster):
        """测试在多台服务器上运行多个容器"""
        cluster.reset()
        server_types = ["c5.large", "c5.xlarge", "c5.2xlarge"]
        containers = [
            Container(
                wf_id=i,
                fn_id=i,
                memory_req=1024,
                memory_alloc=2048,
                computation=1000000000 * (i + 1),
                parallelism=1,
                submission_time=0.0,
                data_transfer_time=0.0,
            )
            for i in range(len(server_types))
        ]

        # 在不同服务器上创建容器
        for i, server_type in enumerate(server_types):
            cluster.on_container_creation(server_type, 0, 0, containers[i])

        # 验证集群状态
        assert cluster.earliest_finished_srv_name in server_types

        # 按顺序完成所有容器
        completed_containers: list[Container] = []
        for _ in range(len(containers)):
            _, c = cluster.on_container_completion()
            completed_containers.append(c)

        # 验证所有容器都被完成
        assert len(completed_containers) == len(containers)

    def test_cluster_reset_after_operations(self, cluster: Cluster):
        """测试在多次操作后重置集群"""
        cluster.reset()
        containers = [
            Container(
                wf_id=i,
                fn_id=i,
                memory_req=1024,
                memory_alloc=2048,
                computation=1000000000,
                parallelism=1,
                submission_time=0.0,
                data_transfer_time=0.0,  # 所有容器在同一时间创建
            )
            for i in range(5)
        ]

        # 执行多次操作
        for container in containers[:3]:
            cluster.on_container_creation("c5.large", 0, 0, container)

        for _ in range(2):
            cluster.on_container_completion()

        # 重置集群
        cluster.reset()

        # 验证所有状态都被重置
        assert cluster.earliest_finished_srv_name is None
        assert cluster.earliest_finished_time == float("inf")

        # 验证可以重新开始
        new_container = Container(
            wf_id=99,
            fn_id=99,
            memory_req=1024,
            memory_alloc=2048,
            computation=1000000000,
            parallelism=1,
            submission_time=0.0,
            data_transfer_time=0.0,
        )
        cluster.on_container_creation("c5.large", 0, 0, new_container)
        assert cluster.earliest_finished_srv_name == "c5.large"
