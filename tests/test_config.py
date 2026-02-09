"""test_config.py

测试 src/faas_workflow_sim/config.py 中的集群配置类
"""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from faas_workflow_sim.config import ClusterConfig, NumaNodeConfig, ServerConfig


class TestClusterConfig:
    """测试 ClusterConfig 配置类"""

    def test_valid_cluster_config(self):
        """测试有效的集群配置"""
        servers = [
            ServerConfig(
                name="large",
                count=2,
                hourly_rate=0.5,
                cold_start_latency=1.5,
                numa_nodes=NumaNodeConfig(count=4, cpu=16, memory=64000),
            ),
            ServerConfig(
                name="medium",
                count=4,
                hourly_rate=0.3,
                cold_start_latency=1.0,
                numa_nodes=NumaNodeConfig(count=2, cpu=8, memory=32000),
            ),
        ]

        cluster_config = ClusterConfig(
            memory_bandwidth=100000000000,
            numa_bandwidth=50000000000,
            network_bandwidth=10000000000,
            single_core_speed=1000000000,
            servers=servers,
        )
        assert cluster_config.memory_bandwidth == 100000000000
        assert cluster_config.numa_bandwidth == 50000000000
        assert cluster_config.network_bandwidth == 10000000000
        assert cluster_config.single_core_speed == 1000000000
        assert len(cluster_config.servers) == 2

    def test_missing_required_fields(self):
        """测试缺少必填字段"""
        servers = [
            ServerConfig(
                name="large",
                count=2,
                hourly_rate=0.5,
                cold_start_latency=1.5,
                numa_nodes=NumaNodeConfig(count=4, cpu=16, memory=64000),
            )
        ]

        with pytest.raises(ValidationError) as exc_info:
            ClusterConfig(memory_bandwidth=100000000000, numa_bandwidth=50000000000, servers=servers)  # type: ignore
        errors = exc_info.value.errors()
        error_fields = {error["loc"][0] for error in errors}
        assert "network_bandwidth" in error_fields
        assert "single_core_speed" in error_fields

    def test_duplicate_server_names(self):
        """测试重复的服务器名称"""
        servers = [
            ServerConfig(
                name="medium",
                count=2,
                hourly_rate=0.3,
                cold_start_latency=1.0,
                numa_nodes=NumaNodeConfig(count=2, cpu=8, memory=32000),
            ),
            ServerConfig(
                name="medium",  # 重复名称
                count=4,
                hourly_rate=0.3,
                cold_start_latency=1.0,
                numa_nodes=NumaNodeConfig(count=2, cpu=8, memory=32000),
            ),
        ]

        with pytest.raises(ValidationError) as exc_info:
            ClusterConfig(
                memory_bandwidth=100000000000,
                numa_bandwidth=50000000000,
                network_bandwidth=10000000000,
                single_core_speed=1000000000,
                servers=servers,
            )
        assert "重复的类型名称" in str(exc_info.value)
        assert "medium" in str(exc_info.value)

    def test_multiple_duplicate_server_names(self):
        """测试多个重复的服务器名称"""
        servers = [
            ServerConfig(
                name="large",
                count=2,
                hourly_rate=0.5,
                cold_start_latency=1.5,
                numa_nodes=NumaNodeConfig(count=4, cpu=16, memory=64000),
            ),
            ServerConfig(
                name="medium",
                count=4,
                hourly_rate=0.3,
                cold_start_latency=1.0,
                numa_nodes=NumaNodeConfig(count=2, cpu=8, memory=32000),
            ),
            ServerConfig(
                name="large",  # 重复
                count=1,
                hourly_rate=0.5,
                cold_start_latency=1.5,
                numa_nodes=NumaNodeConfig(count=4, cpu=16, memory=64000),
            ),
            ServerConfig(
                name="medium",  # 重复
                count=2,
                hourly_rate=0.3,
                cold_start_latency=1.0,
                numa_nodes=NumaNodeConfig(count=2, cpu=8, memory=32000),
            ),
        ]

        with pytest.raises(ValidationError) as exc_info:
            ClusterConfig(
                memory_bandwidth=100000000000,
                numa_bandwidth=50000000000,
                network_bandwidth=10000000000,
                single_core_speed=1000000000,
                servers=servers,
            )
        error_msg = str(exc_info.value)
        assert "重复的类型名称" in error_msg
        assert "large" in error_msg
        assert "medium" in error_msg

    def test_invalid_bandwidth_values(self):
        """测试无效的带宽值"""
        servers = [
            ServerConfig(
                name="large",
                count=2,
                hourly_rate=0.5,
                cold_start_latency=1.5,
                numa_nodes=NumaNodeConfig(count=4, cpu=16, memory=64000),
            )
        ]

        # 测试内存带宽
        with pytest.raises(ValidationError):
            ClusterConfig(
                memory_bandwidth=0,
                numa_bandwidth=50000000000,
                network_bandwidth=10000000000,
                single_core_speed=1000000000,
                servers=servers,
            )

        # 测试 NUMA 带宽
        with pytest.raises(ValidationError):
            ClusterConfig(
                memory_bandwidth=100000000000,
                numa_bandwidth=0,
                network_bandwidth=10000000000,
                single_core_speed=1000000000,
                servers=servers,
            )

        # 测试网络带宽
        with pytest.raises(ValidationError):
            ClusterConfig(
                memory_bandwidth=100000000000,
                numa_bandwidth=50000000000,
                network_bandwidth=0,
                single_core_speed=1000000000,
                servers=servers,
            )

    def test_server_count_can_be_zero(self):
        """测试服务器类型的数量可以为 0（表示没有该类型的服务器）"""
        servers = [
            ServerConfig(
                name="large",
                count=2,
                hourly_rate=0.5,
                cold_start_latency=1.5,
                numa_nodes=NumaNodeConfig(count=4, cpu=16, memory=64000),
            ),
            ServerConfig(
                name="medium",
                count=0,  # 允许为 0
                hourly_rate=0.3,
                cold_start_latency=1.0,
                numa_nodes=NumaNodeConfig(count=2, cpu=8, memory=32000),
            ),
        ]

        cluster_config = ClusterConfig(
            memory_bandwidth=100000000000,
            numa_bandwidth=50000000000,
            network_bandwidth=10000000000,
            single_core_speed=1000000000,
            servers=servers,
        )

        assert len(cluster_config.servers) == 2
        assert cluster_config.servers[0].count == 2
        assert cluster_config.servers[1].count == 0

    def test_total_server_count_must_be_positive(self):
        """测试集群中服务器总数必须至少为 1"""
        servers = [
            ServerConfig(
                name="large",
                count=0,
                hourly_rate=0.5,
                cold_start_latency=1.5,
                numa_nodes=NumaNodeConfig(count=4, cpu=16, memory=64000),
            ),
            ServerConfig(
                name="medium",
                count=0,
                hourly_rate=0.3,
                cold_start_latency=1.0,
                numa_nodes=NumaNodeConfig(count=2, cpu=8, memory=32000),
            ),
        ]

        with pytest.raises(ValidationError) as exc_info:
            ClusterConfig(
                memory_bandwidth=100000000000,
                numa_bandwidth=50000000000,
                network_bandwidth=10000000000,
                single_core_speed=1000000000,
                servers=servers,
            )

        assert "至少需要有一个服务器" in str(exc_info.value)

    def test_multiple_server_types_with_zero_count(self):
        """测试多个服务器类型数量为 0，但总数大于 0 的情况"""
        servers = [
            ServerConfig(
                name="large",
                count=0,
                hourly_rate=0.5,
                cold_start_latency=1.5,
                numa_nodes=NumaNodeConfig(count=4, cpu=16, memory=64000),
            ),
            ServerConfig(
                name="medium",
                count=0,
                hourly_rate=0.3,
                cold_start_latency=1.0,
                numa_nodes=NumaNodeConfig(count=2, cpu=8, memory=32000),
            ),
            ServerConfig(
                name="small",
                count=5,  # 只有这个类型有服务器
                hourly_rate=0.1,
                cold_start_latency=0.5,
                numa_nodes=NumaNodeConfig(count=1, cpu=4, memory=16000),
            ),
        ]

        cluster_config = ClusterConfig(
            memory_bandwidth=100000000000,
            numa_bandwidth=50000000000,
            network_bandwidth=10000000000,
            single_core_speed=1000000000,
            servers=servers,
        )

        # 验证所有服务器类型都被保留
        assert len(cluster_config.servers) == 3
        assert cluster_config.servers[0].count == 0
        assert cluster_config.servers[1].count == 0
        assert cluster_config.servers[2].count == 5


class TestClusterConfigYamlLoading:
    """测试从 YAML 文件加载配置"""

    @pytest.fixture
    def config_file_path(self):
        """获取配置文件路径"""
        return Path(__file__).parent / "data" / "cluster_config.yaml"

    def test_load_from_yaml(self, config_file_path: Path):
        """测试从 YAML 文件加载配置"""
        config = ClusterConfig.from_yaml(str(config_file_path))

        # 验证集群级别配置
        assert config.memory_bandwidth == 100000000000
        assert config.numa_bandwidth == 50000000000
        assert config.network_bandwidth == 10000000000
        assert config.single_core_speed == 1000000000

        # 验证服务器配置
        assert len(config.servers) == 8

        # 验证 c5.large 服务器
        c5_large = next(s for s in config.servers if s.name == "c5.large")
        assert c5_large.count == 1
        assert c5_large.hourly_rate == 0.085
        assert c5_large.cold_start_latency == 0.5
        assert c5_large.numa_nodes.count == 1
        assert c5_large.numa_nodes.cpu == 2
        assert c5_large.numa_nodes.memory == 4096

        # 验证 c5.xlarge 服务器
        c5_xlarge = next(s for s in config.servers if s.name == "c5.xlarge")
        assert c5_xlarge.count == 1
        assert c5_xlarge.hourly_rate == 0.17
        assert c5_xlarge.cold_start_latency == 0.6
        assert c5_xlarge.numa_nodes.count == 1
        assert c5_xlarge.numa_nodes.cpu == 4
        assert c5_xlarge.numa_nodes.memory == 8192

        # 验证 c5.2xlarge 服务器
        c5_2xlarge = next(s for s in config.servers if s.name == "c5.2xlarge")
        assert c5_2xlarge.count == 1
        assert c5_2xlarge.hourly_rate == 0.34
        assert c5_2xlarge.cold_start_latency == 0.8
        assert c5_2xlarge.numa_nodes.count == 2
        assert c5_2xlarge.numa_nodes.cpu == 4
        assert c5_2xlarge.numa_nodes.memory == 8192

        # 验证 c5.4xlarge 服务器
        c5_4xlarge = next(s for s in config.servers if s.name == "c5.4xlarge")
        assert c5_4xlarge.count == 1
        assert c5_4xlarge.hourly_rate == 0.68
        assert c5_4xlarge.cold_start_latency == 1.0
        assert c5_4xlarge.numa_nodes.count == 4
        assert c5_4xlarge.numa_nodes.cpu == 4
        assert c5_4xlarge.numa_nodes.memory == 8192

        # 验证 c5.9xlarge 服务器
        c5_9xlarge = next(s for s in config.servers if s.name == "c5.9xlarge")
        assert c5_9xlarge.count == 1
        assert c5_9xlarge.hourly_rate == 1.53
        assert c5_9xlarge.cold_start_latency == 1.5
        assert c5_9xlarge.numa_nodes.count == 9
        assert c5_9xlarge.numa_nodes.cpu == 4
        assert c5_9xlarge.numa_nodes.memory == 8192

        # 验证 c5.12xlarge 服务器
        c5_12xlarge = next(s for s in config.servers if s.name == "c5.12xlarge")
        assert c5_12xlarge.count == 1
        assert c5_12xlarge.hourly_rate == 2.04
        assert c5_12xlarge.cold_start_latency == 2.0
        assert c5_12xlarge.numa_nodes.count == 12
        assert c5_12xlarge.numa_nodes.cpu == 4
        assert c5_12xlarge.numa_nodes.memory == 8192

        # 验证 c5.18xlarge 服务器
        c5_18xlarge = next(s for s in config.servers if s.name == "c5.18xlarge")
        assert c5_18xlarge.count == 1
        assert c5_18xlarge.hourly_rate == 3.06
        assert c5_18xlarge.cold_start_latency == 3.0
        assert c5_18xlarge.numa_nodes.count == 18
        assert c5_18xlarge.numa_nodes.cpu == 4
        assert c5_18xlarge.numa_nodes.memory == 8192

        # 验证 c5.24xlarge 服务器
        c5_24xlarge = next(s for s in config.servers if s.name == "c5.24xlarge")
        assert c5_24xlarge.count == 1
        assert c5_24xlarge.hourly_rate == 4.08
        assert c5_24xlarge.cold_start_latency == 4.0
        assert c5_24xlarge.numa_nodes.count == 24
        assert c5_24xlarge.numa_nodes.cpu == 4
        assert c5_24xlarge.numa_nodes.memory == 8192

    def test_yaml_file_exists(self, config_file_path: Path):
        """测试 YAML 配置文件是否存在"""
        assert config_file_path.exists()
        assert config_file_path.is_file()

    def test_yaml_file_structure(self, config_file_path: Path):
        """测试 YAML 文件结构"""
        with open(config_file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        # 验证顶层字段
        required_fields = ["memory_bandwidth", "numa_bandwidth", "network_bandwidth", "single_core_speed", "servers"]
        for field in required_fields:
            assert field in data

        # 验证服务器列表
        assert isinstance(data["servers"], list)
        assert len(data["servers"]) > 0

        # 验证每个服务器配置
        for server in data["servers"]:
            assert "name" in server
            assert "count" in server
            assert "hourly_rate" in server
            assert "cold_start_latency" in server
            assert "numa_nodes" in server

            # 验证 NUMA 节点配置
            numa = server["numa_nodes"]
            assert "count" in numa
            assert "cpu" in numa
            assert "memory" in numa

    def test_from_yaml_file_not_found(self, tmp_path: Path):
        """测试 from_yaml 方法处理文件不存在的情况"""
        non_existent_file = tmp_path / "non_existent_config.yaml"

        with pytest.raises(FileNotFoundError):
            ClusterConfig.from_yaml(str(non_existent_file))

    def test_from_yaml_invalid_yaml(self, tmp_path: Path):
        """测试 from_yaml 方法处理无效 YAML 的情况"""
        invalid_yaml_file = tmp_path / "invalid_config.yaml"
        invalid_yaml_file.write_text("invalid: yaml: content: [unclosed")

        with pytest.raises(yaml.YAMLError):
            ClusterConfig.from_yaml(str(invalid_yaml_file))

    def test_from_yaml_invalid_data_format(self, tmp_path: Path):
        """测试 from_yaml 方法处理无效数据格式的情况"""
        invalid_format_file = tmp_path / "invalid_format.yaml"
        invalid_format_file.write_text("memory_bandwidth: 100000000000\nnuma_bandwidth: 50000000000")

        # 缺少必填字段应该引发 ValidationError
        with pytest.raises(ValidationError) as exc_info:
            ClusterConfig.from_yaml(str(invalid_format_file))

        errors = exc_info.value.errors()
        error_fields = {error["loc"][0] for error in errors}
        assert "network_bandwidth" in error_fields
        assert "single_core_speed" in error_fields
        assert "servers" in error_fields
