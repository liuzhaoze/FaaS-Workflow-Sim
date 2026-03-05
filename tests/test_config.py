"""test_config.py

测试 src/serverless_workflow_arena/config.py 中的集群配置类
"""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from serverless_workflow_arena.config import ClusterConfig, NumaNodeConfig, ServerConfig


class TestClusterConfig:
    """测试 ClusterConfig 配置类"""

    def test_valid_cluster_config(self):
        """测试有效的集群配置"""
        data: dict[str, int | list[dict[str, object]]] = {
            "memory_bandwidth": 100000000000,
            "numa_bandwidth": 50000000000,
            "network_bandwidth": 10000000000,
            "single_core_speed": 1000000000,
            "servers": [
                {
                    "name": "large",
                    "count": 2,
                    "hourly_rate": 0.5,
                    "cold_start_latency": 1.5,
                    "numa_nodes": {"count": 4, "cpu": 16, "memory": 64000},
                },
                {
                    "name": "medium",
                    "count": 4,
                    "hourly_rate": 0.3,
                    "cold_start_latency": 1.0,
                    "numa_nodes": {"count": 2, "cpu": 8, "memory": 32000},
                },
            ],
        }

        cluster_config = ClusterConfig.model_validate(data)
        assert cluster_config.memory_bandwidth == 100000000000
        assert cluster_config.numa_bandwidth == 50000000000
        assert cluster_config.network_bandwidth == 10000000000
        assert cluster_config.single_core_speed == 1000000000
        assert len(cluster_config.servers) == 2

    def test_missing_required_fields(self):
        """测试缺少必填字段"""
        data: dict[str, int | list[dict[str, object]]] = {
            "memory_bandwidth": 100000000000,
            "numa_bandwidth": 50000000000,
            "servers": [
                {
                    "name": "large",
                    "count": 2,
                    "hourly_rate": 0.5,
                    "cold_start_latency": 1.5,
                    "numa_nodes": {"count": 4, "cpu": 16, "memory": 64000},
                }
            ],
        }

        with pytest.raises(ValidationError) as exc_info:
            ClusterConfig.model_validate(data)
        errors = exc_info.value.errors()
        error_fields = {error["loc"][0] for error in errors}
        assert "network_bandwidth" in error_fields
        assert "single_core_speed" in error_fields

    def test_duplicate_server_names(self):
        """测试重复的服务器名称"""
        data: dict[str, int | list[dict[str, object]]] = {
            "memory_bandwidth": 100000000000,
            "numa_bandwidth": 50000000000,
            "network_bandwidth": 10000000000,
            "single_core_speed": 1000000000,
            "servers": [
                {
                    "name": "medium",
                    "count": 2,
                    "hourly_rate": 0.3,
                    "cold_start_latency": 1.0,
                    "numa_nodes": {"count": 2, "cpu": 8, "memory": 32000},
                },
                {
                    "name": "medium",  # 重复名称
                    "count": 4,
                    "hourly_rate": 0.3,
                    "cold_start_latency": 1.0,
                    "numa_nodes": {"count": 2, "cpu": 8, "memory": 32000},
                },
            ],
        }

        with pytest.raises(ValidationError) as exc_info:
            ClusterConfig.model_validate(data)
        error_msg = str(exc_info.value)
        assert "重复" in error_msg or "medium" in error_msg

    def test_multiple_duplicate_server_names(self):
        """测试多个重复的服务器名称"""
        data: dict[str, int | list[dict[str, object]]] = {
            "memory_bandwidth": 100000000000,
            "numa_bandwidth": 50000000000,
            "network_bandwidth": 10000000000,
            "single_core_speed": 1000000000,
            "servers": [
                {
                    "name": "large",
                    "count": 2,
                    "hourly_rate": 0.5,
                    "cold_start_latency": 1.5,
                    "numa_nodes": {"count": 4, "cpu": 16, "memory": 64000},
                },
                {
                    "name": "medium",
                    "count": 4,
                    "hourly_rate": 0.3,
                    "cold_start_latency": 1.0,
                    "numa_nodes": {"count": 2, "cpu": 8, "memory": 32000},
                },
                {
                    "name": "large",  # 重复
                    "count": 1,
                    "hourly_rate": 0.5,
                    "cold_start_latency": 1.5,
                    "numa_nodes": {"count": 4, "cpu": 16, "memory": 64000},
                },
                {
                    "name": "medium",  # 重复
                    "count": 2,
                    "hourly_rate": 0.3,
                    "cold_start_latency": 1.0,
                    "numa_nodes": {"count": 2, "cpu": 8, "memory": 32000},
                },
            ],
        }

        with pytest.raises(ValidationError) as exc_info:
            ClusterConfig.model_validate(data)
        error_msg = str(exc_info.value)
        assert "重复" in error_msg or "large" in error_msg

    def test_invalid_bandwidth_values(self):
        """测试无效的带宽值"""
        servers: list[dict[str, object]] = [
            {
                "name": "large",
                "count": 2,
                "hourly_rate": 0.5,
                "cold_start_latency": 1.5,
                "numa_nodes": {"count": 4, "cpu": 16, "memory": 64000},
            }
        ]

        # 测试内存带宽
        with pytest.raises(ValidationError):
            ClusterConfig.model_validate(
                {
                    "memory_bandwidth": 0,
                    "numa_bandwidth": 50000000000,
                    "network_bandwidth": 10000000000,
                    "single_core_speed": 1000000000,
                    "servers": servers,
                }
            )

        # 测试 NUMA 带宽
        with pytest.raises(ValidationError):
            ClusterConfig.model_validate(
                {
                    "memory_bandwidth": 100000000000,
                    "numa_bandwidth": 0,
                    "network_bandwidth": 10000000000,
                    "single_core_speed": 1000000000,
                    "servers": servers,
                }
            )

        # 测试网络带宽
        with pytest.raises(ValidationError):
            ClusterConfig.model_validate(
                {
                    "memory_bandwidth": 100000000000,
                    "numa_bandwidth": 50000000000,
                    "network_bandwidth": 0,
                    "single_core_speed": 1000000000,
                    "servers": servers,
                }
            )

    def test_bandwidth_ordering_memory_not_greater_than_numa(self):
        """测试带宽排序约束：memory_bandwidth 必须大于 numa_bandwidth"""
        servers: list[dict[str, object]] = [
            {
                "name": "large",
                "count": 2,
                "hourly_rate": 0.5,
                "cold_start_latency": 1.5,
                "numa_nodes": {"count": 4, "cpu": 16, "memory": 64000},
            }
        ]

        # memory_bandwidth == numa_bandwidth 应该被拒绝
        with pytest.raises(ValidationError) as exc_info:
            ClusterConfig.model_validate(
                {
                    "memory_bandwidth": 50000000000,
                    "numa_bandwidth": 50000000000,
                    "network_bandwidth": 10000000000,
                    "single_core_speed": 1000000000,
                    "servers": servers,
                }
            )
        assert "带宽配置应满足" in str(exc_info.value)

        # memory_bandwidth < numa_bandwidth 应该被拒绝
        with pytest.raises(ValidationError) as exc_info:
            ClusterConfig.model_validate(
                {
                    "memory_bandwidth": 40000000000,
                    "numa_bandwidth": 50000000000,
                    "network_bandwidth": 10000000000,
                    "single_core_speed": 1000000000,
                    "servers": servers,
                }
            )
        assert "带宽配置应满足" in str(exc_info.value)

    def test_bandwidth_ordering_numa_not_greater_than_network(self):
        """测试带宽排序约束：numa_bandwidth 必须大于 network_bandwidth"""
        servers: list[dict[str, object]] = [
            {
                "name": "large",
                "count": 2,
                "hourly_rate": 0.5,
                "cold_start_latency": 1.5,
                "numa_nodes": {"count": 4, "cpu": 16, "memory": 64000},
            }
        ]

        # numa_bandwidth == network_bandwidth 应该被拒绝
        with pytest.raises(ValidationError) as exc_info:
            ClusterConfig.model_validate(
                {
                    "memory_bandwidth": 100000000000,
                    "numa_bandwidth": 10000000000,
                    "network_bandwidth": 10000000000,
                    "single_core_speed": 1000000000,
                    "servers": servers,
                }
            )
        assert "带宽配置应满足" in str(exc_info.value)

        # numa_bandwidth < network_bandwidth 应该被拒绝
        with pytest.raises(ValidationError) as exc_info:
            ClusterConfig.model_validate(
                {
                    "memory_bandwidth": 100000000000,
                    "numa_bandwidth": 5000000000,
                    "network_bandwidth": 10000000000,
                    "single_core_speed": 1000000000,
                    "servers": servers,
                }
            )
        assert "带宽配置应满足" in str(exc_info.value)

    def test_zero_count_servers_are_removed(self):
        """测试数量为 0 的服务器配置会被自动移除"""
        data: dict[str, int | list[dict[str, object]]] = {
            "memory_bandwidth": 100000000000,
            "numa_bandwidth": 50000000000,
            "network_bandwidth": 10000000000,
            "single_core_speed": 1000000000,
            "servers": [
                {
                    "name": "large",
                    "count": 2,
                    "hourly_rate": 0.5,
                    "cold_start_latency": 1.5,
                    "numa_nodes": {"count": 4, "cpu": 16, "memory": 64000},
                },
                {
                    "name": "medium",
                    "count": 0,  # 会被移除
                    "hourly_rate": 0.3,
                    "cold_start_latency": 1.0,
                    "numa_nodes": {"count": 2, "cpu": 8, "memory": 32000},
                },
                {
                    "name": "small",
                    "count": 0,  # 会被移除
                    "hourly_rate": 0.1,
                    "cold_start_latency": 0.5,
                    "numa_nodes": {"count": 1, "cpu": 4, "memory": 16000},
                },
            ],
        }

        cluster_config = ClusterConfig.model_validate(data)

        # 只有 count > 0 的服务器被保留
        assert len(cluster_config.servers) == 1
        assert cluster_config.servers[0].name == "large"
        assert cluster_config.servers[0].count == 2

    def test_all_servers_with_zero_count_raises_error(self):
        """测试所有服务器数量都为 0 时应该引发错误"""
        data: dict[str, int | list[dict[str, object]]] = {
            "memory_bandwidth": 100000000000,
            "numa_bandwidth": 50000000000,
            "network_bandwidth": 10000000000,
            "single_core_speed": 1000000000,
            "servers": [
                {
                    "name": "large",
                    "count": 0,
                    "hourly_rate": 0.5,
                    "cold_start_latency": 1.5,
                    "numa_nodes": {"count": 4, "cpu": 16, "memory": 64000},
                },
                {
                    "name": "medium",
                    "count": 0,
                    "hourly_rate": 0.3,
                    "cold_start_latency": 1.0,
                    "numa_nodes": {"count": 2, "cpu": 8, "memory": 32000},
                },
            ],
        }

        with pytest.raises(ValidationError) as exc_info:
            ClusterConfig.model_validate(data)

        assert "至少需要有一个服务器" in str(exc_info.value) or "至少" in str(exc_info.value)

    def test_mixed_zero_and_positive_counts(self):
        """测试混合零和非零数量的服务器配置"""
        data: dict[str, int | list[dict[str, object]]] = {
            "memory_bandwidth": 100000000000,
            "numa_bandwidth": 50000000000,
            "network_bandwidth": 10000000000,
            "single_core_speed": 1000000000,
            "servers": [
                {
                    "name": "large",
                    "count": 0,  # 会被移除
                    "hourly_rate": 0.5,
                    "cold_start_latency": 1.5,
                    "numa_nodes": {"count": 4, "cpu": 16, "memory": 64000},
                },
                {
                    "name": "medium",
                    "count": 0,  # 会被移除
                    "hourly_rate": 0.3,
                    "cold_start_latency": 1.0,
                    "numa_nodes": {"count": 2, "cpu": 8, "memory": 32000},
                },
                {
                    "name": "small",
                    "count": 5,  # 会被保留
                    "hourly_rate": 0.1,
                    "cold_start_latency": 0.5,
                    "numa_nodes": {"count": 1, "cpu": 4, "memory": 16000},
                },
            ],
        }

        cluster_config = ClusterConfig.model_validate(data)

        # 只有 count > 0 的服务器被保留
        assert len(cluster_config.servers) == 1
        assert cluster_config.servers[0].name == "small"
        assert cluster_config.servers[0].count == 5

    def test_server_config_objects_can_be_created(self):
        """测试可以直接创建 ServerConfig 对象"""
        server = ServerConfig(
            name="test",
            count=1,
            hourly_rate=0.1,
            cold_start_latency=0.5,
            numa_nodes=NumaNodeConfig(count=1, cpu=2, memory=4096),
        )
        assert server.name == "test"
        assert server.count == 1

    def test_negative_server_count_is_invalid(self):
        """测试负数的服务器数量是无效的"""
        with pytest.raises(ValidationError):
            ServerConfig(
                name="test",
                count=-1,
                hourly_rate=0.1,
                cold_start_latency=0.5,
                numa_nodes=NumaNodeConfig(count=1, cpu=2, memory=4096),
            )


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

    def test_from_yaml_all_servers_with_zero_count(self, tmp_path: Path):
        """测试从 YAML 加载所有服务器 count 为 0 的配置"""
        yaml_content = """
memory_bandwidth: 100000000000
numa_bandwidth: 50000000000
network_bandwidth: 10000000000
single_core_speed: 1000000000
servers:
- name: large
  count: 0
  hourly_rate: 0.5
  cold_start_latency: 1.5
  numa_nodes:
    count: 4
    cpu: 16
    memory: 64000
- name: medium
  count: 0
  hourly_rate: 0.3
  cold_start_latency: 1.0
  numa_nodes:
    count: 2
    cpu: 8
    memory: 32000
"""
        config_file = tmp_path / "all_zero.yaml"
        config_file.write_text(yaml_content)

        with pytest.raises(ValidationError) as exc_info:
            ClusterConfig.from_yaml(str(config_file))

        assert "至少需要有一个服务器" in str(exc_info.value) or "至少" in str(exc_info.value)
