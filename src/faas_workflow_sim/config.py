"""config.py

集群配置模型
"""

import yaml
from pydantic import BaseModel, Field, model_validator


class NumaNodeConfig(BaseModel):
    count: int = Field(..., gt=0, description="NUMA 节点数量")
    cpu: int = Field(..., gt=0, description="CPU 核心数")
    memory: int = Field(..., gt=0, description="内存大小 (MB)")


class ServerConfig(BaseModel):
    name: str = Field(..., description="服务器类型名称")
    count: int = Field(..., gt=0, description="服务器数量")
    hourly_rate: float = Field(..., gt=0, description="每小时租金")
    cold_start_latency: float = Field(..., gt=0, description="冷启动延迟 (秒)")
    numa_nodes: NumaNodeConfig = Field(..., description="NUMA 节点配置")


class ClusterConfig(BaseModel):
    memory_bandwidth: int = Field(..., gt=0, description="内存带宽 (bytes/s)")
    numa_bandwidth: int = Field(..., gt=0, description="NUMA 节点间带宽 (bytes/s)")
    network_bandwidth: int = Field(..., gt=0, description="网络带宽 (bytes/s)")
    single_core_speed: int = Field(..., gt=0, description="单核计算速度")
    servers: list[ServerConfig] = Field(..., description="服务器配置列表")

    @classmethod
    def from_yaml(cls, path: str) -> "ClusterConfig":
        """从 YAML 文件加载集群配置"""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    @model_validator(mode="after")
    def validate_unique_server_names(self) -> "ClusterConfig":
        names = [server.name for server in self.servers]
        unique_names = set(names)

        if len(names) != len(unique_names):
            duplicates = set(n for n in names if names.count(n) > 1)
            raise ValueError(f"服务器配置中存在重复的类型名称: {', '.join(duplicates)}")

        return self
