"""config.py

集群配置模型
"""

from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator


class NumaNodeConfig(BaseModel):
    count: int = Field(..., gt=0, description="NUMA 节点数量")
    cpu: int = Field(..., gt=0, description="CPU 核心数")
    memory: int = Field(..., gt=0, description="内存大小 (MB)")


class ServerConfig(BaseModel):
    name: str = Field(..., description="服务器类型名称")
    count: int = Field(..., ge=0, description="服务器数量")
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

    @model_validator(mode="before")
    def remove_zero_count_servers(cls, data: dict[str, Any]) -> dict[str, Any]:
        """移除数量为 0 的服务器类型"""
        if "servers" in data:
            data["servers"] = [s for s in data["servers"] if s.get("count", 0) > 0]
        return data

    @model_validator(mode="after")
    def validate_bandwidths(self) -> "ClusterConfig":
        if not (self.memory_bandwidth > self.numa_bandwidth > self.network_bandwidth):
            raise ValueError("带宽配置应满足 memory_bandwidth > numa_bandwidth > network_bandwidth")

        return self

    @model_validator(mode="after")
    def validate_unique_server_names(self) -> "ClusterConfig":
        names = [server.name for server in self.servers]
        unique_names = set(names)

        if len(names) != len(unique_names):
            duplicates = set(n for n in names if names.count(n) > 1)
            raise ValueError(f"服务器配置中存在重复的类型名称: {', '.join(duplicates)}")

        return self

    @model_validator(mode="after")
    def validate_total_server_count(self) -> "ClusterConfig":
        total_count = sum(server.count for server in self.servers)

        if total_count <= 0:
            raise ValueError("服务器配置中至少需要有一个服务器")

        return self
