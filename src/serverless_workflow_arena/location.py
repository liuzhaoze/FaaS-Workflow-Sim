"""location.py

函数执行位置记录
"""

from typing import NamedTuple


class Location(NamedTuple):
    """函数执行位置

    Attributes:
        server_name (str): 服务器类型名称
        server_id (int): 服务器ID
        numa_node_id (int): NUMA节点ID
    """

    server_name: str
    server_id: int
    numa_node_id: int
