"""cluster.py

异构服务器集群建模
"""

from .config import ClusterConfig
from .container import Container
from .location import Location
from .server import Server


class Cluster:
    """异构服务器集群模型

    Args:
        config (ClusterConfig): 集群配置
    """

    __slots__ = (
        "memory_bandwidth",
        "numa_bandwidth",
        "network_bandwidth",
        "single_core_speed",
        "_servers",
        "earliest_finished_srv_name",
        "earliest_finished_srv_id",
        "earliest_finished_time",
    )

    def __init__(self, cc: ClusterConfig):
        self.memory_bandwidth: int = cc.memory_bandwidth
        self.numa_bandwidth: int = cc.numa_bandwidth
        self.network_bandwidth: int = cc.network_bandwidth
        self.single_core_speed: int = cc.single_core_speed

        self._servers: dict[str, tuple[Server, ...]] = {
            s.name: tuple(
                Server(
                    i,
                    s.hourly_rate,
                    s.cold_start_latency,
                    s.numa_nodes.count,
                    s.numa_nodes.cpu,
                    s.numa_nodes.memory,
                    cc.single_core_speed,
                )
                for i in range(s.count)
            )
            for s in cc.servers
        }
        self.earliest_finished_srv_name: str | None = None
        self.earliest_finished_srv_id: int | None = None
        self.earliest_finished_time: float = float("inf")

    def __len__(self) -> int:
        return len(self._servers)

    def __contains__(self, server_name: str) -> bool:
        return server_name in self._servers

    def __getitem__(self, server_name: str) -> tuple[Server, ...]:
        return self._servers[server_name]

    def reset(self):
        """重置集群中所有服务器的状态以及集群状态变量"""
        for servers in self._servers.values():
            for s in servers:
                s.reset()

        self.earliest_finished_srv_name = None
        self.earliest_finished_srv_id = None
        self.earliest_finished_time = float("inf")

    def on_container_creation(self, server_name: str, server_id: int, numa_node_id: int, c: Container) -> float:
        """将容器分配到指定服务器的指定 NUMA 节点上执行

        Args:
            server_name (str): 目标服务器类型名称
            server_id (int): 目标服务器 ID
            numa_node_id (int): 目标 NUMA 节点 ID
            c (Container): 待分配的容器

        Returns:
            float: 让指定服务器的租期覆盖容器执行所需的租金
        """

        # 将容器分配到指定服务器的指定 NUMA 节点上执行
        rent = self._servers[server_name][server_id].on_container_creation(numa_node_id, c)

        # 更新集群上最早执行完容器的时间和对应的服务器名称和 ID
        self._update_earliest_finished(server_name, server_id)

        return rent

    def on_container_completion(self) -> tuple[float, Container]:
        """完成集群中所有服务器上最早完成的容器

        Returns:
            (float, Container): 让拥有最早完成容器的服务器的租期覆盖容器执行所需的租金和已完成的容器
        """

        if self.earliest_finished_srv_name is None or self.earliest_finished_srv_id is None:
            raise RuntimeError("No running containers in the cluster")

        # 完成集群中拥有最早完成容器的服务器上的该容器
        rent, c = self._servers[self.earliest_finished_srv_name][
            self.earliest_finished_srv_id
        ].on_container_completion()

        # 更新集群上最早执行完容器的时间和对应的服务器名称和 ID
        self._update_earliest_finished(self.earliest_finished_srv_name, self.earliest_finished_srv_id)

        return rent, c

    def _update_earliest_finished(self, srv_name: str, srv_id: int):
        """更新集群上最早执行完容器的时间和对应的服务器类型名称和 ID

        Args:
            srv_name (str): 发生容器创建或完成操作的服务器类型名称
            srv_id (int): 发生容器创建或完成操作的服务器 ID
        """

        earliest_time = self._servers[srv_name][srv_id].earliest_finished_time

        if (
            self.earliest_finished_srv_name == srv_name
            and self.earliest_finished_srv_id == srv_id
            and earliest_time > self.earliest_finished_time
        ):
            # 之前记录的最早完成容器在本服务器上，但现在该服务器的最早完成时间变晚了
            # 其他服务器可能有更早完成的容器，因此需要遍历所有服务器来更新最早完成时间
            self._update_earliest_scan()
        else:
            # 要么之前记录的最早完成容器不在本服务器上，本服务器的最早完成时间变晚不会影响全局最早完成时间
            # 要么最早完成时间变早了，无论之前记录的最早完成容器是否在本服务器上，都可以直接比较更新
            self._update_earliest_compare(srv_name, srv_id, earliest_time)

    def _update_earliest_compare(self, srv_name: str, srv_id: int, earliest_time: float):
        """使用比较法更新最早执行完容器的时间和对应的服务器类型名称和 ID"""
        if earliest_time < self.earliest_finished_time:
            self.earliest_finished_srv_name = srv_name
            self.earliest_finished_srv_id = srv_id
            self.earliest_finished_time = earliest_time

    def _update_earliest_scan(self):
        """使用遍历法更新最早执行完容器的时间和对应的服务器类型名称和 ID"""
        self.earliest_finished_srv_name = None
        self.earliest_finished_srv_id = None
        self.earliest_finished_time = float("inf")

        for name, servers in self._servers.items():
            for s in servers:
                self._update_earliest_compare(name, s.server_id, s.earliest_finished_time)

    def start_server(self, server_name: str, server_id: int, submission_time: float) -> float:
        """启动指定服务器并获取冷启动时间

        Args:
            server_name (str): 目标服务器类型名称
            server_id (int): 目标服务器 ID
            submission_time (float): 函数提交时间

        Returns:
            float: 冷启动时间
        """

        return self._servers[server_name][server_id].startup_at(submission_time)

    def get_data_transfer_speed(self, src: Location, dst: Location) -> int:
        """获取集群中两个 NUMA 节点之间的数据传输速度

        Args:
            src (Location): 源 NUMA 节点位置
            dst (Location): 目标 NUMA 节点位置

        Returns:
            int: 数据传输速度 (bytes/second)
        """

        if src.server_name != dst.server_name or src.server_id != dst.server_id:
            return self.network_bandwidth

        if src.numa_node_id != dst.numa_node_id:
            return self.numa_bandwidth

        return self.memory_bandwidth
