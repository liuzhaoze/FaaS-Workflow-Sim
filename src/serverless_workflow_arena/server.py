"""server.py

异构服务器建模
"""

import math

from .container import Container
from .numa_node import NumaNode

HOUR = 3600.0


class Server:
    """异构服务器模型

    Args:
        server_id (int): 服务器ID
        hourly_rate (float): 服务器每小时的价格
        cold_start_latency (float): 服务器冷启动的时间
        numa_node_count (int): 服务器上的NUMA节点数量
        numa_node_cpu (int): 每个NUMA节点的CPU数量
        numa_node_memory (int): 每个NUMA节点的内存大小 (MB)
        single_core_speed (int): 单核计算速度 (单个 CPU 核心每秒可执行的计算操作数量)
    """

    __slots__ = (
        "server_id",
        "hourly_rate",
        "cold_start_latency",
        "_numa_nodes",
        "expiration_time",
        "earliest_finished_nn_id",
        "earliest_finished_time",
        "latest_finished_nn_id",
        "latest_finished_time",
    )

    def __init__(
        self,
        server_id: int,
        hourly_rate: float,
        cold_start_latency: float,
        numa_node_count: int,
        numa_node_cpu: int,
        numa_node_memory: int,
        single_core_speed: int,
    ):
        self.server_id: int = server_id
        self.hourly_rate: float = hourly_rate
        self.cold_start_latency: float = cold_start_latency

        self._numa_nodes: tuple[NumaNode, ...] = tuple(
            NumaNode(i, numa_node_cpu, numa_node_memory, single_core_speed) for i in range(numa_node_count)
        )
        self.expiration_time: float = 0.0
        self.earliest_finished_nn_id: int | None = None
        self.earliest_finished_time: float = float("inf")
        self.latest_finished_nn_id: int | None = None
        self.latest_finished_time: float = float("-inf")

    def __len__(self) -> int:
        return len(self._numa_nodes)

    def __getitem__(self, numa_node_id: int) -> NumaNode:
        return self._numa_nodes[numa_node_id]

    def __iter__(self):
        return iter(self._numa_nodes)

    def reset(self):
        """重置服务器中各 NUMA 节点的状态以及服务器的状态"""
        for nn in self._numa_nodes:
            nn.reset()

        self.expiration_time = 0.0
        self.earliest_finished_nn_id = None
        self.earliest_finished_time = float("inf")
        self.latest_finished_nn_id = None
        self.latest_finished_time = float("-inf")

    def on_container_creation(self, numa_node_id: int, c: Container) -> float:
        """在指定 NUMA 节点上创建容器时，需要执行的操作

        Args:
            numa_node_id (int): 目标 NUMA 节点 ID
            c (Container): 待分配的容器

        Returns:
            float: 让服务器租期覆盖容器执行所需的租金
        """

        # 将容器分配到指定 NUMA 节点上执行
        self._numa_nodes[numa_node_id].on_container_creation(c)

        # 更新服务器上最早和最晚执行完容器的时间和对应的 NUMA 节点 ID
        self._update_earliest_and_latest_finished(numa_node_id)

        # 检查当前服务器租期是否能够覆盖所有 NUMA 节点上容器的执行
        # 如果不能，则进行新租/续租，并返回所需的租金
        return self._renew_lease_at(c.submission_time)

    def on_container_completion(self) -> tuple[float, Container]:
        """当服务器中最早完成的容器完成时，需要执行的操作

        Returns:
            (float, Container): 让服务器租期覆盖容器执行所需的租金和已完成的容器
        """

        if self.earliest_finished_nn_id is None:
            raise RuntimeError("No running containers on the server")

        # 完成拥有最早完成容器的 NUMA 节点上的该容器
        c = self._numa_nodes[self.earliest_finished_nn_id].on_container_completion()

        # 因为完成容器后有可能会从等待队列中取出新的容器执行，所以仍然需要
        # 更新服务器上最早和最晚执行完容器的时间和对应的 NUMA 节点 ID
        self._update_earliest_and_latest_finished(self.earliest_finished_nn_id)

        # 同样出于上述原因，需要在新取出的容器开始执行后
        # 检查当前服务器租期是否能够覆盖所有 NUMA 节点上容器的执行
        # 如果不能，则进行续租，并返回所需的租金
        return self._renew_lease_at(c.completion_time), c

    def _update_earliest_and_latest_finished(self, nn_id: int):
        """更新服务器上最早和最晚执行完容器的时间和对应的 NUMA 节点 ID

        Args:
            nn_id (int): 发生容器创建或完成操作的 NUMA 节点 ID
        """

        _, earliest_time = self._numa_nodes[nn_id].get_earliest_finished()
        _, latest_time = self._numa_nodes[nn_id].get_latest_finished()

        if self.earliest_finished_nn_id == nn_id and earliest_time > self.earliest_finished_time:
            # 之前记录的最早完成容器在本节点上，但现在该节点的最早完成时间变晚了
            # 其他节点可能有更早完成的容器，因此需要遍历所有节点来更新最早完成时间
            self._update_earliest_scan()
        else:
            # 要么之前记录的最早完成容器不在本节点上，本节点的最早完成时间变晚不会影响全局最早完成时间
            # 要么最早完成时间变早了，无论之前记录的最早完成容器是否在本节点上，都可以直接比较更新
            self._update_earliest_compare(nn_id, earliest_time)

        if self.latest_finished_nn_id == nn_id and latest_time < self.latest_finished_time:
            # 之前记录的最晚完成容器在本节点上，但现在该节点的最晚完成时间变早了
            # 其他节点可能有更晚完成的容器，因此需要遍历所有节点来更新最晚完成时间
            self._update_latest_scan()
        else:
            # 要么之前记录的最晚完成容器不在本节点上，本节点的最晚完成时间变早不会影响全局最晚完成时间
            # 要么最晚完成时间变晚了，无论之前记录的最晚完成容器是否在本节点上，都可以直接比较更新
            self._update_latest_compare(nn_id, latest_time)

    def _update_earliest_compare(self, nn_id: int, earliest_time: float):
        """使用比较法更新最早执行完容器的时间和对应的 NUMA 节点 ID"""
        if earliest_time < self.earliest_finished_time:
            self.earliest_finished_nn_id = nn_id
            self.earliest_finished_time = earliest_time

    def _update_earliest_scan(self):
        """使用遍历法更新最早执行完容器的时间和对应的 NUMA 节点 ID"""
        self.earliest_finished_nn_id = None
        self.earliest_finished_time = float("inf")

        for nn in self._numa_nodes:
            _, earliest_time = nn.get_earliest_finished()
            self._update_earliest_compare(nn.node_id, earliest_time)

    def _update_latest_compare(self, nn_id: int, latest_time: float):
        """使用比较法更新最晚执行完容器的时间和对应的 NUMA 节点 ID"""
        if latest_time > self.latest_finished_time:
            self.latest_finished_nn_id = nn_id
            self.latest_finished_time = latest_time

    def _update_latest_scan(self):
        """使用遍历法更新最晚执行完容器的时间和对应的 NUMA 节点 ID"""
        self.latest_finished_nn_id = None
        self.latest_finished_time = float("-inf")

        for nn in self._numa_nodes:
            _, latest_time = nn.get_latest_finished()
            self._update_latest_compare(nn.node_id, latest_time)

    def _renew_lease_at(self, time: float) -> float:
        """在指定时间检查服务器的租期是否覆盖所有 NUMA 节点上容器的执行

        如果已覆盖，则不需要续租，返回 0.0

        如果未覆盖，则计算需要新租/续租的时间和费用，更新服务器的到期时间，并返回租金

        Args:
            time (float): 检查时间 (在 `on_container_creation` 中传入函数提交时间；在 `on_container_completion` 中传入容器完成时间)

        Returns:
            float: 让服务器租期覆盖容器执行所需的租金
        """

        if self.expiration_time >= self.latest_finished_time:
            # 服务器租期已经覆盖所有 NUMA 节点上容器的执行，无需续租
            return 0.0

        # 确定续租的起始时间
        # - 在 `on_container_creation` 中调用时，检查时间为函数提交时间
        #   如果函数提交时间早于服务器到期时间，则租期从服务器到期时间开始累加
        #   如果函数提交时间晚于服务器到期时间，则租期从函数提交时间开始累加
        # - 在 `on_container_completion` 中调用时，检查时间为容器完成时间
        #   因为容器完成时间一定不会晚于服务器到期时间，所以租期总是从服务器到期时间开始累加
        renewal_start_time = max(time, self.expiration_time)

        # 计算需要续租的小时数
        rent_hours = math.ceil((self.latest_finished_time - renewal_start_time) / HOUR)

        # 更新服务器的到期时间
        self.expiration_time = renewal_start_time + rent_hours * HOUR

        # 返回租赁费用
        return rent_hours * self.hourly_rate

    def startup_at(self, time: float) -> float:
        """在指定时间启动服务器

        Args:
            time (float): 函数提交时间

        Returns:
            float: 冷启动时间
        """

        if time < self.expiration_time:
            # 服务器仍在租期内，无需冷启动
            return 0.0

        for nn in self._numa_nodes:
            nn.on_server_startup(time, self.cold_start_latency)

        return self.cold_start_latency
