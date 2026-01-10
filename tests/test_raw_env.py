"""test_raw_env.py

测试 RawEnv 中函数执行的时序正确性

验证以下时序约束：
1. 对于每个函数：submission_time <= creation_time <= start_time <= completion_time
2. 对于有前驱节点的函数：前驱节点的 completion_time <= 本函数的 submission_time
"""

from pathlib import Path

import pytest

from faas_workflow_sim import ClusterConfig, RawEnv, WorkflowTemplate

DATA_DIR = Path(__file__).parent / "data"


class TestFunctionExecutionTiming:
    """测试函数执行的时序正确性"""

    @pytest.fixture
    def cluster_config(self):
        """加载集群配置"""
        config_file = DATA_DIR / "cluster_config.yaml"
        return ClusterConfig.from_yaml(str(config_file))

    @pytest.fixture
    def workflow_templates(self, cluster_config: ClusterConfig):
        """加载工作流模板"""
        dax_files = tuple(DATA_DIR.glob("*.dax"))
        return [WorkflowTemplate(str(f), cluster_config.single_core_speed) for f in dax_files]

    @pytest.fixture
    def valid_strategies(self, cluster_config: ClusterConfig):
        """生成合法的资源分配策略"""
        strategies: list[tuple[str, int, int]] = []
        for server_config in cluster_config.servers:
            for server_id in range(server_config.count):
                for numa_node_id in range(server_config.numa_nodes.count):
                    strategies.append((server_config.name, server_id, numa_node_id))
        return strategies

    def test_function_internal_timing_order(
        self,
        cluster_config: ClusterConfig,
        workflow_templates: list[WorkflowTemplate],
        valid_strategies: list[tuple[str, int, int]],
    ):
        """测试函数内部时序：submission_time <= creation_time <= start_time <= completion_time"""
        # 使用简单的测试用例：只有1个工作流
        arrival_times = [0.0]
        env = RawEnv(arrival_times, workflow_templates, cluster_config)
        env.reset()

        # 用于收集所有函数执行记录的字典
        # key: (wf_id, fn_id), value: dict with timing information
        function_records: dict[tuple[int, int], dict[str, float]] = {}
        step_count = 0

        while True:
            # 使用 Round-Robin 为函数分配资源
            server_name, server_id, numa_node_id = valid_strategies[step_count % len(valid_strategies)]
            allocated_memory = 256

            env_log = env.step(server_name, server_id, numa_node_id, allocated_memory)

            # 记录当前函数的时序信息
            key = (env_log.wf_id, env_log.fn_id)
            function_records[key] = {
                "submission_time": env_log.submission_time,
                "creation_time": env_log.creation_time,
            }

            # 记录所有完成函数的时序信息
            for record in env_log.finished_function_records:
                key = (record.wf_id, record.fn_id)
                function_records[key]["start_time"] = record.start_time
                function_records[key]["completion_time"] = record.completion_time

            step_count += 1
            if env_log.terminated:
                break

        # 验证所有函数的时序约束
        errors: list[str] = []
        for (wf_id, fn_id), timings in function_records.items():
            submission_time = timings["submission_time"]
            creation_time = timings["creation_time"]
            start_time = timings["start_time"]
            completion_time = timings["completion_time"]

            # 检查 submission_time <= creation_time
            if submission_time > creation_time:
                errors.append(
                    f"Function ({wf_id}, {fn_id}): submission_time ({submission_time}) > creation_time ({creation_time})"
                )

            # 检查 creation_time <= start_time
            if creation_time > start_time:
                errors.append(
                    f"Function ({wf_id}, {fn_id}): creation_time ({creation_time}) > start_time ({start_time})"
                )

            # 检查 start_time <= completion_time
            if start_time > completion_time:
                errors.append(
                    f"Function ({wf_id}, {fn_id}): start_time ({start_time}) > completion_time ({completion_time})"
                )

            # 完整链路检查：submission_time <= creation_time <= start_time <= completion_time
            if not (submission_time <= creation_time <= start_time <= completion_time):
                errors.append(
                    f"Function ({wf_id}, {fn_id}): timing chain broken - "
                    f"submission_time={submission_time}, creation_time={creation_time}, "
                    f"start_time={start_time}, completion_time={completion_time}"
                )

        # 如果有任何错误，抛出断言失败
        if errors:
            error_msg = "Function internal timing order violations:\n" + "\n".join(errors)
            pytest.fail(error_msg)

        # 验证通过，输出统计信息
        print(f"\n✓ Verified {len(function_records)} functions for internal timing order")
        print(f"  All functions satisfy: submission_time <= creation_time <= start_time <= completion_time")

    def test_predecessor_successor_timing_order(
        self,
        cluster_config: ClusterConfig,
        workflow_templates: list[WorkflowTemplate],
        valid_strategies: list[tuple[str, int, int]],
    ):
        """测试前驱节点与后继节点的时序：前驱节点的 completion_time <= 本函数的 submission_time"""
        # 使用简单的测试用例：只有1个工作流
        arrival_times = [0.0]
        env = RawEnv(arrival_times, workflow_templates, cluster_config)
        env.reset()

        # 用于收集所有函数执行记录的字典
        # key: (wf_id, fn_id), value: dict with timing information
        function_records: dict[tuple[int, int], dict[str, float]] = {}
        step_count = 0

        while True:
            # 使用 Round-Robin 为函数分配资源
            server_name, server_id, numa_node_id = valid_strategies[step_count % len(valid_strategies)]
            allocated_memory = 256

            env_log = env.step(server_name, server_id, numa_node_id, allocated_memory)

            # 记录当前函数的时序信息
            key = (env_log.wf_id, env_log.fn_id)
            function_records[key] = {
                "submission_time": env_log.submission_time,
                "creation_time": env_log.creation_time,
            }

            # 记录所有完成函数的时序信息
            for record in env_log.finished_function_records:
                key = (record.wf_id, record.fn_id)
                function_records[key]["start_time"] = record.start_time
                function_records[key]["completion_time"] = record.completion_time

            step_count += 1
            if env_log.terminated:
                break

        # 获取工作流以检查前驱-后继关系
        workflow = env.workload[0]  # 我们只使用了1个工作流

        # 验证前驱-后继节点的时序约束
        errors: list[str] = []
        for (wf_id, fn_id), timings in function_records.items():
            submission_time = timings["submission_time"]

            # 获取该函数的所有前驱节点
            predecessors = workflow.get_predecessor_functions(fn_id)

            if predecessors:
                # 检查所有前驱节点的 completion_time 是否 <= 本函数的 submission_time
                for pred_fn_id, _, _ in predecessors:
                    pred_key = (wf_id, pred_fn_id)
                    if pred_key in function_records:
                        pred_completion_time = function_records[pred_key]["completion_time"]
                        if pred_completion_time > submission_time:
                            errors.append(
                                f"Function ({wf_id}, {fn_id}): "
                                f"predecessor ({wf_id}, {pred_fn_id}) completion_time ({pred_completion_time}) > "
                                f"submission_time ({submission_time})"
                            )

        # 如果有任何错误，抛出断言失败
        if errors:
            error_msg = "Predecessor-Successor timing order violations:\n" + "\n".join(errors)
            pytest.fail(error_msg)

        # 验证通过，输出统计信息
        print(f"\n✓ Verified {len(function_records)} functions for predecessor-successor timing order")
        print(f"  All functions satisfy: predecessor.completion_time <= successor.submission_time")

    def test_combined_timing_constraints(
        self,
        cluster_config: ClusterConfig,
        workflow_templates: list[WorkflowTemplate],
        valid_strategies: list[tuple[str, int, int]],
    ):
        """综合测试：同时验证所有时序约束"""
        # 使用多个工作流进行测试
        arrival_times = [float(i) for i in range(3)]
        env = RawEnv(arrival_times, workflow_templates, cluster_config)
        env.reset()

        # 用于收集所有函数执行记录的字典
        function_records: dict[tuple[int, int], dict[str, float]] = {}
        step_count = 0

        while True:
            # 使用 Round-Robin 为函数分配资源
            server_name, server_id, numa_node_id = valid_strategies[step_count % len(valid_strategies)]
            allocated_memory = 256

            env_log = env.step(server_name, server_id, numa_node_id, allocated_memory)

            # 记录当前函数的时序信息
            key = (env_log.wf_id, env_log.fn_id)
            function_records[key] = {
                "submission_time": env_log.submission_time,
                "creation_time": env_log.creation_time,
            }

            # 记录所有完成函数的时序信息
            for record in env_log.finished_function_records:
                key = (record.wf_id, record.fn_id)
                function_records[key]["start_time"] = record.start_time
                function_records[key]["completion_time"] = record.completion_time

            step_count += 1
            if env_log.terminated:
                break

        # 验证所有时序约束
        errors: list[str] = []

        # 约束1：函数内部时序
        for (wf_id, fn_id), timings in function_records.items():
            submission_time = timings["submission_time"]
            creation_time = timings["creation_time"]
            start_time = timings["start_time"]
            completion_time = timings["completion_time"]

            if not (submission_time <= creation_time <= start_time <= completion_time):
                errors.append(
                    f"[Internal] Function ({wf_id}, {fn_id}): "
                    f"submission_time={submission_time}, creation_time={creation_time}, "
                    f"start_time={start_time}, completion_time={completion_time}"
                )

        # 约束2：前驱-后继时序
        for wf_id in range(len(arrival_times)):
            workflow = env.workload[wf_id]
            for (wf_id_, fn_id), timings in function_records.items():
                if wf_id_ != wf_id:
                    continue

                submission_time = timings["submission_time"]
                predecessors = workflow.get_predecessor_functions(fn_id)

                if predecessors:
                    for pred_fn_id, _, _ in predecessors:
                        pred_key = (wf_id, pred_fn_id)
                        if pred_key in function_records:
                            pred_completion_time = function_records[pred_key]["completion_time"]
                            if pred_completion_time > submission_time:
                                errors.append(
                                    f"[Predecessor] Function ({wf_id}, {fn_id}): "
                                    f"predecessor ({wf_id}, {pred_fn_id}) completion_time ({pred_completion_time}) > "
                                    f"submission_time ({submission_time})"
                                )

        # 如果有任何错误，抛出断言失败
        if errors:
            error_msg = "Timing constraint violations:\n" + "\n".join(errors)
            pytest.fail(error_msg)

        # 验证通过，输出统计信息
        print(f"\n✓ Verified {len(function_records)} functions across {len(arrival_times)} workflows")
        print(f"  ✓ All internal timing constraints satisfied")
        print(f"  ✓ All predecessor-successor timing constraints satisfied")
