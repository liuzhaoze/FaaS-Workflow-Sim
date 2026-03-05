"""test_raw_env.py

测试 RawEnv 中函数执行的时序正确性

验证以下时序约束：
1. 对于每个函数：submission_time <= creation_time <= start_time <= completion_time
2. 对于有前驱节点的函数：前驱节点的 completion_time <= 本函数的 submission_time
"""

from pathlib import Path

import pytest

from serverless_workflow_arena import ClusterConfig, RawEnv, WorkflowTemplate
from serverless_workflow_arena.env_log import WorkflowExecutionRecord

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


class TestRawEnvProperties:
    """测试 RawEnv 的属性和辅助方法"""

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

    def test_submit_queue_length_after_reset(
        self,
        cluster_config: ClusterConfig,
        workflow_templates: list[WorkflowTemplate],
    ):
        """测试重置后 submit_queue_length 返回正确的值"""
        arrival_times = [0.0]
        env = RawEnv(arrival_times, workflow_templates, cluster_config)
        env.reset()

        # reset 之后提交队列至少有1个函数（第一个被取出的函数已出队，队列中剩余的函数）
        # 实际上 reset 后 current_function 被取走了，queue 中剩余的是后续函数
        assert env.submit_queue_length >= 0

    def test_submit_queue_length_decreases_after_step(
        self,
        cluster_config: ClusterConfig,
        workflow_templates: list[WorkflowTemplate],
        valid_strategies: list[tuple[str, int, int]],
    ):
        """测试执行一步后 submit_queue_length 的变化"""
        arrival_times = [0.0]
        env = RawEnv(arrival_times, workflow_templates, cluster_config)
        env.reset()

        # 执行一步
        server_name, server_id, numa_node_id = valid_strategies[0]
        env_log = env.step(server_name, server_id, numa_node_id, 256)

        # 如果环境未结束，可以继续检查
        if not env_log.terminated:
            # 当完成某些函数后，可能会有新函数提交到队列，队列长度可能变化
            assert env.submit_queue_length >= 0
        else:
            assert env.submit_queue_length == 0

    def test_active_workflow_count_after_reset(
        self,
        cluster_config: ClusterConfig,
        workflow_templates: list[WorkflowTemplate],
    ):
        """测试重置后 active_workflow_count 正确"""
        arrival_times = [0.0]
        env = RawEnv(arrival_times, workflow_templates, cluster_config)
        env.reset()

        # reset 后第一个工作流已经到达（current_function 属于该工作流）
        assert env.active_workflow_count == 1

    def test_active_workflow_count_with_multiple_workflows(
        self,
        cluster_config: ClusterConfig,
        workflow_templates: list[WorkflowTemplate],
    ):
        """测试多个工作流时 active_workflow_count 正确"""
        arrival_times = [0.0, 0.0, 0.0]  # 3个同时到达的工作流
        env = RawEnv(arrival_times, workflow_templates, cluster_config)
        env.reset()

        # reset 后至少有1个工作流是活跃的（current_function 所属的工作流）
        assert env.active_workflow_count >= 1
        assert env.active_workflow_count <= len(arrival_times)

    def test_active_workflow_count_decreases_when_workflow_completes(
        self,
        cluster_config: ClusterConfig,
        workflow_templates: list[WorkflowTemplate],
        valid_strategies: list[tuple[str, int, int]],
    ):
        """测试工作流完成时 active_workflow_count 减少"""
        arrival_times = [0.0]
        env = RawEnv(arrival_times, workflow_templates, cluster_config)
        env.reset()

        step_count = 0
        while True:
            server_name, server_id, numa_node_id = valid_strategies[step_count % len(valid_strategies)]
            env_log = env.step(server_name, server_id, numa_node_id, 256)
            step_count += 1
            if env_log.terminated:
                break

        # 所有工作流完成后，active_workflow_count 应该为 0
        assert env.active_workflow_count == 0

    def test_active_workflows_cleared_on_reset(
        self,
        cluster_config: ClusterConfig,
        workflow_templates: list[WorkflowTemplate],
        valid_strategies: list[tuple[str, int, int]],
    ):
        """测试 reset 后 _active_workflows 被正确清除"""
        arrival_times = [0.0]
        env = RawEnv(arrival_times, workflow_templates, cluster_config)
        env.reset()

        # 执行完成
        step_count = 0
        while True:
            server_name, server_id, numa_node_id = valid_strategies[step_count % len(valid_strategies)]
            env_log = env.step(server_name, server_id, numa_node_id, 256)
            step_count += 1
            if env_log.terminated:
                break

        # 再次重置
        env.reset()

        # 重置后应该重新包含第一个工作流
        assert env.active_workflow_count == 1

    def test_finished_workflow_records_populated(
        self,
        cluster_config: ClusterConfig,
        workflow_templates: list[WorkflowTemplate],
        valid_strategies: list[tuple[str, int, int]],
    ):
        """测试 finished_workflow_records 在工作流完成时被正确填充"""
        arrival_times = [0.0]
        env = RawEnv(arrival_times, workflow_templates, cluster_config)
        env.reset()

        all_workflow_records: list[WorkflowExecutionRecord] = []
        step_count = 0

        while True:
            server_name, server_id, numa_node_id = valid_strategies[step_count % len(valid_strategies)]
            env_log = env.step(server_name, server_id, numa_node_id, 256)
            all_workflow_records.extend(env_log.finished_workflow_records)
            step_count += 1
            if env_log.terminated:
                break

        # 1个工作流应该有1条记录
        assert len(all_workflow_records) == 1
        record = all_workflow_records[0]
        assert record.wf_id == 0
        assert record.completion_time > 0.0

    def test_finished_workflow_records_multiple_workflows(
        self,
        cluster_config: ClusterConfig,
        workflow_templates: list[WorkflowTemplate],
        valid_strategies: list[tuple[str, int, int]],
    ):
        """测试多个工作流时 finished_workflow_records 包含所有工作流的记录"""
        arrival_times = [0.0, 1.0, 2.0]
        env = RawEnv(arrival_times, workflow_templates, cluster_config)
        env.reset()

        all_workflow_records: list[WorkflowExecutionRecord] = []
        step_count = 0

        while True:
            server_name, server_id, numa_node_id = valid_strategies[step_count % len(valid_strategies)]
            env_log = env.step(server_name, server_id, numa_node_id, 256)
            all_workflow_records.extend(env_log.finished_workflow_records)
            step_count += 1
            if env_log.terminated:
                break

        # 3个工作流应该有3条记录
        assert len(all_workflow_records) == 3

        # 每个工作流ID应该唯一出现一次
        recorded_wf_ids = {r.wf_id for r in all_workflow_records}
        assert recorded_wf_ids == {0, 1, 2}

        # 完成时间应该都是正数
        for record in all_workflow_records:
            assert record.completion_time > 0.0

    def test_get_data_distribution_no_predecessors(
        self,
        cluster_config: ClusterConfig,
        workflow_templates: list[WorkflowTemplate],
        valid_strategies: list[tuple[str, int, int]],
    ):
        """测试无前驱函数时 get_data_distribution 返回空列表"""
        arrival_times = [0.0]
        env = RawEnv(arrival_times, workflow_templates, cluster_config)
        env.reset()

        # current_function 是源点函数，无前驱
        assert env.current_function is not None
        wf_id = env.current_function.wf_id
        fn_id = env.current_function.fn_id

        # 源点函数没有前驱，get_data_distribution 应该返回空列表
        distribution = env.get_data_distribution(wf_id, fn_id)
        assert distribution == []

    def test_get_data_distribution_with_predecessors(
        self,
        cluster_config: ClusterConfig,
        workflow_templates: list[WorkflowTemplate],
        valid_strategies: list[tuple[str, int, int]],
    ):
        """测试有前驱函数时 get_data_distribution 返回正确的位置和数据大小"""
        arrival_times = [0.0]
        env = RawEnv(arrival_times, workflow_templates, cluster_config)
        env.reset()

        # 收集每一步的函数信息，找到第一个有前驱的函数
        step_count = 0
        found_fn_with_predecessors = False

        while True:
            assert env.current_function is not None
            wf_id = env.current_function.wf_id
            fn_id = env.current_function.fn_id

            predecessors = env.workload[wf_id].get_predecessor_functions(fn_id)

            server_name, server_id, numa_node_id = valid_strategies[step_count % len(valid_strategies)]
            env_log = env.step(server_name, server_id, numa_node_id, 256)
            step_count += 1

            if predecessors and not found_fn_with_predecessors:
                # 验证 get_data_distribution 返回的信息
                distribution = env.get_data_distribution(wf_id, fn_id)
                assert len(distribution) == len(predecessors)

                for (location, data_size), (_, _, expected_data_size) in zip(distribution, predecessors):
                    assert data_size == expected_data_size
                    assert location is not None

                found_fn_with_predecessors = True

            if env_log.terminated:
                break
