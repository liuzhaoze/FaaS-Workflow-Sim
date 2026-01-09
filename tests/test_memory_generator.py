"""test_memory_generator.py

测试 src/faas_workflow_sim/tools/memory_generator.py 中的函数
"""

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pytest import FixtureRequest

from faas_workflow_sim.tools.memory_generator import generate_memory_requirements, get_memory_requirements


class TestGetMemoryRequirements:
    """测试 get_memory_requirements 函数"""

    def test_get_memory_requirements_zero(self):
        """测试零个内存需求"""
        result = get_memory_requirements(0)
        assert result == []

    def test_get_memory_requirements_one(self):
        """测试一个内存需求"""
        result = get_memory_requirements(1)
        assert result == [128]

    def test_get_memory_requirements_multiple(self):
        """测试多个内存需求"""
        result = get_memory_requirements(5)
        assert result == [128, 128, 128, 128, 128]

    def test_get_memory_requirements_consistency(self):
        """测试内存需求的一致性"""
        n = 10
        result = get_memory_requirements(n)
        assert len(result) == n
        assert all(mem == 128 for mem in result)

    def test_get_memory_requirements_large_number(self):
        """测试大数量的内存需求"""
        n = 1000
        result = get_memory_requirements(n)
        assert len(result) == n
        assert all(mem == 128 for mem in result)


class TestGenerateMemoryRequirements:
    """测试 generate_memory_requirements 函数"""

    # 测试数据目录的绝对路径
    DATA_DIR = Path(__file__).parent / "data"

    @pytest.fixture(params=["CYBERSHAKE.n.200.0.dax", "MONTAGE.n.200.0.dax"])
    def dax_file(self, request: FixtureRequest) -> Path:
        """用于测试的 DAX 文件"""
        return self.DATA_DIR / request.param

    def test_generate_memory_requirements_success(self, dax_file: Path):
        """测试成功生成内存需求 JSON 文件"""
        generate_memory_requirements(str(dax_file))

        # 检查生成的 JSON 文件
        json_path = dax_file.with_suffix(".memory.json")
        assert json_path.exists()

        # 验证 JSON 内容
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "memory_reqs" in data
        assert len(data["memory_reqs"]) == 200

        # 验证内存需求数据
        for i, mem_req in enumerate(data["memory_reqs"]):
            assert mem_req["id"] == i
            assert mem_req["value"] == 128

        # 清理文件
        json_path.unlink()

    def test_generate_memory_requirements_skip_existing(self, dax_file: Path):
        """测试跳过已存在的 JSON 文件"""
        # 首先生成内存需求 JSON 文件
        generate_memory_requirements(str(dax_file))

        json_path = dax_file.with_suffix(".memory.json")
        assert json_path.exists()

        # 记录原始修改时间和内容
        original_mtime = json_path.stat().st_mtime
        original_content = json_path.read_text(encoding="utf-8")

        with patch("builtins.print") as mock_print:
            generate_memory_requirements(str(dax_file))
            mock_print.assert_any_call(
                f"Memory requirements JSON file already exists: {json_path}, skipping generation."
            )

        # 验证文件没有被修改
        assert json_path.stat().st_mtime == original_mtime
        assert json_path.read_text(encoding="utf-8") == original_content

        # 清理文件
        json_path.unlink()

    def test_generate_memory_requirements_json_format(self, dax_file: Path):
        """测试生成的 JSON 文件格式"""
        generate_memory_requirements(str(dax_file))

        json_path = dax_file.with_suffix(".memory.json")
        with open(json_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 验证基本格式
        assert content.startswith("{")
        assert content.endswith("}")
        assert '"memory_reqs"' in content
        assert '"id"' in content
        assert '"value"' in content

        # 验证可以正确解析为 JSON
        data: dict[str, list[dict[str, int]]] = json.loads(content)
        assert isinstance(data, dict)
        assert isinstance(data["memory_reqs"], list)
        assert len(data["memory_reqs"]) == 200

        # 清理文件
        json_path.unlink()

    @patch("builtins.print")
    def test_generate_memory_requirements_prints_messages(self, mock_print: MagicMock, dax_file: Path):
        """测试 generate_memory_requirements 打印的消息"""
        generate_memory_requirements(str(dax_file))

        # 验证打印的消息
        expected_calls = [
            f"Generating memory requirements for DAX file: {dax_file}",
            f"Generated memory requirements JSON file: {dax_file.with_suffix('.memory.json')}",
        ]

        call_args = [str(call[0][0]) for call in mock_print.call_args_list]

        assert any(expected_calls[0] in call for call in call_args)
        assert any(expected_calls[1] in call for call in call_args)

        # 清理文件
        json_path = dax_file.with_suffix(".memory.json")
        if json_path.exists():
            json_path.unlink()

    def test_generate_memory_requirements_sorted_output(self, dax_file: Path):
        """测试输出按 ID 排序"""
        generate_memory_requirements(str(dax_file))

        json_path = dax_file.with_suffix(".memory.json")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 验证按 ID 排序
        ids = [mem_req["id"] for mem_req in data["memory_reqs"]]
        assert ids == sorted(ids)
        assert len(ids) == 200  # 验证是实际的数据文件

        # 清理文件
        json_path.unlink()

    def test_generate_memory_requirements_multiple_runs(self, dax_file: Path):
        """测试多次运行的行为"""
        # 第一次运行
        generate_memory_requirements(str(dax_file))

        json_path = dax_file.with_suffix(".memory.json")
        assert json_path.exists()

        # 记录第一次运行后的修改时间
        first_mtime = json_path.stat().st_mtime

        # 等待一小段时间以确保时间戳不同
        time.sleep(0.1)

        # 第二次运行应该跳过
        with patch("builtins.print") as mock_print:
            generate_memory_requirements(str(dax_file))
            mock_print.assert_any_call(
                f"Memory requirements JSON file already exists: {json_path}, skipping generation."
            )

        # 验证文件没有被修改
        assert json_path.stat().st_mtime == first_mtime

        # 清理文件
        json_path.unlink()
