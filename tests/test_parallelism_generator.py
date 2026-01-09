"""test_parallelism_generator.py

测试 src/faas_workflow_sim/tools/parallelism_generator.py 中的函数
"""

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pytest import FixtureRequest

from faas_workflow_sim.tools.parallelism_generator import generate_parallelisms, get_parallelisms


class TestGetParallelisms:
    """测试 get_parallelisms 函数"""

    def test_get_parallelisms_zero(self):
        """测试零个并行度"""
        result = get_parallelisms(0)
        assert result == []

    def test_get_parallelisms_one(self):
        """测试一个并行度"""
        result = get_parallelisms(1)
        assert result == [1]

    def test_get_parallelisms_multiple(self):
        """测试多个并行度"""
        result = get_parallelisms(5)
        assert result == [1, 1, 1, 1, 1]

    def test_get_parallelisms_consistency(self):
        """测试并行度的一致性"""
        n = 10
        result = get_parallelisms(n)
        assert len(result) == n
        assert all(par == 1 for par in result)

    def test_get_parallelisms_large_number(self):
        """测试大数量的并行度"""
        n = 1000
        result = get_parallelisms(n)
        assert len(result) == n
        assert all(par == 1 for par in result)


class TestGenerateParallelisms:
    """测试 generate_parallelisms 函数"""

    # 测试数据目录的绝对路径
    DATA_DIR = Path(__file__).parent / "data"

    @pytest.fixture(params=["CYBERSHAKE.n.200.0.dax", "MONTAGE.n.200.0.dax"])
    def dax_file(self, request: FixtureRequest) -> Path:
        """用于测试的 DAX 文件"""
        return self.DATA_DIR / request.param

    def test_generate_parallelisms_success(self, dax_file: Path):
        """测试成功生成并行度 JSON 文件"""
        generate_parallelisms(str(dax_file))

        # 检查生成的 JSON 文件
        json_path = dax_file.with_suffix(".parallelism.json")
        assert json_path.exists()

        # 验证 JSON 内容
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "parallelisms" in data
        assert len(data["parallelisms"]) == 200

        # 验证并行度数据
        for i, par in enumerate(data["parallelisms"]):
            assert par["id"] == i
            assert par["value"] == 1

        # 清理文件
        json_path.unlink()

    def test_generate_parallelisms_skip_existing(self, dax_file: Path):
        """测试跳过已存在的 JSON 文件"""
        # 首先生成并行度 JSON 文件
        generate_parallelisms(str(dax_file))

        json_path = dax_file.with_suffix(".parallelism.json")
        assert json_path.exists()

        # 记录原始修改时间和内容
        original_mtime = json_path.stat().st_mtime
        original_content = json_path.read_text(encoding="utf-8")

        with patch("builtins.print") as mock_print:
            generate_parallelisms(str(dax_file))
            mock_print.assert_any_call(f"Parallelisms JSON file already exists: {json_path}, skipping generation.")

        # 验证文件没有被修改
        assert json_path.stat().st_mtime == original_mtime
        assert json_path.read_text(encoding="utf-8") == original_content

        # 清理文件
        json_path.unlink()

    def test_generate_parallelisms_json_format(self, dax_file: Path):
        """测试生成的 JSON 文件格式"""
        generate_parallelisms(str(dax_file))

        json_path = dax_file.with_suffix(".parallelism.json")
        with open(json_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 验证基本格式
        assert content.startswith("{")
        assert content.endswith("}")
        assert '"parallelisms"' in content
        assert '"id"' in content
        assert '"value"' in content

        # 验证可以正确解析为 JSON
        data: dict[str, list[dict[str, int]]] = json.loads(content)
        assert isinstance(data, dict)
        assert isinstance(data["parallelisms"], list)
        assert len(data["parallelisms"]) == 200

        # 清理文件
        json_path.unlink()

    @patch("builtins.print")
    def test_generate_parallelisms_prints_messages(self, mock_print: MagicMock, dax_file: Path):
        """测试 generate_parallelisms 打印的消息"""
        generate_parallelisms(str(dax_file))

        # 验证打印的消息
        expected_calls = [
            f"Generating parallelisms for DAX file: {dax_file}",
            f"Generated parallelisms JSON file: {dax_file.with_suffix('.parallelism.json')}",
        ]

        call_args = [str(call[0][0]) for call in mock_print.call_args_list]

        assert any(expected_calls[0] in call for call in call_args)
        assert any(expected_calls[1] in call for call in call_args)

        # 清理文件
        json_path = dax_file.with_suffix(".parallelism.json")
        if json_path.exists():
            json_path.unlink()

    def test_generate_parallelisms_sorted_output(self, dax_file: Path):
        """测试输出按 ID 排序"""
        generate_parallelisms(str(dax_file))

        json_path = dax_file.with_suffix(".parallelism.json")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 验证按 ID 排序
        ids = [par["id"] for par in data["parallelisms"]]
        assert ids == sorted(ids)
        assert len(ids) == 200  # 验证是实际的数据文件

        # 清理文件
        json_path.unlink()

    def test_generate_parallelisms_multiple_runs(self, dax_file: Path):
        """测试多次运行的行为"""
        # 第一次运行
        generate_parallelisms(str(dax_file))

        json_path = dax_file.with_suffix(".parallelism.json")
        assert json_path.exists()

        # 记录第一次运行后的修改时间
        first_mtime = json_path.stat().st_mtime

        # 等待一小段时间以确保时间戳不同
        time.sleep(0.1)

        # 第二次运行应该跳过
        with patch("builtins.print") as mock_print:
            generate_parallelisms(str(dax_file))
            mock_print.assert_any_call(f"Parallelisms JSON file already exists: {json_path}, skipping generation.")

        # 验证文件没有被修改
        assert json_path.stat().st_mtime == first_mtime

        # 清理文件
        json_path.unlink()
