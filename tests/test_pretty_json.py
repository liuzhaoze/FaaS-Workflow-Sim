"""test_pretty_json.py

测试 src/faas_workflow_sim/tools/pretty_json.py 中的函数
"""

import tempfile
from pathlib import Path

from faas_workflow_sim.tools.dax_parser import EdgeInfo, NodeInfo
from faas_workflow_sim.tools.pretty_json import pretty_json_dump


class TestPrettyJsonDump:
    """测试 pretty_json_dump 函数"""

    def test_pretty_json_dump_single_list(self):
        """测试单个列表的 JSON 输出"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            temp_file = f.name

        try:
            nodes: list[NodeInfo] = [{"id": 0, "runtime": 10.5}, {"id": 1, "runtime": 15.2}]
            pretty_json_dump(temp_file, nodes=nodes)

            # 读取并验证内容
            content = Path(temp_file).read_text(encoding="utf-8")

            # 基本格式检查
            assert content.startswith("{")
            assert content.endswith("}")
            assert '"nodes"' in content
            assert '"id"' in content
            assert '"runtime"' in content

            # 验证包含所有节点
            assert '"id": 0' in content
            assert '"runtime": 10.5' in content
            assert '"id": 1' in content
            assert '"runtime": 15.2' in content

            # 验证格式（缩进和换行）
            lines = content.split("\n")
            assert len(lines) > 4  # 应该有多行
            assert "    " in content  # 应该有缩进

        finally:
            Path(temp_file).unlink()

    def test_pretty_json_dump_multiple_lists(self):
        """测试多个列表的 JSON 输出"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            temp_file = f.name

        try:
            nodes: list[NodeInfo] = [{"id": 0, "runtime": 10.5}]
            edges: list[EdgeInfo] = [{"parent": 0, "child": 1, "size_bytes": 1024}]
            memory_reqs = [{"id": 0, "value": 512}]

            pretty_json_dump(temp_file, nodes=nodes, edges=edges, memory_reqs=memory_reqs)

            # 读取并验证内容
            content = Path(temp_file).read_text(encoding="utf-8")

            # 验证包含所有键
            assert '"nodes"' in content
            assert '"edges"' in content
            assert '"memory_reqs"' in content

            # 验证包含所有数据
            assert '"runtime": 10.5' in content
            assert '"size_bytes": 1024' in content
            assert '"value": 512' in content

            # 验证逗号分隔
            lines = content.split("\n")
            # 找到包含 ], 的行（除了最后一个）
            lines_with_comma = [line for line in lines if "]," in line]
            assert len(lines_with_comma) >= 1  # 至少有一个键后面有逗号

        finally:
            Path(temp_file).unlink()

    def test_pretty_json_dump_empty_lists(self):
        """测试空列表的 JSON 输出"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            temp_file = f.name

        try:
            pretty_json_dump(temp_file, nodes=[], edges=[])

            content = Path(temp_file).read_text(encoding="utf-8")

            assert '"nodes": [\n    ]' in content
            assert '"edges": [\n    ]' in content

        finally:
            Path(temp_file).unlink()

    def test_pretty_json_dump_custom_tab_size(self):
        """测试自定义缩进大小"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            temp_file = f.name

        try:
            nodes: list[NodeInfo] = [{"id": 0, "runtime": 10.5}]
            pretty_json_dump(temp_file, tab_size=2, nodes=nodes)

            content = Path(temp_file).read_text(encoding="utf-8")

            # 验证使用自定义缩进
            assert "  " in content  # 2个空格的缩进

        finally:
            Path(temp_file).unlink()

    def test_pretty_json_dump_complex_objects(self):
        """测试复杂对象的 JSON 输出"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            temp_file = f.name

        try:
            complex_nodes: list[dict[str, object]] = [
                {
                    "id": 0,
                    "runtime": 10.5,
                    "metadata": {
                        "name": "task1",
                        "tags": ["cpu", "memory"],
                        "config": {"threads": 4, "priority": "high"},
                    },
                }
            ]

            pretty_json_dump(temp_file, nodes=complex_nodes)

            content = Path(temp_file).read_text(encoding="utf-8")

            # 验证复杂对象被正确序列化
            assert '"metadata"' in content
            assert '"tags"' in content
            assert '"config"' in content
            assert '"threads": 4' in content
            assert '"priority": "high"' in content

        finally:
            Path(temp_file).unlink()

    def test_pretty_json_dump_unicode_content(self):
        """测试包含 Unicode 字符的内容"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            temp_file = f.name

        try:
            unicode_nodes: list[dict[str, object]] = [
                {"id": 0, "name": "测试任务", "description": "这是一个包含中文的任务"},
                {"id": 1, "name": "Tâche Française", "description": "Une tâche avec des caractères spéciaux"},
            ]

            pretty_json_dump(temp_file, nodes=unicode_nodes)

            content = Path(temp_file).read_text(encoding="utf-8")

            # 验证 Unicode 字符被正确处理
            assert "测试任务" in content
            assert "这是一个包含中文的任务" in content
            assert "Tâche Française" in content
            assert "Une tâche avec des caractères spéciaux" in content

        finally:
            Path(temp_file).unlink()

    def test_pretty_json_dump_special_characters(self):
        """测试特殊字符的处理"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            temp_file = f.name

        try:
            special_nodes: list[dict[str, object]] = [
                {"id": 0, "path": "C:\\Windows\\System32", "regex": "pattern.*match"},
                {"id": 1, "quote": 'He said "Hello"', "newlines": "Line1\nLine2"},
            ]

            pretty_json_dump(temp_file, nodes=special_nodes)

            content = Path(temp_file).read_text(encoding="utf-8")

            # 验证特殊字符被正确转义
            assert "C:\\\\Windows\\\\System32" in content
            assert "pattern.*match" in content
            assert 'He said \\"Hello\\"' in content
            assert "Line1\\nLine2" in content

        finally:
            Path(temp_file).unlink()

    def test_pretty_json_dump_numeric_types(self):
        """测试不同数值类型的处理"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            temp_file = f.name

        try:
            numeric_nodes: list[dict[str, object]] = [
                {"id": 0, "integer": 42, "float": 3.14159, "scientific": 1.23e-4, "zero": 0, "negative": -17}
            ]

            pretty_json_dump(temp_file, nodes=numeric_nodes)

            content = Path(temp_file).read_text(encoding="utf-8")

            # 验证数值类型被正确处理
            assert '"integer": 42' in content
            assert '"float": 3.14159' in content
            assert '"scientific": 0.000123' in content
            assert '"zero": 0' in content
            assert '"negative": -17' in content

        finally:
            Path(temp_file).unlink()
