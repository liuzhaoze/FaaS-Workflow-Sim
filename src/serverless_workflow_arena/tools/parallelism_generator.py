"""parallelism_generator.py

为 Pegasus DAX 文件生成对应的并行度 JSON 文件：

{
    "parallelisms": [
        {"id": 0, "value": 1},
        {"id": 1, "value": 2},
        ...
    ]
}

说明：
- value 为函数的并行度
"""

import xml.etree.ElementTree as ET
from pathlib import Path

from .pretty_json import pretty_json_dump


def generate_parallelisms(path: str) -> str:
    """在同目录下为 DAX 文件生成对应的并行度 JSON 文件

    如果 JSON 文件已存在，则跳过生成。

    Args:
        path (str): DAX 文件路径

    Returns:
        str: 生成的并行度 JSON 文件路径
    """

    dax_path: Path = Path(path)
    print(f"Generating parallelisms for DAX file: {dax_path}")

    json_path = dax_path.with_suffix(".parallelism.json")
    if json_path.exists():
        print(f"Parallelisms JSON file already exists: {json_path}, skipping generation.")
        return str(json_path)

    tree = ET.parse(dax_path)
    root = tree.getroot()

    # 从 root tag 中获取 namespace (此处为 http://pegasus.isi.edu/schema/DAX)
    ns = {"dax": root.tag.split("}")[0][1:]} if "}" in root.tag else {}

    # 获得 job 的总数
    job_xpath = ".//dax:job" if ns else ".//job"
    n_jobs = len(root.findall(job_xpath, ns))

    result = {"parallelisms": [{"id": job_id, "value": par} for job_id, par in enumerate(get_parallelisms(n_jobs))]}

    # 按 ID 排序
    result["parallelisms"].sort(key=lambda x: x["id"])

    # 写入 JSON 文件
    pretty_json_dump(str(json_path), parallelisms=result["parallelisms"])

    print(f"Generated parallelisms JSON file: {json_path}")
    return str(json_path)


def get_parallelisms(n: int) -> list[int]:
    """获得指定数量的并行度

    Args:
        n (int): 并行度的数量

    Returns:
        list[int]: 并行度列表
    """

    # 此处为示例，实际可根据需求调整并行度生成逻辑，或从数据集中读取
    parallelisms = [1] * n

    return parallelisms
