"""memory_generator.py

为 Pegasus DAX 文件生成对应的内存需求 JSON 文件：

{
    "memory_reqs": [
        {"id": 0, "value": 256},
        {"id": 1, "value": 512},
        ...
    ]
}

说明：
- value 为函数所需内存大小 (单位：MB)
"""

import xml.etree.ElementTree as ET
from pathlib import Path

from .pretty_json import pretty_json_dump


def generate_memory_requirements(path: str) -> str:
    """在同目录下为 DAX 文件生成对应的内存需求 JSON 文件

    如果 JSON 文件已存在，则跳过生成。

    Args:
        path (str): DAX 文件路径

    Returns:
        str: 生成的内存需求 JSON 文件路径
    """

    dax_path: Path = Path(path)
    print(f"Generating memory requirements for DAX file: {dax_path}")

    json_path = dax_path.with_suffix(".memory.json")
    if json_path.exists():
        print(f"Memory requirements JSON file already exists: {json_path}, skipping generation.")
        return str(json_path)

    tree = ET.parse(dax_path)
    root = tree.getroot()

    # 从 root tag 中获取 namespace (此处为 http://pegasus.isi.edu/schema/DAX)
    ns = {"dax": root.tag.split("}")[0][1:]} if "}" in root.tag else {}

    # 获得 job 的总数
    job_xpath = ".//dax:job" if ns else ".//job"
    n_jobs = len(root.findall(job_xpath, ns))

    result = {
        "memory_reqs": [
            {"id": job_id, "value": mem_req} for job_id, mem_req in enumerate(get_memory_requirements(n_jobs))
        ]
    }

    # 按 ID 排序
    result["memory_reqs"].sort(key=lambda x: x["id"])

    # 写入 JSON 文件
    pretty_json_dump(str(json_path), memory_reqs=result["memory_reqs"])

    print(f"Generated memory requirements JSON file: {json_path}")
    return str(json_path)


def get_memory_requirements(n: int) -> list[int]:
    """获得指定数量的内存需求

    Args:
        n (int): 内存需求的数量

    Returns:
        list[int]: 内存需求列表 (单位：MB)
    """

    # 此处为示例，实际可根据需求调整内存需求生成逻辑，或从数据集中读取
    memory_reqs = [128] * n

    return memory_reqs
