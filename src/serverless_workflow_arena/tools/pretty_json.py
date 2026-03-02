import json
from typing import Mapping, Sequence


def pretty_json_dump(path: str, tab_size: int = 4, **kwargs: Sequence[Mapping[str, object]]):
    """按行打印列表中的对象

    Args:
        path (str): 输出文件路径
        **kwargs: 需要打印的键值对，值为 list[dict]
    """

    tab = " " * tab_size
    with open(path, "w", encoding="utf-8") as f:
        f.write("{\n")

        for i, (k, v) in enumerate(kwargs.items()):
            # 写入键
            f.write(f'{tab}"{k}": [\n')

            # 写入列表中的每个对象
            for j, item in enumerate(v):
                j_comma = "," if j < len(v) - 1 else ""
                f.write(tab * 2 + json.dumps(item, ensure_ascii=False) + j_comma + "\n")

            i_comma = "," if i < len(kwargs) - 1 else ""
            f.write(f"{tab}]{i_comma}\n")

        f.write("}")
