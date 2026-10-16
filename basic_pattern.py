"""使用 FLEXIS 挖掘单张图中的频繁子图，并编码为 <1>/<2>/...。"""

from __future__ import annotations

import copy
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Hashable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class _Pattern:
    """FLEXIS 输出的一个频繁子图。"""

    graph: Any


class _LabelTable:
    """把 NetworkX 图标签映射为 FLEXIS 所需的连续整数。"""

    def __init__(self) -> None:
        self._ids: Dict[Tuple[type, Hashable], int] = {}
        self.values: List[Any] = []

    def encode(self, value: Any) -> int:
        try:
            key = (type(value), value)
            hash(key)
        except TypeError:
            key = (type(value), repr(value))

        if key not in self._ids:
            self._ids[key] = len(self.values)
            self.values.append(value)
        return self._ids[key]


def _networkx():
    try:
        import networkx as nx
    except ImportError as exc:
        raise RuntimeError(
            "缺少 NetworkX 依赖，请先运行 `pip install -r requirements.txt`。"
        ) from exc
    return nx


def _find_executable(explicit_path: Optional[str], env_name: str, name: str) -> str:
    """从显式参数、环境变量、常见构建目录或 PATH 定位 FLEXIS 可执行文件。"""
    workspace = Path(__file__).resolve().parent
    root_value = os.environ.get("FLEXIS_ROOT")
    candidates: List[Path] = []

    if explicit_path:
        candidates.append(Path(explicit_path))
    if os.environ.get(env_name):
        candidates.append(Path(os.environ[env_name]))
    if root_value:
        candidates.append(Path(root_value) / "build" / "release" / "bin" / name)

    for folder in (
        "external/flexis",
        "third_party/flexis",
        "flexis",
    ):
        candidates.append(workspace / folder / "build" / "release" / "bin" / name)

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())

    path_match = shutil.which(name) or shutil.which(Path(name).stem)
    if path_match:
        return path_match

    env_hint = f"{env_name}"
    raise FileNotFoundError(
        f"找不到 FLEXIS 可执行文件 {name}。请从 "
        "https://github.com/akshit-sharma/flexis 获取源码并按 README 编译，"
        f"再设置环境变量 {env_hint}；也可设置 FLEXIS_ROOT 指向源码目录。"
    )


def _write_flexis_lg(graph: Any, output_path: Path) -> Tuple[List[Any], List[Any]]:
    """将 NetworkX 图写成 FLEXIS 预处理器支持的 GraphAMi .lg 格式。"""
    if graph.is_multigraph():
        raise TypeError("FLEXIS 输入不支持平行边；请先将 MultiGraph 转为简单图。")

    nodes = list(graph.nodes)
    node_ids = {node: index for index, node in enumerate(nodes)}
    node_labels = _LabelTable()
    edge_labels = _LabelTable()
    edges: List[Tuple[int, int, int]] = []

    for source, target, data in graph.edges(data=True):
        edge_label_id = edge_labels.encode(data.get("label", 0))
        source_id = node_ids[source]
        target_id = node_ids[target]
        edges.append((source_id, target_id, edge_label_id))
        if not graph.is_directed() and source_id != target_id:
            edges.append((target_id, source_id, edge_label_id))

    with output_path.open("w", encoding="utf-8", newline="\n") as output:
        output.write("# t 1\n")
        for index, node in enumerate(nodes):
            label_id = node_labels.encode(graph.nodes[node].get("label", 0))
            output.write(f"v {index} {label_id}\n")
        for source_id, target_id, edge_label_id in edges:
            output.write(f"e {source_id} {target_id} {edge_label_id}\n")

    return node_labels.values, edge_labels.values


def _pattern_key(pattern: Any) -> Tuple[Any, ...]:
    """给相同大小的频繁模式一个稳定的选择顺序。"""
    nodes = tuple(
        sorted(
            (repr(node), repr(data.get("label", 0)))
            for node, data in pattern.nodes(data=True)
        )
    )
    edges = tuple(
        sorted(
            (repr(source), repr(target), repr(data.get("label", 0)))
            for source, target, data in pattern.edges(data=True)
        )
    )
    return pattern.number_of_nodes(), pattern.number_of_edges(), nodes, edges


def _parse_frequent_patterns(
    output_path: Path,
    directed: bool,
    node_label_values: Sequence[Any],
    edge_label_values: Sequence[Any],
) -> Optional[_Pattern]:
    """读取 FLEXIS 的 --print-patterns 输出，选最后一个非空层中最大的模式。"""
    nx = _networkx()
    marker = re.compile(r"^Frequent patterns:\s*(\d+)\s*$")
    current_section = False
    section_best = None
    latest_nonempty = None
    current_pattern = None

    def finish_pattern() -> None:
        nonlocal current_pattern, section_best
        if current_pattern is None:
            return

        node_count = current_pattern["node_count"]
        edge_count = current_pattern["edge_count"]
        nodes = current_pattern["nodes"]
        edges = current_pattern["edges"]
        current_pattern = None

        if len(nodes) != node_count or len(edges) != edge_count:
            return
        if node_count < 2 or edge_count == 0:
            return

        pattern_graph = nx.DiGraph() if directed else nx.Graph()
        for node_id in range(node_count):
            label_id = nodes[node_id]
            label = (
                node_label_values[label_id]
                if 0 <= label_id < len(node_label_values)
                else label_id
            )
            pattern_graph.add_node(node_id, label=label)

        for source, target, label_id in edges:
            label = (
                edge_label_values[label_id]
                if 0 <= label_id < len(edge_label_values)
                else label_id
            )
            pattern_graph.add_edge(source, target, label=label)

        candidate = _Pattern(pattern_graph)
        if section_best is None or _pattern_key(candidate.graph) > _pattern_key(
            section_best.graph
        ):
            section_best = candidate

    with output_path.open("r", encoding="utf-8", errors="replace") as output:
        for raw_line in output:
            line = raw_line.strip()
            found = marker.match(line)
            if found:
                finish_pattern()
                if current_section and section_best is not None:
                    latest_nonempty = section_best
                current_section = True
                section_best = None
                continue

            if not current_section:
                continue

            parts = line.split()
            if not parts:
                continue

            if parts[0] == "p" and len(parts) >= 3:
                finish_pattern()
                try:
                    current_pattern = {
                        "node_count": int(parts[1]),
                        "edge_count": int(parts[2]),
                        "nodes": {},
                        "edges": [],
                    }
                except ValueError:
                    current_pattern = None
                continue

            if current_pattern is None:
                continue

            try:
                if parts[0] == "v" and len(parts) >= 3:
                    current_pattern["nodes"][int(parts[1])] = int(parts[2])
                elif parts[0] == "e" and len(parts) >= 4:
                    current_pattern["edges"].append(
                        (int(parts[1]), int(parts[2]), int(parts[3]))
                    )
                else:
                    finish_pattern()
                    continue
            except ValueError:
                finish_pattern()
                continue

            if (
                len(current_pattern["nodes"]) == current_pattern["node_count"]
                and len(current_pattern["edges"]) == current_pattern["edge_count"]
            ):
                finish_pattern()

    finish_pattern()
    if current_section and section_best is not None:
        latest_nonempty = section_best
    return latest_nonempty


def freq_mining(
    graph: Any,
    bound: int,
    *,
    flexis_bin: Optional[str] = None,
    process_bin: Optional[str] = None,
    max_pattern_size: int = 6,
    slider: float = 0.98,
) -> Optional[_Pattern]:
    """调用 FLEXIS 单图频繁子图挖掘，返回一个频繁模式或 None。

    ``bound`` 是 FLEXIS 的整数支持度阈值；``max_pattern_size`` 是模式最大
    顶点数。FLEXIS 的频繁度使用其 MIS 支持度定义，``slider`` 控制该指标。
    """
    if not isinstance(bound, int) or isinstance(bound, bool) or bound < 1:
        raise ValueError("bound 必须是大于等于 1 的整数支持度阈值。")
    if not 2 <= max_pattern_size <= 10:
        raise ValueError("FLEXIS 的 max_pattern_size 必须介于 2 和 10 之间。")
    if not 0.0 <= slider <= 1.0:
        raise ValueError("slider 必须介于 0.0 和 1.0 之间。")
    if graph.number_of_nodes() < 2 or graph.number_of_edges() == 0:
        return None

    flexis_exe = _find_executable(flexis_bin, "FLEXIS_BIN", "flexis.exe")
    process_exe = _find_executable(process_bin, "FLEXIS_PROCESS_BIN", "process.exe")

    with tempfile.TemporaryDirectory(prefix="flexis_pattern_") as temp_dir:
        temp_path = Path(temp_dir)
        lg_path = temp_path / "input.lg"
        binary_prefix = temp_path / "input"
        binary_path = Path(str(binary_prefix) + ".vf3.bin")
        stdout_path = temp_path / "flexis.stdout.txt"
        node_label_values, edge_label_values = _write_flexis_lg(graph, lg_path)

        preprocess_command = [
            process_exe,
            "--input",
            str(lg_path),
            "--output",
            str(binary_prefix),
            "--format",
            "lg",
            "--seed",
            "0",
        ]
        preprocessed = subprocess.run(
            preprocess_command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if preprocessed.returncode != 0 or not binary_path.is_file():
            details = (preprocessed.stderr or preprocessed.stdout).strip()
            raise RuntimeError(
                "FLEXIS 图预处理失败。"
                + (f"\n上游输出：{details[-3000:]}" if details else "")
            )

        mining_command = [
            flexis_exe,
            "--input",
            str(binary_path),
            "--support",
            str(bound),
            "--max-size",
            str(max_pattern_size),
            "--slider",
            str(slider),
            "--print-output",
            "--print-patterns",
        ]
        with stdout_path.open("w", encoding="utf-8", newline="\n") as stdout_file:
            mined = subprocess.run(
                mining_command,
                stdout=stdout_file,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
        if mined.returncode != 0:
            details = (mined.stderr or "").strip()
            raise RuntimeError(
                "FLEXIS 频繁子图挖掘失败。"
                + (f"\n上游错误：{details[-3000:]}" if details else "")
            )

        return _parse_frequent_patterns(
            stdout_path,
            graph.is_directed(),
            node_label_values,
            edge_label_values,
        )


def _edge_provenance(
    source: Any, target: Any, data: Dict[str, Any]
) -> List[Tuple[Any, Any, Dict[str, Any]]]:
    existing = data.get("provenance")
    if existing:
        return list(existing)
    return [(source, target, copy.deepcopy(data))]


def super_node(graph: Any, pattern: _Pattern, token: str) -> Any:
    """把模式的贪心最大不重叠实例收缩为带来源信息的超节点。"""
    nx = _networkx()
    pattern_graph = pattern.graph
    if graph.is_directed() != pattern_graph.is_directed():
        raise ValueError("待收缩图与频繁模式的有向属性不一致。")

    node_match = lambda left, right: left.get("label", 0) == right.get("label", 0)
    edge_match = lambda left, right: left.get("label", 0) == right.get("label", 0)
    if graph.is_directed():
        matcher = nx.algorithms.isomorphism.DiGraphMatcher(
            graph, pattern_graph, node_match=node_match, edge_match=edge_match
        )
    else:
        matcher = nx.algorithms.isomorphism.GraphMatcher(
            graph, pattern_graph, node_match=node_match, edge_match=edge_match
        )

    selected: List[Tuple[Any, ...]] = []
    used_nodes = set()
    for mapping in matcher.subgraph_monomorphisms_iter():
        occurrence = tuple(mapping.keys())
        if len(occurrence) != pattern_graph.number_of_nodes():
            continue
        if any(node in used_nodes for node in occurrence):
            continue
        selected.append(occurrence)
        used_nodes.update(occurrence)

    if not selected:
        raise RuntimeError(
            "FLEXIS 返回了频繁模式，但 NetworkX 未能在当前图中找到可收缩实例。"
        )

    contracted = nx.DiGraph() if graph.is_directed() else nx.Graph()
    contracted.graph.update(graph.graph)
    replacement: Dict[Any, Any] = {}
    cluster_nodes: Dict[Any, Tuple[Any, ...]] = {}

    for node in graph.nodes:
        replacement[node] = node
    for occurrence_index, occurrence in enumerate(selected, start=1):
        supernode = ("__basic_pattern_supernode__", token, occurrence_index)
        while supernode in graph or supernode in cluster_nodes:
            occurrence_index += 1
            supernode = ("__basic_pattern_supernode__", token, occurrence_index)
        cluster_nodes[supernode] = occurrence
        for node in occurrence:
            replacement[node] = supernode

    for node, data in graph.nodes(data=True):
        if node not in used_nodes:
            contracted.add_node(node, **copy.deepcopy(data))

    for supernode, occurrence in cluster_nodes.items():
        members: List[Any] = []
        internal_edges: List[Tuple[Any, Any, Dict[str, Any]]] = []
        occurrence_set = set(occurrence)
        for node in occurrence:
            node_data = graph.nodes[node]
            members.extend(node_data.get("members", (node,)))
            internal_edges.extend(copy.deepcopy(node_data.get("internal_edges", ())))
        for source, target, data in graph.edges(data=True):
            if source in occurrence_set and target in occurrence_set:
                internal_edges.extend(_edge_provenance(source, target, data))

        contracted.add_node(
            supernode,
            label=token,
            kind="basic_pattern_supernode",
            pattern_id=token,
            members=tuple(members),
            internal_edges=tuple(internal_edges),
        )

    grouped_edges: Dict[Tuple[Any, Any], List[Tuple[Any, Any, Dict[str, Any]]]] = {}
    stable_order = {node: index for index, node in enumerate(contracted.nodes)}
    for source, target, data in graph.edges(data=True):
        new_source = replacement[source]
        new_target = replacement[target]
        if new_source == new_target and new_source in cluster_nodes:
            continue
        if not graph.is_directed() and stable_order.get(new_source, 0) > stable_order.get(
            new_target, 0
        ):
            new_source, new_target = new_target, new_source
        grouped_edges.setdefault((new_source, new_target), []).append(
            (source, target, data)
        )

    for (source, target), old_edges in grouped_edges.items():
        provenance: List[Tuple[Any, Any, Dict[str, Any]]] = []
        labels: List[Any] = []
        for old_source, old_target, data in old_edges:
            provenance.extend(_edge_provenance(old_source, old_target, data))
            labels.append(data.get("label", 0))

        if len(old_edges) == 1:
            edge_data = copy.deepcopy(old_edges[0][2])
            edge_data["provenance"] = tuple(provenance)
        else:
            label_counts: Dict[Tuple[str, str], int] = {}
            for label in labels:
                key = (type(label).__qualname__, repr(label))
                label_counts[key] = label_counts.get(key, 0) + 1
            edge_data = {
                "label": ("__superedge__", tuple(sorted(label_counts.items()))),
                "multiplicity": len(old_edges),
                "provenance": tuple(provenance),
            }
        contracted.add_edge(source, target, **edge_data)

    return contracted


def encode(patterns: Sequence[_Pattern]) -> str:
    """按发现顺序为基本模式编号，返回形如 ``<1>/<2>`` 的字符串。"""
    return "/".join(f"<{index}>" for index in range(1, len(patterns) + 1))


def get_encode_basic_pattern(
    G: Any,
    bound: int,
    *,
    flexis_bin: Optional[str] = None,
    process_bin: Optional[str] = None,
    max_pattern_size: int = 6,
    slider: float = 0.98,
) -> str:
    """反复挖掘并收缩频繁子图，最后返回 ``<1>/<2>/...`` 编码。

    Args:
        G: NetworkX ``Graph`` 或 ``DiGraph``；节点和边可用 ``label`` 属性标记。
        bound: FLEXIS 支持度阈值，必须是正整数。
        flexis_bin: 可选的 FLEXIS 可执行文件路径；也可设置 ``FLEXIS_BIN``。
        process_bin: 可选的 FLEXIS 预处理器路径；也可设置 ``FLEXIS_PROCESS_BIN``。
        max_pattern_size: FLEXIS 探索的最大模式顶点数，范围为 2 到 10。
        slider: FLEXIS 的 MIS 支持度参数，范围为 0 到 1。

    Returns:
        以斜杠分隔的模式编号字符串；没有频繁模式时返回空字符串。

    ``bound`` 使用 FLEXIS 的 MIS 支持度定义，不等同于事务图集合中的
    ``min_support``。每轮选择 FLEXIS 最后一个非空模式层中顶点数、边数最多的
    模式，并把其贪心最大不重叠实例收缩为超节点。
    """
    nx = _networkx()
    if not isinstance(G, (nx.Graph, nx.DiGraph)) or G.is_multigraph():
        raise TypeError("G 必须是 NetworkX Graph 或 DiGraph，且不能包含平行边。")
    if not isinstance(bound, int) or isinstance(bound, bool) or bound < 1:
        raise ValueError("bound 必须是大于等于 1 的整数支持度阈值。")
    if not 2 <= max_pattern_size <= 10:
        raise ValueError("FLEXIS 的 max_pattern_size 必须介于 2 和 10 之间。")
    if not 0.0 <= slider <= 1.0:
        raise ValueError("slider 必须介于 0.0 和 1.0 之间。")

    current_graph = G.copy()
    patterns: List[_Pattern] = []
    while current_graph.number_of_nodes() >= 2 and current_graph.number_of_edges() > 0:
        pattern = freq_mining(
            current_graph,
            bound,
            flexis_bin=flexis_bin,
            process_bin=process_bin,
            max_pattern_size=max_pattern_size,
            slider=slider,
        )
        if pattern is None:
            break

        token = f"<{len(patterns) + 1}>"
        reduced_graph = super_node(current_graph, pattern, token)
        if reduced_graph.number_of_nodes() >= current_graph.number_of_nodes():
            raise RuntimeError("超节点收缩没有减少图的节点数，已停止以避免无限循环。")

        patterns.append(pattern)
        current_graph = reduced_graph

    return encode(patterns)
