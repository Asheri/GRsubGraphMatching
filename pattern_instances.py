"""无标签无向模式类型与其在原图中的具体出现实例。"""

from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass
from typing import Any, Dict, Hashable, Iterable, List, Mapping, Optional, Sequence, Tuple

import networkx as nx

from subgraph_sampling import enumerate_query_matches


@dataclass(frozen=True)
class PatternType:
    """按无标签图同构去重后的模式类型。"""

    type_id: str
    token: str
    canonical_code: str
    graph: nx.Graph


@dataclass(frozen=True)
class PatternOccurrence:
    """某模式类型在原图中的一次出现，节点映射以规范模式节点为键。"""

    occurrence_id: str
    type_id: str
    node_mapping: Mapping[int, Hashable]
    nodes: Tuple[Hashable, ...]
    mining_round: int = 0


def _validate_simple_undirected(graph: Any, name: str) -> None:
    if not isinstance(graph, nx.Graph) or graph.is_directed() or graph.is_multigraph():
        raise TypeError(f"{name} 必须是无标签无向 NetworkX Graph。")
    if nx.number_of_selfloops(graph):
        raise ValueError(f"{name} 不允许自环。")


def _adjacency_code(graph: nx.Graph, order: Sequence[Hashable]) -> str:
    return "".join(
        "1" if graph.has_edge(order[left], order[right]) else "0"
        for left in range(len(order))
        for right in range(left + 1, len(order))
    )


def canonicalize_pattern(graph: nx.Graph, *, max_vertices: int = 8) -> Tuple[str, nx.Graph]:
    """返回精确的规范邻接码和规范节点编号图。

    模式规模受限以避免无标签图规范化的阶乘搜索失控。当前合成实验中的基础
    模式远小于该上限；大组合优先以递归规则表达，不进入此规范化路径。
    """
    _validate_simple_undirected(graph, "pattern")
    size = graph.number_of_nodes()
    if size == 0:
        raise ValueError("pattern 至少需要包含一个节点。")
    if size > max_vertices:
        raise ValueError(f"模式最多允许 {max_vertices} 个节点，收到 {size} 个。")

    nodes = list(graph.nodes)
    degree_groups: Dict[int, List[Hashable]] = {}
    for node in nodes:
        degree_groups.setdefault(graph.degree[node], []).append(node)
    degree_values = sorted(degree_groups)
    groups = [degree_groups[degree] for degree in degree_values]

    best_code: Optional[str] = None
    best_order: Optional[Tuple[Hashable, ...]] = None
    group_permutations = [list(itertools.permutations(group)) for group in groups]
    for combination in itertools.product(*group_permutations):
        order = tuple(node for group_order in combination for node in group_order)
        code = _adjacency_code(graph, order)
        if best_code is None or code < best_code:
            best_code = code
            best_order = order

    assert best_code is not None and best_order is not None
    canonical = nx.Graph()
    canonical.add_nodes_from(range(size))
    for left in range(size):
        for right in range(left + 1, size):
            if graph.has_edge(best_order[left], best_order[right]):
                canonical.add_edge(left, right)
    return f"n{size}:e{graph.number_of_edges()}:{best_code}", canonical


def register_pattern_type(
    graph: nx.Graph,
    *,
    token: Optional[str] = None,
    max_vertices: int = 8,
) -> PatternType:
    code, canonical = canonicalize_pattern(graph, max_vertices=max_vertices)
    digest = hashlib.sha256(code.encode("ascii")).hexdigest()[:16]
    type_id = f"p_{digest}"
    return PatternType(
        type_id=type_id,
        token=token or f"<{type_id}>",
        canonical_code=code,
        graph=canonical,
    )


def enumerate_pattern_occurrences(
    graph: nx.Graph,
    pattern_type: PatternType,
    *,
    max_occurrences: Optional[int] = None,
    mining_round: int = 0,
) -> Tuple[List[PatternOccurrence], bool]:
    """枚举某模式在原图中的不同出现，保留规范顶点到原图节点的映射。"""
    _validate_simple_undirected(graph, "G")
    mappings, complete = enumerate_query_matches(
        graph,
        pattern_type.graph,
        max_matches=max_occurrences,
    )
    occurrences = []
    for index, mapping in enumerate(mappings, start=1):
        ordered_nodes = tuple(mapping[position] for position in range(pattern_type.graph.number_of_nodes()))
        occurrences.append(
            PatternOccurrence(
                occurrence_id=f"{pattern_type.type_id}@{index:06d}",
                type_id=pattern_type.type_id,
                node_mapping=dict(mapping),
                nodes=ordered_nodes,
                mining_round=mining_round,
            )
        )
    return occurrences, complete


def build_pattern_vocabulary(
    graph: nx.Graph,
    seed_patterns: Iterable[Tuple[str, nx.Graph]],
    *,
    min_support: int = 1,
    min_vertices: int = 3,
    max_vertices: int = 8,
    max_occurrences_per_type: Optional[int] = None,
) -> Tuple[Dict[str, PatternType], List[PatternOccurrence], Dict[str, bool]]:
    """从预设/query 模式建立有限词表，并保留支持度达标的具体出现。

    ``seed_patterns`` 中的名称作为人类可读 token。相同图同构类型只保留一个
    词表项；类型支持度以非诱导、去除 query 自同构重复后的实例数计算。
    """
    _validate_simple_undirected(graph, "G")
    if min_support < 1:
        raise ValueError("min_support 必须大于等于 1。")
    if min_vertices < 1 or max_vertices < min_vertices:
        raise ValueError("模式节点数范围无效。")
    if max_vertices > 8:
        raise ValueError("当前精确规范化最多支持 8 个节点；更大组合用递归规则表达。")

    types_by_code: Dict[str, PatternType] = {}
    for name, pattern in seed_patterns:
        if not min_vertices <= pattern.number_of_nodes() <= max_vertices:
            continue
        candidate = register_pattern_type(
            pattern,
            token=f"<{name}>",
            max_vertices=max_vertices,
        )
        types_by_code.setdefault(candidate.canonical_code, candidate)

    retained_types: Dict[str, PatternType] = {}
    all_occurrences: List[PatternOccurrence] = []
    completeness: Dict[str, bool] = {}
    for code, pattern_type in sorted(types_by_code.items()):
        occurrences, complete = enumerate_pattern_occurrences(
            graph,
            pattern_type,
            max_occurrences=max_occurrences_per_type,
        )
        completeness[pattern_type.type_id] = complete
        if len(occurrences) < min_support:
            continue
        retained_types[pattern_type.type_id] = pattern_type
        all_occurrences.extend(occurrences)

    return retained_types, all_occurrences, completeness


__all__ = [
    "PatternOccurrence",
    "PatternType",
    "build_pattern_vocabulary",
    "canonicalize_pattern",
    "enumerate_pattern_occurrences",
    "register_pattern_type",
]
