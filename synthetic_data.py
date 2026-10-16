"""生成可复现的无标签无向合成 query 图集和宿主图。"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, Hashable, List, Mapping, Tuple

import networkx as nx


@dataclass
class SyntheticDataset:
    graph: nx.Graph
    queries: Dict[str, nx.Graph]
    base_patterns: Dict[str, nx.Graph]
    planted: List[Dict[str, Any]]
    seed: int


def make_seed_queries() -> Dict[str, nx.Graph]:
    """构造 10 个不同的连通无标签 query 模式。"""
    queries: Dict[str, nx.Graph] = {}

    queries["tri"] = nx.cycle_graph(3)
    queries["square"] = nx.cycle_graph(4)
    queries["star"] = nx.star_graph(3)
    queries["path4"] = nx.path_graph(4)

    triangle_tail = nx.cycle_graph(3)
    triangle_tail.add_edge(0, 3)
    queries["triangle_tail"] = triangle_tail

    butterfly = nx.Graph()
    butterfly.add_edges_from(
        [(0, 1), (1, 2), (2, 0), (0, 3), (3, 4), (4, 0)]
    )
    queries["butterfly"] = butterfly

    diamond = nx.Graph()
    diamond.add_edges_from(
        [(0, 1), (1, 2), (2, 0), (0, 3), (3, 1)]
    )
    queries["diamond"] = diamond

    square_tail = nx.cycle_graph(4)
    square_tail.add_edge(0, 4)
    queries["square_tail"] = square_tail

    triangle_path = nx.cycle_graph(3)
    triangle_path.add_edges_from([(0, 3), (3, 4)])
    queries["triangle_path"] = triangle_path

    queries["pentagon"] = nx.cycle_graph(5)
    return queries


def make_base_patterns() -> Dict[str, nx.Graph]:
    """返回合成实验的原子叶子模式；复合 query 不直接成为叶子。"""
    queries = make_seed_queries()
    return {
        name: queries[name]
        for name in ("tri", "square", "star")
    }


def _copy_with_relation(
    graph: nx.Graph,
    query: nx.Graph,
    previous_mapping: Mapping[Hashable, Hashable],
    copy_index: int,
    query_id: str,
    relation: str,
    next_node_id: int,
) -> Tuple[Dict[Hashable, Hashable], List[Tuple[Hashable, Hashable]], int]:
    query_nodes = list(query.nodes)
    new_mapping: Dict[Hashable, Hashable] = {}

    if relation == "shared_node":
        shared_query_node = query_nodes[0]
        shared_host_node = previous_mapping[shared_query_node]
        new_mapping[shared_query_node] = shared_host_node
    elif relation == "shared_edge":
        query_edge = next(iter(query.edges))
        host_edge = (previous_mapping[query_edge[0]], previous_mapping[query_edge[1]])
        new_mapping[query_edge[0]] = host_edge[0]
        new_mapping[query_edge[1]] = host_edge[1]
    elif relation != "bridge":
        raise ValueError(f"未知的植入关系：{relation}")

    for node in query_nodes:
        if node not in new_mapping:
            new_mapping[node] = f"{query_id}:copy{copy_index}:node{next_node_id}"
            next_node_id += 1

    graph.add_nodes_from(new_mapping.values())
    graph.add_edges_from(
        (new_mapping[source], new_mapping[target])
        for source, target in query.edges
    )

    if relation == "bridge":
        previous_host = previous_mapping[query_nodes[0]]
        new_host = new_mapping[query_nodes[0]]
        graph.add_edge(previous_host, new_host)

    return new_mapping, list(query.edges), next_node_id


def generate_synthetic_dataset(
    *,
    seed: int = 7,
    copies_per_query: int = 5,
    max_extra_nodes: int = 15,
    max_extra_edges: int = 15,
) -> SyntheticDataset:
    """植入 10 类模式并加入有界随机扰动。

    planted 记录只说明实例来源；后续监督真值须在返回的最终图上重新匹配。
    """
    if copies_per_query < 5:
        raise ValueError("每种 query 至少植入 5 次。")
    if max_extra_nodes < 0 or max_extra_nodes > 15:
        raise ValueError("max_extra_nodes 必须在 0 到 15 之间。")
    if max_extra_edges < 0 or max_extra_edges > 15:
        raise ValueError("max_extra_edges 必须在 0 到 15 之间。")

    rng = random.Random(seed)
    queries = make_seed_queries()
    graph = nx.Graph()
    planted: List[Dict[str, Any]] = []

    for query_id, query in queries.items():
        previous_mapping: Dict[Hashable, Hashable] = {}
        relations = ["shared_node", "shared_edge", "bridge", "shared_node"]
        for copy_index in range(copies_per_query):
            if copy_index == 0:
                mapping = {
                    node: f"{query_id}:copy0:node{index}"
                    for index, node in enumerate(query.nodes)
                }
                graph.add_nodes_from(mapping.values())
                graph.add_edges_from(
                    (mapping[source], mapping[target])
                    for source, target in query.edges
                )
                relation = "disjoint_seed"
            else:
                relation = relations[(copy_index - 1) % len(relations)]
                mapping, _, _ = _copy_with_relation(
                    graph,
                    query,
                    previous_mapping,
                    copy_index,
                    query_id,
                    relation,
                    0,
                )
            planted.append(
                {
                    "query_id": query_id,
                    "copy_id": f"{query_id}#{copy_index + 1}",
                    "relation_to_previous": relation,
                    "query_to_graph": dict(mapping),
                }
            )
            previous_mapping = mapping

    if graph.number_of_nodes() + max_extra_nodes >= 500:
        raise ValueError("植入图过大，无法满足最终节点数小于 500 的约束。")

    extra_node_count = rng.randint(0, max_extra_nodes)
    next_index = 0
    while extra_node_count:
        node = f"noise:node{next_index}"
        next_index += 1
        if node not in graph:
            graph.add_node(node)
            extra_node_count -= 1

    extra_edge_count = rng.randint(0, max_extra_edges)
    available_pairs = [
        (left, right)
        for index, left in enumerate(graph.nodes)
        for right in list(graph.nodes)[index + 1 :]
        if not graph.has_edge(left, right)
    ]
    rng.shuffle(available_pairs)
    for left, right in available_pairs[:extra_edge_count]:
        graph.add_edge(left, right)

    if graph.number_of_nodes() >= 500:
        raise ValueError("生成后的 G 必须少于 500 个节点。")

    graph.graph.update({"seed": seed, "directed": False, "labeled": False})
    return SyntheticDataset(
        graph=graph,
        queries=queries,
        base_patterns=make_base_patterns(),
        planted=planted,
        seed=seed,
    )


def graph_to_json(graph: nx.Graph) -> Dict[str, Any]:
    """将无标签无向图转成稳定排序的 JSON 兼容结构。"""
    node_ids = list(graph.nodes)
    stable_key = lambda value: (type(value).__qualname__, repr(value))
    ordered_nodes = sorted(node_ids, key=stable_key)
    ordered_edges = sorted(
        (tuple(sorted((source, target), key=stable_key)) for source, target in graph.edges),
        key=lambda edge: (stable_key(edge[0]), stable_key(edge[1])),
    )
    return {
        "nodes": ordered_nodes,
        "edges": [list(edge) for edge in ordered_edges],
    }


__all__ = [
    "SyntheticDataset",
    "generate_synthetic_dataset",
    "graph_to_json",
    "make_base_patterns",
    "make_seed_queries",
]
