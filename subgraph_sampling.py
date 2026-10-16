"""从大图中随机采样与给定 query 同构的子图。"""

from __future__ import annotations

import copy
import numbers
import random
from typing import Any, Callable, Dict, Hashable, List, Optional, Tuple


def _networkx():
    try:
        import networkx as nx
    except ImportError as exc:
        raise RuntimeError(
            "缺少 NetworkX 依赖，请先运行 `pip install -r requirements.txt`。"
        ) from exc
    return nx


def _label_match(attribute: str, default: Any) -> Callable[[dict, dict], bool]:
    """生成默认的节点或边类别标签匹配器。"""
    return lambda host_attrs, query_attrs: host_attrs.get(
        attribute, default
    ) == query_attrs.get(attribute, default)


def _ordered_shuffled_copy(graph: Any, rng: random.Random) -> Any:
    """随机化图的节点和边插入次序，以改变 VF2 的候选遍历顺序。"""
    nx = _networkx()
    shuffled = nx.DiGraph() if graph.is_directed() else nx.Graph()
    shuffled.graph.update(copy.deepcopy(graph.graph))

    nodes = list(graph.nodes(data=True))
    rng.shuffle(nodes)
    for node, attrs in nodes:
        # VF2 只读取属性字典；浅复制外层即可，避免深拷贝大型属性对象。
        shuffled.add_node(node)
        shuffled.nodes[node].update(attrs)

    edges = list(graph.edges(data=True))
    rng.shuffle(edges)
    for source, target, attrs in edges:
        shuffled.add_edge(source, target)
        shuffled.edges[source, target].update(attrs)
    return shuffled


def _new_matcher(
    graph: Any,
    query: Any,
    node_match: Callable[[dict, dict], bool],
    edge_match: Callable[[dict, dict], bool],
) -> Any:
    """建立 NetworkX VF2 匹配器；映射方向为主图节点到 query 节点。"""
    nx = _networkx()
    matcher_type = (
        nx.algorithms.isomorphism.DiGraphMatcher
        if graph.is_directed()
        else nx.algorithms.isomorphism.GraphMatcher
    )
    return matcher_type(
        graph,
        query,
        node_match=node_match,
        edge_match=edge_match,
    )


def _match_iterator(matcher: Any, induced: bool):
    """选择诱导同构或非诱导单态的 VF2 枚举器。"""
    if induced:
        return matcher.subgraph_isomorphisms_iter()
    return matcher.subgraph_monomorphisms_iter()


def _edge_key(graph: Any, source: Hashable, target: Hashable) -> Any:
    if graph.is_directed():
        return source, target
    return frozenset((source, target))


def _occurrence_key(graph: Any, query: Any, host_to_query: Dict[Any, Any]) -> Any:
    """按实际节点和边集合标识一次出现，折叠 query 自身的对称映射。"""
    query_to_host = {query_node: host_node for host_node, query_node in host_to_query.items()}
    host_nodes = frozenset(query_to_host.values())
    host_edges = frozenset(
        _edge_key(graph, query_to_host[source], query_to_host[target])
        for source, target in query.edges()
    )
    return host_nodes, host_edges


def enumerate_query_matches(
    G: Any,
    query: Any,
    *,
    induced: bool = False,
    max_matches: Optional[int] = None,
) -> Tuple[List[Dict[Hashable, Hashable]], bool]:
    """枚举不同的无标签无向 query 匹配，并返回是否枚举完整。

    每个匹配是 ``query_node -> G_node`` 映射。query 自同构产生的等价映射按
    宿主节点集合及所匹配边集合去重。设置 ``max_matches`` 后，一旦发现第
    ``max_matches + 1`` 个不同实例便停止，并返回 ``complete=False``。
    """
    nx = _networkx()
    if not isinstance(G, nx.Graph) or G.is_directed() or G.is_multigraph():
        raise TypeError("G 必须是无标签无向 NetworkX Graph。")
    if not isinstance(query, nx.Graph) or query.is_directed() or query.is_multigraph():
        raise TypeError("query 必须是无标签无向 NetworkX Graph。")
    if nx.number_of_selfloops(G) or nx.number_of_selfloops(query):
        raise ValueError("G 和 query 必须是无自环简单图。")
    if query.number_of_nodes() == 0:
        raise ValueError("query 至少需要包含一个节点。")
    if max_matches is not None and (
        not isinstance(max_matches, numbers.Integral)
        or isinstance(max_matches, bool)
        or max_matches < 0
    ):
        raise ValueError("max_matches 必须为非负整数或 None。")
    if max_matches == 0:
        return [], False
    if query.number_of_nodes() > G.number_of_nodes():
        return [], True

    always_match = lambda _host, _query: True
    matcher = _new_matcher(G, query, always_match, always_match)
    matches: List[Dict[Hashable, Hashable]] = []
    seen = set()
    for host_to_query in _match_iterator(matcher, induced):
        key = _occurrence_key(G, query, host_to_query)
        if key in seen:
            continue
        if max_matches is not None and len(matches) >= max_matches:
            return matches, False
        seen.add(key)
        matches.append(
            {query_node: host_node for host_node, query_node in host_to_query.items()}
        )
    return matches, True


def _extract_occurrence(
    graph: Any,
    query: Any,
    host_to_query: Dict[Any, Any],
    induced: bool,
) -> Any:
    """复制匹配到的 host 节点和边，返回节点 ID 仍为原图 ID 的子图。"""
    nx = _networkx()
    result = nx.DiGraph() if graph.is_directed() else nx.Graph()
    result.graph.update(copy.deepcopy(graph.graph))
    query_to_host = {query_node: host_node for host_node, query_node in host_to_query.items()}

    for query_node in query.nodes:
        host_node = query_to_host[query_node]
        result.add_node(host_node)
        result.nodes[host_node].update(copy.deepcopy(graph.nodes[host_node]))

    if induced:
        host_nodes = set(query_to_host.values())
        for source, target, attrs in graph.subgraph(host_nodes).edges(data=True):
            result.add_edge(source, target)
            result.edges[source, target].update(copy.deepcopy(attrs))
    else:
        for query_source, query_target in query.edges():
            host_source = query_to_host[query_source]
            host_target = query_to_host[query_target]
            result.add_edge(host_source, host_target)
            result.edges[host_source, host_target].update(
                copy.deepcopy(graph.edges[host_source, host_target])
            )

    # 映射键和值是用户的原始节点 ID，不对标识对象做深拷贝。
    result.graph["query_mapping"] = dict(query_to_host)
    result.graph["matched_host_nodes"] = tuple(
        query_to_host[query_node] for query_node in query.nodes
    )
    result.graph["sampling_induced"] = induced
    return result


def sample_query_subgraphs(
    G: Any,
    query: Any,
    B: int,
    *,
    induced: bool = False,
    strategy: str = "random",
    seed: Optional[int] = None,
    node_match: Optional[Callable[[dict, dict], bool]] = None,
    edge_match: Optional[Callable[[dict, dict], bool]] = None,
    node_label: str = "label",
    edge_label: str = "label",
    default_label: Any = 0,
) -> List[Any]:
    """随机采样至多 ``B`` 个与 ``query`` 同构的子图。

    默认使用 NetworkX 的 VF2 子图单态匹配器。非诱导匹配时，query 要求的边
    必须存在于 ``G``，但匹配节点之间可以有额外边；返回的结果只包含 query
    对应的边。设置 ``induced=True`` 后，匹配节点之间的边必须与 query 完全一致。

    ``strategy="random"`` 会先随机打乱节点和边的插入顺序，再按受该顺序影响的
    VF2 遍历次序枚举，找到 B 个不同实例后立即返回。该策略偏向 VF2 较早找到
    的匹配，不保证全局均匀；随机化会复制 G 和 query 的图结构，额外空间复杂度
    为 O(|V_G| + |E_G| + |V_query| + |E_query|)，不含属性对象自身大小。
    ``strategy="uniform"`` 会遍历 VF2 枚举的全部映射（含之后去重的对称映射），
    对不同的节点/边实例用蓄水池抽样得到无放回的精确均匀样本；运行时间受映射
    总数影响，去重集合的内存随不同实例数及 query 大小增长。

    匹配实例按 host 节点集合及其对应边集合去重，因此 query 的对称映射不会
    重复计数；在非诱导模式下，同一节点集合上的不同匹配边集合仍是不同实例。
    若实例总数少于 ``B``，返回全部实例。

    默认节点和边属性使用 ``label`` 字段比较，字段缺失时使用标签 ``0``。
    传入 ``node_match`` 或 ``edge_match`` 可分别替换对应默认规则；回调参数
    顺序是 ``(G 中的属性字典, query 中的属性字典)``。

    Args:
        G: NetworkX ``Graph`` 或 ``DiGraph``，不支持平行边图。
        query: 与 ``G`` 同为有向或无向的非空 NetworkX 简单图。
        B: 需要采样的实例数，必须为非负整数。
        induced: 是否采用诱导子图同构语义。
        strategy: ``"random"`` 为快速随机顺序搜索；``"uniform"`` 为精确均匀抽样。
        seed: 随机种子，传入整数可复现随机搜索次序和抽样结果。
        node_match: 可选节点属性匹配函数。
        edge_match: 可选边属性匹配函数。
        node_label: 默认匹配的节点标签字段名。
        edge_label: 默认匹配的边标签字段名。
        default_label: 节点或边标签字段缺失时使用的默认值。

    Returns:
        最多 ``B`` 个 NetworkX 图。结果沿用 ``G`` 的节点 ID 和属性；每个图的
        ``graph["query_mapping"]`` 属性记录 query 节点到原图节点的映射。
    """
    nx = _networkx()
    if not isinstance(G, (nx.Graph, nx.DiGraph)) or G.is_multigraph():
        raise TypeError("G 必须是 NetworkX Graph 或 DiGraph，且不能包含平行边。")
    if not isinstance(query, (nx.Graph, nx.DiGraph)) or query.is_multigraph():
        raise TypeError("query 必须是 NetworkX Graph 或 DiGraph，且不能包含平行边。")
    if G.is_directed() != query.is_directed():
        raise ValueError("G 和 query 必须同时为有向图或同时为无向图。")
    if query.number_of_nodes() == 0:
        raise ValueError("query 至少需要包含一个节点。")
    if not isinstance(B, numbers.Integral) or isinstance(B, bool) or B < 0:
        raise ValueError("B 必须是非负整数。")
    B = int(B)
    if strategy not in {"random", "uniform"}:
        raise ValueError("strategy 只能是 'random' 或 'uniform'。")
    if B == 0 or query.number_of_nodes() > G.number_of_nodes():
        return []

    actual_node_match = (
        node_match if node_match is not None else _label_match(node_label, default_label)
    )
    actual_edge_match = (
        edge_match if edge_match is not None else _label_match(edge_label, default_label)
    )
    rng = random.Random(seed)
    found: Dict[Any, Any] = {}

    if strategy == "uniform":
        seen = set()
        reservoir: List[Dict[Any, Any]] = []
        occurrence_count = 0
        matcher = _new_matcher(G, query, actual_node_match, actual_edge_match)
        for mapping in _match_iterator(matcher, induced):
            key = _occurrence_key(G, query, mapping)
            if key in seen:
                continue
            seen.add(key)
            occurrence_count += 1
            # VF2 会继续修改内部状态；留存独立映射，最终只构造抽中的子图。
            saved_mapping = dict(mapping)
            if len(reservoir) < B:
                reservoir.append(saved_mapping)
            else:
                replacement_index = rng.randrange(occurrence_count)
                if replacement_index < B:
                    reservoir[replacement_index] = saved_mapping
        return [
            _extract_occurrence(G, query, mapping, induced)
            for mapping in reservoir
        ]

    # 大图只随机化一次，避免每轮重启都复制整张图；达到 B 个实例即提前停止。
    shuffled_graph = _ordered_shuffled_copy(G, rng)
    shuffled_query = _ordered_shuffled_copy(query, rng)
    matcher = _new_matcher(
        shuffled_graph,
        shuffled_query,
        actual_node_match,
        actual_edge_match,
    )
    for mapping in _match_iterator(matcher, induced):
        key = _occurrence_key(G, query, mapping)
        if key not in found:
            found[key] = _extract_occurrence(G, query, mapping, induced)
        if len(found) >= B:
            break
    return list(found.values())


__all__ = ["enumerate_query_matches", "sample_query_subgraphs"]
