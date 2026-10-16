"""用精确匹配真值评估候选区域的 OracleCoverage 上限。"""

from __future__ import annotations

import numbers
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import networkx as nx

from overlap_hierarchy import Region
from pattern_instances import PatternOccurrence
from subgraph_sampling import enumerate_query_matches


def enumerate_match_records(
    graph: nx.Graph,
    query: nx.Graph,
    occurrences: Sequence[PatternOccurrence],
    *,
    max_matches: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], bool]:
    """获取完整映射并标记匹配是否跨越基础实例边界。"""
    mappings, complete = enumerate_query_matches(
        graph,
        query,
        max_matches=max_matches,
    )
    occurrence_node_sets = [set(occurrence.nodes) for occurrence in occurrences]
    records = []
    for index, mapping in enumerate(mappings, start=1):
        nodes = frozenset(mapping.values())
        host_edges = frozenset(
            tuple(sorted((mapping[source], mapping[target]), key=lambda value: (type(value).__qualname__, repr(value))))
            for source, target in query.edges
        )
        cross_boundary = not any(nodes.issubset(occurrence_nodes) for occurrence_nodes in occurrence_node_sets)
        records.append(
            {
                "match_id": f"match-{index:08d}",
                "query_to_graph": dict(mapping),
                "host_nodes": nodes,
                "host_edges": host_edges,
                "cross_boundary": cross_boundary,
            }
        )
    return records, complete


def _coverage_greedy(
    matches: Sequence[Mapping[str, Any]],
    regions: Sequence[Region],
    *,
    max_regions: Optional[int],
    storage_budget: Optional[int],
    max_region_size: Optional[int],
) -> Tuple[set, List[Region], Dict[str, set], Dict[str, int]]:
    eligible = [
        region
        for region in regions
        if not region.is_virtual
        and (max_region_size is None or len(region.nodes) <= max_region_size)
        and any(set(match["host_nodes"]).issubset(region.nodes) for match in matches)
    ]
    match_sets = [set(match["host_nodes"]) for match in matches]
    match_indices_by_node: Dict[Any, List[int]] = {}
    for index, match_nodes in enumerate(match_sets):
        for node in match_nodes:
            match_indices_by_node.setdefault(node, []).append(index)
    covered_by_region: Dict[str, set] = {}
    for region in eligible:
        possible = {
            index
            for node in region.nodes
            for index in match_indices_by_node.get(node, ())
        }
        covered_by_region[region.region_id] = {
            index for index in possible if match_sets[index].issubset(region.nodes)
        }
    selected: List[Region] = []
    covered: set = set()
    remaining_storage = storage_budget

    while eligible and (max_regions is None or len(selected) < max_regions):
        scored = []
        for region in eligible:
            cost = max(1, len(region.nodes))
            if remaining_storage is not None and cost > remaining_storage:
                continue
            newly_covered = covered_by_region[region.region_id] - covered
            if newly_covered:
                scored.append((len(newly_covered) / cost, len(newly_covered), region.region_id, region, newly_covered, cost))
        if not scored:
            break
        _ratio, _gain, _region_id, chosen, newly_covered, cost = sorted(
            scored, key=lambda item: (-item[0], -item[1], item[2])
        )[0]
        selected.append(chosen)
        covered.update(newly_covered)
        eligible = [region for region in eligible if region.region_id != chosen.region_id]
        if remaining_storage is not None:
            remaining_storage -= cost
    costs = {
        region.region_id: max(1, len(region.nodes))
        for region in regions
        if region.region_id in covered_by_region
    }
    return covered, selected, covered_by_region, costs


def _coverage_upper_bound(
    covered_by_region: Mapping[str, set],
    region_costs: Mapping[str, int],
    target_indices: set,
    *,
    retrieval_budget: Optional[int],
    storage_budget: Optional[int],
) -> int:
    """给有限预算下的最大覆盖数计算一个可证明的松弛上界。"""
    candidate_sets = [covered & target_indices for covered in covered_by_region.values()]
    potential = set().union(*candidate_sets) if candidate_sets else set()
    upper = len(potential)
    counts = sorted((len(covered) for covered in candidate_sets), reverse=True)
    if retrieval_budget is not None:
        upper = min(upper, sum(counts[:retrieval_budget]))
    if storage_budget is not None:
        items = sorted(
            (
                (len(covered), region_costs[region_id])
                for region_id, all_covered in covered_by_region.items()
                if (covered := all_covered & target_indices)
            ),
            key=lambda item: (-item[0] / item[1], item[1]),
        )
        remaining = float(storage_budget)
        fractional_bound = 0.0
        for value, cost in items:
            if remaining <= 0:
                break
            fraction = min(1.0, remaining / cost)
            fractional_bound += value * fraction
            remaining -= cost * fraction
        upper = min(upper, int(fractional_bound + 1e-12))
    return upper


def oracle_coverage(
    graph: nx.Graph,
    matches: Sequence[Mapping[str, Any]],
    regions: Sequence[Region],
    *,
    region_size_limit: int,
    retrieval_budget: Optional[int] = None,
    storage_budget: Optional[int] = None,
) -> Dict[str, Any]:
    """计算有界区域能覆盖的真匹配比例及跨边界分层结果。"""
    if (
        not isinstance(region_size_limit, numbers.Integral)
        or isinstance(region_size_limit, bool)
        or region_size_limit < 1
    ):
        raise ValueError("region_size_limit 必须大于等于 1。")
    if retrieval_budget is not None and (
        not isinstance(retrieval_budget, numbers.Integral)
        or isinstance(retrieval_budget, bool)
        or retrieval_budget < 0
    ):
        raise ValueError("retrieval_budget 必须是非负整数或 None。")
    if storage_budget is not None and (
        not isinstance(storage_budget, numbers.Integral)
        or isinstance(storage_budget, bool)
        or storage_budget < 0
    ):
        raise ValueError("storage_budget 必须是非负整数或 None。")

    covered_indices, selected_regions, covered_by_region, region_costs = _coverage_greedy(
        matches,
        regions,
        max_regions=retrieval_budget,
        storage_budget=storage_budget,
        max_region_size=region_size_limit,
    )
    all_indices = set(range(len(matches)))
    upper_count = _coverage_upper_bound(
        covered_by_region,
        region_costs,
        all_indices,
        retrieval_budget=retrieval_budget,
        storage_budget=storage_budget,
    )
    result: Dict[str, Any] = {
        "region_size_limit": region_size_limit,
        "retrieval_budget": retrieval_budget,
        "storage_budget": storage_budget,
        "match_count": len(matches),
        "covered_match_count": len(covered_indices),
        "greedy_covered_match_count": len(covered_indices),
        "greedy_coverage": len(covered_indices) / len(matches) if matches else None,
        "oracle_covered_match_upper_bound": upper_count,
        "oracle_coverage": upper_count / len(matches) if matches else None,
        "oracle_upper_bound_is_exact": upper_count == len(covered_indices),
        "selected_region_count": len(selected_regions),
        "selected_region_ids": [region.region_id for region in selected_regions],
        "verification_node_work": sum(len(region.nodes) for region in selected_regions),
        "verification_edge_work": sum(
            graph.subgraph(region.nodes).number_of_edges() for region in selected_regions
        ),
        "by_boundary": {},
    }
    for label, predicate in (
        ("cross_boundary", lambda match: bool(match["cross_boundary"])),
        ("within_one_leaf", lambda match: not bool(match["cross_boundary"])),
    ):
        indices = [index for index, match in enumerate(matches) if predicate(match)]
        covered_count = sum(index in covered_indices for index in indices)
        upper_boundary_count = _coverage_upper_bound(
            covered_by_region,
            region_costs,
            set(indices),
            retrieval_budget=retrieval_budget,
            storage_budget=storage_budget,
        )
        result["by_boundary"][label] = {
            "match_count": len(indices),
            "covered_match_count": covered_count,
            "greedy_coverage": covered_count / len(indices) if indices else None,
            "oracle_covered_match_upper_bound": upper_boundary_count,
            "oracle_coverage": upper_boundary_count / len(indices) if indices else None,
        }
    return result


def evaluate_budgets(
    graph: nx.Graph,
    matches: Sequence[Mapping[str, Any]],
    regions: Sequence[Region],
    region_size_limits: Iterable[int],
    *,
    retrieval_budgets: Sequence[Optional[int]] = (None, 5),
    storage_budgets: Sequence[Optional[int]] = (None,),
) -> List[Dict[str, Any]]:
    rows = []
    for size_limit in region_size_limits:
        for retrieval_budget in retrieval_budgets:
            for storage_budget in storage_budgets:
                rows.append(
                    oracle_coverage(
                        graph,
                        matches,
                        regions,
                        region_size_limit=size_limit,
                        retrieval_budget=retrieval_budget,
                        storage_budget=storage_budget,
                    )
                )
    return rows


__all__ = ["enumerate_match_records", "evaluate_budgets", "oracle_coverage"]
