"""递归候选组合关系图与预算受限的单父索引树。"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import numbers
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import networkx as nx

from pattern_instances import (
    PatternOccurrence,
    PatternType,
    canonicalize_pattern,
    register_pattern_type,
)


@dataclass
class Region:
    region_id: str
    level: int
    nodes: frozenset
    structure: nx.Graph
    child_ids: Tuple[str, ...] = ()
    source_occurrence_ids: Tuple[str, ...] = ()
    pattern_type: Optional[str] = None
    composition_rule: Optional[str] = None
    parent_id: Optional[str] = None
    is_virtual: bool = False
    canonical_code: Optional[str] = None
    selected_for_index: bool = False


@dataclass
class CompositionCandidate:
    candidate_id: str
    result_region_id: str
    child_ids: Tuple[str, str]
    child_node_sets: Tuple[Tuple[Any, ...], Tuple[Any, ...]]
    shared_node_ids: Tuple[Any, ...]
    result_pattern_type: Optional[str]
    composition_rule: str
    covered_nodes: Tuple[Any, ...]
    level: int
    relation_kind: str
    selected_for_index: bool = False


@dataclass
class Hierarchy:
    regions: Dict[str, Region]
    candidates: List[CompositionCandidate]
    root_id: Optional[str]
    max_height: int
    repeatability_by_level: Dict[int, float] = field(default_factory=dict)
    promoted_types: Dict[str, PatternType] = field(default_factory=dict)
    exploration_by_level: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    repeatability_budget: float = 1.0


def _stable(value: Any) -> Tuple[str, str]:
    return type(value).__qualname__, repr(value)


def _sorted_nodes(nodes: Iterable[Any]) -> Tuple[Any, ...]:
    return tuple(sorted(set(nodes), key=_stable))


def _edges_between(graph: nx.Graph, left: Set[Any], right: Set[Any]) -> List[Tuple[Any, Any]]:
    return [
        (source, target)
        for source, target in graph.edges
        if (source in left and target in right) or (target in left and source in right)
    ]


def _repeatability(regions: Sequence[Region]) -> float:
    covered = set().union(*(region.nodes for region in regions)) if regions else set()
    if not covered:
        return 1.0
    return sum(len(region.nodes) for region in regions) / len(covered)


def _structure_for_occurrence(
    occurrence: PatternOccurrence,
    types: Mapping[str, PatternType],
) -> nx.Graph:
    pattern_type = types[occurrence.type_id]
    structure = nx.Graph()
    structure.add_nodes_from(occurrence.node_mapping.values())
    structure.add_edges_from(
        (occurrence.node_mapping[source], occurrence.node_mapping[target])
        for source, target in pattern_type.graph.edges
    )
    return structure


def _known_composition_token(
    candidate: CompositionCandidate,
    parent: Region,
    regions: Mapping[str, Region],
) -> Optional[str]:
    """给已约定的三角形组合结构分配稳定别名。"""
    left, right = (regions[region_id] for region_id in candidate.child_ids)
    if left.pattern_type != "<tri>" or right.pattern_type != "<tri>":
        return None
    shared_count = len(candidate.shared_node_ids)
    if shared_count == 1 and len(parent.nodes) == 5 and parent.structure.number_of_edges() == 6:
        return "<butterfly>"
    if shared_count == 2 and len(parent.nodes) == 4 and parent.structure.number_of_edges() == 5:
        return "<diamond>"
    return None


def build_hierarchy(
    graph: nx.Graph,
    occurrences: Sequence[PatternOccurrence],
    pattern_types: Mapping[str, PatternType],
    *,
    tau: float = 0.34,
    repeatability_budget: Optional[float] = None,
    max_height: int = 3,
    min_promotion_support: int = 2,
    max_candidates_per_level: int = 5000,
    allow_overlap: bool = True,
) -> Hierarchy:
    """从具体模式实例构造候选组合关系图和索引树。

    候选图保留每层探索到的所有合格两两组合；索引树按覆盖增益/存储代价排序，
    并确保每个区域在树中至多有一个父节点。组合关系记录不复制边列表。
    """
    if graph.is_directed() or graph.is_multigraph():
        raise TypeError("G 必须是无标签无向简单图。")
    if nx.number_of_selfloops(graph):
        raise ValueError("G 必须是无自环的无标签无向简单图。")
    if (
        isinstance(tau, bool)
        or not isinstance(tau, numbers.Real)
        or not math.isfinite(tau)
        or not 0.0 <= tau <= 1.0
    ):
        raise ValueError("tau 必须介于 0 和 1 之间。")
    if repeatability_budget is not None and (
        isinstance(repeatability_budget, bool)
        or not isinstance(repeatability_budget, numbers.Real)
        or not math.isfinite(repeatability_budget)
        or repeatability_budget < 1.0
    ):
        raise ValueError("repeatability_budget 必须是有限且大于等于 1 的数值。")
    if (
        not isinstance(max_height, numbers.Integral)
        or isinstance(max_height, bool)
        or max_height < 0
    ):
        raise ValueError("max_height 必须是非负整数。")
    if (
        not isinstance(min_promotion_support, numbers.Integral)
        or isinstance(min_promotion_support, bool)
        or min_promotion_support < 1
        or not isinstance(max_candidates_per_level, numbers.Integral)
        or isinstance(max_candidates_per_level, bool)
        or max_candidates_per_level < 1
    ):
        raise ValueError("模式晋升支持度和候选上限必须是正整数。")

    regions: Dict[str, Region] = {}
    frontier: List[str] = []
    for occurrence in occurrences:
        if occurrence.type_id not in pattern_types:
            continue
        region_id = occurrence.occurrence_id
        region = Region(
            region_id=region_id,
            level=0,
            nodes=frozenset(occurrence.nodes),
            structure=_structure_for_occurrence(occurrence, pattern_types),
            source_occurrence_ids=(occurrence.occurrence_id,),
            pattern_type=pattern_types[occurrence.type_id].token,
            canonical_code=pattern_types[occurrence.type_id].canonical_code,
            selected_for_index=True,
        )
        regions[region_id] = region
        frontier.append(region_id)

    candidates: List[CompositionCandidate] = []
    all_promoted_types: Dict[str, PatternType] = {}
    leaf_repeatability = _repeatability([regions[i] for i in frontier])
    if repeatability_budget is None:
        repeatability_budget = leaf_repeatability
    if repeatability_budget + 1e-12 < leaf_repeatability:
        raise ValueError(
            "repeatability_budget 小于固定叶层本身的重复引用率 "
            f"({repeatability_budget:.6g} < {leaf_repeatability:.6g})。"
        )
    repeatability_by_level: Dict[int, float] = {0: leaf_repeatability}
    exploration_by_level: Dict[int, Dict[str, Any]] = {}
    token_by_code = {
        pattern_type.canonical_code: pattern_type.token
        for pattern_type in pattern_types.values()
    }

    for level in range(1, max_height + 1):
        candidate_regions: List[Tuple[CompositionCandidate, Region, float]] = []
        pair_count = 0
        total_possible_pairs = len(frontier) * (len(frontier) - 1) // 2
        budget_hit = total_possible_pairs > max_candidates_per_level
        for left_index, left_id in enumerate(frontier):
            left = regions[left_id]
            for right_id in frontier[left_index + 1 :]:
                if pair_count >= max_candidates_per_level:
                    break
                pair_count += 1
                right = regions[right_id]
                shared = left.nodes.intersection(right.nodes)
                overlap = len(shared) / min(len(left.nodes), len(right.nodes))
                connectors = _edges_between(graph, set(left.nodes), set(right.nodes))
                relation_kind = "overlap" if shared else "bridge"
                if allow_overlap:
                    eligible = overlap >= tau or (not shared and bool(connectors))
                else:
                    eligible = not shared and bool(connectors)
                if not eligible:
                    continue

                parent_structure = nx.compose(left.structure, right.structure)
                if not shared:
                    parent_structure.add_edges_from(connectors)
                parent_nodes = frozenset(parent_structure.nodes)
                child_types = (left.pattern_type or "<composition>", right.pattern_type or "<composition>")
                relation_text = f"shared_nodes={len(shared)}" if shared else "graph_bridge"
                rule = f"{child_types[0]} + {child_types[1]} [{relation_text}]"
                result_token: Optional[str] = None
                try:
                    code, _ = canonicalize_pattern(parent_structure)
                    result_token = token_by_code.get(code)
                except ValueError:
                    code = f"large:{len(parent_nodes)}:{parent_structure.number_of_edges()}"

                candidate_id = f"candidate-{level:02d}-{len(candidate_regions) + 1:06d}"
                candidate = CompositionCandidate(
                    candidate_id=candidate_id,
                    result_region_id=f"region-{level:02d}-{len(candidate_regions) + 1:06d}",
                    child_ids=(left_id, right_id),
                    child_node_sets=(_sorted_nodes(left.nodes), _sorted_nodes(right.nodes)),
                    shared_node_ids=_sorted_nodes(shared),
                    result_pattern_type=result_token,
                    composition_rule=rule,
                    covered_nodes=_sorted_nodes(parent_nodes),
                    level=level,
                    relation_kind=relation_kind,
                )
                parent_id = candidate.result_region_id
                parent = Region(
                    region_id=parent_id,
                    level=level,
                    nodes=parent_nodes,
                    structure=parent_structure,
                    child_ids=(left_id, right_id),
                    source_occurrence_ids=tuple(
                        sorted(
                            set(left.source_occurrence_ids + right.source_occurrence_ids)
                        )
                    ),
                    pattern_type=result_token,
                    composition_rule=rule,
                    canonical_code=code if code.startswith("n") else None,
                )
                novelty = len(parent_nodes) - max(len(left.nodes), len(right.nodes))
                storage_cost = max(1, len(parent_nodes))
                score = novelty / storage_cost
                candidate_regions.append((candidate, parent, score))
            if pair_count >= max_candidates_per_level:
                break

        exploration_by_level[level] = {
            "pair_budget": max_candidates_per_level,
            "possible_pair_count": total_possible_pairs,
            "explored_pair_count": pair_count,
            "budget_hit": budget_hit,
            "candidate_count": len(candidate_regions),
            "exhaustive": not budget_hit,
        }

        if not candidate_regions:
            break

        code_counts: Dict[str, int] = {}
        for _candidate, parent, _score in candidate_regions:
            if parent.canonical_code and parent.canonical_code not in token_by_code:
                code_counts[parent.canonical_code] = code_counts.get(parent.canonical_code, 0) + 1
        promoted_types: Dict[str, PatternType] = {}
        for candidate, parent, _score in candidate_regions:
            code = parent.canonical_code
            if (
                code
                and code not in token_by_code
                and code_counts.get(code, 0) >= min_promotion_support
            ):
                alias = _known_composition_token(candidate, parent, regions)
                pattern_type = register_pattern_type(parent.structure, token=alias)
                token_by_code[code] = pattern_type.token
                promoted_types[pattern_type.type_id] = pattern_type
                all_promoted_types[pattern_type.type_id] = pattern_type
        for candidate, parent, _score in candidate_regions:
            if parent.canonical_code in token_by_code:
                parent.pattern_type = token_by_code[parent.canonical_code]
                candidate.result_pattern_type = parent.pattern_type

        selected_children: Set[str] = set()
        selected_regions: List[Region] = []
        next_frontier_ids: List[str] = []
        for candidate, parent, _score in sorted(
            candidate_regions,
            key=lambda entry: (-entry[2], entry[0].candidate_id),
        ):
            regions[parent.region_id] = parent
            candidates.append(candidate)

        for candidate, parent, _score in sorted(
            candidate_regions,
            key=lambda entry: (-entry[2], entry[0].candidate_id),
        ):
            if any(child_id in selected_children for child_id in candidate.child_ids):
                continue
            prospective = selected_regions + [parent]
            carried = [
                regions[region_id]
                for region_id in frontier
                if region_id not in selected_children and region_id not in candidate.child_ids
            ]
            if _repeatability(prospective + carried) > repeatability_budget:
                continue
            candidate.selected_for_index = True
            parent.selected_for_index = True
            selected_children.update(candidate.child_ids)
            selected_regions.append(parent)

        for parent in selected_regions:
            for child_id in parent.child_ids:
                regions[child_id].parent_id = parent.region_id
            next_frontier_ids.append(parent.region_id)
        next_frontier_ids.extend(
            region_id for region_id in frontier if region_id not in selected_children
        )
        if not selected_regions:
            break
        frontier = next_frontier_ids
        repeatability_by_level[level] = _repeatability([regions[i] for i in frontier])

    if not frontier:
        root_id = None
    elif len(frontier) == 1:
        root_id = frontier[0]
    else:
        root_id = "<bos>"
        regions[root_id] = Region(
            region_id=root_id,
            level=max_height + 1,
            nodes=frozenset().union(*(regions[region_id].nodes for region_id in frontier)),
            structure=nx.compose_all([regions[region_id].structure for region_id in frontier]),
            child_ids=tuple(frontier),
            source_occurrence_ids=tuple(
                sorted(
                    {
                        occurrence_id
                        for region_id in frontier
                        for occurrence_id in regions[region_id].source_occurrence_ids
                    }
                )
            ),
            pattern_type="<bos>",
            is_virtual=True,
        )
        for region_id in frontier:
            regions[region_id].parent_id = root_id

    return Hierarchy(
        regions=regions,
        candidates=candidates,
        root_id=root_id,
        max_height=max_height,
        repeatability_by_level=repeatability_by_level,
        promoted_types=all_promoted_types,
        exploration_by_level=exploration_by_level,
        repeatability_budget=repeatability_budget,
    )


def build_neighborhood_regions(graph: nx.Graph, radius: int = 1) -> List[Region]:
    if radius < 0:
        raise ValueError("radius 必须是非负整数。")
    regions = []
    for index, node in enumerate(sorted(graph.nodes, key=_stable), start=1):
        nodes = frozenset(nx.single_source_shortest_path_length(graph, node, cutoff=radius))
        regions.append(
            Region(
                region_id=f"neighborhood-{index:06d}",
                level=0,
                nodes=nodes,
                structure=graph.subgraph(nodes).copy(),
                pattern_type="<neighborhood>",
            )
        )
    return regions


def build_disjoint_regions(
    occurrences: Sequence[PatternOccurrence],
    pattern_types: Mapping[str, PatternType],
) -> List[Region]:
    """用确定性贪心选出节点互不相交的基础实例，作为非重叠基线。"""
    regions = []
    used_nodes: Set[Any] = set()
    ordered = sorted(
        occurrences,
        key=lambda occurrence: (
            len(occurrence.nodes),
            occurrence.type_id,
            occurrence.occurrence_id,
        ),
    )
    for occurrence in ordered:
        if used_nodes.intersection(occurrence.nodes):
            continue
        used_nodes.update(occurrence.nodes)
        regions.append(
            Region(
                region_id=f"disjoint-{occurrence.occurrence_id}",
                level=0,
                nodes=frozenset(occurrence.nodes),
                structure=_structure_for_occurrence(occurrence, pattern_types),
                source_occurrence_ids=(occurrence.occurrence_id,),
                pattern_type=pattern_types[occurrence.type_id].token,
            )
        )
    return regions


def region_records(hierarchy: Hierarchy, *, index_only: bool = False) -> List[Dict[str, Any]]:
    regions = [
        region
        for region in hierarchy.regions.values()
        if not index_only or region.selected_for_index or region.is_virtual
    ]
    return [
        {
            "region_id": region.region_id,
            "parent_id": region.parent_id,
            "children": list(region.child_ids),
            "source_occurrence_ids": list(region.source_occurrence_ids),
            "covered_nodes": list(_sorted_nodes(region.nodes)),
            "result_pattern_type": region.pattern_type,
            "composition_rule": region.composition_rule,
            "level": region.level,
            "is_virtual": region.is_virtual,
            "selected_for_index": region.selected_for_index,
        }
        for region in sorted(regions, key=lambda item: item.region_id)
    ]


def candidate_records(hierarchy: Hierarchy) -> List[Dict[str, Any]]:
    return [
        {
            "candidate_id": candidate.candidate_id,
            "result_region_id": candidate.result_region_id,
            "child_ids": list(candidate.child_ids),
            "child_node_sets": [list(nodes) for nodes in candidate.child_node_sets],
            "shared_node_ids": list(candidate.shared_node_ids),
            "result_pattern_type": candidate.result_pattern_type,
            "composition_rule": candidate.composition_rule,
            "covered_nodes": list(candidate.covered_nodes),
            "level": candidate.level,
            "relation_kind": candidate.relation_kind,
            "exploration_status": "explored",
            "selected_for_index": candidate.selected_for_index,
        }
        for candidate in hierarchy.candidates
    ]


__all__ = [
    "CompositionCandidate",
    "Hierarchy",
    "Region",
    "build_hierarchy",
    "build_disjoint_regions",
    "build_neighborhood_regions",
    "candidate_records",
    "region_records",
]
