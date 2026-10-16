"""构建合成数据、重叠层次索引和 OracleCoverage 报告。"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import networkx as nx

from oracle_coverage import enumerate_match_records, evaluate_budgets
from overlap_hierarchy import (
    Hierarchy,
    Region,
    build_disjoint_regions,
    build_hierarchy,
    build_neighborhood_regions,
    candidate_records,
    region_records,
)
from pattern_instances import PatternOccurrence, PatternType, build_pattern_vocabulary
from synthetic_data import SyntheticDataset, generate_synthetic_dataset, graph_to_json


def _stable_key(value: Any):
    return type(value).__qualname__, repr(value)


def _json_graph(graph: nx.Graph) -> Dict[str, Any]:
    return graph_to_json(graph)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _pattern_type_record(pattern_type: PatternType) -> Dict[str, Any]:
    return {
        "type_id": pattern_type.type_id,
        "token": pattern_type.token,
        "canonical_code": pattern_type.canonical_code,
        "graph": _json_graph(pattern_type.graph),
    }


def _occurrence_record(occurrence: PatternOccurrence) -> Dict[str, Any]:
    return {
        "occurrence_id": occurrence.occurrence_id,
        "type_id": occurrence.type_id,
        "node_mapping": {
            str(node): occurrence.node_mapping[node]
            for node in sorted(occurrence.node_mapping)
        },
        "nodes": list(occurrence.nodes),
        "mining_round": occurrence.mining_round,
    }


def _match_record(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "match_id": record["match_id"],
        "query_to_graph": {
            str(node): record["query_to_graph"][node]
            for node in sorted(record["query_to_graph"])
        },
        "host_nodes": sorted(record["host_nodes"], key=_stable_key),
        "host_edges": [
            list(edge)
            for edge in sorted(
                record["host_edges"],
                key=lambda edge: tuple(_stable_key(node) for node in edge),
            )
        ],
        "cross_boundary": record["cross_boundary"],
    }


def _write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def run_experiment(
    *,
    output_dir: Path,
    seed: int = 7,
    copies_per_query: int = 5,
    extra_nodes: int = 15,
    extra_edges: int = 15,
    min_pattern_vertices: int = 3,
    max_pattern_vertices: int = 8,
    min_support: int = 3,
    tau: float = 0.34,
    repeatability_budget: Optional[float] = None,
    max_height: int = 3,
    min_promotion_support: int = 2,
    max_candidates_per_level: int = 5000,
    max_matches_per_query: Optional[int] = 100000,
    flexis_bound: Optional[int] = None,
    flexis_bin: Optional[str] = None,
    process_bin: Optional[str] = None,
    neighborhood_radius: int = 1,
    region_size_limits: Sequence[int] = (4, 6, 8, 12),
    retrieval_budgets: Sequence[Optional[int]] = (None, 5),
    storage_budgets: Sequence[Optional[int]] = (None, 250),
) -> Dict[str, Any]:
    experiment_started = time.perf_counter()
    dataset: SyntheticDataset = generate_synthetic_dataset(
        seed=seed,
        copies_per_query=copies_per_query,
        max_extra_nodes=extra_nodes,
        max_extra_edges=extra_edges,
    )
    seed_patterns = list(dataset.base_patterns.items())
    if flexis_bound is not None:
        from basic_pattern import freq_mining

        mined = freq_mining(
            dataset.graph,
            flexis_bound,
            flexis_bin=flexis_bin,
            process_bin=process_bin,
            max_pattern_size=min(10, max_pattern_vertices),
        )
        if mined is not None:
            seed_patterns.append(("frequent", mined.graph))
    pattern_types, occurrences, occurrence_completeness = build_pattern_vocabulary(
        dataset.graph,
        seed_patterns,
        min_support=min_support,
        min_vertices=min_pattern_vertices,
        max_vertices=max_pattern_vertices,
    )
    if not occurrences:
        raise RuntimeError("没有满足最低支持度的基础模式实例。")

    hierarchy = build_hierarchy(
        dataset.graph,
        occurrences,
        pattern_types,
        tau=tau,
        repeatability_budget=repeatability_budget,
        max_height=max_height,
        min_promotion_support=min_promotion_support,
        max_candidates_per_level=max_candidates_per_level,
    )
    disjoint_regions = build_disjoint_regions(occurrences, pattern_types)
    neighborhood_regions = build_neighborhood_regions(
        dataset.graph,
        radius=neighborhood_radius,
    )
    all_pattern_types = dict(pattern_types)
    all_pattern_types.update(hierarchy.promoted_types)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "graph.json", {
        "graph_id": "G0",
        "seed": seed,
        "directed": False,
        "labeled": False,
        **_json_graph(dataset.graph),
    })
    _write_json(output_dir / "queries.json", {
        query_id: _json_graph(query)
        for query_id, query in sorted(dataset.queries.items())
    })
    _write_json(output_dir / "planted_provenance.json", dataset.planted)
    _write_json(output_dir / "pattern_vocabulary.json", [
        _pattern_type_record(pattern_type)
        for pattern_type in sorted(all_pattern_types.values(), key=lambda item: item.type_id)
    ])
    _write_json(output_dir / "pattern_occurrences.json", [
        _occurrence_record(occurrence)
        for occurrence in sorted(occurrences, key=lambda item: item.occurrence_id)
    ])
    _write_json(output_dir / "occurrence_enumeration.json", occurrence_completeness)
    _write_json(output_dir / "index_regions.json", region_records(hierarchy, index_only=True))
    _write_json(output_dir / "composition_regions.json", region_records(hierarchy))
    _write_json(output_dir / "composition_candidates.json", candidate_records(hierarchy))
    _write_json(output_dir / "hierarchy_summary.json", {
        "root_id": hierarchy.root_id,
        "max_height": hierarchy.max_height,
        "repeatability_by_level": hierarchy.repeatability_by_level,
        "tau": tau,
        "requested_repeatability_budget": repeatability_budget,
        "effective_repeatability_budget": hierarchy.repeatability_budget,
        "exploration_by_level": hierarchy.exploration_by_level,
        "index_build_seconds": time.perf_counter() - experiment_started,
    })

    matches_jsonl = []
    oracle_rows = []
    baselines: Dict[str, List[Region]] = {
        "non_overlap": disjoint_regions,
        "overlap_index_tree": [
            region
            for region in hierarchy.regions.values()
            if region.selected_for_index and not region.is_virtual
        ],
        "overlap_candidate_graph": [
            region
            for region in hierarchy.regions.values()
            if not region.is_virtual
        ],
        "neighborhood": neighborhood_regions,
    }
    enumeration_summary = {}
    for query_id, query in sorted(dataset.queries.items()):
        matches, complete = enumerate_match_records(
            dataset.graph,
            query,
            occurrences,
            max_matches=max_matches_per_query,
        )
        enumeration_summary[query_id] = {
            "complete": complete,
            "match_count": len(matches),
            "max_matches": max_matches_per_query,
        }
        for record in matches:
            matches_jsonl.append({"graph_id": "G0", "query_id": query_id, **_match_record(record)})
        for baseline_name, regions in baselines.items():
            rows = evaluate_budgets(
                dataset.graph,
                matches,
                regions,
                region_size_limits,
                retrieval_budgets=retrieval_budgets,
                storage_budgets=storage_budgets,
            )
            for row in rows:
                row.update({
                    "graph_id": "G0",
                    "query_id": query_id,
                    "baseline": baseline_name,
                    "available_region_count": len(regions),
                    "available_storage_node_references": sum(len(region.nodes) for region in regions),
                    "match_enumeration_complete": complete,
                    "metric_is_estimate": not complete,
                })
                oracle_rows.append(row)

    _write_jsonl(output_dir / "matches.jsonl", matches_jsonl)
    _write_json(output_dir / "match_enumeration.json", enumeration_summary)
    _write_json(output_dir / "oracle_coverage.json", oracle_rows)
    return {
        "output_dir": str(output_dir.resolve()),
        "graph_nodes": dataset.graph.number_of_nodes(),
        "graph_edges": dataset.graph.number_of_edges(),
        "query_count": len(dataset.queries),
        "pattern_type_count": len(all_pattern_types),
        "occurrence_count": len(occurrences),
        "candidate_count": len(hierarchy.candidates),
        "root_id": hierarchy.root_id,
        "index_build_seconds": time.perf_counter() - experiment_started,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/overlap_experiment"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--copies-per-query", type=int, default=5)
    parser.add_argument("--extra-nodes", type=int, default=15)
    parser.add_argument("--extra-edges", type=int, default=15)
    parser.add_argument("--min-pattern-vertices", type=int, default=3)
    parser.add_argument("--max-pattern-vertices", type=int, default=8)
    parser.add_argument("--min-support", type=int, default=3)
    parser.add_argument("--tau", type=float, default=0.34)
    parser.add_argument(
        "--repeatability-budget",
        type=float,
        default=None,
        help="全局重复引用率上限；默认取固定叶层的最低可行值。显式值不能低于叶层率。",
    )
    parser.add_argument("--max-height", type=int, default=3)
    parser.add_argument("--min-promotion-support", type=int, default=2)
    parser.add_argument("--max-candidates-per-level", type=int, default=5000)
    parser.add_argument("--max-matches-per-query", type=int, default=100000)
    parser.add_argument("--flexis-bound", type=int)
    parser.add_argument("--flexis-bin")
    parser.add_argument("--process-bin")
    parser.add_argument("--neighborhood-radius", type=int, default=1)
    parser.add_argument("--region-size-limits", type=int, nargs="+", default=[4, 6, 8, 12])
    parser.add_argument("--retrieval-budgets", type=int, nargs="+", default=[5])
    parser.add_argument("--storage-budgets", type=int, nargs="+", default=[250])
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = run_experiment(
        output_dir=args.output_dir,
        seed=args.seed,
        copies_per_query=args.copies_per_query,
        extra_nodes=args.extra_nodes,
        extra_edges=args.extra_edges,
        min_pattern_vertices=args.min_pattern_vertices,
        max_pattern_vertices=args.max_pattern_vertices,
        min_support=args.min_support,
        tau=args.tau,
        repeatability_budget=args.repeatability_budget,
        max_height=args.max_height,
        min_promotion_support=args.min_promotion_support,
        max_candidates_per_level=args.max_candidates_per_level,
        max_matches_per_query=args.max_matches_per_query,
        flexis_bound=args.flexis_bound,
        flexis_bin=args.flexis_bin,
        process_bin=args.process_bin,
        neighborhood_radius=args.neighborhood_radius,
        region_size_limits=args.region_size_limits,
        retrieval_budgets=(None, *args.retrieval_budgets),
        storage_budgets=(None, *args.storage_budgets),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
