"""将精确匹配实例转换为 BART 的区域路径监督数据。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import networkx as nx


SPECIAL_TOKENS = [
    "<pad>", "<bos>", "<eos>", "<unk>", "<q>", "<qnode>", "<edge>",
    "<index>", "<region>", "<pattern>", "<nodes>", "<stats>",
    "<path_sep>", "<root>", "<tri>", "<square>", "<star>", "<diamond>",
    "<butterfly>", "<composition>", "<neighborhood>",
]


class FixedVocabulary:
    """实验期间固定的结构 token 词表，越界时明确失败。"""

    def __init__(self, graph_node_limit: int = 499, child_limit: int = 2047, query_node_limit: int = 7):
        tokens = list(SPECIAL_TOKENS)
        tokens.extend(f"<qnode_{index}>" for index in range(query_node_limit + 1))
        tokens.extend(f"<node_{index}>" for index in range(graph_node_limit + 1))
        tokens.extend(f"<child_{index}>" for index in range(child_limit + 1))
        tokens.extend(f"<count_{index}>" for index in range(501))
        tokens.extend(f"<edges_{index}>" for index in range(2049))
        self.token_to_id = {token: index for index, token in enumerate(tokens)}
        self.id_to_token = tokens

    def id(self, token: str) -> int:
        if token not in self.token_to_id:
            raise ValueError(f"token 不在固定词表中：{token}")
        return self.token_to_id[token]

    def encode(self, tokens: Iterable[str]) -> List[int]:
        return [self.id(token) for token in tokens]

    def dump(self, path: Path) -> None:
        path.write_text(json.dumps(self.token_to_id, ensure_ascii=False, indent=2), encoding="utf-8")


def _stable(value: Any) -> Tuple[str, str]:
    return type(value).__qualname__, repr(value)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _region_paths(regions: Sequence[Mapping[str, Any]], root_id: str) -> Dict[str, Tuple[int, ...]]:
    by_id = {region["region_id"]: region for region in regions}
    if root_id not in by_id:
        raise ValueError(f"索引树缺少根节点 {root_id}")
    paths: Dict[str, Tuple[int, ...]] = {root_id: ()}
    queue = [root_id]
    while queue:
        parent_id = queue.pop(0)
        children = sorted(by_id[parent_id].get("children", []))
        for child_index, child_id in enumerate(children):
            if child_id in paths:
                raise ValueError(f"索引结构出现多父节点或环：{child_id}")
            paths[child_id] = paths[parent_id] + (child_index,)
            queue.append(child_id)
    return paths


def _select_diverse_matches(
    graph: nx.Graph, matches: Sequence[Mapping[str, Any]], limit: int = 5, radius: int = 1
) -> List[Mapping[str, Any]]:
    """用图邻域最大新增覆盖贪心选取至多 B 个实例。"""
    neighborhoods = []
    for match in matches:
        covered = set(match["host_nodes"])
        for node in match["host_nodes"]:
            covered.update(nx.single_source_shortest_path_length(graph, node, cutoff=radius))
        neighborhoods.append(covered)
    selected: List[int] = []
    covered_nodes = set()
    remaining = set(range(len(matches)))
    while remaining and len(selected) < limit:
        chosen = min(remaining, key=lambda index: (-len(neighborhoods[index] - covered_nodes), index))
        selected.append(chosen)
        covered_nodes.update(neighborhoods[chosen])
        remaining.remove(chosen)
    return [matches[index] for index in selected]


def build_graph_examples(graph_dir: Path, *, split: str, vocabulary: FixedVocabulary) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    graph_json = _load_json(graph_dir / "graph.json")
    graph = nx.Graph()
    graph.add_nodes_from(graph_json["nodes"])
    graph.add_edges_from(tuple(edge) for edge in graph_json["edges"])
    node_ids = sorted(graph.nodes, key=_stable)
    node_index = {node: index for index, node in enumerate(node_ids)}
    if len(node_ids) > 500:
        raise ValueError(f"图节点数超出词表容量：{len(node_ids)}")

    query_json = _load_json(graph_dir / "queries.json")
    regions = _load_json(graph_dir / "index_regions.json")
    summary = _load_json(graph_dir / "hierarchy_summary.json")
    root_id = summary["root_id"]
    if root_id is None:
        raise ValueError("索引树没有根节点")
    paths = _region_paths(regions, root_id)
    region_by_id = {region["region_id"]: region for region in regions}
    (graph_dir / "region_paths.json").write_text(
        json.dumps([
            {
                "region_id": region_id,
                "path": list(paths[region_id]),
                "result_pattern_type": region_by_id[region_id].get("result_pattern_type"),
                "children": region_by_id[region_id].get("children", []),
                "covered_nodes": region_by_id[region_id].get("covered_nodes", []),
            }
            for region_id in sorted(paths, key=lambda rid: (len(paths[rid]), paths[rid], rid))
        ], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    index_context: List[str] = ["<index>"]
    for region_id in sorted(paths, key=lambda rid: (len(paths[rid]), paths[rid], rid)):
        region = region_by_id[region_id]
        index_context.extend(["<region>", "<root>" if region_id == root_id else "<pattern>"])
        pattern_token = region.get("result_pattern_type") or "<composition>"
        index_context.append(pattern_token if pattern_token in vocabulary.token_to_id else "<composition>")
        index_context.extend(f"<child_{child}>" for child in paths[region_id])
        index_context.append("<nodes>")
        covered = sorted((node_index[node] for node in region.get("covered_nodes", [])))
        index_context.extend(f"<node_{index}>" for index in covered)
        induced_edges = graph.subgraph(region.get("covered_nodes", [])).number_of_edges()
        index_context.extend([
            "<stats>", f"<count_{min(len(covered), 500)}>", f"<edges_{min(induced_edges, 2048)}>",
        ])

    all_matches = _read_jsonl(graph_dir / "matches.jsonl")
    enumeration = _load_json(graph_dir / "match_enumeration.json")
    examples: List[Dict[str, Any]] = []
    uncovered: Dict[str, List[str]] = {}
    for query_id, query_record in sorted(query_json.items()):
        query_nodes = sorted(query_record["nodes"], key=_stable)
        query_index = {node: index for index, node in enumerate(query_nodes)}
        query_tokens: List[str] = ["<q>"]
        for index in range(len(query_nodes)):
            query_tokens.extend(["<qnode>", f"<qnode_{index}>"])
        for source, target in sorted(query_record["edges"], key=lambda edge: (_stable(edge[0]), _stable(edge[1]))):
            left, right = sorted((query_index[source], query_index[target]))
            query_tokens.extend(["<edge>", f"<qnode_{left}>", f"<qnode_{right}>"])
        input_ids = vocabulary.encode(query_tokens + index_context)
        query_matches = [record for record in all_matches if record["query_id"] == query_id]
        region_options = []
        for region_id, region in region_by_id.items():
            region_nodes = set(region.get("covered_nodes", []))
            if region_nodes:
                region_options.append((len(region_nodes), len(paths[region_id]), region_id, region_nodes))
        targets = []
        missing = []
        for match in query_matches:
            host = set(match["host_nodes"])
            candidates = [option for option in region_options if host.issubset(option[3])]
            if not candidates:
                missing.append(match["match_id"])
                continue
            targets.append((match, min(candidates)[:3]))
        if missing:
            uncovered[query_id] = missing
        selected = _select_diverse_matches(graph, [item[0] for item in targets], limit=5, radius=1)
        selected_ids = {item["match_id"] for item in selected}
        selected_targets = [(match, label) for match, label in targets if match["match_id"] in selected_ids]
        selected_targets.sort(key=lambda item: item[0]["match_id"])
        target_tokens: List[str] = []
        source_matches = []
        for match, (_size, _depth, region_id) in selected_targets:
            target_tokens.extend(
                ["<root>"] if not paths[region_id]
                else [f"<child_{index}>" for index in paths[region_id]]
            )
            target_tokens.append("<path_sep>")
            source_matches.append(match["match_id"])
        target_tokens.append("<eos>")
        examples.append({
            "graph_id": graph_json["graph_id"], "query_id": query_id, "split": split,
            "input_ids": input_ids, "target_ids": vocabulary.encode(target_tokens),
            "target_paths": [list(paths[label[2]]) for match, label in selected_targets],
            "target_region_ids": [label[2] for match, label in selected_targets],
            "source_match_ids": source_matches,
            "match_enumeration_complete": enumeration[query_id]["complete"],
            "positive_match_count": len(query_matches), "selected_match_count": len(selected_targets),
        })
    (graph_dir / "path_examples.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in examples),
        encoding="utf-8",
    )
    return examples, {
        "graph_id": graph_json["graph_id"], "split": split, "graph_nodes": len(node_ids),
        "query_count": len(query_json), "example_count": len(examples),
        "uncovered_matches_by_query": {key: len(value) for key, value in uncovered.items()},
        "max_input_tokens": max((len(row["input_ids"]) for row in examples), default=0),
        "max_target_tokens": max((len(row["target_ids"]) for row in examples), default=0),
    }


__all__ = ["FixedVocabulary", "build_graph_examples"]
