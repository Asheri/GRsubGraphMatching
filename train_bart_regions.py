"""构建跨图路径数据、并行训练三份小型 BART 并精确评估检索。"""

from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import networkx as nx
import torch
import transformers
from transformers import BartForConditionalGeneration

from bart_region_model import generate_paths, make_model, set_seed
from region_path_data import (
    MAX_RETRIEVAL_REGION_SIZE,
    FixedVocabulary,
    _region_paths,
    build_graph_examples,
)
from run_experiment import run_experiment
from subgraph_sampling import enumerate_query_matches


TRAIN_SPECS = [("G_train_0", 7), ("G_train_1", 8), ("G_train_2", 9)]
TEST_SPEC = ("G_test_0", 10)
TRAIN_SEEDS = [(17, 0), (23, 1), (29, 2)]
MAX_INPUT_LENGTH = 4096
MAX_TARGET_LENGTH = 128


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def prepare(
    output_dir: Path,
    *,
    max_matches_per_query: int = 100000,
    reuse_graph_artifacts: bool = False,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    graph_specs = [(graph_id, seed, "train") for graph_id, seed in TRAIN_SPECS]
    graph_specs.append((TEST_SPEC[0], TEST_SPEC[1], "test"))
    summaries = []
    for graph_id, seed, split in graph_specs:
        graph_dir = output_dir / split / graph_id
        if reuse_graph_artifacts:
            required = ["graph.json", "queries.json", "index_regions.json", "matches.jsonl", "match_enumeration.json"]
            missing = [name for name in required if not (graph_dir / name).is_file()]
            if missing:
                raise FileNotFoundError(f"{graph_id} 缺少可复用产物：{', '.join(missing)}")
            graph_record = _read_json(graph_dir / "graph.json")
            hierarchy_record = _read_json(graph_dir / "hierarchy_summary.json")
            summaries.append({
                "output_dir": str(graph_dir.resolve()), "graph_id": graph_id,
                "graph_nodes": len(graph_record["nodes"]), "graph_edges": len(graph_record["edges"]),
                "query_count": len(_read_json(graph_dir / "queries.json")),
                "pattern_type_count": len(_read_json(graph_dir / "pattern_vocabulary.json")),
                "occurrence_count": len(_read_json(graph_dir / "pattern_occurrences.json")),
                "candidate_count": len(_read_json(graph_dir / "composition_candidates.json")),
                "root_id": hierarchy_record["root_id"],
                "index_build_seconds": hierarchy_record["index_build_seconds"],
                "all_match_nodes_covered": all(
                    row["all_match_nodes_covered"]
                    for row in _read_json(graph_dir / "match_node_coverage.json").values()
                ),
            })
            continue
        summaries.append(run_experiment(
            output_dir=graph_dir,
            seed=seed,
            graph_id=graph_id,
            copies_per_query=5,
            extra_nodes=15,
            extra_edges=15,
            min_pattern_vertices=3,
            max_pattern_vertices=8,
            min_support=3,
            tau=0.30,
            max_height=2,
            max_candidates_per_level=5000,
            max_index_occurrences=512,
            max_matches_per_query=max_matches_per_query,
            region_size_limits=(4, 6, 8, 12),
            retrieval_budgets=(None, 5),
            storage_budgets=(None, 250),
        ))

    vocabulary = FixedVocabulary()
    train_rows = []
    test_rows = []
    data_summaries = []
    for graph_id, _seed, split in graph_specs:
        rows, summary = build_graph_examples(
            output_dir / split / graph_id, split=split, vocabulary=vocabulary
        )
        if summary["max_input_tokens"] > MAX_INPUT_LENGTH:
            raise ValueError(
                f"{graph_id} 输入长度 {summary['max_input_tokens']} 超过 BART 上限 {MAX_INPUT_LENGTH}。"
            )
        if summary["max_target_tokens"] > MAX_TARGET_LENGTH:
            raise ValueError(
                f"{graph_id} 目标长度 {summary['max_target_tokens']} 超过 BART 上限 {MAX_TARGET_LENGTH}。"
            )
        (train_rows if split == "train" else test_rows).extend(rows)
        data_summaries.append(summary)

    vocabulary.dump(output_dir / "path_vocabulary.json")
    _write_jsonl(output_dir / "path_examples_train.jsonl", train_rows)
    _write_jsonl(output_dir / "path_examples_test.jsonl", test_rows)
    manifest = {
        "split": {"train_graphs": [spec[0] for spec in TRAIN_SPECS], "test_graph": TEST_SPEC[0]},
        "train_seeds": {graph_id: seed for graph_id, seed in TRAIN_SPECS},
        "test_seed": TEST_SPEC[1],
        "query_ids": sorted(_read_json(output_dir / "train" / TRAIN_SPECS[0][0] / "queries.json")),
        "tau": 0.30,
        "R": "每张图按基础叶引用计算的最低可行重复预算",
        "H": 2,
        "candidate_budget_per_level": 5000,
        "max_retrieval_region_size": MAX_RETRIEVAL_REGION_SIZE,
        "test_graph_in_training_examples": False,
        "train_example_count": len(train_rows),
        "test_example_count": len(test_rows),
        "vocabulary_size": len(vocabulary.id_to_token),
        "data_summaries": data_summaries,
        "graph_summaries": summaries,
        "enumeration": {
            graph_id: _read_json(output_dir / split / graph_id / "match_enumeration.json")
            for graph_id, _seed, split in graph_specs
        },
    }
    _write_json(output_dir / "split_manifest.json", manifest)
    return {
        "output_dir": str(output_dir.resolve()),
        "train_examples": len(train_rows),
        "test_examples": len(test_rows),
        "vocabulary_size": len(vocabulary.id_to_token),
        "data_summaries": data_summaries,
    }


def _pad_batch(rows: Sequence[Mapping[str, Any]], pad_id: int, device: torch.device):
    input_length = max(len(row["input_ids"]) for row in rows)
    target_length = max(len(row["target_ids"]) for row in rows)
    input_ids = torch.full((len(rows), input_length), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(rows), input_length), dtype=torch.long)
    labels = torch.full((len(rows), target_length), -100, dtype=torch.long)
    for index, row in enumerate(rows):
        source = torch.tensor(row["input_ids"], dtype=torch.long)
        target = torch.tensor(row["target_ids"], dtype=torch.long)
        input_ids[index, :len(source)] = source
        attention_mask[index, :len(source)] = 1
        labels[index, :len(target)] = target
    return input_ids.to(device), attention_mask.to(device), labels.to(device)


def train(data_dir: Path, run_dir: Path, *, seed: int, device_name: str, epochs: int = 30) -> Dict[str, Any]:
    if not device_name.startswith("cuda:"):
        raise ValueError("训练设备必须显式指定为 --device cuda:i。")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，拒绝在 CPU 上静默训练。")
    device_index = int(device_name.split(":", 1)[1])
    if device_index >= torch.cuda.device_count():
        raise RuntimeError(f"请求 {device_name}，但当前只有 {torch.cuda.device_count()} 张 GPU。")
    device = torch.device(device_name)
    set_seed(seed)
    vocab = _read_json(data_dir / "path_vocabulary.json")
    rows = _read_jsonl(data_dir / "path_examples_train.jsonl")
    if not rows:
        raise ValueError("训练集为空。")
    model = make_model(len(vocab), max_input_length=MAX_INPUT_LENGTH, max_target_length=MAX_TARGET_LENGTH).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    best_loss = float("inf")
    history = []
    started = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        indices = list(range(len(rows)))
        random.Random(seed + epoch).shuffle(indices)
        loss_sum = 0.0
        for index in indices:
            batch = _pad_batch([rows[index]], int(vocab["<pad>"]), device)
            optimizer.zero_grad(set_to_none=True)
            output = model(input_ids=batch[0], attention_mask=batch[1], labels=batch[2])
            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(output.loss.detach().cpu())
        epoch_loss = loss_sum / len(rows)
        history.append({"epoch": epoch + 1, "train_loss": epoch_loss})
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            run_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(run_dir / "best_model")
            _write_json(run_dir / "best_checkpoint.json", {"epoch": epoch + 1, "train_loss": epoch_loss})
        print(json.dumps({"seed": seed, "device": device_name, "epoch": epoch + 1, "train_loss": epoch_loss}, ensure_ascii=False), flush=True)
    run_metadata = {
        "seed": seed, "device": device_name, "gpu_name": torch.cuda.get_device_name(device_index),
        "nvidia_driver": subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        ).stdout.splitlines()[device_index].strip(),
        "epochs": epochs, "best_train_loss": best_loss, "train_examples": len(rows),
        "model": {"d_model": 128, "encoder_layers": 2, "decoder_layers": 2, "heads": 4},
        "optimizer": "AdamW", "learning_rate": 3e-4, "weight_decay": 0.01,
        "batch_size": 1, "max_input_length": MAX_INPUT_LENGTH, "max_target_length": MAX_TARGET_LENGTH,
        "torch": torch.__version__, "transformers": transformers.__version__,
        "cuda_runtime": torch.version.cuda,
        "seconds": time.perf_counter() - started,
    }
    _write_json(run_dir / "training_history.json", history)
    _write_json(run_dir / "run_metadata.json", run_metadata)
    return run_metadata


def _instance_key(query: nx.Graph, mapping: Mapping[Any, Any]) -> Tuple[Tuple[str, ...], Tuple[Tuple[str, str], ...]]:
    nodes = tuple(sorted((repr(node) for node in mapping.values())))
    edges = tuple(sorted(tuple(sorted((repr(mapping[left]), repr(mapping[right])))) for left, right in query.edges))
    return nodes, edges


def _path_sequences(output_ids: Sequence[int], id_to_token: Sequence[str]) -> Tuple[List[Tuple[int, ...]], List[str]]:
    token_by_id = list(id_to_token)
    paths: List[Tuple[int, ...]] = []
    errors: List[str] = []
    current: List[int] = []
    saw_content = False
    tokens = list(output_ids)
    if tokens and 0 <= tokens[0] < len(token_by_id) and token_by_id[tokens[0]] == "<eos>":
        tokens = tokens[1:]
    for token_id in tokens:
        token = token_by_id[token_id] if 0 <= token_id < len(token_by_id) else "<invalid>"
        if token == "<eos>":
            break
        if token in {"<pad>", "<bos>"}:
            continue
        if token == "<path_sep>":
            if saw_content:
                paths.append(tuple(current))
            else:
                errors.append("empty_path")
            current = []
            saw_content = False
            continue
        if token == "<root>":
            if saw_content or current:
                errors.append("mixed_root_path")
            current = []
            saw_content = True
            continue
        if token.startswith("<child_") and token.endswith(">"): 
            try:
                current.append(int(token[len("<child_"):-1]))
                saw_content = True
            except ValueError:
                errors.append("invalid_child_token")
            continue
        errors.append("unexpected_token")
    if saw_content:
        paths.append(tuple(current))
    return paths, errors


def _evaluate_regions(
    graph: nx.Graph,
    query: nx.Graph,
    predicted_region_ids: Sequence[str],
    regions: Mapping[str, Mapping[str, Any]],
    *,
    max_regions: int = 5,
    node_budget: int = 250,
) -> Tuple[List[Dict[str, Any]], int, int, set]:
    found: Dict[Any, Dict[str, Any]] = {}
    verification_nodes = 0
    verification_edges = 0
    verified_regions = 0
    regions_with_matches = set()
    for region_id in predicted_region_ids:
        if verified_regions >= max_regions:
            break
        nodes = set(regions[region_id].get("covered_nodes", []))
        if (
            not nodes
            or regions[region_id].get("is_virtual", False)
            or len(nodes) > MAX_RETRIEVAL_REGION_SIZE
            or verification_nodes + len(nodes) > node_budget
        ):
            continue
        verified_regions += 1
        verification_nodes += len(nodes)
        induced = graph.subgraph(nodes).copy()
        verification_edges += induced.number_of_edges()
        mappings, _complete = enumerate_query_matches(induced, query, max_matches=1000)
        if mappings:
            regions_with_matches.add(region_id)
        for mapping in mappings:
            key = _instance_key(query, mapping)
            if len(found) < 5:
                found.setdefault(key, {"region_id": region_id, "query_to_graph": mapping})
    return list(found.values()), verification_nodes, verification_edges, regions_with_matches


def _truth_keys(graph_dir: Path, query_id: str, query: nx.Graph) -> Tuple[set, bool]:
    rows = [row for row in _read_jsonl(graph_dir / "matches.jsonl") if row["query_id"] == query_id]
    keys = set()
    for row in rows:
        nodes = tuple(sorted(repr(node) for node in row["host_nodes"]))
        edges = tuple(sorted(tuple(sorted((repr(edge[0]), repr(edge[1])))) for edge in row["host_edges"]))
        keys.add((nodes, edges))
    complete = _read_json(graph_dir / "match_enumeration.json")[query_id]["complete"]
    return keys, complete


def _metric_row(query_id: str, truth: set, retrieved: Sequence[Mapping[str, Any]], first_has_match: bool, valid_paths: int, generated_count: int, verification_nodes: int, verification_edges: int, seconds: float, complete: bool) -> Dict[str, Any]:
    retrieved_keys = {_instance_key_from_record(item) for item in retrieved}
    hit_count = len(retrieved_keys & truth)
    recall5 = hit_count / len(truth) if truth else None
    precision5 = hit_count / 5.0 if truth else None
    f1 = (2 * precision5 * recall5 / (precision5 + recall5)) if truth and (precision5 + recall5) else 0.0
    return {
        "query_id": query_id,
        "truth_count": len(truth),
        "retrieved_instance_count": len(retrieved),
        "Recall@1": int(first_has_match) if truth else None,
        "Recall@5": recall5,
        "Precision@5": precision5,
        "F1@5": f1,
        "PathValidity@5": min(valid_paths, 5) / max(1, generated_count),
        "valid_path_count": min(valid_paths, 5),
        "generated_path_count": generated_count,
        "verification_node_work": verification_nodes,
        "verification_edge_work": verification_edges,
        "retrieval_seconds": seconds,
        "match_enumeration_complete": complete,
        "metric_is_estimate": not complete,
    }


def _instance_key_from_record(item: Mapping[str, Any]):
    return item["instance_key"]


def _evaluate_one_seed(data_dir: Path, runs_dir: Path, output_dir: Path, seed: int, device_name: str) -> List[Dict[str, Any]]:
    vocab_record = _read_json(data_dir / "path_vocabulary.json")
    id_to_token = [token for token, _id in sorted(vocab_record.items(), key=lambda pair: pair[1])]
    test_dir = data_dir / "test" / TEST_SPEC[0]
    graph_record = _read_json(test_dir / "graph.json")
    graph = nx.Graph()
    graph.add_nodes_from(graph_record["nodes"])
    graph.add_edges_from(tuple(edge) for edge in graph_record["edges"])
    query_records = _read_json(test_dir / "queries.json")
    region_records = _read_json(test_dir / "index_regions.json")
    region_by_id = {row["region_id"]: row for row in region_records}
    region_paths = _region_paths(region_records, _read_json(test_dir / "hierarchy_summary.json")["root_id"])
    path_to_region = {path: region_id for region_id, path in region_paths.items()}
    test_examples = {row["query_id"]: row for row in _read_jsonl(data_dir / "path_examples_test.jsonl")}
    output_dir.mkdir(parents=True, exist_ok=True)
    predicted_rows = []
    retrieved_rows = []
    all_metric_rows = []
    run_dir = runs_dir / f"seed{seed}"
    model = BartForConditionalGeneration.from_pretrained(run_dir / "best_model").to(device_name)
    device = next(model.parameters()).device
    for query_id, query_record in sorted(query_records.items()):
        query = nx.Graph()
        query.add_nodes_from(query_record["nodes"])
        query.add_edges_from(tuple(edge) for edge in query_record["edges"])
        example = test_examples[query_id]
        started = time.perf_counter()
        outputs = generate_paths(model, example["input_ids"], device=device)
        inference_seconds = time.perf_counter() - started
        top_path_region_id = None
        if outputs:
            top_paths, _ = _path_sequences(outputs[0], id_to_token)
            if top_paths:
                candidate_id = path_to_region.get(top_paths[0])
                if candidate_id is not None:
                    candidate = region_by_id[candidate_id]
                    if (
                        not candidate.get("is_virtual", False)
                        and len(candidate.get("covered_nodes", [])) <= MAX_RETRIEVAL_REGION_SIZE
                    ):
                        top_path_region_id = candidate_id
        predicted_region_ids = []
        valid_paths = 0
        invalid_reasons: Dict[str, int] = {}
        for output in outputs:
            if len(predicted_region_ids) >= 5:
                break
            paths, parse_errors = _path_sequences(output, id_to_token)
            for reason in parse_errors:
                invalid_reasons[reason] = invalid_reasons.get(reason, 0) + 1
            if not paths:
                invalid_reasons["no_path"] = invalid_reasons.get("no_path", 0) + 1
            for path in paths:
                if len(predicted_region_ids) >= 5:
                    break
                region_id = path_to_region.get(path)
                if region_id is None:
                    invalid_reasons["unreachable_path"] = invalid_reasons.get("unreachable_path", 0) + 1
                elif region_id in predicted_region_ids:
                    invalid_reasons["duplicate_path"] = invalid_reasons.get("duplicate_path", 0) + 1
                elif region_by_id[region_id].get("is_virtual", False):
                    invalid_reasons["virtual_root_not_retrievable"] = invalid_reasons.get("virtual_root_not_retrievable", 0) + 1
                elif len(region_by_id[region_id].get("covered_nodes", [])) > MAX_RETRIEVAL_REGION_SIZE:
                    invalid_reasons["oversize_region"] = invalid_reasons.get("oversize_region", 0) + 1
                else:
                    predicted_region_ids.append(region_id)
        query_started = time.perf_counter()
        retrieved, verify_nodes, verify_edges, regions_with_matches = _evaluate_regions(
            graph, query, predicted_region_ids, region_by_id
        )
        valid_paths = len(regions_with_matches)
        match_seconds = time.perf_counter() - query_started
        truth, complete = _truth_keys(test_dir, query_id, query)
        retrieved_with_keys = []
        for item in retrieved:
            key = _instance_key(query, item["query_to_graph"])
            retrieved_with_keys.append({**item, "instance_key": key})
            retrieved_rows.append({
                "seed": seed, "query_id": query_id, "region_id": item["region_id"],
                "query_to_graph": item["query_to_graph"],
            })
        first_has_match = False
        if top_path_region_id is not None:
            first_results, _, _, _ = _evaluate_regions(
                graph, query, [top_path_region_id], region_by_id, max_regions=1
            )
            first_has_match = any(_instance_key(query, item["query_to_graph"]) in truth for item in first_results)
        metrics = _metric_row(
            query_id, truth, retrieved_with_keys, first_has_match, valid_paths,
            len(outputs), verify_nodes, verify_edges, inference_seconds + match_seconds, complete,
        )
        metrics.update({"seed": seed, "method": "bart"})
        all_metric_rows.append(metrics)
        predicted_rows.append({
            "seed": seed, "query_id": query_id, "predicted_region_ids": predicted_region_ids,
            "predicted_paths": [list(region_paths[region_id]) for region_id in predicted_region_ids],
            "valid_path_count": valid_paths, "invalid_reasons": invalid_reasons,
        })
    model.to("cpu")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    _write_jsonl(output_dir / "predicted_paths.jsonl", predicted_rows)
    _write_jsonl(output_dir / "retrieved_matches.jsonl", retrieved_rows)
    _write_jsonl(output_dir / "metrics_by_query.jsonl", all_metric_rows)
    with (output_dir / "metrics_by_query.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        columns = sorted({key for row in all_metric_rows for key in row})
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(all_metric_rows)
    _write_json(output_dir / "seed_summary.json", {"seed": seed, "metrics": _macro_metrics(all_metric_rows)})
    return all_metric_rows


def evaluate(data_dir: Path, runs_dir: Path, output_dir: Path) -> Dict[str, Any]:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 3:
        raise RuntimeError("评估三份模型前要求至少 3 张可用 GPU，并行分配到 cuda:0/1/2。")
    from concurrent.futures import ThreadPoolExecutor

    output_dir.mkdir(parents=True, exist_ok=True)
    futures = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        for seed, device_index in TRAIN_SEEDS:
            seed_output = output_dir / f"seed{seed}"
            futures.append(pool.submit(
                _evaluate_one_seed, data_dir, runs_dir, seed_output, seed, f"cuda:{device_index}"
            ))
        seed_rows = [future.result() for future in futures]
    all_metric_rows = [row for rows in seed_rows for row in rows]
    summary_by_seed = [
        {"seed": seed, "metrics": _macro_metrics(rows)}
        for (seed, _gpu), rows in zip(TRAIN_SEEDS, seed_rows)
    ]
    test_dir = data_dir / "test" / TEST_SPEC[0]
    graph_record = _read_json(test_dir / "graph.json")
    graph = nx.Graph()
    graph.add_nodes_from(graph_record["nodes"])
    graph.add_edges_from(tuple(edge) for edge in graph_record["edges"])
    query_records = _read_json(test_dir / "queries.json")
    region_records = _read_json(test_dir / "index_regions.json")
    region_paths = _region_paths(region_records, _read_json(test_dir / "hierarchy_summary.json")["root_id"])
    baseline_rows = _evaluate_baselines(graph, query_records, test_dir, region_records, region_paths)
    combined_rows = all_metric_rows + baseline_rows
    _write_jsonl(output_dir / "metrics_by_query.jsonl", combined_rows)
    with (output_dir / "metrics_by_query.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        columns = sorted({key for row in combined_rows for key in row})
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(combined_rows)
    # 合并每个 seed 的细节产物，方便按 query/seed 查看预测与返回实例。
    for filename in ("predicted_paths.jsonl", "retrieved_matches.jsonl"):
        _write_jsonl(output_dir / filename, [
            row for seed in (17, 23, 29)
            for row in _read_jsonl(output_dir / f"seed{seed}" / filename)
        ])
    report = {
        "bart_by_seed": summary_by_seed,
        "bart_seed_mean_std": _mean_std([row["metrics"] for row in summary_by_seed]),
        "baselines": _macro_metrics_by_method(baseline_rows),
        "test_graph": TEST_SPEC[0], "test_query_count": len(query_records),
        "metrics_note": "Recall@1/5、Precision@5、F1@5 基于测试图最终图上的精确 VF2 真值。",
    }
    _write_json(output_dir / "metrics_summary.json", report)
    _write_json(output_dir / "baseline_metrics.json", report["baselines"])
    return report


def _macro_metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    names = ["Recall@1", "Recall@5", "Precision@5", "F1@5", "PathValidity@5", "retrieval_seconds", "verification_node_work", "verification_edge_work"]
    result = {}
    for name in names:
        values = [float(row[name]) for row in rows if row.get(name) is not None]
        result[name] = sum(values) / len(values) if values else None
        result[f"{name}_n"] = len(values)
    return result


def _mean_std(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    output = {}
    for name in ("Recall@1", "Recall@5", "Precision@5", "F1@5", "PathValidity@5"):
        values = [float(row[name]) for row in rows if row.get(name) is not None]
        mean = sum(values) / len(values) if values else None
        variance = sum((value - mean) ** 2 for value in values) / len(values) if len(values) > 1 else 0.0
        output[name] = {"mean": mean, "std": variance ** 0.5 if mean is not None else None}
    return output


def _evaluate_baselines(graph, query_records, test_dir, region_records, region_paths):
    rows = []
    region_by_id = {row["region_id"]: row for row in region_records}
    regular = [
        row for row in region_records
        if not row.get("is_virtual") and len(row.get("covered_nodes", [])) <= MAX_RETRIEVAL_REGION_SIZE
    ]
    neighborhood = []
    for index, node in enumerate(sorted(graph.nodes, key=repr)):
        nodes = set(nx.single_source_shortest_path_length(graph, node, cutoff=1))
        neighborhood.append({"region_id": f"baseline-neighbor-{index}", "covered_nodes": list(nodes)})
    for query_id, query_record in sorted(query_records.items()):
        query = nx.Graph()
        query.add_nodes_from(query_record["nodes"])
        query.add_edges_from(tuple(edge) for edge in query_record["edges"])
        truth, complete = _truth_keys(test_dir, query_id, query)
        random_order = [row["region_id"] for row in regular]
        random.Random(0).shuffle(random_order)
        methods = {
            "smallest_region": [row["region_id"] for row in sorted(regular, key=lambda row: (len(row["covered_nodes"]), row["region_id"]))],
            "random_region": random_order,
        }
        for method, ordered_ids in methods.items():
            chosen = []
            refs = 0
            for region_id in ordered_ids:
                if len(chosen) >= 5:
                    break
                cost = len(region_by_id[region_id]["covered_nodes"])
                if refs + cost <= 250:
                    refs += cost
                    chosen.append(region_id)
            retrieved, work_nodes, work_edges, valid_regions = _evaluate_regions(graph, query, chosen, region_by_id)
            keys = {_instance_key(query, item["query_to_graph"]) for item in retrieved}
            true_hits = len(keys & truth)
            recall = true_hits / len(truth) if truth else None
            precision = true_hits / 5 if truth else None
            first, _, _, _ = _evaluate_regions(graph, query, chosen[:1], region_by_id, max_regions=1)
            row = {
                "query_id": query_id, "truth_count": len(truth), "retrieved_instance_count": len(retrieved),
                "Recall@1": int(any(_instance_key(query, item["query_to_graph"]) in truth for item in first)) if truth else None,
                "Recall@5": recall, "Precision@5": precision,
                "F1@5": 2 * precision * recall / (precision + recall) if truth and precision + recall else 0.0,
                "PathValidity@5": len(valid_regions) / 5, "valid_path_count": len(valid_regions),
                "generated_path_count": 5, "verification_node_work": work_nodes,
                "verification_edge_work": work_edges, "retrieval_seconds": None,
                "match_enumeration_complete": complete, "metric_is_estimate": not complete,
                "method": method,
            }
            rows.append(row)
        # 固定半径邻域对照不要求该区域属于路径树。
        found = {}
        work_nodes = work_edges = 0
        checked_regions = 0
        valid_neighborhoods = 0
        for region in neighborhood:
            if checked_regions >= 5:
                break
            cost = len(region["covered_nodes"])
            if cost > MAX_RETRIEVAL_REGION_SIZE or work_nodes + cost > 250:
                continue
            work_nodes += cost
            checked_regions += 1
            subgraph = graph.subgraph(region["covered_nodes"]).copy()
            work_edges += subgraph.number_of_edges()
            matches, _ = enumerate_query_matches(subgraph, query, max_matches=1000)
            if matches:
                valid_neighborhoods += 1
            for mapping in matches:
                if len(found) < 5:
                    found.setdefault(_instance_key(query, mapping), {"query_to_graph": mapping})
        keys = set(found) & truth
        recall = len(keys) / len(truth) if truth else None
        precision = len(keys) / 5 if truth else None
        rows.append({
            "query_id": query_id, "truth_count": len(truth), "retrieved_instance_count": len(found),
            "Recall@1": int(bool(keys)) if truth else None, "Recall@5": recall,
            "Precision@5": precision,
            "F1@5": 2 * precision * recall / (precision + recall) if truth and precision + recall else 0.0,
            "PathValidity@5": valid_neighborhoods / 5, "valid_path_count": valid_neighborhoods,
            "generated_path_count": 5, "verification_node_work": work_nodes,
            "verification_edge_work": work_edges, "retrieval_seconds": None,
            "match_enumeration_complete": complete, "metric_is_estimate": not complete,
            "method": "radius1_neighborhood",
        })
    return rows


def _macro_metrics_by_method(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {method: _macro_metrics([row for row in rows if row["method"] == method]) for method in sorted({row["method"] for row in rows})}


def _launch_all(args: argparse.Namespace) -> int:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 3:
        raise RuntimeError(f"要求恰好 3 张可用 GPU，当前发现 {torch.cuda.device_count() if torch.cuda.is_available() else 0} 张。")
    processes = []
    for seed, device_index in TRAIN_SEEDS:
        run_dir = args.runs_dir / f"seed{seed}"
        command = [sys.executable, str(Path(__file__).resolve()), "train", "--data-dir", str(args.data_dir), "--run-dir", str(run_dir), "--seed", str(seed), "--device", f"cuda:{device_index}", "--epochs", str(args.epochs)]
        log_path = run_dir / "training.log"
        run_dir.mkdir(parents=True, exist_ok=True)
        stream = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
        processes.append((seed, process, stream, log_path))
    failures = []
    for seed, process, stream, log_path in processes:
        code = process.wait()
        stream.close()
        if code:
            failures.append({"seed": seed, "exit_code": code, "log": str(log_path)})
    if failures:
        print(json.dumps({"status": "incomplete", "failures": failures}, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps({"status": "complete", "runs": [str(args.runs_dir / f"seed{seed}") for seed, _ in TRAIN_SEEDS]}, ensure_ascii=False, indent=2))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--output-dir", type=Path, required=True)
    prepare_parser.add_argument("--max-matches-per-query", type=int, default=100000)
    prepare_parser.add_argument("--reuse-graph-artifacts", action="store_true")
    train_parser = commands.add_parser("train")
    train_parser.add_argument("--data-dir", type=Path, required=True)
    train_parser.add_argument("--run-dir", type=Path, required=True)
    train_parser.add_argument("--seed", type=int, required=True)
    train_parser.add_argument("--device", required=True)
    train_parser.add_argument("--epochs", type=int, default=30)
    all_parser = commands.add_parser("train-all")
    all_parser.add_argument("--data-dir", type=Path, required=True)
    all_parser.add_argument("--runs-dir", type=Path, required=True)
    all_parser.add_argument("--epochs", type=int, default=30)
    eval_parser = commands.add_parser("evaluate")
    eval_parser.add_argument("--data-dir", type=Path, required=True)
    eval_parser.add_argument("--runs-dir", type=Path, required=True)
    eval_parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(
            args.output_dir,
            max_matches_per_query=args.max_matches_per_query,
            reuse_graph_artifacts=args.reuse_graph_artifacts,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "train":
        result = train(args.data_dir, args.run_dir, seed=args.seed, device_name=args.device, epochs=args.epochs)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "train-all":
        raise SystemExit(_launch_all(args))
    else:
        result = evaluate(args.data_dir, args.runs_dir, args.output_dir)
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
