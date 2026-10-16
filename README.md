# 基本模式挖掘与编码

`basic_pattern.py` 面向单张 NetworkX 图，复用 FLEXIS 频繁子图挖掘程序，按发现顺序收缩基本模式，并输出 `<1>/<2>/...` 编码。

`subgraph_sampling.py` 提供从大图中采样与给定 query 同构的子图接口，复用 NetworkX 的 VF2 子图匹配器。

`run_experiment.py` 生成合成 query/宿主图，建立频繁模式实例、候选组合关系图和有预算的索引树，并输出 OracleCoverage。它使用无标签、无向、无自环简单图；模式叶子是原图中具体的频繁模式实例，组合候选保留节点重叠和递归标签，不复制边列表。索引树仅选择单父合并，候选关系图会保留预算内探索过的备选组合。

## 安装

项目固定使用 uv 0.12.23、Python 3.12.15 和 NetworkX 3.7。服务器需先安装指定版本的 uv，然后在工作区根目录执行：

```bash
curl -LsSf https://astral.sh/uv/0.12.23/install.sh | sh
uv sync
```

`uv sync` 会读取 `.python-version` 和 `uv.lock`，在项目目录创建 `.venv` 并安装锁定的依赖。服务器缺少 Python 3.12.15 时，`uv` 会自动下载托管版本；项目配置只使用 uv 托管的解释器。后续运行脚本可使用 `uv run`，例如 `uv run python basic_pattern.py`；也可激活 `.venv` 后使用其中的 Python。

`requirements.txt` 同步固定了 NetworkX 版本，供不使用 `uv` 的环境参考；项目的可复现安装以 `uv.lock` 为准。

FLEXIS 使用 C++ 实现。先安装 CMake、Ninja 和 vcpkg，然后在工作区执行：

```powershell
git clone https://github.com/akshit-sharma/flexis external/flexis
cd external/flexis
cmake --preset release
cmake --build --preset release
```

确保 `VCPKG_ROOT` 已指向 vcpkg 目录。构建完成后，程序默认查找：

```text
external/flexis/build/release/bin/flexis.exe
external/flexis/build/release/bin/process.exe
```

也可以用环境变量指定路径：

```powershell
$env:FLEXIS_ROOT = (Resolve-Path .\external\flexis).Path
# 或分别指定两个可执行文件：
$env:FLEXIS_BIN = (Resolve-Path .\external\flexis\build\release\bin\flexis.exe).Path
$env:FLEXIS_PROCESS_BIN = (Resolve-Path .\external\flexis\build\release\bin\process.exe).Path
```

## 使用

```python
import networkx as nx
from basic_pattern import get_encode_basic_pattern

graph = nx.Graph()
graph.add_node("a", label="person")
graph.add_node("b", label="person")
graph.add_edge("a", "b", label="friend")

# bound 是 FLEXIS 的 MIS 支持度整数阈值。
code = get_encode_basic_pattern(graph, bound=2)
print(code)  # 例如：<1>/<2>
```

节点和边的 `label` 可使用字符串或整数；未指定时默认标签为 `0`。无向图写入 FLEXIS 时会转换为双向边。有向图保持边方向。当前不接受 NetworkX 的 `MultiGraph` / `MultiDiGraph`。

`max_pattern_size` 默认是 6，可设置为 2 到 10；`slider` 默认是 FLEXIS 的 0.98。每轮从 FLEXIS 最后一个非空模式层中选择顶点数、边数最多的模式。FLEXIS 按 MIS 支持度判定频繁性，不等同于图事务集合中的常规定义。

收缩后的超节点会在节点属性中保存 `members`、`internal_edges` 和 `pattern_id`；收缩后合并的边会保存 `provenance`。多个边合并到同一超边时，`label` 会记录边标签及重数摘要。

收缩结构参考了 GCon 的无损收缩表示：超节点保留原节点映射和可供恢复的边信息。这里的频繁模式选择与不重叠实例匹配是本模块逻辑。

## Query 子图采样

```python
import networkx as nx
from subgraph_sampling import sample_query_subgraphs

G = nx.Graph()
G.add_edges_from([(1, 2), (2, 3), (3, 1), (3, 4)])
query = nx.path_graph(3)

# 默认是非诱导匹配；三角形中的路径也算 query 的匹配实例。
samples = sample_query_subgraphs(G, query, B=2, seed=7)
for sample in samples:
    print(sample.nodes, sample.edges, sample.graph["query_mapping"])

# 若要求匹配节点之间不能出现 query 没有的额外边：
induced_samples = sample_query_subgraphs(
    G, query, B=2, induced=True, seed=7
)

# 需要跨全部不同实例精确均匀抽样时使用 uniform；它会完整枚举匹配。
uniform_samples = sample_query_subgraphs(
    G, query, B=2, strategy="uniform", seed=7
)
```

函数最多返回 B 个实例；匹配实例少于 B 个时返回全部实例。默认 `strategy="random"` 会随机化一次节点和边的插入顺序，再按受该顺序影响的 VF2 遍历顺序取样，找到 B 个实例后停止；它不承诺全局均匀。该策略会复制 G 和 query 的图结构，额外空间复杂度为 O(|V_G| + |E_G| + |V_query| + |E_query|)，不含属性对象自身大小。`strategy="uniform"` 使用蓄水池抽样从全部不同实例中无放回均匀采样，需要遍历 VF2 枚举的所有映射（包括之后去重的对称映射）；运行时间受映射总数影响，去重集合的内存随不同实例数及 query 大小增长。两种策略都会折叠 query 自同构造成的重复映射；非诱导模式下，同一组主图节点对应的不同匹配边集合会视为不同实例。节点和边默认按 `label` 属性匹配，缺失时视为标签 `0`。结果保留主图节点 ID 和属性，并在图级属性 `query_mapping` 中记录 query 到主图的节点映射。

当前匹配器为 NetworkX VF2，适合无需编译的通用 Python 集成；对超大或稠密图有更高性能要求时，可评估 [VF3 官方 C++ 库](https://github.com/MiviaLab/vf3lib)。

## 重叠层次索引与 OracleCoverage

在工作区根目录运行合成实验：

```bash
uv run python run_experiment.py --output-dir artifacts/overlap_experiment --seed 7
```

默认生成 10 个无标签无向 query，每个至少植入 5 次，再添加至多 15 个节点和 15 条边。最终 query 真值从扰动后的图用 VF2 重新枚举。可以通过参数调整支持度、重叠阈值 `tau`、全局重复引用预算、最大合并高度、匹配枚举上限和区域大小上限：

```powershell
uv run python run_experiment.py `
  --seed 7 `
  --min-support 3 `
  --min-pattern-vertices 3 `
  --max-pattern-vertices 8 `
  --tau 0.34 `
  --max-height 3 `
  --region-size-limits 4 6 8 12 `
  --retrieval-budgets 5 `
  --storage-budgets 250
```

`R` 默认采用基础叶实例本身的重复引用率，作为所有叶子都保留时的最低可行预算。若显式指定 `--repeatability-budget`，该值必须不小于叶层重复率，否则程序会报出最低可行值。层次摘要会记录生效的 `R`、每层候选配对探索数，以及候选搜索是否触及预算上限。

若已构建 FLEXIS，可选传入 `--flexis-bound N` 让 FLEXIS 从合成宿主图中额外提取一个频繁模式候选；不传该参数时，基础词表来自生成的 query 模式。

输出目录包括 `graph.json`、`queries.json`、`planted_provenance.json`、`pattern_vocabulary.json`、`pattern_occurrences.json`、`occurrence_enumeration.json`、`composition_candidates.json`、`composition_regions.json`、`index_regions.json`、`hierarchy_summary.json`、`matches.jsonl`、`match_enumeration.json` 和 `oracle_coverage.json`。`oracle_coverage.json` 按非重叠区域、重叠索引树、完整候选组合图及固定半径邻域对照，并标记匹配枚举是否因上限而不完整；有限预算下同时报告可行贪心覆盖率下界和由松弛优化得到的 Oracle 覆盖上界，上下界相等时最优值已确定；`hierarchy_summary.json` 标记每层候选搜索是否触及预算上限。验证边工作量按原图中区域诱导子图的边数计算。`--max-pattern-vertices` 当前最多为 8；更大的结构保留递归组合表达。

## 参考

- FLEXIS 论文发表于 KDD 2025，仓库提供了 C++ 频繁子图挖掘实现：[论文 DOI](https://doi.org/10.1145/3711896.3736966)、[代码仓库](https://github.com/akshit-sharma/flexis)。
- 上游命令行、输入格式和模式文本格式见 FLEXIS 的 [README](https://github.com/akshit-sharma/flexis#readme)。
- GCon 对超节点、摘要信息和逆收缩的定义见 [Making graphs compact by lossless contraction](https://link.springer.com/article/10.1007/s00778-022-00731-7)。
- query 子图匹配使用 NetworkX [VF2 文档](https://networkx.org/documentation/stable/reference/algorithms/isomorphism.vf2.html)；VF3 官方库为 [MiviaLab/vf3lib](https://github.com/MiviaLab/vf3lib)。
