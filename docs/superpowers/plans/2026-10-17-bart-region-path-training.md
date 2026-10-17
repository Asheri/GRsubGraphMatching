# BART 区域路径生成训练实现计划

> **面向 AI 代理的工作者：** 按任务顺序在当前工作区执行。步骤使用复选框记录进度。训练运行本身是用户要求的实验评估；不新增或运行独立测试套件。

**目标：** 在 3 张独立训练图上训练小型 BART，使其根据 query 和目标图索引上下文生成区域路径，并在第 4 张新图上评估实例检索效果。

**架构：** 每张图独立生成固定的 10 个 query、宿主图、基础模式实例和层次索引。数据构建器为每条 query 匹配选择最小包含区域作为路径标签；BART 接收 query 与目标图索引序列，产生至多 5 条候选路径；精确 VF2 匹配器在候选区域中生成最终实例。三次独立随机种子训练分别使用三张 GPU。

**技术栈：** Python 3.12.15、uv 0.12.23、NetworkX 3.7、PyTorch、Transformers、NetworkX VF2。

---

## 文件边界

- 修改 `pyproject.toml`、`uv.lock`：固定 PyTorch、Transformers 及训练依赖，确保服务器 `uv sync` 可复现。
- 修改 `synthetic_data.py`：为不同宿主图添加图级节点命名空间，保持 `make_seed_queries()` 的 10 个 query 完全一致。
- 修改 `run_experiment.py`：接受图 ID 与模型实验所需的树高度/阈值/候选预算，将图 ID 写入所有产物。
- 新建 `region_path_data.py`：生成树路径、索引上下文、路径监督样本和可验证的样本元数据。
- 新建 `bart_region_model.py`：实现确定性 token 词表、图/query 序列化、小型随机初始化 BART 和 beam path 解码。
- 新建 `train_bart_regions.py`：构建 3 张训练图及 1 张测试图，运行三个 GPU 训练种子、精确检索、基线比较并写报告。
- 修改 `README.md`：加入 `uv sync`、数据设置、GPU 运行命令、指标解释和产物说明。

## 任务 1：固定跨图训练环境

**文件：**
- 修改：`pyproject.toml`
- 修改：`uv.lock`

- [ ] 根据远程服务器的 NVIDIA 驱动和 CUDA 能力确定兼容的固定 PyTorch/Transformers 版本；训练依赖作为项目依赖纳入 lock，服务器运行 `uv sync` 后即可导入。
- [ ] 配置 PyTorch CUDA wheel 来源，使 `uv sync` 在 Linux x86_64 上安装 CUDA 可用的 PyTorch，而不是意外安装 CPU-only wheel；锁文件记录确切 URL 和哈希。
- [ ] 保留 `requires-python == 3.12.15`、`uv == 0.12.23` 和 `networkx == 3.7`，重新生成 lock 文件，并记录 GPU/驱动/运行库版本检查命令。
- [ ] 在服务器项目目录安装固定的 uv 版本，再执行锁定安装；服务器尚无 uv 时使用 `curl -LsSf https://astral.sh/uv/0.12.23/install.sh | sh`，随后将 uv 的用户 bin 目录加入当前 shell 的 `PATH`。

安装检查命令：

```bash
uv --version
uv sync --locked
uv run python -c "import torch, transformers, networkx; print(torch.__version__, transformers.__version__, networkx.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
```

预期：`uv` 为 0.12.23，NetworkX 为 3.7，服务器能够发现 3 张 CUDA GPU；版本号以生成的 lock 为准。

## 任务 2：构造固定 query 的 3+1 张图

**文件：**
- 修改：`synthetic_data.py`
- 修改：`run_experiment.py`
- 新建：`train_bart_regions.py`

- [ ] 为 `generate_synthetic_dataset()` 增加 `graph_id: str = "G0"`；将宿主节点 ID 与 planted provenance 改为包含图 ID 的稳定命名，例如 `G_train_0:tri:copy0:node0`。`make_seed_queries()` 不依赖 graph seed，10 个 query 结构与 ID 在四张图中完全相同。
- [ ] 为 `run_experiment()` 增加 `graph_id` 参数，传入数据生成器，并将其用于 `graph.json`、`matches.jsonl`、Oracle 行及摘要；保持既有默认行为 `G0`。
- [ ] 每个植入 query 实例额外加确定性覆盖脚手架：3 节点补成三角形、4 节点补成三叶星、5 节点补成共享中心的两个三角形；原 query 边保留，脚手架边单独写入 provenance。扰动边只在模式覆盖节点间采样，噪声节点保持隔离。
- [ ] 精确枚举每个 query 的最终匹配后，校验每个匹配节点集合都被三角形/四环/三叶星基础实例覆盖；违反时中止数据构建并写明未覆盖实例。
- [ ] 新建编排函数，分别以图种子 `7, 8, 9` 生成 `G_train_0..2`，以种子 `10` 生成 `G_test_0`；四张图使用相同的 query 结构、植入次数和扰动上限，各自独立构造模式实例及层次树。
- [ ] 使用 `tau=0.30`、`max_height=2`、`max_candidates_per_level=5000` 做最小规模运行；每张图的 `hierarchy_summary.json` 必须原样保留 `possible_pair_count`、`explored_pair_count`、`budget_hit` 和 `exhaustive`，不把预算截断结果误报为穷举。
- [ ] 将四张图的图 ID、随机种子、query ID、split、`tau/R/H`、候选预算、VF2 完整枚举状态写入 `split_manifest.json`；测试图数据只放测试目录，训练数据加载接口禁止枚举该目录。

产物布局：

```text
artifacts/bart_region_path/
  train/G_train_0/...
  train/G_train_1/...
  train/G_train_2/...
  test/G_test_0/...
  split_manifest.json
```

## 任务 3：生成区域路径标签与 BART 输入

**文件：**
- 新建：`region_path_data.py`
- 修改：`overlap_hierarchy.py`（仅当需要稳定导出根路径或子节点排序时）

- [ ] 从 `index_regions.json` 读取仅被索引树选中的区域；以 `hierarchy_summary.json.root_id` 为根，按稳定 `region_id` 排序子节点，计算每个可访问区域的路径。虚拟根使用 `<root>`，路径段采用 `<child_i>`，叶子路径和中间路径都可作为目标。
- [ ] 对每条完整 VF2 匹配，筛选 `host_nodes` 完全包含于索引树区域的候选，按 `(区域节点数, 路径深度, region_id)` 选择最小区域；若没有普通区域满足，则允许使用覆盖匹配的虚拟 BOS 根。无法定位的匹配保留为未覆盖样本并计数，不伪造路径。
- [ ] 每个 `(graph_id, query_id)` 输入样本包括确定性 query 边序列和对应索引树上下文；每个图的 host node ID 先按稳定排序转换为图内整数编号。索引上下文包含所有可达区域的路径、组合 token、子节点、覆盖节点编号及由该区域诱导子图计算的节点数/边数摘要。
- [ ] 为路径 token 和图结构 token 建立固定词表：保留 query/edge/region/模式 token，预分配 `<qnode_0>` 至 `<qnode_7>`、`<node_0>` 至 `<node_499>`、`<child_0>` 至 `<child_2047>`、区域节点数与边数 token、`<root>`、`<path_sep>`、`<eos>`、`<unk>`。超出预分配范围时明确报错，不静默截断或映射成普通文本。
- [ ] 每个 query 在每张训练图上从完整匹配中用半径 1 的覆盖贪心采样至多 5 个监督实例；把对应路径按稳定顺序连接为目标序列。测试真值不用于词表训练、路径监督、模型选择或早停。
- [ ] 写出 `path_examples.jsonl`、`path_vocabulary.json`、`region_paths.json`，逐条包含图 ID、query ID、输入 token ID、目标路径、来源 match ID、匹配枚举完整标记和未覆盖原因。

## 任务 4：实现小型 BART 和三卡训练

**文件：**
- 新建：`bart_region_model.py`
- 修改：`train_bart_regions.py`

- [ ] 通过 `BartConfig` 构建随机初始化模型：`d_model=128`、encoder/decoder 各 2 层、`encoder_attention_heads=decoder_attention_heads=4`、`encoder_ffn_dim=decoder_ffn_dim=512`、dropout `0.1`；词表大小来自固定词表，输入最大长度 4096，输出最大长度 128。位置嵌入容量考虑 BART 的 offset 2。
- [ ] 使用 `BartForConditionalGeneration` 和固定 seed 训练；batch 内对输入/目标做右侧 padding，target padding 置 `-100`；优化器使用 AdamW，学习率 `3e-4`，weight decay `0.01`，最多 200 epoch，按训练 loss 保存最优 checkpoint，不读取测试指标做早停或调参。
- [ ] 每个 epoch 用确定性的训练样本顺序或显式记录 shuffle seed；运行配置记录模型结构、随机种子、数据摘要、学习率、batch size、最大长度、epoch 数、beam 数及软件/驱动版本。
- [ ] 提供 `--device cuda:i` 参数。GPU 数量为 3 时分别启动 seed `17, 23, 29`，显式映射到 `cuda:0/1/2` 并行运行；任意 GPU 不可用时在启动训练前失败，不让多个训练进程争抢同一设备。
- [ ] 推理用 `num_beams=10`、`num_return_sequences=5` 的确定性 beam search；解析至多 5 条 `<child_i>` 路径，丢弃非法、不可达和重复路径，并记录每种无效原因。

服务器命令：

```bash
uv run python train_bart_regions.py prepare --output-dir artifacts/bart_region_path
uv run python train_bart_regions.py train --data-dir artifacts/bart_region_path --run-dir artifacts/bart_region_path/runs/seed17 --seed 17 --device cuda:0
uv run python train_bart_regions.py train --data-dir artifacts/bart_region_path --run-dir artifacts/bart_region_path/runs/seed23 --seed 23 --device cuda:1
uv run python train_bart_regions.py train --data-dir artifacts/bart_region_path --run-dir artifacts/bart_region_path/runs/seed29 --seed 29 --device cuda:2
uv run python train_bart_regions.py evaluate --data-dir artifacts/bart_region_path --runs-dir artifacts/bart_region_path/runs --output-dir artifacts/bart_region_path/report
```

三个训练命令并发启动；`prepare` 和 `evaluate` 为 CPU/VF2 工作，在训练开始前完成准备，在训练完成后执行评估。

## 任务 5：精确验证、基线及指标报告

**文件：**
- 新建：`train_bart_regions.py`
- 复用：`subgraph_sampling.py`、`oracle_coverage.py`

- [ ] 对每个测试 query 按路径顺序读取最多 5 个预测区域，在原图限制为该区域节点的诱导子图上调用既有非诱导 VF2 枚举；把映射转换成规范实例 key，按实例去重，最多保留 5 个实例。
- [ ] 真值只从测试图最终图的完整 VF2 枚举取得。记录 `Recall@1`、`Recall@5`、`Precision@5`、`F1@5`、`PathValidity@5`、返回实例数、路径无效率、验证节点/边数、推理/匹配耗时及枚举完整性；匹配枚举不完整时把依赖它的指标标为估计值。
- [ ] 实现随机区域、按区域大小排序和固定半径邻域三种基线；所有方法均从区域内用同一个 VF2 实现返回实例，使用相同的 5 区域和验证成本上限。
- [ ] 在测试图每个 query 上先计算指标，再宏平均；无匹配 query 的 Precision/Recall/F1 标记为不适用，不参与宏平均。对 seed 17/23/29 分别汇总均值与标准差，并同时保存逐 query CSV/JSON。
- [ ] 输出 `predicted_paths.jsonl`、`retrieved_matches.jsonl`、`metrics_by_query.csv`、`metrics_summary.json`、`baseline_metrics.json` 和环境元数据；确认 split manifest 记录测试图从未进入任何训练样本。

## 任务 6：文档、服务器运行与结果说明

**文件：**
- 修改：`README.md`
- 更新：`uv.lock`

- [ ] 写明 3 张训练图/1 张测试图、固定 10-query 的评估边界、路径定义、指标公式、预算截断状态和结果仅是合成跨图可行性证据。
- [ ] 文档列出先 `uv sync --locked`、再 `prepare`、三 GPU 并行 `train`、最后 `evaluate` 的服务器命令，以及所有输出文件解释。
- [ ] SSH 服务器运行前记录 `nvidia-smi`、GPU 数量、CUDA 驱动、`uv --version`、Python 与依赖版本；确认 GPU 数为 3 后再启动训练。
- [ ] 服务器原 `/home/zuohaonan/GRsubGraphMatching` 不是独立 Git 根目录，其 `.git` 位于 `/home/zuohaonan`，且父仓库状态异常；不在该目录同步或安装。新建隔离运行目录 `/home/zuohaonan/GRsubGraphMatching-bart-run`，只同步已提交代码和配置，再在该目录中安装与训练。
- [ ] 如训练阶段发生任何错误，先按全局 `AGENTS.md` 要求汇总此前已知训练错误，并追加到 `C:\Users\12062\OneDrive\Desktop\items\training-errors.md`；后续每次新错误继续追加时间、阶段、日志、根因、修复和验证信息。
- [ ] 完成 3 个随机种子运行后，按指标与基线如实报告结果；若某进程失败，保留其它已完成结果并将整体实验标为不完整，不伪造均值。

## 计划自检

- 规格中的固定 10 个 query、3 张训练图/1 张新测试图、节点命名空间、`tau=0.30` 和候选截断记录对应任务 2。
- 中间树节点路径、最小包含区域标签、query 与索引树输入、最多 5 个监督 match、固定词表对应任务 3。
- 小型随机初始化 BART、确定性 beam、三个 GPU seed 对应任务 4。
- Recall@1/5、Precision@5、F1@5、PathValidity@5、验证成本、匹配枚举完整性、基线和 seed 均值/标准差对应任务 5。
- 锁定运行环境、服务器准备、训练错误档案和结果说明对应任务 1、6。
- 不新增独立测试套件；运行 `prepare/train/evaluate` 是用户请求的训练可行性实验本身。
