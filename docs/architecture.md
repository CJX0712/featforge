# FeatForge · Architecture

> 模块化 AutoML 特征工程系统 · 作者 晨星 (CJX0712) · CPU 可跑 · 确定性可复现

## 1. 设计目标
- **复用顶级开源**：以 `scikit-learn`（底层 `numpy`/`scipy`）为 SOTA 后端，不自研公式化 PMDarima/AutoML 引擎（符合 Forge 系列契约）。
- **两个原创旗舰创新**：
  - `FeatForgeSynth` —— 相关性引导的自动特征合成；
  - `FeatForgeSelect` —— mRMR 去冗余预筛 + 交叉验证驱动的贪心前向选择。
- **离线兜底**：所有度量（相关 / 互信息替代 / 下游 CV）均有纯 `numpy` 实现；sklearn 不可用时自动降级为手写 k 折逻辑回归 / 最小二乘。
- **单向依赖**：`cli → pipeline → {data, synth, select, eval} → core`。

## 2. 目录骨架
```
featforge/
  core/        types(Dataset/FeatureFrame/Result) · errors(E100-E500) · config(FF_* 覆盖) · interfaces(Protocol)
  data/        synthetic(5 个合成难度梯度) · loaders(csv/npy/json)
  synth/       synthesizer(FeatForgeSynth 旗舰1) · registry
  select/      selector(FeatForgeSelect 旗舰2 · mRMR + 前向选择)
  eval/        metrics(下游 CV · 互信息 · |corr|，numpy 兜底)
  pipeline/    FeatPipeline.evaluate / benchmark / 报告
  cli.py       argparse 入口(benchmark / features)
  examples/    run_demo.py(端到端 + 落盘 benchmark.json)
tests/         pytest 单测（41 个，全绿）
docs/          architecture.md
```

## 3. 调用单向无环
```
cli → pipeline → {data, synth, select, eval} → core
```
`pipeline` 不继承任何模型，只按名解析并统一测量，**三臂共用一个模型脚手架**（`StandardScaler + LogisticRegression`），
保证任何差异都归因于“特征质量”而非“模型不同”。

## 4. 旗舰创新细节

### 4.1 FeatForgeSynth（旗舰 1 · 相关性引导合成）
1. **变换库**（逐列，带域安全守卫）：`x²`、`|x|`、`log1p|x|`、`sqrt|x|`、`sigmoid(x)`、`1/(|x|+1)`。
   全部主值分支天然避免 NaN/Inf，输出再经 `nan_to_num` 兜底 ⇒ **不变量：合成矩阵恒为有限值**。
2. **交互发现**：所有成对乘积 `xi*xj`（受 `max_pair_scan` 上限保护防御内存爆炸）。
3. **相关性引导裁剪**：用 **互信息**（sklearn `mutual_info_classif/regression`；离线退化为 `|Pearson corr|`）
   给候选列打分，保留 top-k。
4. **预算分配**：交互有**独立预留预算**（交互是核心创新且数量远小于逐列变换），但该预留
   **不得超过 `synth_top_k` 总上限** —— 保证 `n_synth <= synth_top_k` 契约恒成立。

**不变量**：合成列数 ≤ `synth_top_k`；输出恒有限；`source` 字典标注每列 `raw`/`synth` 来源。

### 4.2 FeatForgeSelect（旗舰 2 · mRMR + 前向选择）
1. **mRMR 预筛**：`score(j) = relevance(j) − mean(|corr|(j, 已选))`。
   最大化与目标的相关性、最小化与已选列的冗余 ⇒ **近乎重复的列无法挤占真实信号**。
2. **贪心前向选择**：
   - 起点：单列 CV 得分最高者；
   - 每轮把使**下游指标提升最多**的列加入；
   - 仅当增益 > `tol`(1e-4) 才接受，否则停止 ⇒ **不变量：`forward_gains` 严格单调递增**。
3. 结果带完整 `forward_gains` 轨迹与 `relevance` 映射，**每个幸存列都可被审计**。

### 4.3 关键设计：为什么三臂用同一个模型
切换评测模型会混淆“特征增益”与“模型容量差”。因此 `downstream_cv` 内部固定为
`Pipeline(StandardScaler, LogisticRegression/LinearRegression)`，raw / synth_all / selected 三臂完全一致。
**结论：分数差 = 特征质量的差。**

## 5. 评测口径（诚实基线）
- **三臂**：`raw`（原始列，基线）、`synth_all`（原始 + 全候选）、`selected`（mRMR + 前向选择后的精简集）。
- 5 折交叉验证，`random_state=42` 固定 ⇒ 结果逐位可复现（sklearn 与 numpy 兜底两路均确定）。
- 每次运行的 `benchmark.json` 由写入时加盖时间戳与后端标注，**所有 README 数字必须同口径实测留存**。

## 6. 复现命令
```bash
python -m venv .venv && .venv/Scripts/python -m pip install -r requirements.txt
python -m pytest -q -W ignore::UserWarning      # 41 全绿
python -m featforge.examples.run_demo            # 生成 benchmark.json
python -m featforge.cli features --dataset xor   # 查看该数据集的合成/精选列
```

## 7. 已知边界（诚实声明）
- 前向选择为贪心：20 列以上的候选池建议把 `select_prefilter` 调小以控时。
- `friedman` 数据集上 `selected`(0.8600) 略低于 `synth_all`(0.8717)：这是**精度 vs 紧凑度的取舍**
  （特征数 7 vs 35），非机制缺陷；跨 5 数据集聚合 `selected` 仍最优（0.9402）。
- `sigmoid` / `inv` 等有界变换对极端量纲敏感，已用 `clip`/`+1` 守卫。
- 互信息打分随机性由 `random_state` 固定；离线 `|corr|` 退化版本**只看线性相关**，
  在纯非线性结构上会低估 —— 那属于兜底路径的正常上限，sklearn 可用时不触发。
