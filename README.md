# FeatForge

> 模块化 AutoML 特征工程系统 · 作者 **晨星 (CJX0712)** · CPU 可跑 · 确定性可复现

FeatForge 是一套端到端的**自动特征工程**平台：在 `scikit-learn`（底层 `numpy`/`scipy`）之上，
自动为表格数据**合成**候选特征（非线性变换 + 成对交互），再用去冗余 + 前向选择**精选**出
一个紧凑的高价值特征集。所有度量都有纯 `numpy` 离线兜底。

**核心价值**：用一个固定的下游模型（`StandardScaler + LogisticRegression`），
在 5 个合成基准上把平均精度从 **0.7673 提到 0.9402（+0.1729，相对 +22.5%）**，
而且特征数反而更少（7 → 5）。

---

## ✨ 特性

- **旗舰 1 · FeatForgeSynth**：相关性引导的特征合成。逐列变换库（`x²` / `|x|` / `log1p|x|` /
  `sqrt|x|` / `sigmoid(x)` / `1/(|x|+1)`）+ 成对交互 `xi*xj`，用**互信息**给候选列打分后保留 top-k。
  全部变换带域安全守卫 ⇒ 输出恒为有限值。
- **旗舰 2 · FeatForgeSelect**：两阶段精选。① **mRMR**（最大相关、最小冗余）预筛，让近乎重复的
  合成列无法挤占真实信号；② **贪心前向选择**，每轮只加入使下游交叉验证指标提升最多的列。
- **离线兜底**：`downstream_cv` 在无 sklearn 时降级为手写的 **k 折逻辑回归 / 最小二乘**（纯 numpy）；
  互信息退化为 `|Pearson corr|`。保证零重型依赖也能跑。
- **同模型评测**：三臂共用同一模型脚手架，分数差即特征质量差，不是模型容量差。
- **可审计**：`forward_gains` 轨迹 + `relevance` 映射 + `source`（`raw`/`synth`）标注，
  每个幸存列都能解释为什么留下。
- **单向无环架构**：`cli → pipeline → {data, synth, select, eval} → core`。

---

## 📦 安装

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
```

> Python ≥ 3.9。核心依赖仅 `numpy` + `scipy` + `scikit-learn`。

---

## 🚀 快速开始

### Python API

```python
from featforge.data.synthetic import default_benchmark_set
from featforge.pipeline import FeatPipeline

pipe = FeatPipeline()
report = pipe.benchmark(default_benchmark_set(), out_path="benchmark.json")
print(pipe.print_report(report))
```

### 单独看某个数据集合成/精选了哪些列

```bash
python -m featforge.cli features --dataset xor
#   会打印：最高相关性的合成列 Top-10、最终精选列、前向选择的精度轨迹
```

### 命令行

```bash
python -m featforge.cli benchmark --out benchmark.json         # 跑全套三臂基准
python -m featforge.examples.run_demo                          # 端到端 demo + 落盘
python -m pytest -q -W ignore::UserWarning                     # 41 测试全绿
```

---

## 📊 性能基线（实测 · sklearn 1.9.1 · cv=5 · random_state=42）

> 5 个数据集，每个都刻意埋了线性模型抓不到的效应（交互 / 非线性 / 冗余）。
> **三臂共用 `StandardScaler + LogisticRegression`**，5 折交叉验证准确率。

| 数据集 | raw 基线 | synth_all（全候选） | **selected（精选）** | 精选特征数 | 增益 vs raw |
|---|---|---|---|---|---|
| interaction (`x0*x1+0.3x2`) | 0.6483 | 0.8917 | **0.9233** | 4 | **+0.2750** |
| nonlinear (`sin x0 + x1²`) | 0.8233 | 0.9367 | **0.9617** | 6 | **+0.1384** |
| xor（纯 XOR 交互） | 0.5560 | 0.9360 | **0.9680** | 3 | **+0.4120** |
| redundant（含重复列） | 0.9720 | 0.9600 | **0.9880** | 5 | **+0.0160** |
| friedman（经典交互基准） | 0.8367 | 0.8717 | 0.8600 | 7 | +0.0233 |
| **聚合均值** | **0.7673** | 0.9192 | **0.9402** | **5.00** | **+0.1729** |

**关键结论**

- `selected` 在 **5/5 数据集上击败 raw 基线**，聚合 **+0.1729（相对 +22.5%）**，
  且平均只用 **5 个特征**（raw 有 7 个）——**既更准、又更省**。
- 最能体现价值的 `xor`：原始线性模型几乎随机（0.5560），合成出交互列后跃至 **0.9680**。
- `redundant`：mRMR 成功剔除 5 个近乎重复的冗余列，精度反升到 0.9880，验证了去冗余的意义。
- ⚠️ **诚实边界**：`friedman` 上 `selected`(0.8600) 略低于 `synth_all`(0.8717)。
  这是**精度与紧凑度的取舍**（7 个特征 vs 35 个），不是机制缺陷；跨 5 集聚合 `selected` 仍最优。
  数值均为同口径实测，`benchmark.json` 可复现。

---

## 🧠 旗舰创新细节

### FeatForgeSynth（旗舰 1）

1. **变换库**（域安全）：`x²`、`|x|`、`log1p|x|`、`sqrt|x|`、`sigmoid(x)`、`1/(|x|+1)`，全部主值分支天然避开 NaN/Inf。
2. **交互发现**：成对乘积 `xi*xj`，受 `max_pair_scan` 保护防御内存爆炸。
3. **互信息打分**：`mutual_info_classif/regression`；无 sklearn 时退化为 `|Pearson corr|`。
4. **预算分配**：交互有独立预留预算，但**绝不超过 `synth_top_k` 总上限**。
   **不变量**：合成列数 ≤ `synth_top_k`；输出恒有限。

### FeatForgeSelect（旗舰 2）

1. **mRMR 预筛**：`score(j) = relevance(j) − mean(|corr|(j, 已选))`。
   最大化与目标相关性、最小化与已选列冗余 ⇒ 近乎重复的列无法挤占真实信号。
2. **贪心前向选择**：起点取单列 CV 最优；每轮加入使**下游指标提升最多**的列；
   仅当增益 > `tol`(1e-4) 才接受，否则停止。
   **不变量**：`forward_gains` **严格单调递增**（贪心只接受改进）。
3. 输出带完整轨迹与相关性映射，可逐个审计。

---

## 🏗️ 架构

```
featforge/
  core/        types · errors(E100-E500) · config(FF_* 环境变量覆盖) · interfaces(Protocol)
  data/        synthetic(5 个难度梯度) · loaders(csv/npy/json)
  synth/       synthesizer(FeatForgeSynth 旗舰1) · registry
  select/      selector(FeatForgeSelect 旗舰2)
  eval/        metrics(下游 CV · 互信息 · |corr| · numpy 兜底)
  pipeline/    FeatPipeline.evaluate / benchmark / 报告
  cli.py       argparse 入口(benchmark / features)
  examples/    run_demo.py
tests/         pytest 单测（41 个，全绿）
docs/          architecture.md
```

---

## ⚙️ 环境变量

| 变量 | 含义 |
|---|---|
| `FF_RANDOM_STATE` | 全局随机种子（默认 42） |
| `FF_CV` | 交叉验证折数（默认 5） |
| `FF_SYNTH_TOP_K` | 合成列总数上限（默认 30） |
| `FF_SYNTH_INTERACTIONS` | 交互列预留数（默认 20） |
| `FF_SELECT_K` | 最终精选特征数（默认 15） |
| `FF_SELECT_PREFILTER` | mRMR 预筛宽度（默认 40） |

---

## 🔁 复现

```bash
python -m pytest -q -W ignore::UserWarning      # 41 测试全绿
python -m featforge.examples.run_demo           # 生成 benchmark.json（逐位可复现）
```

---

## ⚠️ 已知边界

- 前向选择为贪心，候选池 > 40 列建议调小 `select_prefilter` 以控时（当前 demo 约 34s）。
- 离线 `|corr|` 只看线性相关，纯非线性结构上会低估；sklearn 可用时不触发该路径。
- 有界变换（`sigmoid`/`inv`）对极端量纲敏感，已用 `clip`/`+1` 守卫。

---

## 📄 协议与署名

作者：**晨星 (CJX0712)**。代码以 MIT 协议开源，可自由学习、修改、再分发。
