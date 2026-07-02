# Multi-Agent Collaboration Benchmark (MAB)

**DAI 2026 Industry Track — Junze Li**

一个框架无关的多智能体AI系统基准测试工具。回答一个问题：*"哪个模型 + 哪种协作策略 最适合我的任务？"*

---

## 这个工具能做什么

| 功能 | 说明 |
|------|------|
| **运行标准化基准测试** | 9个内置任务，覆盖问答/代码/数据分析/研究/规划 |
| **对比模型和协作策略** | 支持单Agent、多步流水线、角色扮演、真实CrewAI |
| **自动采集指标** | 分数、token消耗、耗时、成本、编排开销 |
| **可视化对比** | Streamlit 仪表盘，热力图+柱状图+成本分析 |
| **自定义扩展** | 加新任务 ~20行代码，加新框架适配器 ~100行代码 |

## 架构

```
┌──────────────────────────────────────┐
│  Streamlit 可视化仪表盘               │
├──────────────────────────────────────┤
│  基准测试运行器 & 报告生成器           │
├──────────────────────────────────────┤
│  框架适配器                           │
│  ├─ Single-Agent (基线，无框架依赖)    │
│  ├─ Multi-Step (自研三步流水线)        │
│  ├─ Role-Based (CrewAI 风格模拟)      │
│  ├─ CrewAI (真实库，需 Python 3.10+)   │
│  ├─ AutoGen (需安装)                  │
│  └─ LangGraph (需安装)                │
├──────────────────────────────────────┤
│  任务套件 (9个任务, 4个类别)           │
├──────────────────────────────────────┤
│  插桩式 LLM 层 (自动指标采集)          │
└──────────────────────────────────────┘
```

---

## 快速开始

### 1. 安装

```bash
# 核心依赖（必装）
pip install -r requirements.txt

# 可选：第三方多智能体框架
pip install crewai         # CrewAI 适配器（需要 Python 3.10+）
pip install pyautogen      # AutoGen 适配器
pip install langgraph langchain langchain-anthropic  # LangGraph 适配器
```

### 2. 设置 API Key

```bash
# DeepSeek（推荐，最便宜）
# Windows PowerShell:
$env:DEEPSEEK_API_KEY = "sk-..."
# Linux/Mac:
export DEEPSEEK_API_KEY="sk-..."

# OpenAI
export OPENAI_API_KEY="sk-..."

# 或者使用中转代理
export OPENAI_API_KEY="sk-..."
export OPENAI_API_BASE="https://your-proxy.com/v1"
```

### 3. 运行实验

```bash
# === 最常用 ===

# 单Agent基线测试（免装第三方框架，开箱即用）：
python experiments/run_experiment.py

# DeepSeek V4 Flash 多策略对比（推荐）：
python experiments/run_deepseek_full.py

# 多模型跨厂商对比（DeepSeek + OpenAI）：
python experiments/run_multimodel.py

# 真实 CrewAI 测试（需 Python 3.12 + crewai 已安装）：
python experiments/run_real_crewai.py

# === 使用 CLI ===

# 列出所有可用任务和适配器
python -m benchmark_toolkit.cli.main list

# 运行指定任务
python -m benchmark_toolkit.cli.main run --task mhq-1 --task code-1

# 运行指定适配器
python -m benchmark_toolkit.cli.main run --adapter single-agent --adapter crewai
```

### 4. 查看结果

```bash
# 启动可视化仪表盘
streamlit run benchmark_toolkit/dashboard/app.py
```

### 5. 生成论文图表

```bash
python paper/generate_figures.py
# 图表输出到 paper/figures/
```

---

## 不用 API Key 也能体验

```bash
# 生成模拟数据
python experiments/generate_mock_results.py

# 启动仪表盘，上传 experiments/results/mock_benchmark.json
streamlit run benchmark_toolkit/dashboard/app.py
```

---

## 内置适配器

| 适配器 | 文件 | 依赖 | 说明 |
|--------|------|------|------|
| **Single-Agent** | `single_agent.py` | 无 | 一次 LLM 调用，作为基线 |
| **Multi-Step** | `multi_step_adapter.py` | 无 | 自研三阶段流水线（Research→Analyze→Write） |
| **Role-Based** | `role_based_adapter.py` | 无 | CrewAI 风格：含 backstory、delegation |
| **CrewAI** | `crewai_adapter.py` | crewai | 真实 CrewAI 库（需 Python 3.10+） |
| **AutoGen** | `autogen_adapter.py` | pyautogen | 微软多智能体框架 |
| **LangGraph** | `langgraph_adapter.py` | langgraph | LangChain 图式编排 |

## 内置任务

| ID | 任务 | 类别 | 难度 | 区分度 |
|----|------|------|------|:--:|
| MHQ-1 | AlphaGo 总部所在国的官方语言 | 多跳问答 | 中等 | ⭐高 |
| MHQ-2 | 第一个登月者的出生年份 | 多跳问答 | 中等 | 低（天花板） |
| MHQ-3 | 爱因斯坦同名元素的化学符号 | 多跳问答 | 中等 | 低 |
| CODE-1 | 回文判断函数（含边界条件） | 代码生成 | 中等 | ⭐高 |
| CODE-2 | 斐波那契数列函数 | 代码生成 | 中等 | ⭐高 |
| DATA-1 | 周销售数据分析报告 | 数据分析 | 中等 | 中 |
| RES-1 | Transformer vs Mamba 架构对比 | 研究综述 | 困难 | 低（天花板） |
| RES-2 | 推测解码原理解析 | 研究综述 | 困难 | ⭐高（厂商分裂） |
| PLAN-1 | 500人3天技术会议策划 | 规划类 | 中等 | 低（天花板） |

---

## 采集的指标

| 指标 | 说明 |
|------|------|
| **Score** | 任务成功度 (0.0–1.0，自动化评分) |
| **Total Tokens** | 输入+输出 token 总量 |
| **Prompt Tokens** | 输入上下文消耗的 token |
| **Completion Tokens** | 输出生成的 token |
| **Total Time (ms)** | 端到端耗时 |
| **Agent Interactions** | 智能体间的消息交换次数 |
| **Estimated Cost (USD)** | 基于模型官方定价估算 |
| **Orchestration Overhead** | LLM 调用之外的编排时间 |
| **Efficiency Ratio** | 输出 token / 总 token |

---

## 论文核心发现

12 种配置、108 次 API 调用、跨 DeepSeek 和 OpenAI 两个厂商。

### 结果排名

| 配置 | 得分 | 成功率 | Token | 耗时 | 成本 |
|------|------|--------|-------|------|------|
| **V4 Flash Multi** 🥇 | **0.91** | **100%** | 69,584 | 41.5s | **$0.016** |
| V4 Flash CrewAI | 0.89 | 89% | -- | 44.6s | -- |
| GPT-4o Multi | 0.85 | 100% | 30,496 | 19.7s | $0.184 |
| V4 Pro Multi | 0.85 | 89% | 89,386 | 164.2s | $0.064 |
| V4 Flash Role-Based | 0.84 | 89% | 77,799 | 60.7s | $0.017 |
| GPT-4o Mini Multi | 0.83 | 78% | 37,254 | 28.1s | $0.014 |
| V4 Pro Single | 0.80 | 89% | 13,007 | 27.1s | $0.011 |
| V4 Flash Single | 0.79 | 78% | 13,615 | 11.6s | $0.004 |
| GPT-4o Single | 0.79 | 89% | 5,972 | 5.5s | $0.047 |
| GPT-4o Role-Based | 0.76 | 78% | 35,586 | 20.2s | $0.203 |
| GPT-4o Mini Role-Based | 0.70 | 67% | 36,470 | 25.3s | $0.012 |
| GPT-4o Mini Single | 0.67 | 67% | 6,256 | 8.3s | $0.003 |

### 六条核心结论

1. **协作红利跨厂商成立** — 所有模型从多步协作中获益 (+0.05 ~ +0.16)
2. **越弱的模型获益越大** — GPT-4o Mini +0.16，GPT-4o 仅 +0.06
3. **简单 > 复杂** — 三步流水线 > 角色扮演 > 单Agent，跨 3 个模型成立
4. **便宜模型 + 协作 > 贵模型单打** — V4 Flash Multi ($0.016) 完胜 GPT-4o Multi ($0.184)
5. **协作增强推理，不能创造知识** — RES-2 上 GPT 全系 0.49-0.63，加多少 Agent 都没用
6. **代码任务收益最大** — GPT-4o Mini CODE-1: 0.00 → 0.92

---

## 自定义扩展

### 添加新任务 (~20 行代码)

```python
from benchmark_toolkit.core.task import Task, TaskInput

def my_input():
    return TaskInput(task_id="my-task", description="解释量子计算的基本原理",
                     input_data={"topic": "quantum computing"})

def my_eval(output: str) -> float:
    keywords = ["qubit", "superposition", "entanglement"]
    return sum(1 for k in keywords if k in output.lower()) / len(keywords)

my_task = Task(
    task_id="my-task", name="量子计算解释", description="解释量子计算",
    category="qa", difficulty="hard",
    input_generator=my_input, evaluator=my_eval,
    min_agents_recommended=2,
)
```

### 添加新框架适配器 (~100 行代码)

```python
from benchmark_toolkit.core.adapter import BaseAdapter, AdapterConfig
from benchmark_toolkit.core.task import TaskInput

class MyAdapter(BaseAdapter):
    def solve(self, task_input: TaskInput) -> str:
        # 用 self.llm.chat() 调用你的框架逻辑
        # self.llm 会自动记录所有 token/时间/成本
        result = self.llm.chat(
            messages=[{"role": "user", "content": task_input.description}],
            agent_name="MyAgent", purpose="solving"
        )
        return result

    def agent_names(self) -> list[str]:
        return ["MyAgent"]
```

### 用 JSON 结果做二次分析

```python
from benchmark_toolkit.core.metrics import BenchmarkReport

report = BenchmarkReport.from_json("experiments/results/your_result.json")

# 查某个任务的所有配置得分
for adapter in report.adapters:
    r = report.get_result(adapter, "code-1")
    print(f"{adapter}: {r.score:.2f} ({r.metrics.total_tokens} tokens)")

# 导出为字典做自定义分析
data = report.to_dict()
```

---

## 项目结构

```
hk/
├── benchmark_toolkit/          # 主包
│   ├── core/                   # Task, Adapter, Runner, Metrics
│   ├── adapters/               # 框架适配器（6个）
│   ├── tasks/                  # 任务注册表
│   ├── dashboard/              # Streamlit 可视化
│   └── cli/                    # 命令行工具
├── experiments/                # 实验脚本 & 结果
│   ├── run_experiment.py       # 单Agent基线
│   ├── run_deepseek_full.py    # DeepSeek 多策略对比
│   ├── run_multimodel.py       # 跨厂商多模型对比
│   ├── run_real_crewai.py      # 真实 CrewAI 实验 (Python 3.12)
│   ├── generate_mock_results.py # 模拟数据（免 API Key）
│   └── results/                # JSON 结果文件
├── paper/                      # DAI 2026 论文
│   ├── main.tex                # LaTeX 源码
│   ├── main.pdf                # 编译好的 PDF
│   └── generate_figures.py     # 从 JSON 生成论文图表
├── zh_for_review/              # 中文审阅版（不提交）
│   ├── paper_CN_full.md       # 完整中文翻译
│   ├── paper_CN.md            # 中文摘要版
│   └── README_CN.md           # 中文说明
├── tests/                      # 45 个单元测试
├── requirements.txt
└── pyproject.toml
```

---

## 常见问题

**Q: 我没有 API Key 能用吗？**
可以体验仪表盘。运行 `python experiments/generate_mock_results.py` 生成模拟数据，然后 `streamlit run benchmark_toolkit/dashboard/app.py`。

**Q: 支持哪些 LLM 提供商？**
DeepSeek、OpenAI、Anthropic，以及任何 OpenAI 兼容的中转代理。在 `AdapterConfig` 中设置 `llm_provider` 和 `api_base` 即可。

**Q: CrewAI/AutoGen/LangGraph 适配器真的能用吗？**
CrewAI 已验证可用（需 Python 3.12 + `pip install crewai`）。AutoGen 和 LangGraph 适配器代码已完成但未经过完整实验验证——它们可以运行但可能需要根据最新 API 微调。

**Q: 怎么用自己的任务替换内置任务？**
参考上面"添加新任务"的代码示例。创建 `TaskSuite([your_tasks])` 传给 `BenchmarkRunner` 即可。

**Q: 实验结果在哪里？**
JSON 文件在 `experiments/results/`。论文使用的最终数据在 `final_combined.json`。

---

## 引用

```bibtex
@inproceedings{mab2026,
  title={Multi-Agent Collaboration Benchmark: A Framework-Agnostic
         Toolkit for Comparing Agentic AI Systems},
  author={Junze Li},
  booktitle={Proceedings of the 8th International Conference on
             Distributed Artificial Intelligence (DAI 2026)},
  year={2026},
}
```

## 许可证

MIT License
