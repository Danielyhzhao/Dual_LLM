# test/ — 实验数据文件夹

本文件夹专门用于**量化评估 Dual-LLM 教学系统的有效性**。所有实验输入（题目、
参考答案、带 bug 的学生代码）和实验脚本都放在这里，与系统源码（`config.py`、
`pipeline.py` 等）分离，方便复现和对比。

## 目录结构

```
test/
├── README.md              # 本说明
├── run_experiment.py      # 实验驱动脚本(可离线 / 可接 DeepSeek)
├── problems/              # 实验输入:每题一个 JSON
│   ├── two_sum.json
│   ├── is_palindrome.json
│   ├── is_anagram.json
│   └── longest_common_subsequence.json
├── runs/                  # 每次实验的原始输出(按时间戳, .gitkeep 占位)
└── metrics/               # 汇总指标 summary.json (.gitkeep 占位)
```

## 两个核心指标

### 1. 缺陷检出率 (Bug-catch rate) — 离线即可测
把一批**带 bug 的学生代码**交给系统，看考官 LLM-1 生成的测试用例能否让 bug 暴露
（即测试不通过）。

- 对每个题目：先跑**参考答案**作为对照组，确认生成的测试本身是有效的
  （参考答案应通过率 ≈ 100%）。
- 再跑每个 bug 变体：若生成的测试**不通过** → 该 bug 被"抓到"。
- `缺陷检出率 = 被抓到的 bug 变体数 / 总 bug 变体数`

> 这个指标衡量的是"考官出题准不准"，是系统有效性的第一道关口。

### 2. 通过率提升 (Lift) — 需要真实 DeepSeek Key
用导师 LLM-2 的建议 + 一个"修复者"把 bug 代码改对，对比修复前后的通过率。

- `before` = 原始 bug 代码在系统生成的测试上的通过率（离线测量，确定性）
- `after`  = 修复后代码在同一套测试上的通过率（离线测量，确定性）
- `lift = after - before`，跨样本取平均

> 这个指标衡量的是"导师建议是否真的帮学生把代码改对了"，是系统教学价值的直接证据。

## 怎么跑

```bash
# 进入项目根
cd dualllmv2

# (1) 离线基线实验 —— 不需要任何 API Key，立刻能跑
python test/run_experiment.py --mode offline

# (2) 接入真实 DeepSeek 测"通过率提升" —— 需要 .env 配好 LLM1/LLM2 Key
cp .env.example .env      # 然后填入你的 Key
python test/run_experiment.py --mode lift

# (3) 两个都跑
python test/run_experiment.py --mode both
```

运行结果：
- `test/runs/<时间戳>.json` —— 本次实验的原始数据
- `test/metrics/summary.json` —— 汇总指标，可跨多次运行对比

## 如何扩展实验

- **加题目**：在 `problems/` 新增一个 JSON，字段为
  `name` / `problem_desc` / `reference`(正确代码) / `buggy`(数组, 每项 `label`+`code`)。
  若题目类型已在 `llm_backends._ORACLES` 注册（two sum、palindrome、anagram、
  LCS、reverse、climb stairs 等），生成的测试会带"标准答案"断言，信号最强。
- **加对照组（单 LLM 基线）**：可新增一个只用 LLM-1 建议、跳过 LLM-2 对抗的
  流水线模式，对比"双 LLM 对抗" vs "单 LLM"的提升差异——这是下一步最有价值的实验。

## 安全说明
本文件夹不含任何 API Key。`runs/` 与 `metrics/` 下的生成文件已在根目录
`.gitignore` 中忽略，不会上传到 GitHub。
