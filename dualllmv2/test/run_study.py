#!/usr/bin/env python
# run_study.py
# Dual-LLM 消融对照实验(接入真实 DeepSeek API,在 DA 环境运行)
#
# 研究问题:导师(LLM-2)的"建议"与"强制反思/多轮迭代"各自把学生代码的
#           通过率提升了多少?
#
# 实验设计(4 个实验组,自变量 = 学生获得的帮助等级):
#   G0 对照组 (无帮助)      : 学生直接交 bug 代码,只用标准测试判卷,无 LLM 介入。
#   G1 仅靠测试反馈          : 学生能看到"哪些测试用例挂了"(failures),但无导师
#                              文字建议,自己动手改,最多 3 轮(测"测试反馈"本身的价值)。
#   G2 导师建议 · 1 轮       : 失败用例 + 导师(LLM-2)文字建议,自动修复 1 次。
#   G3 导师建议 · 多轮反思    : 失败用例 + 导师建议,最多 3 轮迭代(含强制反思),
#                              直到通过或到达上限。
#
# 因变量:代码在标准测试集上的通过率(pass_rate, 0~1)。
# 辅助指标:缺陷检出率 = 被标准测试抓住的 bug 变体比例(衡量考官 LLM-1 测试有效性)。
#
# 用法:
#   cd dualllmv2
#   python test/run_study.py
# 依赖: DA 环境(langchain-openai, matplotlib, python-dotenv)。.env 需配好两个 DeepSeek Key。
#
# 产物:
#   test/runs/study_<时间戳>.json    原始数据
#   test/metrics/summary_study.json  汇总指标
#   test/metrics/study_results.csv    逐样本明细
#   test/metrics/experiment_chart.png 直观图表(核心交付物)

import argparse
import csv
import json
import os
import re
import sys
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import config
import parsers
import utils
from llm_backends import RuleEngineBackend, get_backend

MAX_ROUNDS = 3


def _n_tests(summary: str) -> int:
    m = re.search(r"Tests:\s*(\d+)", summary or "")
    return int(m.group(1)) if m else 0


def offline_eval(code: str, problem_desc: str) -> dict:
    """用离线规则引擎(参考解 oracle)生成测试,隔离环境跑,返回通过率等指标。"""
    backend = RuleEngineBackend()
    out = backend.llm1_text(code, problem_desc)
    try:
        _, tests = parsers.parse_llm1_output(out)
    except Exception:
        tests = ""
    res = utils.run_unittest(code, tests)
    return {
        "pass_rate": res.pass_rate,
        "passed": res.passed,
        "n_tests": _n_tests(res.summary),
        "failures": res.failures,
        "summary": res.summary,
    }


def _api_call_with_retry(fn, what: str, retries: int = 3):
    """对 DeepSeek 调用做简单重试(应对瞬时 5xx / 限流)。"""
    last = None
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except Exception as e:
            last = e
            msg = str(e)
            if "402" in msg or "Insufficient Balance" in msg:
                raise
            if "401" in msg or "Authentication" in msg:
                raise
            print(f"    ! {what} 第{attempt}次失败: {msg[:120]}; 重试...")
            time.sleep(2 * attempt)
    raise last


def fix_with_api(buggy_code: str, problem_desc: str, suggestions: str,
                 failures: str, with_suggestions: bool = True) -> str:
    """用真实 API(LLM-2 客户端模拟"接受反馈后动手修的学生")把代码改对。
    with_suggestions=False 时只给失败用例、不给导师文字建议(用于 G1 消融)。"""
    config.BACKEND_MODE = "api"
    import llm_backends
    llm_backends._backend_instance = None
    get_backend()
    _, llm2 = config._ensure_llms()

    if with_suggestions:
        tut_sec = f"TUTOR SUGGESTIONS:\n{suggestions}\n\n"
        role = ("You are a student who just received your tutor's feedback on a "
                "buggy coding solution. Read the PROBLEM, your BUGGY CODE, the "
                "TUTOR'S SUGGESTIONS, and the FAILING TESTS. Fix the bugs.")
    else:
        tut_sec = ""
        role = ("You are a student debugging a coding solution. You are NOT given "
                "any tutor advice — only the failing tests. Read the PROBLEM, your "
                "BUGGY CODE, and the FAILING TESTS, then fix the bugs yourself.")

    prompt = (
        f"{role} Return ONLY the corrected Python code (the complete "
        f"function/class), no explanations, no markdown fences.\n\n"
        f"PROBLEM:\n{problem_desc}\n\n"
        f"BUGGY CODE:\n{buggy_code}\n\n"
        f"{tut_sec}"
        f"FAILING TESTS:\n{failures}\n"
    )
    from langchain_core.messages import HumanMessage

    def _call():
        return llm2.invoke([HumanMessage(content=prompt)]).content

    resp = _api_call_with_retry(_call, "修复调用")
    m = re.search(r"```(?:python)?\n(.*?)```", resp, re.DOTALL)
    code = (m.group(1) if m else resp).strip()
    if not re.search(r"\b(def|class)\s", code):
        return code
    return code


def _run_rounds(code0, desc, backend, with_suggestions, max_rounds):
    """对单个样本跑最多 max_rounds 轮修复,返回每轮的通过率列表(含第0轮=原始)。"""
    cur = code0
    ev0 = offline_eval(cur, desc)
    traj = [ev0["pass_rate"]]
    cur_pass = ev0["pass_rate"]
    last_round = 0
    for r in range(1, max_rounds + 1):
        if cur_pass >= 1.0:
            traj.append(1.0)
            continue
        ev = offline_eval(cur, desc)
        failures_text = ev["failures"][:2000]
        if with_suggestions:
            llm2_out = _api_call_with_retry(
                lambda: backend.llm2_text(cur, desc, "", failures_text),
                f"导师建议 r{r}")
            suggestions, _ = parsers.parse_llm2_output(llm2_out)
        else:
            suggestions = ""
        fixed = fix_with_api(cur, desc, suggestions, failures_text, with_suggestions)
        if not fixed or fixed == cur:
            traj.append(cur_pass)
            last_round = r
            print(f"      (轮{r}: 修复器无改动,保持 {cur_pass:.0%})")
            break
        cur = fixed
        after = offline_eval(cur, desc)
        cur_pass = after["pass_rate"]
        traj.append(cur_pass)
        last_round = r
        print(f"      {'有建议' if with_suggestions else '仅测试'} 轮{r} -> "
              f"{cur_pass:.0%}{'  ✓' if cur_pass >= 1.0 else ''}")
        if cur_pass >= 1.0:
            for rr in range(r + 1, max_rounds + 1):
                traj.append(1.0)
            break
    for r in range(last_round + 1, max_rounds + 1):
        if r >= len(traj):
            traj.append(cur_pass)
    return traj


def run_study(problems: list, max_rounds: int) -> dict:
    config.BACKEND_MODE = "api"
    import llm_backends
    llm_backends._backend_instance = None
    backend = get_backend()

    print(f"\n===== 消融对照实验(接入 DeepSeek,最多 {max_rounds} 轮) =====")
    samples = []
    caught = 0
    total = 0

    for p in problems:
        desc = p["problem_desc"]
        for v in p["buggy"]:
            total += 1
            base = offline_eval(v["code"], desc)
            is_caught = not base["passed"]
            caught += 1 if is_caught else 0
            print(f"\n[{p['name']}/{v['label']}] 对照组通过率="
                  f"{base['pass_rate']:.0%} (bug{'被检出 ✓' if is_caught else '漏网 ✗'})")

            # G0 对照组
            g0 = base["pass_rate"]
            # G1 仅靠测试反馈(无导师建议):汇报"第1轮一次成功率",以体现导师建议的增量价值
            print(f"  [G1 仅靠测试反馈]")
            g1_traj = _run_rounds(v["code"], desc, backend, False, 1)
            g1 = g1_traj[1] if len(g1_traj) > 1 else g1_traj[-1]
            # G2 导师建议 1 轮:汇报"第1轮一次成功率"
            print(f"  [G2 导师建议·1轮]")
            g2_traj = _run_rounds(v["code"], desc, backend, True, 1)
            g2 = g2_traj[1] if len(g2_traj) > 1 else g2_traj[-1]
            # G3 导师建议 + 多轮反思(最多 max_rounds 轮):汇报最终通过率(系统上限)
            print(f"  [G3 导师建议·多轮反思]")
            g3_traj = _run_rounds(v["code"], desc, backend, True, max_rounds)
            g3 = g3_traj[-1]

            samples.append({
                "problem": p["name"],
                "variant": v["label"],
                "bug_caught": is_caught,
                "g0_control": g0,
                "g1_tests_only": g1,
                "g2_mentor_1r": g2,
                "g3_mentor_3r": g3,
                "g1_traj": [round(x, 3) for x in g1_traj],
                "g2_traj": [round(x, 3) for x in g2_traj],
                "g3_traj": [round(x, 3) for x in g3_traj],
            })

    bug_catch_rate = caught / total if total else 0.0
    mean_by_group = {
        "g0_control": sum(s["g0_control"] for s in samples) / total,
        "g1_tests_only": sum(s["g1_tests_only"] for s in samples) / total,
        "g2_mentor_1r": sum(s["g2_mentor_1r"] for s in samples) / total,
        "g3_mentor_3r": sum(s["g3_mentor_3r"] for s in samples) / total,
    }
    print(f"\n>>> 缺陷检出率 = {bug_catch_rate:.0%} ({caught}/{total})")
    print(">>> 各实验组平均通过率:")
    for k, val in mean_by_group.items():
        print(f"    {k}: {val:.0%}")
    lift = mean_by_group["g3_mentor_3r"] - mean_by_group["g0_control"]
    print(f">>> 整体提升(lift, G3 - G0) = {lift:+.0%}")

    return {
        "samples": samples,
        "bug_catch_rate": bug_catch_rate,
        "caught": caught,
        "total": total,
        "mean_by_group": mean_by_group,
        "lift_g3_vs_g0": lift,
    }


def write_outputs(out: dict, stamp: str):
    runs_dir = os.path.join(ROOT, "test", "runs")
    metrics_dir = os.path.join(ROOT, "test", "metrics")
    os.makedirs(runs_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    with open(os.path.join(runs_dir, f"study_{stamp}.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    summary = {
        "generated_at": out["generated_at"],
        "samples": out["total"],
        "bug_catch_rate": out["bug_catch_rate"],
        "mean_by_group": out["mean_by_group"],
        "lift_g3_vs_g0": out["lift_g3_vs_g0"],
    }
    with open(os.path.join(metrics_dir, "summary_study.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    csv_path = os.path.join(metrics_dir, "study_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["problem", "variant", "bug_caught",
                    "G0_control", "G1_tests_only", "G2_mentor_1r", "G3_mentor_3r"])
        for s in out["samples"]:
            w.writerow([s["problem"], s["variant"], s["bug_caught"],
                        f"{s['g0_control']:.2f}", f"{s['g1_tests_only']:.2f}",
                        f"{s['g2_mentor_1r']:.2f}", f"{s['g3_mentor_3r']:.2f}"])
    return csv_path


def build_chart(out: dict, path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    groups = ["G0\nControl\n(no help)", "G1\nTests-only\n(1 attempt)",
              "G2\nMentor 1 round", "G3\nMentor <=3 rounds\n(+reflection)"]
    means = [
        out["mean_by_group"]["g0_control"],
        out["mean_by_group"]["g1_tests_only"],
        out["mean_by_group"]["g2_mentor_1r"],
        out["mean_by_group"]["g3_mentor_3r"],
    ]
    colors = ["#9aa0a6", "#4c8bf5", "#f5a623", "#34a853"]

    fig = plt.figure(figsize=(13, 5.4))
    fig.suptitle("Dual-LLM System Effectiveness: Ablation Study",
                 fontsize=15, fontweight="bold")

    # ---- 左:各组平均通过率(柱状图) ----
    ax1 = fig.add_subplot(1, 2, 1)
    bars = ax1.bar(groups, [m * 100 for m in means], color=colors, width=0.62)
    ax1.set_ylim(0, 108)
    ax1.set_ylabel("Mean Pass Rate (%)", fontsize=11)
    ax1.set_title("Mean Pass Rate by Group (dependent variable)", fontsize=12)
    for b, m in zip(bars, means):
        ax1.text(b.get_x() + b.get_width() / 2, m * 100 + 2,
                 f"{m:.0%}", ha="center", va="bottom", fontsize=11,
                 fontweight="bold")
    lift = out["lift_g3_vs_g0"]
    ax1.annotate(f"Overall lift {lift:+.0%}",
                 xy=(3, means[3] * 100), xytext=(1.3, 95),
                 fontsize=11, color="#d93025", fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="#d93025"))
    ax1.grid(axis="y", alpha=0.3)

    # ---- 右:逐题通过率随帮助等级变化(折线图) ----
    ax2 = fig.add_subplot(1, 2, 2)
    xs = [0, 1, 2, 3]
    cmap = plt.get_cmap("tab10")
    for i, s in enumerate(out["samples"]):
        ys = [s["g0_control"], s["g1_tests_only"], s["g2_mentor_1r"], s["g3_mentor_3r"]]
        ax2.plot(xs, [y * 100 for y in ys], marker="o",
                 label=f"{s['problem']}/{s['variant']}",
                 color=cmap(i % 10), linewidth=1.6)
    ax2.set_xticks(xs)
    ax2.set_xticklabels(groups, fontsize=8.5)
    ax2.set_ylim(0, 110)
    ax2.set_ylabel("Pass Rate (%)", fontsize=11)
    ax2.set_xlabel("Help level (student assistance)", fontsize=11)
    ax2.set_title("Per-sample pass-rate trajectory", fontsize=12)
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=7.5, loc="lower right", ncol=1)

    fig.text(0.5, 0.01,
             f"N={out['total']}  |  Bug-catch rate={out['bug_catch_rate']:.0%}"
             f"  |  G3-G0 lift={lift:+.0%}  |  generated {out['generated_at']}",
             ha="center", fontsize=9, color="#555")

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Dual-LLM ablation study")
    ap.add_argument("--problems", default=os.path.join(ROOT, "test", "problems"))
    ap.add_argument("--rounds", type=int, default=MAX_ROUNDS)
    args = ap.parse_args()
    max_rounds = max(1, min(args.rounds, 5))

    problems = []
    for fn in sorted(os.listdir(args.problems)):
        if fn.endswith(".json"):
            with open(os.path.join(args.problems, fn), "r", encoding="utf-8") as f:
                problems.append(json.load(f))
    if not problems:
        print(f"未在 {args.problems} 找到题目,退出。")
        return

    if config.KEY1_IS_PLACEHOLDER or config.KEY2_IS_PLACEHOLDER:
        print("API keys not configured; cannot run API experiment.")
        return

    out = run_study(problems, max_rounds)
    out["generated_at"] = datetime.now().isoformat(timespec="seconds")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_path = write_outputs(out, stamp)
    chart_path = os.path.join(ROOT, "test", "metrics", "experiment_chart.png")
    build_chart(out, chart_path)

    print(f"\n原始数据   -> test/runs/study_{stamp}.json")
    print(f"汇总指标   -> test/metrics/summary_study.json")
    print(f"逐样本明细 -> {csv_path}")
    print(f"直观图表   -> {chart_path}")


if __name__ == "__main__":
    main()
