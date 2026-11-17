#!/usr/bin/env python
# run_experiment.py
# Dual-LLM 系统有效性实验驱动脚本
#
# 这个脚本把系统当成"教学有效性测量仪":用同一批带 bug 的学生代码跑流水线,
# 量化两个核心指标:
#   1) 缺陷检出率(bug-catch rate):考官 LLM-1 生成的测试,能抓到多少比例的 bug?
#   2) 通过率提升(lift):导师 LLM-2 给建议 + 一个"修复者"把代码改对后,
#      通过率提升多少?(需要真实 DeepSeek Key)
#
# 用法:
#   # 离线基线实验(无需任何 API Key,立刻可跑):
#   python run_experiment.py --mode offline
#
#   # 接入真实 DeepSeek 测量"通过率提升"(需要 dualllmv2/.env 配好两个 Key):
#   python run_experiment.py --mode lift
#
#   # 两个都跑:
#   python run_experiment.py --mode both
#
# 结果写入:
#   test/runs/<时间戳>.json      本次实验原始数据
#   test/metrics/summary.json    汇总指标(可跨次累加对比)

import argparse
import json
import os
import re
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import config
import parsers
import utils
from llm_backends import RuleEngineBackend, get_backend


def _n_tests(summary: str) -> int:
    m = re.search(r"Tests:\s*(\d+)", summary or "")
    return int(m.group(1)) if m else 0


def evaluate_offline(code: str, problem_desc: str) -> dict:
    """用离线规则引擎生成 ISSUES+TESTS,在隔离环境跑测试,返回通过率等指标。"""
    backend = RuleEngineBackend()
    out = backend.llm1_text(code, problem_desc)
    try:
        issues, tests = parsers.parse_llm1_output(out)
    except Exception:
        issues, tests = [], ""
    res = utils.run_unittest(code, tests)
    return {
        "pass_rate": res.pass_rate,
        "passed": res.passed,
        "n_tests": _n_tests(res.summary),
        "n_issues": len(issues),
        "summary": res.summary,
    }


def load_problems(folder: str) -> list:
    problems = []
    for fn in sorted(os.listdir(folder)):
        if fn.endswith(".json"):
            with open(os.path.join(folder, fn), "r", encoding="utf-8") as f:
                problems.append(json.load(f))
    return problems


def run_offline(problems: list) -> dict:
    print("\n===== 离线基线实验:考官(LLM-1)的缺陷检出能力 =====")
    per_problem = []
    caught_flags = []
    for p in problems:
        ref = evaluate_offline(p["reference"], p["problem_desc"])
        print(f"\n[{p['name']}] 参考答案通过率 = {ref['pass_rate']:.0%} "
              f"({ref['n_tests']} 测试用例, 参考解作为对照组)")
        variants = []
        for v in p["buggy"]:
            ev = evaluate_offline(v["code"], p["problem_desc"])
            caught = not ev["passed"]
            caught_flags.append(caught)
            variants.append({
                "label": v["label"],
                "pass_rate": ev["pass_rate"],
                "passed": ev["passed"],
                "caught": caught,
                "n_tests": ev["n_tests"],
            })
            print(f"  - 变体 {v['label']:<16} 通过率={ev['pass_rate']:.0%}  "
                  f"缺陷检出={'是 ✓' if caught else '否 ✗'}")
        catch_rate = sum(1 for x in variants if x["caught"]) / len(variants)
        per_problem.append({
            "problem": p["name"],
            "ref_pass_rate": ref["pass_rate"],
            "n_tests": ref["n_tests"],
            "catch_rate": catch_rate,
            "variants": variants,
        })
    overall = sum(caught_flags) / len(caught_flags) if caught_flags else 0.0
    print(f"\n>>> 总体缺陷检出率 (bug-catch rate) = {overall:.0%} "
          f"({sum(caught_flags)}/{len(caught_flags)} 个 bug 变体被抓到)")
    return {"per_problem": per_problem, "overall_catch_rate": overall}


def fixer_api(buggy_code: str, problem_desc: str, issues_text: str,
              failures_text: str) -> str:
    """用 LLM-2 的客户端,根据导师建议把代码改对。仅用于测量提升,不进仓库。"""
    config.BACKEND_MODE = "api"
    import llm_backends
    llm_backends._backend_instance = None
    backend = get_backend()  # APIBackend
    _, llm2 = config._ensure_llms()
    prompt = (
        "You are a code tutor. Below is a student's buggy solution, the problem, "
        "the tutor's suggestions, and the failing tests.\n"
        "Rewrite the code to FIX the bugs. Return ONLY the corrected Python code, "
        "no explanations, no markdown fences.\n\n"
        f"PROBLEM:\n{problem_desc}\n\nSTUDENT CODE:\n{buggy_code}\n\n"
        f"TUTOR SUGGESTIONS:\n{issues_text}\n\nFAILING TESTS:\n{failures_text}\n"
    )
    from langchain_core.messages import HumanMessage
    resp = llm2.invoke([HumanMessage(content=prompt)]).content
    m = re.search(r"```(?:python)?\n(.*?)```", resp, re.DOTALL)
    return (m.group(1) if m else resp).strip()


def run_lift(problems: list) -> dict:
    print("\n===== 提升实验(接入 DeepSeek):导师建议 -> 代码通过率提升 =====")
    rows = []
    for p in problems:
        for v in p["buggy"]:
            before = evaluate_offline(v["code"], p["problem_desc"])
            try:
                config.BACKEND_MODE = "api"
                import llm_backends
                llm_backends._backend_instance = None
                backend = get_backend()  # APIBackend
                llm1_out = backend.llm1_text(v["code"], p["problem_desc"])
                sug_out = backend.llm2_text(v["code"], p["problem_desc"], llm1_out, "")
                suggestions, failures = parsers.parse_llm2_output(sug_out)
                fixed = fixer_api(v["code"], p["problem_desc"], suggestions, failures)
            except Exception as e:
                print(f"  ! {p['name']}/{v['label']} API 调用失败: {e}")
                continue
            after = evaluate_offline(fixed, p["problem_desc"])
            lift = after["pass_rate"] - before["pass_rate"]
            rows.append({
                "problem": p["name"],
                "variant": v["label"],
                "before": before["pass_rate"],
                "after": after["pass_rate"],
                "lift": lift,
            })
            print(f"  - {p['name']}/{v['label']:<16} "
                  f"通过率 {before['pass_rate']:.0%} -> {after['pass_rate']:.0%}  "
                  f"(提升 {lift:+.0%})")
    mean_lift = sum(r["lift"] for r in rows) / len(rows) if rows else 0.0
    if rows:
        print(f"\n>>> 平均通过率提升 (lift) = {mean_lift:+.0%} "
              f"({len(rows)} 个样本)")
    else:
        print("\n>>> 未获得任何提升样本(可能 API Key 未配置或网络失败)")
    return {"rows": rows, "mean_lift": mean_lift}


def main():
    ap = argparse.ArgumentParser(description="Dual-LLM 系统有效性实验")
    ap.add_argument("--mode", choices=["offline", "lift", "both"], default="offline")
    ap.add_argument("--problems",
                    default=os.path.join(ROOT, "test", "problems"))
    args = ap.parse_args()

    problems = load_problems(args.problems)
    if not problems:
        print(f"未在 {args.problems} 找到题目 JSON,退出。")
        return

    out = {"problems_count": len(problems),
           "generated_at": datetime.now().isoformat(timespec="seconds")}
    if args.mode in ("offline", "both"):
        out["offline"] = run_offline(problems)
    if args.mode in ("lift", "both"):
        out["lift"] = run_lift(problems)

    runs_dir = os.path.join(ROOT, "test", "runs")
    metrics_dir = os.path.join(ROOT, "test", "metrics")
    os.makedirs(runs_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(runs_dir, f"{stamp}.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # 汇总指标
    summary = {
        "generated_at": out["generated_at"],
        "problems_count": len(problems),
    }
    if "offline" in out:
        summary["overall_catch_rate"] = out["offline"]["overall_catch_rate"]
        summary["per_problem_catch_rate"] = {
            pp["problem"]: pp["catch_rate"] for pp in out["offline"]["per_problem"]
        }
    if "lift" in out:
        summary["mean_lift"] = out["lift"]["mean_lift"]
        summary["lift_samples"] = len(out["lift"]["rows"])
    with open(os.path.join(metrics_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n实验原始数据 -> test/runs/{stamp}.json")
    print(f"汇总指标     -> test/metrics/summary.json")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
