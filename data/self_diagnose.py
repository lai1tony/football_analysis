"""自检脚本：跑一遍数据库统计 + 新策略对历史样本的回放，打印诊断报告。

直接执行：
    python data/self_diagnose.py

只读不写，不会改任何数据。安全。
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path

# 所有模块都从 data/ 同级 import（与 app.py 一致）
import collection_repository as repo
import learning_engine as le
import prediction_engine as pe
from feature_engine import safe_float


def section(title: str) -> None:
    print()
    print("=" * 64)
    print(title)
    print("=" * 64)


def kv(label: str, value, width: int = 28) -> None:
    print(f"  {label.ljust(width)} {value}")


def main() -> None:
    db_path: Path = repo.PRIMARY_DB_PATH
    if not db_path.exists():
        print(f"找不到数据库：{db_path}")
        return

    section("[1/6] 数据库现状")
    with closing(sqlite3.connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        for table in ("matches", "analyses", "prediction_runs", "feedback_logs", "feature_snapshots"):
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            kv(f"{table} 行数", count)

        issues = [r[0] for r in conn.execute(
            "SELECT issue, COUNT(*) FROM matches WHERE issue<>'' GROUP BY issue ORDER BY issue DESC"
        ).fetchall()]
        kv("matches 中期号", "、".join(issues) or "-")

        # dedup 检查：feedback_logs 是否每场只剩一条
        dup_rows = conn.execute(
            """
            SELECT match_id, COUNT(*) cnt
            FROM feedback_logs
            GROUP BY match_id
            HAVING cnt > 1
            """
        ).fetchall()
        kv("feedback_logs 重复 match", len(dup_rows))
        if dup_rows:
            print("    ⚠ 仍有重复行（dedup 没生效或 init_db 还没跑过新版本）")

        # feature_snapshots dedup 检查
        snap_dup = conn.execute(
            """
            SELECT match_id, COUNT(*) cnt
            FROM feature_snapshots
            GROUP BY match_id
            HAVING cnt > 1
            """
        ).fetchall()
        kv("feature_snapshots 重复 match", len(snap_dup))

        # 已结算样本数
        settled = conn.execute("SELECT COUNT(*) FROM feedback_logs").fetchone()[0]
        kv("已结算样本（可回测）", settled)

    section("[2/6] 新策略常量自检（确认改造已生效）")
    from action_policy import (
        ACTION_CONTROL,
        DEFAULT_LIGHT_GATE,
        DEFAULT_MAIN_GATE,
        DEFAULT_PROMOTE_GATE,
    )
    from outcome_policy import OUTCOME_CONTROL

    expected = {
        "DEFAULT_MAIN_GATE.ev (期望 0.10)": DEFAULT_MAIN_GATE["ev"],
        "DEFAULT_MAIN_GATE.confidence (期望 0.62)": DEFAULT_MAIN_GATE["confidence"],
        "DEFAULT_LIGHT_GATE.ev (期望 0.04)": DEFAULT_LIGHT_GATE["ev"],
        "ACTION_CONTROL.main_probability_margin (期望 -0.060)": ACTION_CONTROL["main_probability_margin"],
        "ACTION_CONTROL.legacy_gap_quality_floor (期望 0.70)": ACTION_CONTROL["legacy_gap_quality_floor"],
        "OUTCOME_CONTROL.legacy_gap_quality_floor (期望 0.70)": OUTCOME_CONTROL["legacy_gap_quality_floor"],
        "OUTCOME_CONTROL.switch_legacy_gap (应不存在)": OUTCOME_CONTROL.get("switch_legacy_gap", "已删除 ✓"),
    }
    for key, value in expected.items():
        kv(key, value)

    section("[3/6] 当前回测 — 由 summarize_backtest 用新策略代码重算")
    backtest = pe.summarize_backtest()
    if not backtest.get("total_settled"):
        print("  暂无已结算样本，无法回测。")
        return

    kv("已结算样本", backtest["total_settled"])
    kv("最终命中率", f"{backtest['recommendation_hit_rate']*100:.1f}%")
    kv("最终总 ROI", f"{backtest['total_roi']:.4f}")
    kv("正 EV 占比", f"{backtest['positive_ev_share']*100:.1f}%")
    print()
    print("  algorithm 桶（不含 LLM 复核）:")
    for k in ("action_count", "hit_rate", "total_roi", "avg_roi", "avg_stake_pct"):
        v = backtest["algorithm"].get(k, 0)
        if isinstance(v, float):
            kv(f"  {k}", f"{v:.4f}" if k != "hit_rate" else f"{v*100:.1f}%")
        else:
            kv(f"  {k}", v)
    print("  final 桶（含历史 LLM 复核）:")
    for k in ("action_count", "hit_rate", "total_roi", "avg_roi", "avg_stake_pct"):
        v = backtest["final"].get(k, 0)
        if isinstance(v, float):
            kv(f"  {k}", f"{v:.4f}" if k != "hit_rate" else f"{v*100:.1f}%")
        else:
            kv(f"  {k}", v)

    print("  概率模型对比（越低越好）:")
    for label in ("market", "legacy", "independent"):
        b = backtest[label]
        kv(
            f"  {label}",
            f"Brier {b['brier_score']:.4f}  LogLoss {b['log_loss']:.4f}  Hit {b['hit_rate']*100:.1f}%",
        )

    # xG coverage on the latest prediction_runs (D 第 4 步)
    try:
        with closing(sqlite3.connect(db_path)) as conn:
            conn.row_factory = sqlite3.Row
            recent_runs = conn.execute(
                """
                SELECT p.match_id, m.league, m.home_team, m.away_team
                FROM prediction_runs p
                JOIN matches m ON m.match_id = p.match_id
                JOIN feedback_logs f ON f.match_id = p.match_id
                WHERE f.prediction_run_id = p.run_id
                """
            ).fetchall()

        from team_name_aliases import league_codes_for_500_label, resolve_team_aliases
        from source_understat_client import lookup_team_in_league, load_league_xg, UNDERSTAT_LEAGUES

        league_payloads: dict = {}
        coverable_leagues = {
            code for r in recent_runs
            for code in league_codes_for_500_label(str(r["league"] or ""))
        }
        for code in coverable_leagues:
            try:
                league_payloads[code] = load_league_xg(code)
            except Exception:
                league_payloads[code] = {"teams": {}}

        total = 0
        coverable = 0
        matched = 0
        for r in recent_runs:
            total += 1
            codes = league_codes_for_500_label(str(r["league"] or ""))
            if not codes:
                continue
            coverable += 1
            home_a = resolve_team_aliases(str(r["home_team"] or ""))
            away_a = resolve_team_aliases(str(r["away_team"] or ""))
            for c in codes:
                p = league_payloads.get(c) or {}
                if lookup_team_in_league(p, home_a) and lookup_team_in_league(p, away_a):
                    matched += 1
                    break

        print()
        print("  xG 覆盖（D 改造）:")
        kv("  已结算样本数", total)
        kv("  联赛 understat 可覆盖", f"{coverable} ({coverable/max(total,1)*100:.1f}%)")
        kv("  实际匹配两队", f"{matched} ({matched/max(total,1)*100:.1f}%)")
        if matched < coverable * 0.85 and coverable > 0:
            print("    ⚠ 队名匹配率偏低，跑 inspect_xg_match.py 看缺口")
    except Exception as exc:  # noqa: BLE001
        print(f"  xG 覆盖检查失败: {exc}")

    section("[4/6] 假装训练一次 calibrator — 看 independent 是否能向市场靠拢")
    rows = le._sorted_learning_rows(window_issue_count=le.MAX_LEARNING_WINDOW_ISSUE_COUNT)
    train_rows, val_rows = le._split_rows(rows)
    if len(rows) < 30:
        print(f"  训练样本仅 {len(rows)} 条，不足以跑 calibrator（最少 60）")
    else:
        cal_fit = le._fit_calibrator(train_rows)
        cal_val = le._validate_calibrator(val_rows, cal_fit)
        kv("calibrator 训练状态", cal_fit["status"])
        kv("calibrator 训练原因", cal_fit.get("reason") or "-")
        if cal_fit["train_metrics"].get("baseline"):
            base = cal_fit["train_metrics"]["baseline"]
            cand = cal_fit["train_metrics"]["candidate"]
            print(f"  训练段 LogLoss: {base['log_loss']:.4f} -> {cand['log_loss']:.4f}")
            print(f"  训练段 Brier  : {base['brier_score']:.4f} -> {cand['brier_score']:.4f}")
        kv("validation 决策", cal_val["status"])
        if cal_val.get("baseline"):
            print(f"  验证段 LogLoss: {cal_val['baseline']['log_loss']:.4f} -> {cal_val['candidate']['log_loss']:.4f}")
            print(f"  验证段 Brier  : {cal_val['baseline']['brier_score']:.4f} -> {cal_val['candidate']['brier_score']:.4f}")

    section("[5/6] 市场先验 shrinkage 预演 — 不重跑预测，只用历史 final/market 概率推算")
    # 用历史 prediction_runs 里已经存好的 final_*_prob 与 market_*_prob，
    # 模拟新版 blend_predictions（market 作为先验）会得到什么 Brier/Hit。
    from prediction_engine import _market_prior_alpha

    with closing(sqlite3.connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT
                f.actual_result,
                p.final_home_prob, p.final_draw_prob, p.final_away_prob,
                p.market_home_prob, p.market_draw_prob, p.market_away_prob,
                p.quality_score, p.model_agreement
            FROM feedback_logs f
            JOIN prediction_runs p ON p.run_id = f.prediction_run_id
            WHERE p.market_home_prob > 0
              AND p.market_draw_prob > 0
              AND p.market_away_prob > 0
            """
        ).fetchall()

    if not rows:
        print("  缺少 market_*_prob 历史数据，无法预演（旧 run 可能没存这一列）。")
    else:
        import math

        def metrics(probs_list, actuals):
            n = len(probs_list)
            if n == 0:
                return None
            brier = 0.0
            logloss = 0.0
            hits = 0
            for p, actual in zip(probs_list, actuals):
                idx = {"home": 0, "draw": 1, "away": 2}.get(actual, 0)
                truth = [0.0, 0.0, 0.0]
                truth[idx] = 1.0
                brier += sum((p[i] - truth[i]) ** 2 for i in range(3)) / 3.0
                logloss += -math.log(max(p[idx], 1e-6))
                if [p[0], p[1], p[2]].index(max(p)) == idx:
                    hits += 1
            return {
                "brier": brier / n,
                "logloss": logloss / n,
                "hit": hits / n,
                "n": n,
            }

        old_probs, new_probs, market_probs_list, actuals = [], [], [], []
        alpha_values = []
        for r in rows:
            old = (
                safe_float(r["final_home_prob"]),
                safe_float(r["final_draw_prob"]),
                safe_float(r["final_away_prob"]),
            )
            mkt = (
                safe_float(r["market_home_prob"]),
                safe_float(r["market_draw_prob"]),
                safe_float(r["market_away_prob"]),
            )
            alpha = _market_prior_alpha(
                safe_float(r["quality_score"]),
                safe_float(r["model_agreement"]),
            )
            blended = tuple(alpha * old[i] + (1 - alpha) * mkt[i] for i in range(3))
            total = sum(blended)
            blended = tuple(x / total for x in blended) if total > 0 else mkt

            old_probs.append(old)
            new_probs.append(blended)
            market_probs_list.append(mkt)
            actuals.append(str(r["actual_result"] or ""))
            alpha_values.append(alpha)

        old_m = metrics(old_probs, actuals)
        new_m = metrics(new_probs, actuals)
        mkt_m = metrics(market_probs_list, actuals)
        avg_alpha = sum(alpha_values) / len(alpha_values)

        kv("可用样本", old_m["n"])
        kv("平均 alpha (越大越偏模型)", f"{avg_alpha:.3f}")
        print()
        print(f"                   Brier     LogLoss   Hit")
        print(f"  旧 final     :  {old_m['brier']:.4f}   {old_m['logloss']:.4f}   {old_m['hit']*100:.1f}%")
        print(f"  新 final     :  {new_m['brier']:.4f}   {new_m['logloss']:.4f}   {new_m['hit']*100:.1f}%")
        print(f"  市场基准     :  {mkt_m['brier']:.4f}   {mkt_m['logloss']:.4f}   {mkt_m['hit']*100:.1f}%")

        if new_m["brier"] < old_m["brier"]:
            improvement = (old_m["brier"] - new_m["brier"]) / old_m["brier"] * 100
            print(f"\n  ✓ 新 blend Brier 改善 {improvement:.1f}%")
        else:
            degradation = (new_m["brier"] - old_m["brier"]) / old_m["brier"] * 100
            print(f"\n  ⚠ 新 blend Brier 反而恶化 {degradation:.1f}% — 需要再调 alpha 范围")

        if new_m["hit"] > old_m["hit"]:
            print(f"  ✓ 新 blend 命中率从 {old_m['hit']*100:.1f}% → {new_m['hit']*100:.1f}%")

    section("[6/6] 核心诊断 — 一句话结论")
    indep_brier = backtest["independent"]["brier_score"]
    market_brier = backtest["market"]["brier_score"]
    indep_hit = backtest["independent"]["hit_rate"]
    market_hit = backtest["market"]["hit_rate"]
    algo_roi = backtest["algorithm"]["total_roi"]
    algo_actions = backtest["algorithm"]["action_count"]

    diagnosis = []

    if indep_brier > market_brier:
        diagnosis.append(
            f"❌ independent Brier ({indep_brier:.4f}) 比 market ({market_brier:.4f}) 高 "
            f"{(indep_brier-market_brier)/market_brier*100:.1f}%。"
            "概率模型本身没有 alpha — 即使策略层放开也救不了。"
        )
    else:
        diagnosis.append(f"✓ independent 概率模型在 Brier 上击败市场。")

    if indep_hit < market_hit - 0.03:
        diagnosis.append(
            f"❌ independent 命中率 ({indep_hit*100:.1f}%) 比 market ({market_hit*100:.1f}%) 低 "
            f"{(market_hit-indep_hit)*100:.1f}pp。"
        )

    if algo_actions == 0:
        diagnosis.append("❌ algorithm 桶动作数为 0：策略层即便放开也找不到信号。")
    elif algo_roi < 0:
        diagnosis.append(
            f"❌ algorithm 桶 ROI 为 {algo_roi:.4f}（负 EV）。"
            "新策略已正常生效（动作数活跃），但下游概率不准导致下注亏损。"
        )
    else:
        diagnosis.append(f"✓ algorithm 桶 ROI 为 {algo_roi:.4f}，策略层挑出了正 EV 信号。")

    main_count = sum(
        1 for k in ("algorithm", "final")
        if backtest[k].get("avg_stake_pct", 0) > 1.0
    )
    if main_count > 0:
        diagnosis.append(f"✓ 仓位上限放开生效（avg_stake > 1%），不再被压扁到 1% 以下。")

    print()
    for line in diagnosis:
        print(f"  {line}")

    print()
    print("  下一步建议：")
    if indep_brier > market_brier:
        print("    1. 不要急着用历史数据 retrain 阈值。先解决 independent 模型 alpha 缺失：")
        print("       - 启用 calibrator（学习面板点训练 → 启用），把 independent 向市场拉近")
        print("       - 或临时把策略权重转向 legacy/market（legacy 命中率比 independent 还高）")
        print("    2. 中长期补 xG、近期对手强度加权、closing line 跟踪")
    else:
        print("    1. 模型有 alpha 但 ROI 仍负 → 调高 main_gate.ev 抬到 0.14，挑更窄的边际")
        print("    2. 加 hours_to_kickoff 特征（赛前 vs 早盘 alpha 差异大）")
    print()


if __name__ == "__main__":
    main()
