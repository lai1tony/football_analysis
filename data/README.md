# data 模块说明

`data/` 是当前仓库唯一真实可运行的应用主体，负责足球比赛数据同步、采集、预测、学习闭环、回测、赛后反馈和 Flask 检阅页面。

## 模块边界

| 文件 | 职责 |
| --- | --- |
| `app.py` | Flask Web 入口，页面渲染、表单动作、后台任务和进度轮询 |
| `source_500_client.py` | 500.com / odds.500.com 页面抓取、重试和 `gb18030` 解码 |
| `collection_strategy.py` | 统一采集策略，按字段记录 `主源/已有数据 -> playwright-cli -> anysearch` 的来源和质量 |
| `collection_service.py` | 采集编排、HTML 解析、结构化字段生成 |
| `collection_repository.py` | SQLite schema、读写、统计、学习配置、TOP3 持久化 |
| `source_market_value_client.py` | 球队球员身价采集，Transfermarkt/Playwright 优先，AnySearch 抽取 Transfermarkt 页面兜底 |
| `source_supplement_client.py` | Flashscore、Soccerway、Understat 等公开源补采入口 |
| `source_lineup_client.py` | 伤停和预计阵容补采，Flashscore 优先，RotoWire 作为缺失时的替补源 |
| `source_understat_client.py` | Understat 数据抓取和缓存 |
| `feature_engine.py` | 特征快照构建、概率和数值辅助函数 |
| `outcome_policy.py` | 三向赛果概率、市场概率、价值评分 |
| `action_policy.py` | 动作门禁、EV、仓位、风险动作 |
| `learning_engine.py` | 学习画像、概率校准、让球盘目标策略搜索、当前策略回放和启用门槛 |
| `replay_backfill.py` | 历史学习闭环回放补数，同步更早期号、逐场超时采集、预测、结算并标记反馈来源 |
| `prediction_engine.py` | 预测主流程、LLM 复核、二级仲裁、报告、回测摘要 |
| `run_prediction.py` | 预测、反馈、摘要、回测 CLI |
| `run_full_pipeline.py` | 初始化、同步、批量采集 CLI |
| `collector_store.py` | 兼容旧导入的 facade，新代码不应继续扩展它 |

## 数据流

```text
500.com 当前期列表
  -> sync_matches()
  -> matches
  -> collect_match()
  -> collection_strategy.apply_unified_collection_strategy()
  -> analyses
  -> predict_match()
  -> feature_snapshots + prediction_runs
  -> apply_target_batch_strategy_to_issue()
  -> record_feedback()
  -> feedback_logs
  -> replay_backfill_learning_feedback()  # 可选：补齐更早期历史闭环反馈
  -> feedback_logs[roi_source='replay_backfill']
  -> train_learning_profile()
  -> learning_profiles
  -> compute_issue_top_picks()
  -> issue_top_picks
```

## 核心数据表

- `matches`: 对阵、期号、联赛、比赛时间、列表页赔率和热度。
- `analyses`: 采集后的近期状态、交锋、赔率、热度、阵容、球队身价、补采摘要和 `collection_quality_summary`。
- `feature_snapshots`: 每次预测使用的赛前特征快照。
- `prediction_runs`: 每次预测 run 的概率、EV、动作、LLM 复核、仲裁、目标批量策略执行动作和报告。
- `feedback_logs`: 赛果、命中、ROI 和备注。
- `learning_profiles`: 学习候选、已启用配置、策略诊断和目标门槛。
- `issue_top_picks`: 每期 TOP3 精选，按每场最新 run 重新计算；页面会结合 `matches` 和 `feedback_logs` 显示命中/错误/待结算状态。

## 预测与动作策略

预测结果是三向赛果: `home`、`draw`、`away`。
基础实力维度包含 `market_value_summary`。当主客队双方身价都可解析时，`feature_engine` 会生成身价差和身价评分差，并交给量化、ML 和 legacy 市场模型使用；只有单方身价或失败备注不会强行影响模型。

推荐动作是执行强度:

- `主推`: 执行动作，优先级最高。
- `轻仓`: 执行动作，保守参与。
- `观望`: 不计入执行动作。

`coverage_draw_rescue` 是当前目标批量生产层。`predict_match()` 保存单场 run 后会对同期期数未结算的 latest runs 应用该策略；`predict_issue()` 整期完成后会再统一应用一次，保证最终执行动作按完整期数重算。策略会写回 `recommendation`、`recommended_outcome`、`suggested_stake_pct`、`effective_recommendation`、`effective_stake_pct`，并把 `effective_action_source` 标记为 `target_batch_strategy`。

策略计算读取原始 `algo_recommendation`、`algo_recommended_outcome`、`algo_suggested_stake_pct` 作为单场模型底稿，不读取自己上次写回的最终动作，避免重复应用时自我反馈。已有 `feedback_logs` 的 run 会被 `apply_target_batch_strategy_to_issue()` 跳过；比赛开赛后再生成的新预测由 `save_prediction_run()` 追加保存，不替换赛前 canonical run，也不搬运旧反馈。页面和 TOP3 优先使用 `effective_recommendation`，为空时才回退到原始 `recommendation`。TOP3 会优先选 `主推/轻仓`，可执行不足 3 场时才用 `观望` 补齐。

Web 批量采集、预测和赛果结算表单支持 `selected_match_ids`。有勾选时，`collect_all_matches()`、`predict_issue()`、`settle_issue_results()` 和目标批量策略只处理这些 match_id；未勾选时才按当前期全量处理。维护这条链路时要避免选中预测改写未勾选比赛的 latest run。

`suggested_stake_pct` 是胜平负主推荐链路的最终仓位字段，会被 `coverage_draw_rescue` 写回。让球盘目前没有单独持久化仓位列；Flask 页面在 `_decorate_prediction_run()` 中基于让球推荐侧、覆盖率、赔率、EV、信心和 `action_policy.stake_for_action()` 临时派生 `handicap_suggested_stake_pct` 供展示。两者都是分数 Kelly 口径，不是 full Kelly。

当前期赛果同步结算完成后，`settle_issue_results()` 会重算当期 TOP3。首页 TOP3 卡片会对已结算场次显示命中/错误、实际赛果和比分；顶部命中率只使用已结算 TOP3 作为分母，待结算场次不参与统计。

## 学习闭环现状

- 点击“训练学习候选”会基于学习窗口内的已结算 canonical runs 进行离线搜索。
- 点击“回放补齐历史期”会补入学习窗口仍缺少的更早期号，默认目标是凑够 90 期完整已结算闭环样本。
- 回放补数不是导入凭空记录；它会重新同步历史期对赛，用历史赛前可得字段重跑预测，再同步赛果并把反馈标记为 `roi_source='replay_backfill'`。
- 每场采集在子进程内执行并有超时保护；重跑会跳过已有采集/反馈，支持断点续跑。若某期只生成部分反馈，该期进入 `partial_issues`，不应当成完整学习闭环样本。
- 当前训练目标针对让球盘推荐，候选必须满足命中率 `>= 70%`、执行占比 `>= 60%`，才允许“启用候选”。
- 当前 active profile 为 #45，策略类型 `handicap_bucket_table`。它根据让球盘、两侧覆盖率差、EV 差和客队水位分桶，命中桶表时输出让球侧和轻仓动作，未命中时降为让球观望。
- 当前全量历史回放使用真实 `handicap_actual_result` 评估 #45：1232 条样本、964 个动作、执行占比 78.25%、执行命中率 74.27%。
- 首页“历史反馈”统计的是 `feedback_logs` 中已保存 prediction runs 的真实累计反馈，可能混合多个 profile；“当前策略回放”才是 active profile #45 对同一历史样本的即时回放口径。
- 未来预测仍会生成基线概率与原始 `algo_*` 底稿；#45 只改写让球盘 `handicap_risk`，胜平负最终执行动作继续由 `coverage_draw_rescue` 目标批量生产层落库。
- 截至 2026-06-15，本地 SQLite 内 `26059` 到 `26085` 共 27 期、378 条已采集样本的核心采集维度已补齐，`market_value_summary` 覆盖 378/378。
- 截至 2026-06-15，`coverage_draw_rescue` 在 374 条已结算样本上给出 224 个动作，执行占比 59.89%，命中率 70.54%，滚动 10 期最低命中率 70.11%；当前已落库 final action 口径为 228 个动作、69.74% 命中率。

## 运行入口

从仓库根目录运行:

```powershell
python data/app.py
python data/run_full_pipeline.py
python data/run_prediction.py backtest --json
cd data; python replay_backfill.py
python data/check_runtime.py
```

不要直接改成 `python -m data.app`，当前脚本依赖本地绝对导入。

## 测试

当前可直接运行的回归测试主要在 `data/test_*.py`:

```powershell
python -m unittest data.test_action_policy data.test_outcome_policy data.test_learning_engine data.test_feedback_backtest
cd data
python -m unittest test_collection_quality
```

如果本机虚拟环境损坏，可以使用项目运行时 Python，并把 venv site-packages 放入 `PYTHONPATH`。

## 常见排障

- 页面打不开但 5050 端口有监听: 检查是否有多个 `data/app.py` 进程，重复服务会导致 SQLite `database is locked`。
- 首页请求超时: 不应在首页路径触发 `fit_target_strategy()` 这类离线搜索；训练搜索只应由学习训练动作触发。
- 回放补数卡住: 优先检查单场采集是否超时；`replay_backfill.py` 会对子进程设超时并保留已采集 rows，修复后可直接重跑续接。
- 采集失败: 先跑 `list_current_matches.py` 或 `collect_current_first_match.py` 验证上游页面结构。
- 500.com 编码异常: `source_500_client.fetch_html()` 默认按 `gb18030` 解码并重试，除非确认 bug，否则不要随意改。
