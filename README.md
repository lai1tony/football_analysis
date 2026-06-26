# 足球分析与预测原型

当前仓库的真实可运行系统是 `data/` 下的 Python + Flask + SQLite 应用。根目录中曾经出现过面向更大前端系统或旧 TypeScript 实现的说明，但当前 checkout 不包含可运行的 Next.js/TypeScript 应用，维护时以本文件和 `data/README.md` 为准。

## 当前状态

- Web UI: `python data/app.py`，默认访问 `http://127.0.0.1:5050/`。
- 数据库: `data/football_data.db`。
- Public GitHub 仓库只包含源码、文档、脚本和 `.env.example` 占位模板；真实 `.env`、SQLite 数据库、日志、虚拟环境、构建输出和安装包只保留在本地或私有分发包中。
- 数据来源: 500.com 胜负彩页面及 odds.500.com 详情页；缺失字段通过统一策略按 `主源/已有数据 -> playwright-cli -> anysearch` 补采，公开辅助源包括 Flashscore、RotoWire、Understat、Soccerway 和 Transfermarkt。
- 基础实力新增球队球员身价: `market_value_summary` 记录主客队总身价和差值，预测特征会把双方身价差纳入实力评估。身价源以 Transfermarkt 为准，Playwright 未命中时由 AnySearch 抽取 Transfermarkt 页面兜底。
- 采集质量记录: `analyses.collection_quality_summary` 保存每个字段的采集阶段、来源和质量分，便于复盘哪些数据来自主源、Playwright 或 AnySearch。
- 预测流程: 特征快照、量化概率、启发式概率、融合概率、动作策略、LLM 复核、二级仲裁、目标批量策略落库、赛后反馈。
- 学习闭环: 训练候选只在点击“训练学习候选”时执行离线策略搜索；首页会区分“历史保存反馈”和“当前策略回放”，避免把旧 prediction runs 的累计反馈误读为 active profile 的回测成绩。
- 历史回放补数: 点击“回放补齐历史期”会按缺口向更早期号回放，流程为同步对赛、逐场超时采集、整期预测、同步赛果、写入 `roi_source='replay_backfill'` 的反馈；部分闭环期不会计作完整学习样本。
- 让球盘学习策略: 当前 active learning profile 为 #45，策略类型 `handicap_bucket_table`，用于让球盘推荐。当前全量历史回放样本 1232 场，执行 964 场，执行占比 78.25%，执行命中率 74.27%；训练目标为让球盘命中率 `>= 70%`、执行占比 `>= 60%`。只有验证达标的学习候选才能启用。
- 目标批量生产层: `coverage_draw_rescue` 会在 `predict_match()` 单场保存后、`predict_issue()` 整期完成后应用到同期期数未结算 latest prediction runs，写回 `recommendation`、`recommended_outcome`、`suggested_stake_pct` 和 `effective_*`，来源标记为 `target_batch_strategy`。策略计算读取原始 `algo_*` 底稿，避免重复应用时把自身写回结果当成新输入；已有 `feedback_logs` 的赛前 canonical run 会被跳过，防止赛后重预测改写赛前基准。
- 本期精选 TOP3: 首页按当前期 `issue_top_picks` 展示精选场次；执行当前期赛果同步结算后会刷新 TOP3，已结算场次显示命中/错误、实际赛果和比分，命中率只按已结算 TOP3 计算。
- 历史样本: 截至 2026-06-15，本地 SQLite 内 `26059` 到 `26085` 共 27 期、378 条已采集样本的核心采集维度已补齐，其中 `market_value_summary` 覆盖 378/378。
- 历史保存反馈: 首页“历史反馈”统计的是 `feedback_logs` 中已经保存的旧 prediction runs，可能混合多个 learning profile，不会随 #45 启用自动改写。要查看 #45 口径，应看同一面板中的“当前策略回放”。
- 未来预测: active learning profile #45 会在生成让球盘 `handicap_risk` 后应用到新 prediction runs；胜平负最终执行动作仍由 `coverage_draw_rescue` 目标批量生产层落库。
- 批量操作范围: 首页对赛列表支持勾选多场；勾选后采集、预测、同步赛果并结算只处理选中对赛，未勾选时仍按当前期全量处理。
- 页面口径: “赛后结果与赛前预测对比”分开展示胜平负和让球盘的已结算、正确、错误、命中率和 ROI；AI 预测中台分开展示胜平负建议仓位和让球盘建议仓位，两者均为分数 Kelly 口径。

## 快速启动

从仓库根目录运行:

```powershell
start_app.bat
```

或手动启动:

```powershell
python data/app.py
```

如果 Windows 上的 `python` 命中 Microsoft Store 占位程序或 `py` launcher 指向失效版本，先运行:

```powershell
C:\Users\15696\AppData\Local\Programs\Python\Python312\python.exe data\check_runtime.py
```

确认依赖可用后，用同一个 Python 跑下面的 CLI 命令。

访问:

```text
http://127.0.0.1:5050/
```

停止服务:

```powershell
stop_app.bat
```

如果 5050 端口仍被占用:

```powershell
stop_app_port.bat
```

## 常用命令

批量同步和采集当前期:

```powershell
python data/run_full_pipeline.py
```

生成预测:

```powershell
python data/run_prediction.py predict --match-id <match_id>
python data/run_prediction.py predict --issue <issue>
python data/run_prediction.py predict --match-id <match_id> --collect
python data/run_prediction.py predict --match-id <match_id> --json
```

记录赛后反馈:

```powershell
python data/run_prediction.py feedback --run-id <run_id> --match-id <match_id> --actual-result home|draw|away --actual-score "2-1"
```

查看摘要和回测:

```powershell
python data/run_prediction.py summary
python data/run_prediction.py summary --json
python data/run_prediction.py backtest --json
```

回填历史身价和采集质量:

```powershell
python data/backfill_market_values.py
python data/backfill_collection_quality.py
python data/backfill_collection_quality.py --force
```

回放补齐学习闭环历史期:

```powershell
cd data
python replay_backfill.py
```

也可以在首页点击“回放补齐历史期”。完成后再点击“训练学习候选”，用补齐后的完整已结算期重新搜索候选。

抓取探针:

```powershell
python data/list_current_matches.py
python data/collect_current_first_match.py
```

## 主要文档

- `data/README.md`: 当前 Python 数据与预测模块的技术说明。
- `SETUP_GUIDE.md`: 环境、启动、运行、排障手册。
- `data/PLAYWRIGHT_CLI_INTEGRATION.md`: Playwright CLI 抓取和统一补采策略说明。
- `AGENTS.md`: 维护规则、入口、边界和 GitNexus 要求。

## 维护注意

- 不要把入口改成 `python -m ...`，当前脚本使用本地绝对导入。
- 新代码优先依赖 `source_500_client.py`、`collection_service.py`、`collection_repository.py`、`prediction_engine.py`。
- `collector_store.py` 只作为兼容 facade。
- 不要提交真实 `.env`、`data/*.db`、日志、虚拟环境或 `dist/`/`build/` 产物；公开配置示例只更新 `.env.example`。
- 修改抓取或解析逻辑前，先用小探针验证 500.com 页面结构。
- SQLite 被锁时，先停止重复的 `data/app.py` Python 进程，再重新启动服务。
