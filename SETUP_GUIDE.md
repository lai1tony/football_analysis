# 安装、运行与维护手册

本文档描述当前 Python + Flask + SQLite 原型的实际运行方式。

## 环境要求

- Windows 10/11。
- Python 3.11+，推荐 3.12。
- 核心依赖: `flask`、`requests`、`beautifulsoup4`。
- 可选依赖: Playwright CLI、AnySearch、numpy、onnxruntime 等补采或增强组件。

## 安装依赖

```powershell
python -m venv data\.myenv
data\.myenv\Scripts\activate
pip install flask requests beautifulsoup4
```

本仓库没有稳定的 npm/Next.js 前端运行入口。即使根目录存在 `package.json` 或 `.next` 残留，也不要把它当作当前主应用。

## 配置

`.env` 位于仓库根目录，由 `data/config_service.py` 读取和保存。Web 配置页为:

```text
http://127.0.0.1:5050/config
```

常用配置:

```ini
OPENAI_BASE_URL=https://your-openai-compatible-api/v1
OPENAI_API_KEY=sk-...
OPENAI_MODEL_RESEARCH=gpt-4o

COLLECTION_BASE_URL=https://your-openai-compatible-api/v1
COLLECTION_APIKEY=sk-...
COLLECTION_MODEL=gpt-4o-mini

LLM_REVIEW_ENABLED=true
FOOTBALL_SCRAPER_BACKEND=requests
```

核心 500.com 抓取默认使用 `requests`；分析字段补采使用统一策略 `主源/已有数据 -> playwright-cli -> anysearch`。阵容/伤停优先 Flashscore，缺失时可回退到 RotoWire；球队身价以 Transfermarkt 为准，Playwright 未命中时由 AnySearch 抽取 Transfermarkt 页面兜底。LLM 和外部搜索是增强能力，缺失时应记录来源/质量或失败备注，不应伪造字段。

## 启动 Web

推荐:

```powershell
start_app.bat
```

手动:

```powershell
python data/app.py
```

本机如果 `python` 打开 Microsoft Store 或 `py` launcher 报找不到旧版本，使用明确的 Python 3.12 路径:

```powershell
C:\Users\15696\AppData\Local\Programs\Python\Python312\python.exe data\check_runtime.py
C:\Users\15696\AppData\Local\Programs\Python\Python312\python.exe data\app.py
```

访问:

```text
http://127.0.0.1:5050/
```

停止:

```powershell
stop_app.bat
```

清理 5050 占用:

```powershell
stop_app_port.bat
```

## CLI

批量同步和采集:

```powershell
python data/run_full_pipeline.py
```

预测:

```powershell
python data/run_prediction.py predict --match-id <match_id>
python data/run_prediction.py predict --issue <issue>
python data/run_prediction.py predict --match-id <match_id> --collect
python data/run_prediction.py predict --match-id <match_id> --json
```

预测 run 保存后会自动应用当前期目标批量生产层 `coverage_draw_rescue`。单场预测会重算同期期数未结算的 latest runs；整期预测完成后会再统一重算一次，并把最终执行动作写回 `prediction_runs.recommendation`、`recommended_outcome`、`suggested_stake_pct` 和 `effective_*`，来源为 `target_batch_strategy`。已有 `feedback_logs` 的赛前 canonical run 会被跳过；比赛开赛后再生成的新 run 会追加保存，不会替换赛前基准。

反馈:

```powershell
python data/run_prediction.py feedback --run-id <run_id> --match-id <match_id> --actual-result home|draw|away --actual-score "2-1"
```

摘要和回测:

```powershell
python data/run_prediction.py summary
python data/run_prediction.py summary --json
python data/run_prediction.py backtest --json
```

历史补采与质量归档:

```powershell
python data/backfill_market_values.py
python data/backfill_collection_quality.py
python data/backfill_collection_quality.py --force
```

`--force` 会重写已有 `analyses.collection_quality_summary` 的逐字段来源行，用于历史数据质量摘要格式升级；不会重新抓取外部数据。

## 学习候选使用

1. 打开首页的“学习闭环”区域。
2. 设置学习窗口。
3. 点击“训练学习候选”。
4. 只有候选满足让球盘命中率和执行占比目标时，页面才允许启用。
5. 启用后会影响后续新生成的让球盘 prediction runs；历史已保存 runs 不会自动改写。

当前默认目标:

- 让球盘命中率 `>= 70%`
- 让球盘执行占比 `>= 60%`

当前 active learning profile 为 #43，策略类型 `handicap_bucket_table`，只用于让球盘推荐。它在生成 `handicap_risk` 后按当前桶表策略决定让球侧和是否执行，未命中桶表时降为让球观望。胜平负最终动作仍由 `coverage_draw_rescue` 目标批量生产层落库。

截至 2026-06-20，本地 SQLite 的 #43 当前策略全量历史回放为 1245 条真实已结算样本、751 个动作、执行占比 60.32%、执行命中率 72.17%。首页“历史反馈”统计的是 `feedback_logs` 已保存 prediction runs 的累计真实反馈，可能混合旧 profile；“当前策略回放”才是 active profile #43 的即时回放口径。

## TOP3 重建

当前期赛果同步结算会自动刷新当期 TOP3 状态。首页 TOP3 的命中率只统计已结算精选场次；未开赛或未同步赛果的精选场次显示为待结算，不进入命中率分母。

在 `data/` 目录或仓库根目录均可运行:

```powershell
@'
from collection_repository import list_issues, compute_issue_top_picks

for issue in sorted(list_issues()):
    compute_issue_top_picks(issue)
'@ | python -
```

## 测试

常用回归测试:

```powershell
python -m unittest data.test_action_policy data.test_outcome_policy data.test_learning_engine data.test_feedback_backtest
cd data
python -m unittest test_collection_quality
```

若本地 `data\.myenv\Scripts\python.exe` 损坏，可使用可用 Python，并设置:

```powershell
$env:PYTHONPATH='D:\football_analysis\data\.myenv\Lib\site-packages'
```

## 排障

### 页面打不开

1. 确认访问的是 `http://127.0.0.1:5050/`。
2. 检查是否有多个 Python 服务:

```powershell
Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python|pythonw' } | Select-Object ProcessId,CommandLine
```

3. 如果有多个 `data/app.py`，先停止重复进程再重启。

### database is locked

通常是重复 Flask 服务或后台脚本未退出。停止相关 Python 进程后再启动。

### 首页变慢

首页不应执行离线策略搜索。`fit_target_strategy()` 只应在训练学习候选时运行。

### 采集异常

先运行:

```powershell
python data/check_runtime.py
python data/list_current_matches.py
python data/collect_current_first_match.py
```

如果探针失败，优先检查 500.com 页面结构或网络状态。
如果单个分析字段缺失，查看 `analyses.collection_quality_summary` 判断该字段停在主源、Playwright 还是 AnySearch 阶段。
