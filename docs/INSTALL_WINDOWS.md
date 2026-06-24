# Football Analysis Windows 安装部署说明

## 安装包内容

安装包包含当前项目代码、`data/football_data.db` 历史数据库、当前 `.env` 配置、Python 依赖清单、Node/Playwright 依赖清单、安装脚本和启动脚本。

注意：本包按你的要求包含当前 `.env`，其中可能有 LLM、AnySearch、搜索服务等 API Key。请只在受控环境中分发。

## 新环境前置要求

安装脚本会自动创建 Python 虚拟环境并安装依赖，但新机器需要先具备：

1. Windows 10/11 x64。
2. Python 3.10 或更高版本，并勾选 Add Python to PATH。
3. Node.js LTS，包含 npm。
4. 能访问 Python/npm/Playwright 下载源以及 500.com、odds.500.com、Flashscore、RotoWire、Transfermarkt、Understat 等数据源。

## 一键安装

如果使用 `FootballAnalysisSetup.exe`：运行安装器，保持勾选“Run dependency installer after copying files”。安装完成后会自动执行依赖安装并启动系统。

如果使用 zip 包：解压到目标目录，右键 PowerShell 进入目录，执行：

```powershell
powershell -ExecutionPolicy Bypass -File installer\install.ps1
```

安装完成后运行：

```bat
start_app.bat
```

浏览器访问：

```text
http://127.0.0.1:5050/
```

## Playwright / playwright_cli

项目使用 npm 包 `playwright` 和 `@playwright/cli`。安装脚本会执行：

```powershell
npm install
npx playwright install chromium
```

如果新环境网络不能下载 Chromium，可以先在能联网环境中安装好 Playwright 浏览器缓存，或设置企业 npm/Playwright 下载镜像后重跑安装脚本。

## AnySearch / LLM / 外部搜索

项目通过 `.env` 中的配置调用外部能力。安装包已按要求携带当前 `.env`。如果新环境需要替换密钥，编辑安装目录下的 `.env`，然后重启 `start_app.bat`。

常见配置包括 OpenAI-compatible LLM、AnySearch/collection API、Tavily 或其它搜索入口。具体键名可参考 `.env.example`。

## 数据库

当前历史数据库位于：

```text
data\football_data.db
```

安装包已携带当前数据库。备份时直接复制该文件即可。系统启动后会继续在该 SQLite 文件中写入新采集、新预测和反馈。

## 常用操作

启动系统：运行 `start_app.bat`。

停止系统：关闭启动窗口，或运行 `stop_app_port.bat` 释放 5050 端口。

检查依赖：

```powershell
.venv\Scripts\python.exe data\check_runtime.py
```

重新安装依赖：

```powershell
powershell -ExecutionPolicy Bypass -File installer\install.ps1
```

## 故障排查

如果提示找不到 Python：安装 Python 3.10+ 并加入 PATH，或者执行 `installer\install.ps1 -PythonCommand "C:\Path\To\python.exe"`。

如果提示找不到 Node/npm：安装 Node.js LTS 后重新打开 PowerShell。

如果 Playwright 下载失败：检查网络、代理、npm 源，或设置 `PLAYWRIGHT_DOWNLOAD_HOST` 后重新执行安装脚本。

如果 SQLite 显示只读或 locked：确认没有多个程序同时打开 `data\football_data.db`，删除残留的 `data\football_data.db-journal` 前请先备份数据库。

