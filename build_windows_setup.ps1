param(
    [string]$PythonExe = ""
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$DataDir = Join-Path $Root "data"
$BuildDir = Join-Path $Root "build"
$DistDir = Join-Path $Root "dist"
$InstallerSourceDir = Join-Path $BuildDir "installer_src"
$PayloadStageDir = Join-Path $BuildDir "payload_stage"
$AppStageDir = Join-Path $PayloadStageDir "app"
$PayloadZip = Join-Path $InstallerSourceDir "FootballAnalysisPayload.zip"
$InstallerExe = Join-Path $DistDir "FootballAnalysisSetup.exe"

if (-not $PythonExe) {
    $LocalPython = Join-Path $DataDir ".myenv\Scripts\python.exe"
    if (Test-Path $LocalPython) {
        $PythonExe = $LocalPython
    } else {
        $PythonExe = "python"
    }
}

function Write-Info($Message) {
    Write-Host "[build] $Message"
}

function Set-Utf8NoBom($Path, $Lines) {
    $Text = ($Lines -join [Environment]::NewLine) + [Environment]::NewLine
    [System.IO.File]::WriteAllText($Path, $Text, [System.Text.UTF8Encoding]::new($false))
}

function New-SanitizedEnvExample {
    $Source = Join-Path $Root ".env"
    $Target = Join-Path $Root ".env.example"
    if (Test-Path $Source) {
        $Lines = Get-Content -LiteralPath $Source | ForEach-Object {
            if ($_ -match "^\s*([^#=][^=]*)=(.*)$") {
                $Key = $Matches[1].Trim()
                if ($Key -eq "COLLECTION_APIKEY") { "$Key=" } else { $_ }
            } else {
                $_
            }
        }
        Set-Utf8NoBom $Target $Lines
        return
    }

    Set-Utf8NoBom $Target @(
        "OPENAI_BASE_URL="
        "OPENAI_API_KEY="
        "OPENAI_MODEL_RESEARCH="
        "COLLECTION_BASE_URL="
        "COLLECTION_APIKEY="
        "COLLECTION_MODEL="
        "COLLECTION_REVIEW_MAX_TOKENS=900"
        "LLM_REVIEW_ENABLED=false"
        "TAVILY_API_KEY="
        "FOOTBALL_API_KEY="
        "NETWORK_SEARCH_URL="
        "MODEL_OPTIONS="
        "AGENT_MODEL_MAP="
        "PLAYWRIGHT_CLI_BIN="
        "PLAYWRIGHT_CLI_HEADED=1"
        "PLAYWRIGHT_CLI_WAIT_MS=800"
        "PLAYWRIGHT_CLI_TIMEOUT_MS=120000"
    )
}

function New-LauncherFiles {
    $StartCmd = Join-Path $AppStageDir "Start Football Analysis.cmd"
    $ConfigCmd = Join-Path $AppStageDir "Open Config.cmd"
    $Readme = Join-Path $AppStageDir "README_INSTALL.txt"

    @(
        "@echo off"
        "cd /d ""%~dp0"""
        "start ""Football Analysis Server"" ""%~dp0FootballAnalysis.exe"""
        "timeout /t 3 /nobreak >nul"
        "start """" ""http://127.0.0.1:5050/"""
    ) | Set-Content -LiteralPath $StartCmd -Encoding ASCII

    @(
        "@echo off"
        "start """" ""http://127.0.0.1:5050/config"""
    ) | Set-Content -LiteralPath $ConfigCmd -Encoding ASCII

    @(
        "Football Analysis"
        ""
        "Run: Start Football Analysis.cmd"
        "URL: http://127.0.0.1:5050/"
        "Config: http://127.0.0.1:5050/config"
        ""
        "The installer keeps summary-model config but clears the review model API key. On first run, open Config and fill the review model key again."
    ) | Set-Content -LiteralPath $Readme -Encoding UTF8
}

function Invoke-IExpress {
    $SedPath = Join-Path $BuildDir "FootballAnalysisSetup.sed"
    $InstallCmd = Join-Path $Root "packaging\windows\install.cmd"
    $InstallPs1 = Join-Path $Root "packaging\windows\install.ps1"
    $InstallPlaywrightRuntimePs1 = Join-Path $Root "packaging\windows\install_playwright_runtime.ps1"
    $InstallPlaywrightRuntimeCmd = Join-Path $Root "packaging\windows\install_playwright_runtime.cmd"
    $IExpress = Join-Path $env:SystemRoot "System32\iexpress.exe"

    if (-not (Test-Path $IExpress)) {
        throw "iexpress.exe was not found."
    }

    $InstallerSource = $InstallerSourceDir.TrimEnd("\")
    $Sed = @"
[Version]
Class=IEXPRESS
SEDVersion=3
[Options]
PackagePurpose=InstallApp
ShowInstallProgramWindow=0
HideExtractAnimation=1
UseLongFileName=1
InsideCompressed=0
CAB_FixedSize=0
CAB_ResvCodeSigning=0
RebootMode=N
InstallPrompt=
DisplayLicense=
FinishMessage=
TargetName=$InstallerExe
FriendlyName=Football Analysis Setup
AppLaunched=install.cmd
PostInstallCmd=<None>
AdminQuietInstCmd=
UserQuietInstCmd=
SourceFiles=SourceFiles
[SourceFiles]
SourceFiles0=$InstallerSource
[SourceFiles0]
FootballAnalysisPayload.zip=
install.cmd=
install.ps1=
install_playwright_runtime.ps1=
install_playwright_runtime.cmd=
"@

    if (-not (Test-Path $InstallPlaywrightRuntimePs1)) {
        throw "Playwright runtime installer was not found: $InstallPlaywrightRuntimePs1"
    }
    if (-not (Test-Path $InstallPlaywrightRuntimeCmd)) {
        throw "Playwright runtime CMD wrapper was not found: $InstallPlaywrightRuntimeCmd"
    }

    Copy-Item -LiteralPath $InstallCmd -Destination (Join-Path $InstallerSourceDir "install.cmd") -Force
    Copy-Item -LiteralPath $InstallPs1 -Destination (Join-Path $InstallerSourceDir "install.ps1") -Force
    Copy-Item -LiteralPath $InstallPlaywrightRuntimePs1 -Destination (Join-Path $InstallerSourceDir "install_playwright_runtime.ps1") -Force
    Copy-Item -LiteralPath $InstallPlaywrightRuntimeCmd -Destination (Join-Path $InstallerSourceDir "install_playwright_runtime.cmd") -Force
    Set-Content -LiteralPath $SedPath -Value $Sed -Encoding ASCII

    & $IExpress /N /Q $SedPath
    $FallbackInstaller = Join-Path $Root "FootballAnalysisSetup.exe"
    if (-not (Test-Path $InstallerExe) -and (Test-Path $FallbackInstaller)) {
        Copy-Item -LiteralPath $FallbackInstaller -Destination $InstallerExe -Force
    }
}

Write-Info "creating sanitized .env.example"
New-SanitizedEnvExample

Write-Info "building PyInstaller app"
$AddData = @(
    "data\templates;templates",
    "data\static;static",
    "data\football_data.db;.",
    "data\football_data_live.db;.",
    ".env.example;."
)

& $PythonExe -m PyInstaller `
    --noconfirm `
    --clean `
    --onedir `
    --name FootballAnalysis `
    --paths data `
    --add-data $AddData[0] `
    --add-data $AddData[1] `
    --add-data $AddData[2] `
    --add-data $AddData[3] `
    --add-data $AddData[4] `
    data\app.py
Copy-Item -LiteralPath (Join-Path $Root ".env.example") -Destination (Join-Path $DistDir "FootballAnalysis\.env") -Force

Write-Info "staging installer payload"
Remove-Item -LiteralPath $InstallerSourceDir -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath $PayloadStageDir -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Path $InstallerSourceDir | Out-Null
New-Item -ItemType Directory -Path $PayloadStageDir | Out-Null
Copy-Item -LiteralPath (Join-Path $DistDir "FootballAnalysis") -Destination $AppStageDir -Recurse -Force
Copy-Item -LiteralPath (Join-Path $Root ".env.example") -Destination (Join-Path $AppStageDir ".env") -Force
New-LauncherFiles

Write-Info "creating payload zip"
Compress-Archive -Path (Join-Path $PayloadStageDir "*") -DestinationPath $PayloadZip -Force

Write-Info "creating setup executable"
Invoke-IExpress

Write-Info "done: $InstallerExe"
