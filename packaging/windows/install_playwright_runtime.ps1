param(
    [string]$AppDir = "",
    [switch]$SkipNodeInstall
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

function Resolve-DefaultAppDir {
    $ScriptDir = Split-Path -Parent $MyInvocation.ScriptName
    if ($ScriptDir -and (Test-Path (Join-Path $ScriptDir "FootballAnalysis.exe"))) {
        return $ScriptDir
    }

    $LocalAppData = [Environment]::GetFolderPath("LocalApplicationData")
    return (Join-Path $LocalAppData "FootballAnalysis\app")
}

function Write-Log {
    param([string]$Message)
    $Stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$Stamp] $Message"
}

function Resolve-CommandPath {
    param([string[]]$Names)
    foreach ($Name in $Names) {
        $Command = Get-Command $Name -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($Command -and $Command.Source) {
            return $Command.Source
        }
    }
    return $null
}

function Refresh-ProcessPath {
    $MachinePath = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $UserPath = [Environment]::GetEnvironmentVariable("Path", "User")
    $Parts = @()
    if ($MachinePath) { $Parts += $MachinePath }
    if ($UserPath) { $Parts += $UserPath }
    if ($Parts.Count -gt 0) {
        $env:Path = ($Parts -join ";")
    }
}

function Invoke-External {
    param(
        [string]$FilePath,
        [string[]]$Arguments,
        [string]$StepName
    )

    Write-Log ("{0}: {1} {2}" -f $StepName, $FilePath, ($Arguments -join " "))
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw ("{0} failed with exit code {1}" -f $StepName, $LASTEXITCODE)
    }
}

function Ensure-NodeAndNpm {
    $NodeCmd = Resolve-CommandPath @("node.exe", "node.cmd", "node")
    $NpmCmd = Resolve-CommandPath @("npm.cmd", "npm.exe", "npm")

    if ($NodeCmd -and $NpmCmd) {
        Write-Log "Node.js and npm were found."
        return @{ Node = $NodeCmd; Npm = $NpmCmd }
    }

    if ($SkipNodeInstall) {
        throw "Node.js/npm not found, and -SkipNodeInstall was set. Install Node.js LTS first."
    }

    $Winget = Resolve-CommandPath @("winget.exe", "winget")
    if (-not $Winget) {
        throw "Node.js/npm not found and winget is unavailable. Install Node.js LTS from https://nodejs.org, then run this script again."
    }

    Write-Log "Node.js/npm not found. Installing Node.js LTS with winget."
    try {
        Invoke-External $Winget @("install", "--id", "OpenJS.NodeJS.LTS", "-e", "--accept-package-agreements", "--accept-source-agreements", "--scope", "user") "Install Node.js LTS"
    } catch {
        Write-Log "User-scope winget install failed; retrying with default winget scope."
        Invoke-External $Winget @("install", "--id", "OpenJS.NodeJS.LTS", "-e", "--accept-package-agreements", "--accept-source-agreements") "Install Node.js LTS"
    }

    Refresh-ProcessPath
    $NodeCmd = Resolve-CommandPath @("node.exe", "node.cmd", "node")
    $NpmCmd = Resolve-CommandPath @("npm.cmd", "npm.exe", "npm")

    if (-not ($NodeCmd -and $NpmCmd)) {
        throw "Node.js/npm still not found after installation. Open a new terminal or restart Windows, then run this script again."
    }

    return @{ Node = $NodeCmd; Npm = $NpmCmd }
}

function Set-Utf8NoBom {
    param(
        [string]$Path,
        [string[]]$Lines
    )
    $Parent = Split-Path -Parent $Path
    if ($Parent -and -not (Test-Path $Parent)) {
        New-Item -ItemType Directory -Path $Parent -Force | Out-Null
    }
    $Encoding = New-Object System.Text.UTF8Encoding $false
    $Text = ($Lines -join [Environment]::NewLine) + [Environment]::NewLine
    [System.IO.File]::WriteAllText($Path, $Text, $Encoding)
}

function Set-EnvFileValue {
    param(
        [string]$Path,
        [string]$Key,
        [string]$Value
    )

    $Lines = @()
    if (Test-Path $Path) {
        $Lines = @(Get-Content -LiteralPath $Path)
    }

    $Found = $false
    $Pattern = "^\s*" + [regex]::Escape($Key) + "\s*="
    $NewLines = @()
    foreach ($Line in $Lines) {
        if ($Line -match $Pattern) {
            $NewLines += ("{0}={1}" -f $Key, $Value)
            $Found = $true
        } else {
            $NewLines += $Line
        }
    }

    if (-not $Found) {
        $NewLines += ("{0}={1}" -f $Key, $Value)
    }

    Set-Utf8NoBom $Path $NewLines
}

function Ensure-PlaywrightCliRuntime {
    param(
        [string]$TargetAppDir,
        [string]$NpmCmd
    )

    if (-not (Test-Path $TargetAppDir)) {
        throw "Application directory was not found: $TargetAppDir"
    }

    $RuntimeDir = Join-Path $TargetAppDir "_runtime\playwright-cli"
    New-Item -ItemType Directory -Path $RuntimeDir -Force | Out-Null

    $PackageJson = Join-Path $RuntimeDir "package.json"
    $PackageJsonContent = @'
{
  "private": true,
  "description": "Runtime dependencies for Football Analysis playwright-cli scraping",
  "dependencies": {
    "@playwright/cli": "latest",
    "playwright": "latest"
  }
}
'@
    $Encoding = New-Object System.Text.UTF8Encoding $false
    [System.IO.File]::WriteAllText($PackageJson, $PackageJsonContent, $Encoding)

    Invoke-External $NpmCmd @("install", "--prefix", $RuntimeDir, "--omit=dev", "--no-audit", "--no-fund") "Install @playwright/cli and Playwright"

    $CliBin = Join-Path $RuntimeDir "node_modules\.bin\playwright-cli.cmd"
    if (-not (Test-Path $CliBin)) {
        $AltCliBin = Join-Path $RuntimeDir "node_modules\.bin\playwright-cli"
        if (Test-Path $AltCliBin) {
            $CliBin = $AltCliBin
        } else {
            throw "@playwright/cli was installed, but playwright-cli executable was not found under $RuntimeDir\node_modules\.bin"
        }
    }

    $PlaywrightBin = Join-Path $RuntimeDir "node_modules\.bin\playwright.cmd"
    if (Test-Path $PlaywrightBin) {
        Invoke-External $PlaywrightBin @("install", "chromium") "Install Playwright Chromium browser"
    } else {
        Write-Log "Playwright browser installer was not found; skipping browser download."
    }

    Invoke-External $CliBin @("--help") "Verify playwright-cli"

    $EnvFile = Join-Path $TargetAppDir ".env"
    Set-EnvFileValue $EnvFile "PLAYWRIGHT_CLI_BIN" $CliBin
    Set-EnvFileValue $EnvFile "PLAYWRIGHT_CLI_HEADED" "1"
    Set-EnvFileValue $EnvFile "PLAYWRIGHT_CLI_WAIT_MS" "800"
    Set-EnvFileValue $EnvFile "PLAYWRIGHT_CLI_TIMEOUT_MS" "120000"

    [Environment]::SetEnvironmentVariable("PLAYWRIGHT_CLI_BIN", $CliBin, "User")
    [Environment]::SetEnvironmentVariable("PLAYWRIGHT_CLI_HEADED", "1", "User")

    return $CliBin
}

if (-not $AppDir) {
    $AppDir = Resolve-DefaultAppDir
}

Write-Log "Installing/checking Football Analysis browser runtime."
Write-Log "Application directory: $AppDir"

$Tools = Ensure-NodeAndNpm
Invoke-External $Tools.Node @("--version") "Verify Node.js"
Invoke-External $Tools.Npm @("--version") "Verify npm"

$CliPath = Ensure-PlaywrightCliRuntime -TargetAppDir $AppDir -NpmCmd $Tools.Npm

Write-Host ""
Write-Host "Playwright CLI runtime is ready."
Write-Host "PLAYWRIGHT_CLI_BIN=$CliPath"
Write-Host "The setting was also written to: $(Join-Path $AppDir '.env')"
