# Football Analysis Windows installer/bootstrap
# Run from the installed project root: powershell -ExecutionPolicy Bypass -File installer\install.ps1
param(
    [string]$PythonCommand = "",
    [switch]$SkipPlaywrightBrowsers
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

function Stop-IfNativeFailed {
    param([string]$Step)
    if ($LASTEXITCODE -ne 0) {
        throw "$Step failed with exit code $LASTEXITCODE."
    }
}

function Split-CommandLine {
    param([string]$CommandLine)
    $parts = @()
    foreach ($part in ($CommandLine -split "\s+")) {
        if ($part.Trim()) { $parts += $part.Trim() }
    }
    return $parts
}

function Test-PythonCandidate {
    param(
        [string]$Exe,
        [string[]]$Args
    )
    try {
        $probe = @(
            "import sys",
            "print(sys.executable)",
            "print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')",
            "raise SystemExit(0 if sys.version_info >= (3, 10) else 1)"
        ) -join "; "
        $output = & $Exe @Args -c $probe 2>$null
        if ($LASTEXITCODE -ne 0 -or $output.Count -lt 2) {
            return $null
        }
        return [pscustomobject]@{
            Exe = $Exe
            Args = $Args
            Path = [string]$output[0]
            Version = [string]$output[1]
            CommandText = (($Exe, $Args) -join " ").Trim()
        }
    } catch {
        return $null
    }
}

function Add-PythonPathCandidate {
    param(
        [System.Collections.ArrayList]$Candidates,
        [string]$Path
    )
    if ($Path -and (Test-Path $Path)) {
        [void]$Candidates.Add(@($Path, @()))
    }
}

function Resolve-Python {
    $candidates = [System.Collections.ArrayList]::new()
    if ($PythonCommand.Trim()) {
        $parts = Split-CommandLine $PythonCommand
        if ($parts.Count -gt 0) {
            [void]$candidates.Add(@($parts[0], @($parts | Select-Object -Skip 1)))
        }
    }
    [void]$candidates.Add(@("py", @("-3.12")))
    [void]$candidates.Add(@("py", @("-3.11")))
    [void]$candidates.Add(@("py", @("-3.10")))
    [void]$candidates.Add(@("py", @("-3")))
    [void]$candidates.Add(@("python", @()))
    [void]$candidates.Add(@("python3", @()))

    $localPrograms = Join-Path $env:LOCALAPPDATA "Programs\Python"
    foreach ($dir in @($localPrograms, "C:\Program Files", "C:\Program Files (x86)")) {
        if (Test-Path $dir) {
            Get-ChildItem -Path $dir -Directory -Filter "Python*" -ErrorAction SilentlyContinue |
                Sort-Object Name -Descending |
                ForEach-Object { Add-PythonPathCandidate -Candidates $candidates -Path (Join-Path $_.FullName "python.exe") }
        }
    }
    Add-PythonPathCandidate -Candidates $candidates -Path (Join-Path $env:LOCALAPPDATA "Microsoft\WindowsApps\python.exe")
    Add-PythonPathCandidate -Candidates $candidates -Path (Join-Path $env:LOCALAPPDATA "Microsoft\WindowsApps\python3.exe")

    foreach ($candidate in $candidates) {
        $result = Test-PythonCandidate -Exe $candidate[0] -Args $candidate[1]
        if ($null -ne $result) {
            return $result
        }
    }

    throw "Python 3.10+ not found. Install Python 3.10+ from https://www.python.org/downloads/windows/ and enable 'Add python.exe to PATH', or run: powershell -NoProfile -ExecutionPolicy Bypass -File installer\install.ps1 -PythonCommand 'C:\Path\To\python.exe'"
}

Write-Host "[1/7] Checking Python..."
$Python = Resolve-Python
Write-Host "  Python $($Python.Version)"
Write-Host "  Executable: $($Python.Path)"
Write-Host "  Command: $($Python.CommandText)"

Write-Host "[2/7] Creating virtual environment..."
if ((Test-Path ".venv") -and -not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "  Removing incomplete .venv directory..."
    Remove-Item -Recurse -Force ".venv"
}
& $Python.Exe @($Python.Args + @("-m", "venv", ".venv"))
Stop-IfNativeFailed "Creating virtual environment"

$Py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) {
    throw "Virtual environment was not created correctly: $Py is missing."
}

Write-Host "[3/7] Upgrading pip..."
& $Py -m pip install --upgrade pip setuptools wheel
Stop-IfNativeFailed "Upgrading pip"

Write-Host "[4/7] Installing Python dependencies..."
& $Py -m pip install -r requirements.txt
Stop-IfNativeFailed "Installing Python dependencies"

Write-Host "[5/7] Checking Node.js/npm..."
try {
    $nodeVersion = node --version
    Stop-IfNativeFailed "Checking Node.js"
    $npmVersion = npm --version
    Stop-IfNativeFailed "Checking npm"
} catch {
    throw "Node.js/npm not found. Install Node.js LTS from https://nodejs.org/ then re-run this script."
}
Write-Host "  node $nodeVersion"
Write-Host "  npm $npmVersion"

Write-Host "[6/7] Installing Node dependencies including Playwright CLI..."
npm install
Stop-IfNativeFailed "Installing Node dependencies"
if (-not $SkipPlaywrightBrowsers) {
    Write-Host "  Installing Playwright Chromium browser..."
    npx playwright install chromium
    Stop-IfNativeFailed "Installing Playwright Chromium browser"
}

Write-Host "[7/7] Verifying runtime..."
& $Py data\check_runtime.py
Stop-IfNativeFailed "Runtime check"
& $Py -m py_compile data\app.py data\collection_repository.py data\collection_service.py data\feature_engine.py data\prediction_engine.py
Stop-IfNativeFailed "Python compile check"

Write-Host ""
Write-Host "Installation completed. Start with: start_app.bat"
Write-Host "Then open: http://127.0.0.1:5050/"
