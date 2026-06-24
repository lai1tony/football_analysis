$ErrorActionPreference = "Stop"

$SourceDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PayloadZip = Join-Path $SourceDir "FootballAnalysisPayload.zip"
$InstallRoot = Join-Path $env:LOCALAPPDATA "FootballAnalysis"
$AppDir = Join-Path $InstallRoot "app"
$BackupDir = Join-Path $InstallRoot ("backup_" + (Get-Date -Format "yyyyMMdd_HHmmss"))

function Copy-IfExists($Source, $Destination) {
    if (Test-Path $Source) {
        $Parent = Split-Path -Parent $Destination
        if (-not (Test-Path $Parent)) {
            New-Item -ItemType Directory -Path $Parent | Out-Null
        }
        Copy-Item -LiteralPath $Source -Destination $Destination -Force
    }
}

function Set-Utf8NoBom($Path, $Lines) {
    $Text = ($Lines -join [Environment]::NewLine) + [Environment]::NewLine
    [System.IO.File]::WriteAllText($Path, $Text, [System.Text.UTF8Encoding]::new($false))
}

function Read-EnvMap($Path) {
    $Map = @{}
    if (-not (Test-Path $Path)) {
        return $Map
    }
    Get-Content -LiteralPath $Path | ForEach-Object {
        if ($_ -match "^\s*([^#=][^=]*)=(.*)$") {
            $Key = $Matches[1].Trim().TrimStart([char]0xFEFF)
            $Map[$Key] = $Matches[2].Trim()
        }
    }
    return $Map
}

function Merge-MissingEnvValues($Path, $DefaultsPath) {
    if (-not (Test-Path $Path) -or -not (Test-Path $DefaultsPath)) {
        return
    }
    $Current = Read-EnvMap $Path
    $Defaults = Read-EnvMap $DefaultsPath
    $Seen = @{}
    $Lines = Get-Content -LiteralPath $Path | ForEach-Object {
        if ($_ -match "^\s*([^#=][^=]*)=(.*)$") {
            $Key = $Matches[1].Trim().TrimStart([char]0xFEFF)
            $Seen[$Key] = $true
            if (
                $Key -ne "COLLECTION_APIKEY" -and
                [string]::IsNullOrWhiteSpace([string]$Current[$Key]) -and
                -not [string]::IsNullOrWhiteSpace([string]$Defaults[$Key])
            ) {
                "$Key=$($Defaults[$Key])"
            } else {
                "$Key=$($Current[$Key])"
            }
        } else {
            $_
        }
    }
    foreach ($Key in $Defaults.Keys) {
        if (-not $Seen.ContainsKey($Key)) {
            $Value = if ($Key -eq "COLLECTION_APIKEY") { "" } else { $Defaults[$Key] }
            $Lines += "$Key=$Value"
        }
    }
    Set-Utf8NoBom $Path $Lines
}

function New-Shortcut($Path, $Target, $WorkingDirectory, $IconLocation) {
    $Shell = New-Object -ComObject WScript.Shell
    $Shortcut = $Shell.CreateShortcut($Path)
    $Shortcut.TargetPath = $Target
    $Shortcut.WorkingDirectory = $WorkingDirectory
    if ($IconLocation) {
        $Shortcut.IconLocation = $IconLocation
    }
    $Shortcut.Save()
}

function Clear-ReviewApiKey($Path) {
    if (-not (Test-Path $Path)) {
        return
    }
    $Lines = Get-Content -LiteralPath $Path | ForEach-Object {
        if ($_ -match "^\s*COLLECTION_APIKEY\s*=") {
            "COLLECTION_APIKEY="
        } else {
            $_
        }
    }
    Set-Utf8NoBom $Path $Lines
}

if (-not (Test-Path $PayloadZip)) {
    throw "Installer payload was not found: $PayloadZip"
}

New-Item -ItemType Directory -Path $InstallRoot -Force | Out-Null

$ExistingEnv = Join-Path $AppDir ".env"
$ExistingDb = Join-Path $AppDir "_internal\football_data.db"
$ExistingLiveDb = Join-Path $AppDir "_internal\football_data_live.db"

if ((Test-Path $ExistingEnv) -or (Test-Path $ExistingDb) -or (Test-Path $ExistingLiveDb)) {
    New-Item -ItemType Directory -Path $BackupDir -Force | Out-Null
    Copy-IfExists $ExistingEnv (Join-Path $BackupDir ".env")
    Copy-IfExists $ExistingDb (Join-Path $BackupDir "football_data.db")
    Copy-IfExists $ExistingLiveDb (Join-Path $BackupDir "football_data_live.db")
}

if (Test-Path $AppDir) {
    Remove-Item -LiteralPath $AppDir -Recurse -Force
}

Expand-Archive -LiteralPath $PayloadZip -DestinationPath $InstallRoot -Force

$NewEnv = Join-Path $AppDir ".env"
$ExampleEnv = Join-Path $AppDir "_internal\.env.example"
if (-not (Test-Path $NewEnv) -and (Test-Path $ExampleEnv)) {
    Copy-Item -LiteralPath $ExampleEnv -Destination $NewEnv -Force
}

if (Test-Path $BackupDir) {
    Copy-IfExists (Join-Path $BackupDir ".env") $NewEnv
    Copy-IfExists (Join-Path $BackupDir "football_data.db") (Join-Path $AppDir "_internal\football_data.db")
    Copy-IfExists (Join-Path $BackupDir "football_data_live.db") (Join-Path $AppDir "_internal\football_data_live.db")
}
Merge-MissingEnvValues $NewEnv $ExampleEnv
Clear-ReviewApiKey $NewEnv

$PlaywrightRuntimeInstaller = Join-Path $SourceDir "install_playwright_runtime.ps1"
if (Test-Path $PlaywrightRuntimeInstaller) {
    Write-Host ""
    Write-Host "Installing browser collection runtime dependencies..."
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $PlaywrightRuntimeInstaller -AppDir $AppDir
    if ($LASTEXITCODE -ne 0) {
        throw "Browser collection runtime dependency installation failed."
    }
} else {
    Write-Warning "install_playwright_runtime.ps1 was not found. Browser collection may fail until playwright-cli is installed."
}

$StartCmd = Join-Path $AppDir "Start Football Analysis.cmd"
$Exe = Join-Path $AppDir "FootballAnalysis.exe"
$Desktop = [Environment]::GetFolderPath("DesktopDirectory")
$Programs = [Environment]::GetFolderPath("Programs")
$StartMenuDir = Join-Path $Programs "Football Analysis"
New-Item -ItemType Directory -Path $StartMenuDir -Force | Out-Null

New-Shortcut `
    (Join-Path $Desktop "Football Analysis.lnk") `
    $StartCmd `
    $AppDir `
    $Exe

New-Shortcut `
    (Join-Path $StartMenuDir "Football Analysis.lnk") `
    $StartCmd `
    $AppDir `
    $Exe

Write-Host ""
Write-Host "Football Analysis installed to:"
Write-Host "  $AppDir"
Write-Host ""
Write-Host "Start menu and desktop shortcuts were created."
Write-Host "The installer clears the review model API key. Configure it at:"
Write-Host "  http://127.0.0.1:5050/config"
