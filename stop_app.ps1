[CmdletBinding(SupportsShouldProcess = $true)]
param(
  [int]$Port = 5050,
  [switch]$PortOnly
)

$ErrorActionPreference = "Stop"

$Root = (Resolve-Path -LiteralPath $PSScriptRoot).Path.TrimEnd("\")
$AppScript = (Join-Path $Root "data\app.py")
$VenvRoot = (Join-Path $Root "data\.myenv")

function Normalize-Text {
  param([string]$Text)
  if ([string]::IsNullOrWhiteSpace($Text)) {
    return ""
  }
  return $Text.Replace("/", "\")
}

function Contains-Text {
  param(
    [string]$Text,
    [string]$Needle
  )
  if ([string]::IsNullOrWhiteSpace($Text) -or [string]::IsNullOrWhiteSpace($Needle)) {
    return $false
  }
  return $Text.IndexOf($Needle, [StringComparison]::OrdinalIgnoreCase) -ge 0
}

$processes = @(Get-CimInstance Win32_Process)
$byPid = @{}
foreach ($process in $processes) {
  $byPid[[int]$process.ProcessId] = $process
}

$candidatePids = [System.Collections.Generic.HashSet[int]]::new()
$depthCache = @{}

function Add-Candidate {
  param([int]$ProcessId)
  if ($ProcessId -gt 0 -and $ProcessId -ne $PID) {
    [void]$candidatePids.Add($ProcessId)
  }
}

function Is-Skippable-Process {
  param($Process)
  return [string]::Equals([string]$Process.Name, "conhost.exe", [StringComparison]::OrdinalIgnoreCase)
}

function Get-Depth {
  param([int]$ProcessId)

  if ($depthCache.ContainsKey($ProcessId)) {
    return [int]$depthCache[$ProcessId]
  }

  $depth = 0
  if ($byPid.ContainsKey($ProcessId)) {
    $parentPid = [int]$byPid[$ProcessId].ParentProcessId
    if ($parentPid -gt 0 -and $parentPid -ne $ProcessId -and $byPid.ContainsKey($parentPid)) {
      $depth = 1 + (Get-Depth $parentPid)
    }
  }

  $depthCache[$ProcessId] = $depth
  return $depth
}

$normalizedRoot = Normalize-Text $Root
$normalizedAppScript = Normalize-Text $AppScript
$normalizedVenvRoot = Normalize-Text $VenvRoot

foreach ($process in $processes) {
  $commandLine = Normalize-Text ([string]$process.CommandLine)
  $executablePath = Normalize-Text ([string]$process.ExecutablePath)
  $processName = [string]$process.Name
  $isPythonProcess = [string]::Equals($processName, "python.exe", [StringComparison]::OrdinalIgnoreCase) -or
    [string]::Equals($processName, "pythonw.exe", [StringComparison]::OrdinalIgnoreCase)
  $isCmdProcess = [string]::Equals($processName, "cmd.exe", [StringComparison]::OrdinalIgnoreCase)
  $isStopCommand = (Contains-Text $commandLine "stop_app.bat") -or
    (Contains-Text $commandLine "stop_app.ps1") -or
    (Contains-Text $commandLine "stop_app_port.bat")

  if ($isStopCommand) {
    continue
  }

  $matchesAppScript = $isPythonProcess -and (
    (Contains-Text $commandLine $normalizedAppScript) -or
    (Contains-Text $commandLine "data\app.py") -or
    (Contains-Text $commandLine "data\\app.py")
  )
  $matchesStartScript = $isCmdProcess -and (Contains-Text $commandLine "start_app.bat")
  $matchesProjectVenv = (Contains-Text $executablePath $normalizedVenvRoot) -and
    ((Contains-Text $commandLine "app.py") -or (Contains-Text $commandLine "data\app.py"))

  if ($matchesAppScript -or $matchesStartScript -or $matchesProjectVenv) {
    Add-Candidate ([int]$process.ProcessId)
  }
}

try {
  $listeners = @(Get-NetTCPConnection -LocalAddress 127.0.0.1 -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
} catch {
  $listeners = @()
}

foreach ($listener in $listeners) {
  $listenerPid = [int]$listener.OwningProcess
  $listenerProcess = $byPid[$listenerPid]
  $listenerCommandLine = Normalize-Text ([string]$listenerProcess.CommandLine)
  if ($PortOnly -or (Contains-Text $listenerCommandLine "data\app.py") -or (Contains-Text $listenerCommandLine $normalizedAppScript)) {
    Add-Candidate $listenerPid
  } elseif ($listenerPid -gt 0) {
    Write-Warning "Port $Port is used by PID $listenerPid, but it does not look like this project app. Use -PortOnly to stop it anyway."
  }
}

$added = $true
while ($added) {
  $added = $false
  foreach ($process in $processes) {
    $pidValue = [int]$process.ProcessId
    $parentPid = [int]$process.ParentProcessId
    if (
      $candidatePids.Contains($parentPid) -and
      -not $candidatePids.Contains($pidValue) -and
      $pidValue -ne $PID -and
      -not (Is-Skippable-Process $process)
    ) {
      [void]$candidatePids.Add($pidValue)
      $added = $true
    }
  }
}

if ($candidatePids.Count -eq 0) {
  Write-Host "No football_analysis app process found."
  exit 0
}

$targets = foreach ($candidatePid in $candidatePids) {
  if ($byPid.ContainsKey($candidatePid)) {
    $byPid[$candidatePid]
  }
}

$targets = @(
  $targets |
    ForEach-Object {
      $_ | Add-Member -PassThru NoteProperty Depth (Get-Depth ([int]$_.ProcessId))
    } |
    Sort-Object -Property Depth, ProcessId -Descending
)

foreach ($target in $targets) {
  $processId = [int]$target.ProcessId
  $commandLine = [string]$target.CommandLine
  $label = "PID $processId"
  if (-not [string]::IsNullOrWhiteSpace($commandLine)) {
    $label = "$label - $commandLine"
  }

  if ($PSCmdlet.ShouldProcess($label, "Stop-Process")) {
    try {
      Stop-Process -Id $processId -Force -ErrorAction Stop
      Write-Host "Stopped $label"
    } catch {
      Write-Warning "Failed to stop PID ${processId}: $($_.Exception.Message)"
    }
  }
}
