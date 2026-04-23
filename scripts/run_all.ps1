# Runs all benchmark executables and records their results.

[CmdletBinding()]
param(
    [string] $OutputDir   = "",
    [string] $BuildDir    = "build",
    [string] $MinTime     = "0.5s",
    [int]    $Repetitions = 10,
    [string] $Filter      = ""
)

$ErrorActionPreference = "Stop"

function Format-CmdArg([string]$s) {
    if ($null -eq $s) { return '""' }
    if ($s -eq "")    { return '""' }
    if ($s -notmatch '[\s"]') { return $s }

    $escaped = $s -replace '\\(?=\\*")', '\\' -replace '"', '\"'
    return '"' + $escaped + '"'
}

$RepoRoot   = Resolve-Path (Join-Path $PSScriptRoot "..")
$BuildPath  = Join-Path $RepoRoot $BuildDir
$ReleaseDir = Join-Path $BuildPath "Release"

if (-not (Test-Path $ReleaseDir)) {
    throw "Release build directory not found: $ReleaseDir. Build first with 'cmake --build $BuildDir --config Release'."
}

if ([string]::IsNullOrEmpty($OutputDir)) {
    $stamp     = (Get-Date).ToUniversalTime().ToString("yyyyMMdd-HHmmss")
    $OutputDir = Join-Path $RepoRoot "results/$stamp"
}
else {
    $OutputDir = if ([System.IO.Path]::IsPathRooted($OutputDir)) { $OutputDir } else { Join-Path $RepoRoot $OutputDir }
}
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

Write-Host "Output directory: $OutputDir"

$sysInfoPath = Join-Path $OutputDir "system-info.txt"
$sysLines = @()
$sysLines += "# Run metadata"
$sysLines += "Timestamp (UTC) : " + (Get-Date).ToUniversalTime().ToString("o")
$sysLines += "Hostname        : $env:COMPUTERNAME"
$sysLines += "User            : $env:USERNAME"
$sysLines += ""
$sysLines += "# OS"
$os = Get-CimInstance Win32_OperatingSystem
$sysLines += "OS              : $($os.Caption) $($os.Version) (build $($os.BuildNumber))"
$sysLines += ""
$sysLines += "# CPU"
$cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
$sysLines += "CPU             : $($cpu.Name)"
$sysLines += "Cores/Threads   : $($cpu.NumberOfCores)/$($cpu.NumberOfLogicalProcessors)"
$sysLines += "Max clock (MHz) : $($cpu.MaxClockSpeed)"
$sysLines += ""
$sysLines += "# Memory"
$ramGB = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 1)
$sysLines += "RAM (GB)        : $ramGB"
$sysLines += ""
$sysLines += "# Toolchain"
try {
    $cmakeVer = (& cmake --version 2>$null | Select-Object -First 1)
    $sysLines += "CMake           : $cmakeVer"
}
catch {
    $sysLines += "CMake           : (not found on PATH)"
}
$sysLines += ""
$sysLines += "# Git"
Push-Location $RepoRoot
try {
    $commit = (& git rev-parse HEAD 2>$null)
    $branch = (& git rev-parse --abbrev-ref HEAD 2>$null)
    $dirty  = (& git status --porcelain 2>$null)
    $sysLines += "Branch          : $branch"
    $sysLines += "Commit          : $commit"
    $sysLines += "Working tree    : " + ($(if ([string]::IsNullOrEmpty($dirty)) { "clean" } else { "dirty (uncommitted changes present)" }))
}
catch {
    $sysLines += "(git unavailable)"
}
finally { Pop-Location }
$sysLines += ""
$sysLines += "# Run parameters"
$sysLines += "MinTime         : $MinTime"
$sysLines += "Repetitions     : $Repetitions"
$sysLines += "Filter          : " + ($(if ([string]::IsNullOrEmpty($Filter)) { "(none)" } else { $Filter }))
$sysLines += "ProcessorAffinity: 0x1 (logical core 0)"
$sysLines += "PriorityClass    : High"

Set-Content -Path $sysInfoPath -Value $sysLines -Encoding utf8
Write-Host "Wrote $sysInfoPath"

$exes = Get-ChildItem -Path $ReleaseDir -Filter "bench_*.exe" -File | Sort-Object Name
if ($exes.Count -eq 0) {
    throw "No bench_*.exe files in $ReleaseDir. Build first."
}
Write-Host ("Discovered {0} benchmark executables." -f $exes.Count)

foreach ($exe in $exes) {
    $name        = [System.IO.Path]::GetFileNameWithoutExtension($exe.Name)
    $jsonOut     = Join-Path $OutputDir "$name.json"
    $consoleOut  = Join-Path $OutputDir "$name.console.log"

    $argList = @(
        "--benchmark_format=json",
        "--benchmark_out=$jsonOut",
        "--benchmark_min_time=$MinTime",
        "--benchmark_repetitions=$Repetitions",
        "--benchmark_report_aggregates_only=true"
    )
    if (-not [string]::IsNullOrEmpty($Filter)) {
        $argList += "--benchmark_filter=$Filter"
    }

    Write-Host ""
    Write-Host ("=== {0} ===" -f $name)
    Write-Host ("    -> {0}" -f $jsonOut)

    $argString = ($argList | ForEach-Object { Format-CmdArg $_ }) -join ' '

    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName               = $exe.FullName
    $psi.Arguments              = $argString
    $psi.UseShellExecute        = $false
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError  = $true
    $psi.WorkingDirectory       = $ReleaseDir

    $proc = New-Object System.Diagnostics.Process
    $proc.StartInfo = $psi
    [void]$proc.Start()

    try { $proc.ProcessorAffinity = [IntPtr]::new(0x1) }
    catch { Write-Warning ("Affinity not pinned on {0}: {1}" -f $name, $_.Exception.Message) }
    try { $proc.PriorityClass     = [System.Diagnostics.ProcessPriorityClass]::AboveNormal }
    catch { Write-Warning ("Priority not raised on {0}: {1}" -f $name, $_.Exception.Message) }

    $stdOut = $proc.StandardOutput.ReadToEnd()
    $stdErr = $proc.StandardError.ReadToEnd()
    $proc.WaitForExit()

    Set-Content -Path $consoleOut -Value ($stdOut + "`n--- STDERR ---`n" + $stdErr) -Encoding utf8

    if ($proc.ExitCode -ne 0) {
        Write-Warning ("{0} exited with code {1}. See {2}." -f $name, $proc.ExitCode, $consoleOut)
    }
    else {
        Write-Host ("    OK")
    }
}

Write-Host ""
Write-Host ("Done. Results in: {0}" -f $OutputDir)
