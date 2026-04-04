param(
    [string]$Server = "root@200.200.201.4",
    [string]$RemoteDir = "/root/macd-overlay",
    [string]$PythonBin = "python3",
    [switch]$NoStart
)

$ErrorActionPreference = "Stop"

function Require-Command {
    param([string]$Name)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Missing required command: $Name"
    }
}

Require-Command "ssh"
Require-Command "scp"

function Invoke-RemoteBash {
    param(
        [string]$Target,
        [string]$ScriptText
    )
    $oneLine = ($ScriptText -replace "`r", "" -replace "`n", "; ").Trim()
    $oneLine = ($oneLine -replace "&\s*;", "& ").Trim()
    $escaped = $oneLine.Replace("'", "'\''")
    ssh $Target "bash -lc '$escaped'"
}

$workspace = (Resolve-Path (Join-Path $PSScriptRoot "..\.." )).Path
$stage = Join-Path $env:TEMP "p3_edge_research_deploy"
$bundle = Join-Path $env:TEMP "p3_edge_research_bundle.zip"

if (Test-Path $stage) {
    Remove-Item $stage -Recurse -Force
}
if (Test-Path $bundle) {
    Remove-Item $bundle -Force
}

New-Item -ItemType Directory -Path $stage | Out-Null
New-Item -ItemType Directory -Path (Join-Path $stage "ml") | Out-Null

Copy-Item -Path (Join-Path $workspace "ml\p3_edge_research") -Destination (Join-Path $stage "ml") -Recurse -Force
Copy-Item -Path (Join-Path $workspace "ml\p3.py") -Destination (Join-Path $stage "ml\p3.py") -Force
Copy-Item -Path (Join-Path $workspace "requirements.txt") -Destination (Join-Path $stage "requirements.txt") -Force

Compress-Archive -Path (Join-Path $stage "*") -DestinationPath $bundle -Force

$remoteBundle = "$RemoteDir/p3_edge_research_bundle.zip"

Write-Host "[1/4] Ensure remote directory exists"
Invoke-RemoteBash -Target $Server -ScriptText "mkdir -p '$RemoteDir'"

Write-Host "[2/4] Upload bundle"
scp $bundle "$Server`:$remoteBundle"

Write-Host "[3/4] Unpack and install dependencies"
$setupCmd = @"
set -e
cd '$RemoteDir'
rm -rf ml/p3_edge_research
unzip -o p3_edge_research_bundle.zip
$PythonBin -m venv .venv || true
. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
"@
Invoke-RemoteBash -Target $Server -ScriptText $setupCmd

if (-not $NoStart) {
    Write-Host "[4/4] Start auto_018 live-test launcher in background"
    $runCmd = @"
set -e
cd '$RemoteDir'
. .venv/bin/activate
nohup $PythonBin ml/p3_edge_research/launch_auto018_live_test.py --python-exe $PythonBin --max-files 60 --wf-max-folds 4 --sleep-seconds 3600 > auto018_live_test.log 2>&1 &
echo STARTED
"@
    Invoke-RemoteBash -Target $Server -ScriptText $runCmd
} else {
    Write-Host "[4/4] Skip start (--NoStart set)"
}

Write-Host "Done. Useful remote commands:"
Write-Host "ssh $Server 'tail -n 100 $RemoteDir/auto018_live_test.log'"
Write-Host "ssh $Server 'ps -ef | grep launch_auto018_live_test.py | grep -v grep'"
