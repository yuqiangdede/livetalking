param(
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
    [string]$PythonExe = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path $ProjectRoot).Path
$TargetRoot = Join-Path $ProjectRoot "models\tts\coqui\xtts_v2"

function Ensure-Directory {
    param([string]$Path)
    New-Item -ItemType Directory -Force -Path $Path | Out-Null
}

function Resolve-DownloadPython {
    if (-not [string]::IsNullOrWhiteSpace($PythonExe)) {
        return (Resolve-Path $PythonExe).Path
    }

    $venvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
    if (Test-Path -Path $venvPython -PathType Leaf) {
        return $venvPython
    }

    $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($pyLauncher) {
        return $pyLauncher.Source + " -3.10"
    }

    throw "Python 3.10 not found. Pass -PythonExe explicitly."
}

function Invoke-PythonCommand {
    param(
        [string]$PythonCommand,
        [string[]]$Arguments
    )

    if ($PythonCommand.Contains(" -3.10")) {
        $parts = $PythonCommand.Split(" ", 2)
        & $parts[0] $parts[1] @Arguments
    } else {
        & $PythonCommand @Arguments
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed: $PythonCommand $($Arguments -join ' ')"
    }
}

function Ensure-PythonModule {
    param(
        [string]$PythonCommand,
        [string]$ModuleName,
        [string]$InstallName
    )

    try {
        Invoke-PythonCommand -PythonCommand $PythonCommand -Arguments @("-c", "import $ModuleName")
    } catch {
        Write-Host "Installing Python helper package: $InstallName"
        Invoke-PythonCommand -PythonCommand $PythonCommand -Arguments @("-m", "pip", "install", $InstallName)
    }
}

function Assert-CoquiFiles {
    param([string]$BasePath)

    $configPath = Join-Path $BasePath "config.json"
    if (-not (Test-Path -Path $configPath -PathType Leaf)) {
        throw "Coqui XTTS v2 missing config.json: $configPath"
    }

    $checkpoint = Get-ChildItem -Path $BasePath -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Extension -in @(".pth", ".pt") } |
        Select-Object -First 1
    if (-not $checkpoint) {
        throw "Coqui XTTS v2 missing checkpoint file (*.pth or *.pt) in $BasePath"
    }
}

$resolvedPython = Resolve-DownloadPython
Ensure-Directory -Path $TargetRoot
Ensure-PythonModule -PythonCommand $resolvedPython -ModuleName "huggingface_hub" -InstallName "huggingface_hub"

$downloadScript = Join-Path $env:TEMP "download_coqui_xtts_v2_models.py"
@'
import shutil
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

target_dir = Path(sys.argv[1])
repo_id = "coqui/XTTS-v2"

if target_dir.exists():
    shutil.rmtree(target_dir)
target_dir.parent.mkdir(parents=True, exist_ok=True)
snapshot_download(repo_id=repo_id, local_dir=str(target_dir), local_dir_use_symlinks=False)
'@ | Set-Content -Path $downloadScript -Encoding UTF8

Write-Host "Downloading Coqui XTTS v2 model ..."
Invoke-PythonCommand -PythonCommand $resolvedPython -Arguments @($downloadScript, $TargetRoot)

Assert-CoquiFiles -BasePath $TargetRoot

Write-Host "Coqui XTTS v2 model download complete:"
Write-Host "  Model: $TargetRoot"
