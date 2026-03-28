param(
    [ValidateSet("HuggingFace", "ModelScope")]
    [string]$Source = "HuggingFace",
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
    [string]$PythonExe = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path $ProjectRoot).Path
$TargetRoot = Join-Path $ProjectRoot "models\tts\qwen"
$ModelTarget = Join-Path $TargetRoot "Qwen3-TTS-12Hz-0.6B-CustomVoice"
$TokenizerTarget = Join-Path $TargetRoot "Qwen3-TTS-Tokenizer-12Hz"

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

function Assert-RequiredFiles {
    param(
        [string]$BasePath,
        [string[]]$RequiredFiles,
        [string]$Label
    )

    foreach ($item in $RequiredFiles) {
        $full = Join-Path $BasePath $item
        if (-not (Test-Path -Path $full)) {
            throw "$Label missing required file: $full"
        }
    }
}

$resolvedPython = Resolve-DownloadPython
Ensure-Directory -Path $TargetRoot

if ($Source -eq "HuggingFace") {
    Ensure-PythonModule -PythonCommand $resolvedPython -ModuleName "huggingface_hub" -InstallName "huggingface_hub"
} else {
    Ensure-PythonModule -PythonCommand $resolvedPython -ModuleName "modelscope" -InstallName "modelscope"
}

$downloadScript = Join-Path $env:TEMP "download_qwen_tts_models.py"
@'
import os
import sys
import shutil
from pathlib import Path

source = sys.argv[1]
model_target = Path(sys.argv[2])
tokenizer_target = Path(sys.argv[3])

model_repo = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
tokenizer_repo = "Qwen/Qwen3-TTS-Tokenizer-12Hz"

def clear_target(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)

def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

def download_hf(repo_id: str, target_dir: Path) -> None:
    from huggingface_hub import snapshot_download

    clear_target(target_dir)
    ensure_parent(target_dir)
    snapshot_download(repo_id=repo_id, local_dir=str(target_dir), local_dir_use_symlinks=False)

def download_modelscope(model_id: str, target_dir: Path) -> None:
    from modelscope.hub.snapshot_download import snapshot_download

    clear_target(target_dir)
    ensure_parent(target_dir)
    snapshot_download(model_id=model_id, local_dir=str(target_dir))

if source == "HuggingFace":
    download_hf(model_repo, model_target)
    download_hf(tokenizer_repo, tokenizer_target)
else:
    download_modelscope(model_repo, model_target)
    download_modelscope(tokenizer_repo, tokenizer_target)
'@ | Set-Content -Path $downloadScript -Encoding UTF8

Write-Host "Downloading Qwen TTS model from $Source ..."
Invoke-PythonCommand -PythonCommand $resolvedPython -Arguments @($downloadScript, $Source, $ModelTarget, $TokenizerTarget)

Assert-RequiredFiles -BasePath $ModelTarget -RequiredFiles @("config.json") -Label "Qwen model"
Assert-RequiredFiles -BasePath $TokenizerTarget -RequiredFiles @("config.json") -Label "Qwen tokenizer"

Write-Host "Qwen TTS model download complete:"
Write-Host "  Model: $ModelTarget"
Write-Host "  Tokenizer: $TokenizerTarget"
