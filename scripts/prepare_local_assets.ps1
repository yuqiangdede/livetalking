param(
    [string]$ParaformerSource = "",
    [string]$PuncSource = "",
    [string]$SherpaTtsSource = "",
    [string]$SherpaZhEnTtsSource = "",
    [string]$Wav2LipCheckpoint = "",
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
)

$ErrorActionPreference = "Stop"

function Ensure-Directory {
    param([string]$Path)
    New-Item -ItemType Directory -Force -Path $Path | Out-Null
}

function Copy-DirectoryIfProvided {
    param(
        [string]$Source,
        [string]$Target,
        [string]$Label
    )

    if ([string]::IsNullOrWhiteSpace($Source)) {
        Write-Host "Skip $Label: 未提供来源目录。"
        return
    }

    if (-not (Test-Path $Source)) {
        throw "$Label 来源不存在: $Source"
    }

    Ensure-Directory -Path $Target
    Copy-Item -Path (Join-Path $Source "*") -Destination $Target -Recurse -Force
    Write-Host "Copied $Label -> $Target"
}

function Copy-FileIfProvided {
    param(
        [string]$Source,
        [string]$Target,
        [string]$Label
    )

    if ([string]::IsNullOrWhiteSpace($Source)) {
        Write-Host "Skip $Label: 未提供来源文件。"
        return
    }

    if (-not (Test-Path $Source)) {
        throw "$Label 来源不存在: $Source"
    }

    Ensure-Directory -Path (Split-Path $Target -Parent)
    Copy-Item -Path $Source -Destination $Target -Force
    Write-Host "Copied $Label -> $Target"
}

function Assert-RequiredFiles {
    param(
        [string]$BasePath,
        [string[]]$RequiredFiles,
        [string]$Label
    )

    $missing = @()
    foreach ($item in $RequiredFiles) {
        $full = Join-Path $BasePath $item
        if (-not (Test-Path $full)) {
            $missing += $item
        }
    }

    if ($missing.Count -gt 0) {
        throw "$Label 缺少文件: $($missing -join ', ')"
    }

    Write-Host "$Label 校验通过: $BasePath"
}

$paraformerTarget = Join-Path $ProjectRoot "models\asr\speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
$puncTarget = Join-Path $ProjectRoot "models\asr\punc_ct-transformer_zh-cn-common-vocab272727-pytorch"
$sherpaTarget = Join-Path $ProjectRoot "models\tts\sherpa-onnx-vits-zh-ll"
$sherpaZhEnTarget = Join-Path $ProjectRoot "models\tts\vits-melo-tts-zh_en"
$wav2lipTarget = Join-Path $ProjectRoot "models\avatar\wav2lip256\wav2lip256.pth"
$runtimeTemp = Join-Path $ProjectRoot "runtime\tmp"

Ensure-Directory -Path $paraformerTarget
Ensure-Directory -Path $puncTarget
Ensure-Directory -Path $sherpaTarget
Ensure-Directory -Path $sherpaZhEnTarget
Ensure-Directory -Path (Split-Path $wav2lipTarget -Parent)
Ensure-Directory -Path $runtimeTemp

Copy-DirectoryIfProvided -Source $ParaformerSource -Target $paraformerTarget -Label "Paraformer"
Copy-DirectoryIfProvided -Source $PuncSource -Target $puncTarget -Label "Punctuation"
Copy-DirectoryIfProvided -Source $SherpaTtsSource -Target $sherpaTarget -Label "Sherpa TTS"
Copy-DirectoryIfProvided -Source $SherpaZhEnTtsSource -Target $sherpaZhEnTarget -Label "Sherpa zh/en TTS"
Copy-FileIfProvided -Source $Wav2LipCheckpoint -Target $wav2lipTarget -Label "Wav2Lip checkpoint"

Assert-RequiredFiles -BasePath $paraformerTarget -RequiredFiles @("model.pt", "config.yaml", "tokens.json") -Label "Paraformer"
Assert-RequiredFiles -BasePath $puncTarget -RequiredFiles @("model.int8.onnx", "config.yaml", "tokens.json") -Label "Punctuation"
Assert-RequiredFiles -BasePath $sherpaTarget -RequiredFiles @("model.onnx", "lexicon.txt", "tokens.txt") -Label "Sherpa TTS"
Assert-RequiredFiles -BasePath $sherpaZhEnTarget -RequiredFiles @("model.onnx", "lexicon.txt", "tokens.txt") -Label "Sherpa zh/en TTS"
Assert-RequiredFiles -BasePath (Split-Path $wav2lipTarget -Parent) -RequiredFiles @("wav2lip256.pth") -Label "Wav2Lip"

Write-Host "本地资源准备完成。"
