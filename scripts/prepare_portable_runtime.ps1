param(
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
    [string]$PythonExe = "",
    [string]$FfmpegBinDir = "",
    [switch]$SkipDependencyInstall,
    [switch]$IncludeQwenTts,
    [switch]$IncludeCoquiTts
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path $ProjectRoot).Path
$PortablePythonDir = Join-Path $ProjectRoot "python"
$PortableFfmpegBinDir = Join-Path $ProjectRoot "tools\ffmpeg\bin"
$TorchRequirements = Join-Path $ProjectRoot "requirements-torch-cu128.txt"
$AppRequirements = Join-Path $ProjectRoot "requirements.txt"
$QwenRequirements = Join-Path $ProjectRoot "requirements-qwen-tts.txt"
$CoquiRequirements = Join-Path $ProjectRoot "requirements-coqui-tts.txt"
$AppConfigPath = Join-Path $ProjectRoot "configs\app.yaml"
$PythonReadyMarker = Join-Path $PortablePythonDir ".portable-runtime-ready"
$DependenciesReadyMarker = Join-Path $PortablePythonDir ".portable-dependencies-ready"
$QwenDependenciesReadyMarker = Join-Path $PortablePythonDir ".portable-qwen-dependencies-ready"
$CoquiDependenciesReadyMarker = Join-Path $PortablePythonDir ".portable-coqui-dependencies-ready"
$FfmpegReadyMarker = Join-Path $PortableFfmpegBinDir ".portable-runtime-ready"
$QwenModelDir = Join-Path $ProjectRoot "models\tts\qwen\Qwen3-TTS-12Hz-0.6B-CustomVoice"
$QwenTokenizerDir = Join-Path $ProjectRoot "models\tts\qwen\Qwen3-TTS-Tokenizer-12Hz"
$CoquiModelDir = Join-Path $ProjectRoot "models\tts\coqui\xtts_v2"

function Ensure-Directory {
    param([string]$Path)
    New-Item -ItemType Directory -Force -Path $Path | Out-Null
}

function Invoke-RobocopyMirror {
    param(
        [string]$Source,
        [string]$Target
    )

    Ensure-Directory -Path $Target
    $arguments = @(
        $Source,
        $Target,
        "/MIR",
        "/R:1",
        "/W:1",
        "/NFL",
        "/NDL",
        "/NJH",
        "/NJS",
        "/NP",
        "/XD",
        "__pycache__"
    )
    & robocopy @arguments | Out-Null
    $exitCode = $LASTEXITCODE
    if ($exitCode -ge 8) {
        throw "robocopy failed: source=$Source target=$Target exit_code=$exitCode"
    }
}

function Test-PortablePythonReady {
    param([string]$PythonRoot)

    $pythonExe = Join-Path $PythonRoot "python.exe"
    $pythonDll = Join-Path $PythonRoot "python310.dll"
    $sitePackages = Join-Path $PythonRoot "Lib\site-packages"
    $marker = Join-Path $PythonRoot ".portable-runtime-ready"

    return (
        (Test-Path -Path $pythonExe -PathType Leaf) -and
        (Test-Path -Path $pythonDll -PathType Leaf) -and
        (Test-Path -Path $sitePackages -PathType Container) -and
        (Test-Path -Path $marker -PathType Leaf)
    )
}

function Test-PortablePythonFilesPresent {
    param([string]$PythonRoot)

    $pythonExe = Join-Path $PythonRoot "python.exe"
    $pythonDll = Join-Path $PythonRoot "python310.dll"
    $sitePackages = Join-Path $PythonRoot "Lib\site-packages"

    return (
        (Test-Path -Path $pythonExe -PathType Leaf) -and
        (Test-Path -Path $pythonDll -PathType Leaf) -and
        (Test-Path -Path $sitePackages -PathType Container)
    )
}

function Test-PortableFfmpegReady {
    param([string]$FfmpegRoot)

    $ffmpegExe = Join-Path $FfmpegRoot "ffmpeg.exe"
    $ffprobeExe = Join-Path $FfmpegRoot "ffprobe.exe"
    $marker = Join-Path $FfmpegRoot ".portable-runtime-ready"

    return (
        (Test-Path -Path $ffmpegExe -PathType Leaf) -and
        (Test-Path -Path $ffprobeExe -PathType Leaf) -and
        (Test-Path -Path $marker -PathType Leaf)
    )
}

function Test-PortableFfmpegFilesPresent {
    param([string]$FfmpegRoot)

    $ffmpegExe = Join-Path $FfmpegRoot "ffmpeg.exe"
    $ffprobeExe = Join-Path $FfmpegRoot "ffprobe.exe"

    return (
        (Test-Path -Path $ffmpegExe -PathType Leaf) -and
        (Test-Path -Path $ffprobeExe -PathType Leaf)
    )
}

function Test-PortableDependenciesReady {
    param([string]$PythonExe)

    if (-not (Test-Path -Path $DependenciesReadyMarker -PathType Leaf)) {
        return $false
    }

    $probe = & $PythonExe -c "import numpy, numba; print(numpy.__version__); print(numba.__version__)"
    if ($LASTEXITCODE -ne 0) {
        return $false
    }

    $versions = @($probe -split "`r?`n" | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
    return ($versions.Count -ge 2 -and $versions[0].Trim() -eq "1.22.0" -and $versions[1].Trim().Length -gt 0)
}

function Test-PortableDependenciesInstalled {
    param([string]$PythonExe)

    $probe = & $PythonExe -c "import numpy, numba; print(numpy.__version__); print(numba.__version__)"
    if ($LASTEXITCODE -ne 0) {
        return $false
    }

    $versions = @($probe -split "`r?`n" | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
    return ($versions.Count -ge 2 -and $versions[0].Trim() -eq "1.22.0" -and $versions[1].Trim().Length -gt 0)
}

function Test-QwenAssetsPresent {
    return (
        (Test-Path -Path $QwenModelDir -PathType Container) -and
        (Test-Path -Path $QwenTokenizerDir -PathType Container)
    )
}

function Test-QwenDependenciesReady {
    param([string]$PythonExe)

    if (-not (Test-Path -Path $QwenDependenciesReadyMarker -PathType Leaf)) {
        return $false
    }

    & $PythonExe -c "from qwen_tts import Qwen3TTSModel; print(Qwen3TTSModel.__name__)"
    return $LASTEXITCODE -eq 0
}

function Test-CoquiModelPresent {
    return Test-Path -Path $CoquiModelDir -PathType Container
}

function Test-CoquiDependenciesReady {
    param([string]$PythonExe)

    if (-not (Test-Path -Path $CoquiDependenciesReadyMarker -PathType Leaf)) {
        return $false
    }

    & $PythonExe -c "from TTS.api import TTS; print(TTS.__name__)"
    return $LASTEXITCODE -eq 0
}

function Get-ConfiguredTtsEngine {
    param([string]$ConfigPath)

    if (-not (Test-Path -Path $ConfigPath -PathType Leaf)) {
        return ""
    }

    $match = Select-String -Path $ConfigPath -Pattern '^\s*tts:\s*["'']?([^"'']+)["'']?\s*$' -ErrorAction Stop | Select-Object -First 1
    if ($null -eq $match) {
        return ""
    }

    return $match.Matches[0].Groups[1].Value.Trim()
}

function Resolve-DefaultPythonExe {
    if (-not [string]::IsNullOrWhiteSpace($PythonExe)) {
        return (Resolve-Path $PythonExe).Path
    }

    $venvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
    if (Test-Path -Path $venvPython -PathType Leaf) {
        return $venvPython
    }

    $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($pyLauncher) {
        try {
            $resolved = & $pyLauncher.Source -3.10 -c "import sys; print(sys.executable)"
            if ($LASTEXITCODE -eq 0 -and -not [string]::IsNullOrWhiteSpace($resolved)) {
                return $resolved.Trim()
            }
        } catch {
        }
    }

    $fallbacks = @()
    $localAppData = [Environment]::GetFolderPath("LocalApplicationData")
    if (-not [string]::IsNullOrWhiteSpace($localAppData)) {
        $fallbacks += (Join-Path $localAppData "Programs\Python\Python310\python.exe")
    }
    $fallbacks += "C:\Python310\python.exe"
    foreach ($candidate in $fallbacks) {
        if (Test-Path -Path $candidate -PathType Leaf) {
            return $candidate
        }
    }

    throw "Python 3.10 not found. Pass -PythonExe explicitly."
}

function Resolve-PythonInstallDir {
    param([string]$PythonExePath)

    try {
        $basePrefix = & $PythonExePath -c "import sys; print(sys.base_prefix)"
        if ($LASTEXITCODE -eq 0 -and -not [string]::IsNullOrWhiteSpace($basePrefix)) {
            $resolvedBasePrefix = Resolve-Path $basePrefix.Trim()
            if ($resolvedBasePrefix) {
                return $resolvedBasePrefix.Path
            }
        }
    } catch {
    }

    return (Split-Path -Parent $PythonExePath)
}

function Resolve-DefaultFfmpegBinDir {
    if (-not [string]::IsNullOrWhiteSpace($FfmpegBinDir)) {
        return (Resolve-Path $FfmpegBinDir).Path
    }

    $ffmpegCommand = Get-Command ffmpeg -ErrorAction SilentlyContinue
    if ($ffmpegCommand) {
        return Split-Path -Parent $ffmpegCommand.Source
    }

    $fallbacks = @(
        "D:\Program Files\ffmpeg\bin",
        "C:\Program Files\ffmpeg\bin"
    )
    foreach ($candidate in $fallbacks) {
        if (Test-Path -Path $candidate -PathType Container) {
            return $candidate
        }
    }

    throw "ffmpeg bin directory not found. Pass -FfmpegBinDir explicitly."
}

$resolvedPythonExe = Resolve-DefaultPythonExe
$pythonInstallDir = Resolve-PythonInstallDir -PythonExePath $resolvedPythonExe
if (-not (Test-Path -Path (Join-Path $pythonInstallDir "python310.dll") -PathType Leaf)) {
    throw "Python install root is missing python310.dll: $pythonInstallDir"
}

$resolvedFfmpegBinDir = Resolve-DefaultFfmpegBinDir
if (-not (Test-Path -Path (Join-Path $resolvedFfmpegBinDir "ffmpeg.exe") -PathType Leaf)) {
    throw "ffmpeg.exe not found in $resolvedFfmpegBinDir"
}
if (-not (Test-Path -Path (Join-Path $resolvedFfmpegBinDir "ffprobe.exe") -PathType Leaf)) {
    throw "ffprobe.exe not found in $resolvedFfmpegBinDir"
}

Write-Host "Portable Python source: $pythonInstallDir"
Write-Host "Portable ffmpeg source: $resolvedFfmpegBinDir"

$portablePythonExe = Join-Path $PortablePythonDir "python.exe"
if (Test-PortablePythonReady -PythonRoot $PortablePythonDir) {
    Write-Host "Portable Python already ready, skipping copy."
} elseif (Test-PortablePythonFilesPresent -PythonRoot $PortablePythonDir) {
    Write-Host "Portable Python files already present, creating ready marker."
    New-Item -ItemType File -Force -Path $PythonReadyMarker | Out-Null
} else {
    Invoke-RobocopyMirror -Source $pythonInstallDir -Target $PortablePythonDir
    if (-not (Test-PortablePythonFilesPresent -PythonRoot $PortablePythonDir)) {
        throw "Portable python.exe or required files not found after copy: $PortablePythonDir"
    }
    New-Item -ItemType File -Force -Path $PythonReadyMarker | Out-Null
}

if (Test-PortableFfmpegReady -FfmpegRoot $PortableFfmpegBinDir) {
    Write-Host "Portable ffmpeg already ready, skipping copy."
} elseif (Test-PortableFfmpegFilesPresent -FfmpegRoot $PortableFfmpegBinDir) {
    Write-Host "Portable ffmpeg files already present, creating ready marker."
    New-Item -ItemType File -Force -Path $FfmpegReadyMarker | Out-Null
} else {
    Invoke-RobocopyMirror -Source $resolvedFfmpegBinDir -Target $PortableFfmpegBinDir
    if (-not (Test-PortableFfmpegFilesPresent -FfmpegRoot $PortableFfmpegBinDir)) {
        throw "Portable ffmpeg files not found after copy: $PortableFfmpegBinDir"
    }
    New-Item -ItemType File -Force -Path $FfmpegReadyMarker | Out-Null
}

if (-not (Test-Path -Path $portablePythonExe -PathType Leaf)) {
    throw "Portable python.exe not found: $portablePythonExe"
}

if ($SkipDependencyInstall) {
    Write-Host "Dependency installation skipped by request."
} elseif (Test-PortableDependenciesReady -PythonExe $portablePythonExe) {
    Write-Host "Portable Python dependencies already ready, skipping install."
    if (-not (Test-Path -Path $DependenciesReadyMarker -PathType Leaf)) {
        New-Item -ItemType File -Force -Path $DependenciesReadyMarker | Out-Null
    }
} elseif (Test-PortableDependenciesInstalled -PythonExe $portablePythonExe) {
    Write-Host "Portable Python dependencies already installed, creating ready marker."
    New-Item -ItemType File -Force -Path $DependenciesReadyMarker | Out-Null
} else {
    & $portablePythonExe -m pip install --upgrade pip wheel
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to upgrade pip/wheel in portable runtime."
    }

    & $portablePythonExe -m pip install "setuptools<81"
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to pin setuptools in portable runtime."
    }

    & $portablePythonExe -m pip install -r $TorchRequirements
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install torch dependencies into portable runtime."
    }

    & $portablePythonExe -m pip install -r $AppRequirements
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install app dependencies into portable runtime."
    }

    & $portablePythonExe -m pip install --upgrade --force-reinstall --no-deps "numpy==1.22.0"
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to pin numpy in portable runtime."
    }

    New-Item -ItemType File -Force -Path $DependenciesReadyMarker | Out-Null
}

$configuredTts = Get-ConfiguredTtsEngine -ConfigPath $AppConfigPath

if ($IncludeQwenTts) {
    if ($configuredTts -ne "qwen3_customvoice") {
        throw "Qwen TTS runtime install was requested, but configs/app.yaml does not use qwen3_customvoice."
    }
    if ((Test-Path -Path $QwenRequirements -PathType Leaf) -and (Test-QwenAssetsPresent)) {
        if (Test-QwenDependenciesReady -PythonExe $portablePythonExe) {
            Write-Host "Qwen TTS dependencies already ready, skipping install."
        } else {
            Write-Host "Installing optional Qwen TTS dependencies..."
            & $portablePythonExe -m pip install -r $QwenRequirements
            if ($LASTEXITCODE -ne 0) {
                throw "Failed to install Qwen TTS dependencies into portable runtime."
            }
            New-Item -ItemType File -Force -Path $QwenDependenciesReadyMarker | Out-Null
        }
    } else {
        throw "Qwen TTS runtime install was requested, but Qwen model/tokenizer assets are missing."
    }
} elseif ($configuredTts -eq "qwen3_customvoice") {
    throw "configs/app.yaml uses qwen3_customvoice, but Qwen TTS is excluded by default from the portable runtime. Re-run with -IncludeQwenTts."
} else {
    Write-Host "Configured TTS is '$configuredTts', skip Qwen dependency install."
}

if ($IncludeCoquiTts) {
    if ($configuredTts -ne "coqui_xtts_v2") {
        throw "Coqui XTTS v2 runtime install was requested, but configs/app.yaml does not use coqui_xtts_v2."
    }
    if ((Test-Path -Path $CoquiRequirements -PathType Leaf) -and (Test-CoquiModelPresent)) {
        if (Test-CoquiDependenciesReady -PythonExe $portablePythonExe) {
            Write-Host "Coqui XTTS v2 dependencies already ready, skipping install."
        } else {
            Write-Host "Installing optional Coqui XTTS v2 dependencies..."
            & $portablePythonExe -m pip install -r $CoquiRequirements
            if ($LASTEXITCODE -ne 0) {
                throw "Failed to install Coqui XTTS v2 dependencies into portable runtime."
            }
            New-Item -ItemType File -Force -Path $CoquiDependenciesReadyMarker | Out-Null
        }
    } else {
        throw "Coqui XTTS v2 runtime install was requested, but the model directory is missing."
    }
} elseif ($configuredTts -eq "coqui_xtts_v2") {
    throw "configs/app.yaml uses coqui_xtts_v2, but Coqui XTTS v2 is excluded by default from the portable runtime. Re-run with -IncludeCoquiTts."
} else {
    Write-Host "Configured TTS is '$configuredTts', skip Coqui dependency install."
}

Write-Host "Portable runtime ready:"
Write-Host "  Python: $PortablePythonDir"
Write-Host "  ffmpeg: $PortableFfmpegBinDir"
