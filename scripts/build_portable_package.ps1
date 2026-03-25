param(
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
    [string]$StageDir = "",
    [string]$ZipPath = "",
    [string]$LogPath = "",
    [switch]$SkipZip
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path $ProjectRoot).Path
$DistRoot = Join-Path $ProjectRoot "dist"
$PackageName = "LiveTalking-portable-win64-nvidia-cu128"
$script:StepIndex = 0
$script:StepCount = if ($SkipZip) { 5 } else { 6 }

if ([string]::IsNullOrWhiteSpace($StageDir)) {
    $StageDir = Join-Path $DistRoot $PackageName
}
if ([string]::IsNullOrWhiteSpace($ZipPath)) {
    $ZipPath = Join-Path $DistRoot "$PackageName.zip"
}
if ([string]::IsNullOrWhiteSpace($LogPath)) {
    $LogPath = Join-Path $ProjectRoot "runtime\logs\build_portable_package.log"
}

function Ensure-Directory {
    param([string]$Path)
    New-Item -ItemType Directory -Force -Path $Path | Out-Null
}

Ensure-Directory -Path (Split-Path $LogPath -Parent)

function Write-BuildLog {
    param(
        [string]$Message,
        [string]$Level = "INFO"
    )

    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[$timestamp] [$Level] $Message"
    Write-Host $line
    Add-Content -Path $LogPath -Value $line -Encoding UTF8
}

function Start-BuildStep {
    param([string]$Message)
    $script:StepIndex++
    Write-BuildLog -Message ("[{0}/{1}] {2}" -f $script:StepIndex, $script:StepCount, $Message)
}

function Format-Size {
    param([Int64]$Bytes)

    if ($Bytes -lt 1KB) { return "$Bytes B" }
    if ($Bytes -lt 1MB) { return ("{0:N2} KB" -f ($Bytes / 1KB)) }
    if ($Bytes -lt 1GB) { return ("{0:N2} MB" -f ($Bytes / 1MB)) }
    return ("{0:N2} GB" -f ($Bytes / 1GB))
}

function Copy-DirectoryContent {
    param(
        [string]$Source,
        [string]$Target
    )

    if (-not (Test-Path -Path $Source -PathType Container)) {
        throw "Portable package build failed: missing directory $Source"
    }

    Ensure-Directory -Path $Target
    Copy-Item -Path (Join-Path $Source "*") -Destination $Target -Recurse -Force
}

function Copy-FileIfExists {
    param(
        [string]$Source,
        [string]$Target
    )

    if (-not (Test-Path -Path $Source -PathType Leaf)) {
        throw "Portable package build failed: missing file $Source"
    }

    Ensure-Directory -Path (Split-Path $Target -Parent)
    Copy-Item -Path $Source -Destination $Target -Force
}

function Remove-JunkArtifacts {
    param([string]$Root)

    Get-ChildItem -Path $Root -Directory -Recurse -Force -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -in @("__pycache__", ".git", ".idea") } |
        Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

    Get-ChildItem -Path $Root -File -Recurse -Force -ErrorAction SilentlyContinue |
        Where-Object { $_.Extension -in @(".pyc", ".pyo", ".log") } |
        Remove-Item -Force -ErrorAction SilentlyContinue
}

function Reset-RuntimeDirectories {
    param([string]$Root)

    foreach ($relative in @("runtime\logs", "runtime\tmp")) {
        $full = Join-Path $Root $relative
        if (Test-Path -Path $full) {
            Get-ChildItem -Path $full -Force -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
        } else {
            Ensure-Directory -Path $full
        }
        New-Item -ItemType File -Force -Path (Join-Path $full ".gitkeep") | Out-Null
    }
}

function Assert-Leaf {
    param(
        [string]$Path,
        [string]$Label
    )

    if (-not (Test-Path -Path $Path -PathType Leaf)) {
        throw "Portable package check failed: missing $Label ($Path)"
    }
}

function Assert-DirectoryHasFiles {
    param(
        [string]$Path,
        [string]$Label
    )

    if (-not (Test-Path -Path $Path -PathType Container)) {
        throw "Portable package check failed: missing $Label ($Path)"
    }

    $firstFile = Get-ChildItem -Path $Path -File -Recurse -Force -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($null -eq $firstFile) {
        throw "Portable package check failed: $Label is empty ($Path)"
    }
}

function Get-DefaultAvatarId {
    param([string]$ConfigPath)

    $match = Select-String -Path $ConfigPath -Pattern '^\s*avatar_id:\s*["'']?([^"'']+)["'']?\s*$' -ErrorAction Stop | Select-Object -First 1
    if ($null -eq $match) {
        throw "Portable package check failed: could not find avatar_id in $ConfigPath"
    }
    return $match.Matches[0].Groups[1].Value.Trim()
}

function Start-CompressionProgressJob {
    param(
        [string]$ZipFile,
        [datetime]$StartedAt
    )

    return Start-Job -ScriptBlock {
        param($Path, $StartTime)
        while ($true) {
            Start-Sleep -Seconds 5
            if (Test-Path -Path $Path -PathType Leaf) {
                $size = (Get-Item $Path).Length
                $elapsed = (Get-Date) - $StartTime
                [PSCustomObject]@{
                    Elapsed = $elapsed.ToString("hh\:mm\:ss")
                    SizeBytes = [Int64]$size
                }
            } else {
                $elapsed = (Get-Date) - $StartTime
                [PSCustomObject]@{
                    Elapsed = $elapsed.ToString("hh\:mm\:ss")
                    SizeBytes = [Int64]0
                }
            }
        }
    } -ArgumentList $ZipFile, $StartedAt
}

function Stop-CompressionProgressJob {
    param($Job)

    if ($null -eq $Job) {
        return
    }

    Stop-Job -Job $Job -ErrorAction SilentlyContinue | Out-Null
    Remove-Job -Job $Job -Force -ErrorAction SilentlyContinue | Out-Null
}

function Write-PendingCompressionProgress {
    param($Job)

    if ($null -eq $Job) {
        return
    }

    $updates = Receive-Job -Job $Job -ErrorAction SilentlyContinue
    foreach ($update in $updates) {
        $sizeText = Format-Size -Bytes $update.SizeBytes
        Write-BuildLog -Message ("[zip] elapsed={0} size={1}" -f $update.Elapsed, $sizeText)
    }
}

function Compress-Package {
    param(
        [string]$SourceRoot,
        [string]$DirectoryName,
        [string]$DestinationZip
    )

    if (Test-Path -Path $DestinationZip) {
        Remove-Item -Path $DestinationZip -Force
    }

    $startedAt = Get-Date
    $progressJob = Start-CompressionProgressJob -ZipFile $DestinationZip -StartedAt $startedAt

    try {
        $tarCommand = Get-Command tar -ErrorAction SilentlyContinue
        if ($tarCommand) {
            Write-BuildLog -Message "Compressing portable package with tar.exe ..."
            $tarJob = Start-Job -ScriptBlock {
                param($TarExe, $ZipFile, $Root, $DirName)
                & $TarExe -a -cf $ZipFile -C $Root $DirName
                exit $LASTEXITCODE
            } -ArgumentList $tarCommand.Source, $DestinationZip, $SourceRoot, $DirectoryName

            while ($tarJob.State -eq "Running") {
                Start-Sleep -Seconds 5
                Write-PendingCompressionProgress -Job $progressJob
                $tarJob = Get-Job -Id $tarJob.Id
            }
            Write-PendingCompressionProgress -Job $progressJob
            $jobResult = Receive-Job -Job $tarJob -Keep -ErrorAction SilentlyContinue
            $tarExitCode = 0
            if ($tarJob.ChildJobs.Count -gt 0) {
                $tarExitCode = $tarJob.ChildJobs[0].JobStateInfo.Reason.HResult
                if ($null -eq $tarJob.ChildJobs[0].JobStateInfo.Reason) {
                    $tarExitCode = 0
                }
            }
            Remove-Job -Job $tarJob -Force -ErrorAction SilentlyContinue | Out-Null
            if ($tarExitCode -ne 0 -and -not (Test-Path -Path $DestinationZip -PathType Leaf)) {
                throw "tar.exe failed to create zip: $DestinationZip"
            }
            return
        }

        Write-BuildLog -Message "tar.exe not found, falling back to Compress-Archive ..."
        $compressJob = Start-Job -ScriptBlock {
            param($Path, $Destination)
            Compress-Archive -Path $Path -DestinationPath $Destination -Force
        } -ArgumentList (Join-Path $SourceRoot $DirectoryName), $DestinationZip

        while ($compressJob.State -eq "Running") {
            Start-Sleep -Seconds 5
            Write-PendingCompressionProgress -Job $progressJob
            $compressJob = Get-Job -Id $compressJob.Id
        }
        Write-PendingCompressionProgress -Job $progressJob
        Receive-Job -Job $compressJob -ErrorAction Stop | Out-Null
        Remove-Job -Job $compressJob -Force -ErrorAction SilentlyContinue | Out-Null
    } finally {
        Stop-CompressionProgressJob -Job $progressJob
    }
}

"" | Set-Content -Path $LogPath -Encoding UTF8
Write-BuildLog -Message "Portable package build started."

if (Test-Path -Path $StageDir) {
    Write-BuildLog -Message "Removing existing stage directory: $StageDir"
    Remove-Item -Path $StageDir -Recurse -Force
}
Ensure-Directory -Path $StageDir
Write-BuildLog -Message "Stage directory: $StageDir"

$topLevelFiles = @(
    "app.py",
    "LICENSE",
    "README.md",
    "pyproject.toml",
    "requirements.txt",
    "requirements-torch-cu128.txt",
    "start.ps1",
    "start.bat",
    "start_nvidia_gpu.bat"
)

$topLevelDirs = @(
    "livetalking",
    "configs",
    "docs",
    "scripts",
    "tools",
    "web",
    "wav2lip",
    "models",
    "data",
    "python"
)

Start-BuildStep -Message "Copy top-level files"
foreach ($file in $topLevelFiles) {
    Write-BuildLog -Message "Copy file: $file"
    Copy-FileIfExists -Source (Join-Path $ProjectRoot $file) -Target (Join-Path $StageDir $file)
}

Start-BuildStep -Message "Copy project directories"
foreach ($dir in $topLevelDirs) {
    Write-BuildLog -Message "Copy directory: $dir"
    Copy-DirectoryContent -Source (Join-Path $ProjectRoot $dir) -Target (Join-Path $StageDir $dir)
}

Start-BuildStep -Message "Reset runtime directories and clean junk files"
Ensure-Directory -Path (Join-Path $StageDir "runtime\logs")
Ensure-Directory -Path (Join-Path $StageDir "runtime\tmp")
Reset-RuntimeDirectories -Root $StageDir
Remove-JunkArtifacts -Root $StageDir

Start-BuildStep -Message "Resolve default avatar from config"
$defaultAvatarId = Get-DefaultAvatarId -ConfigPath (Join-Path $ProjectRoot "configs\app.yaml")
Write-BuildLog -Message "Default avatar: $defaultAvatarId"

$requiredChecks = @(
    "app.py",
    "start.ps1",
    "start.bat",
    "start_nvidia_gpu.bat",
    "configs\app.yaml",
    "web\index.html",
    "livetalking\app.py",
    "python\python.exe",
    "python\python310.dll",
    "tools\ffmpeg\bin\ffmpeg.exe",
    "tools\ffmpeg\bin\ffprobe.exe"
)

$requiredDirectories = @(
    @{ Path = "python\Lib"; Label = "Bundled Python Lib directory" },
    @{ Path = "python\Scripts"; Label = "Bundled Python Scripts directory" },
    @{ Path = "python\Lib\site-packages"; Label = "Bundled Python site-packages" },
    @{ Path = "models\avatar\wav2lip256"; Label = "Wav2Lip model directory" },
    @{ Path = "models\asr\speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"; Label = "Paraformer model directory" },
    @{ Path = "models\asr\punc_ct-transformer_zh-cn-common-vocab272727-pytorch"; Label = "Punctuation model directory" },
    @{ Path = "models\tts\sherpa-onnx-vits-zh-ll"; Label = "Sherpa Chinese TTS directory" },
    @{ Path = "models\tts\vits-melo-tts-zh_en"; Label = "Sherpa zh/en TTS directory" },
    @{ Path = "data\avatars"; Label = "Avatar root directory" },
    @{ Path = "data\avatars\$defaultAvatarId"; Label = "Default avatar directory" },
    @{ Path = "data\avatars\$defaultAvatarId\full_imgs"; Label = "Default avatar full_imgs directory" },
    @{ Path = "data\avatars\$defaultAvatarId\face_imgs"; Label = "Default avatar face_imgs directory" }
)

$requiredFiles = @(
    @{ Path = "models\avatar\wav2lip256\wav2lip256.pth"; Label = "Wav2Lip checkpoint" },
    @{ Path = "models\asr\speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch\model.pt"; Label = "Paraformer model.pt" },
    @{ Path = "models\asr\speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch\config.yaml"; Label = "Paraformer config.yaml" },
    @{ Path = "models\asr\speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch\tokens.json"; Label = "Paraformer tokens.json" },
    @{ Path = "models\asr\punc_ct-transformer_zh-cn-common-vocab272727-pytorch\model.int8.onnx"; Label = "Punctuation model.int8.onnx" },
    @{ Path = "models\asr\punc_ct-transformer_zh-cn-common-vocab272727-pytorch\config.yaml"; Label = "Punctuation config.yaml" },
    @{ Path = "models\asr\punc_ct-transformer_zh-cn-common-vocab272727-pytorch\tokens.json"; Label = "Punctuation tokens.json" },
    @{ Path = "models\tts\sherpa-onnx-vits-zh-ll\model.onnx"; Label = "Chinese TTS model.onnx" },
    @{ Path = "models\tts\sherpa-onnx-vits-zh-ll\lexicon.txt"; Label = "Chinese TTS lexicon.txt" },
    @{ Path = "models\tts\sherpa-onnx-vits-zh-ll\tokens.txt"; Label = "Chinese TTS tokens.txt" },
    @{ Path = "models\tts\vits-melo-tts-zh_en\model.onnx"; Label = "zh/en TTS model.onnx" },
    @{ Path = "models\tts\vits-melo-tts-zh_en\lexicon.txt"; Label = "zh/en TTS lexicon.txt" },
    @{ Path = "models\tts\vits-melo-tts-zh_en\tokens.txt"; Label = "zh/en TTS tokens.txt" },
    @{ Path = "data\avatars\$defaultAvatarId\coords.pkl"; Label = "Default avatar coords.pkl" }
)

Start-BuildStep -Message "Validate bundled runtime and assets"
foreach ($item in $requiredChecks) {
    Assert-Leaf -Path (Join-Path $StageDir $item) -Label $item
}
foreach ($dirInfo in $requiredDirectories) {
    Assert-DirectoryHasFiles -Path (Join-Path $StageDir $dirInfo.Path) -Label $dirInfo.Label
}
foreach ($fileInfo in $requiredFiles) {
    Assert-Leaf -Path (Join-Path $StageDir $fileInfo.Path) -Label $fileInfo.Label
}

Start-BuildStep -Message "Ensure dist root exists"
Ensure-Directory -Path $DistRoot

if (-not $SkipZip) {
    Start-BuildStep -Message "Create zip package"
    Compress-Package -SourceRoot $DistRoot -DirectoryName $PackageName -DestinationZip $ZipPath
}

Write-BuildLog -Message "Portable package staged at: $StageDir"
if ($SkipZip) {
    Write-BuildLog -Message "Portable zip skipped by -SkipZip."
} else {
    Write-BuildLog -Message "Portable zip created at: $ZipPath"
}
Write-BuildLog -Message "Default avatar validated: $defaultAvatarId"
Write-BuildLog -Message "Portable package build completed."
