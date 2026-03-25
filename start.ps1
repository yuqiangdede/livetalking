param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$AppArgs
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path $PSScriptRoot).Path
Set-Location $ProjectRoot

$PortablePythonDir = Join-Path $ProjectRoot "python"
$PortablePython = Join-Path $PortablePythonDir "python.exe"
$PortablePythonScripts = Join-Path $PortablePythonDir "Scripts"
$PortableFfmpegBin = Join-Path $ProjectRoot "tools\ffmpeg\bin"
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$AppEntry = Join-Path $ProjectRoot "app.py"

function Test-Leaf {
    param([string]$Path)
    return Test-Path -Path $Path -PathType Leaf
}

function Prepend-PathEntry {
    param([string]$Path)
    if (-not (Test-Path -Path $Path)) {
        return
    }

    $entries = @($env:PATH -split ';') | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
    if ($entries -contains $Path) {
        return
    }
    $env:PATH = "$Path;$env:PATH"
}

function Ensure-RuntimeDirectories {
    $runtimeDirs = @(
        (Join-Path $ProjectRoot "runtime\logs"),
        (Join-Path $ProjectRoot "runtime\tmp")
    )

    foreach ($dir in $runtimeDirs) {
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
        $keepFile = Join-Path $dir ".gitkeep"
        if (-not (Test-Path -Path $keepFile -PathType Leaf)) {
            New-Item -ItemType File -Force -Path $keepFile | Out-Null
        }
    }
}

function Test-AppArgPresent {
    param([string]$Name)

    foreach ($arg in $AppArgs) {
        if ($arg -eq $Name) {
            return $true
        }
        if ($arg.StartsWith("$Name=")) {
            return $true
        }
    }
    return $false
}

Ensure-RuntimeDirectories
Prepend-PathEntry -Path $PortablePythonDir
Prepend-PathEntry -Path $PortablePythonScripts
Prepend-PathEntry -Path $PortableFfmpegBin

$runtimePython = $null
if (Test-Leaf $PortablePython) {
    $runtimePython = $PortablePython
} elseif (Test-Leaf $VenvPython) {
    $runtimePython = $VenvPython
} else {
    throw "Python runtime not found. Expected python\python.exe or .venv\Scripts\python.exe."
}

if (-not (Test-AppArgPresent "--batch_size")) {
    $AppArgs = @("--batch_size", "4") + @($AppArgs)
}

& $runtimePython $AppEntry @AppArgs
exit $LASTEXITCODE
