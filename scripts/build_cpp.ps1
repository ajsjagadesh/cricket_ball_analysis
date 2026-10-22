param(
    [string]$Python = "",
    [string]$BuildType = "Release",
    [string]$Generator = "Ninja",
    [string]$VsDevCmd = ""
)

$ErrorActionPreference = 'Stop'

if (-not $Python) {
    $Python = (Get-Command python).Source
}

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$buildDir = Join-Path $projectRoot 'build'
$sourceDir = Join-Path $projectRoot 'cpp'

if (-not $VsDevCmd) {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (Test-Path $vswhere) {
        $installationPath = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        if ($LASTEXITCODE -eq 0 -and $installationPath) {
            $VsDevCmd = Join-Path $installationPath 'Common7\Tools\VsDevCmd.bat'
        }
    }
}

if (-not $VsDevCmd -or -not (Test-Path $VsDevCmd)) {
    throw 'Visual Studio C++ Build Tools were not found. Supply -VsDevCmd with the path to VsDevCmd.bat.'
}

if (-not (Test-Path $sourceDir)) {
    throw "C++ source directory not found: $sourceDir"
}

if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
    throw 'CMake is required but was not found on PATH.'
}

& $Python -c 'import pybind11' 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host 'pybind11 not installed; installing it into the active environment.'
    & $Python -m pip install pybind11
    if ($LASTEXITCODE -ne 0) { throw 'Failed to install pybind11.' }
}

$pybindDir = & $Python -m pybind11 --cmakedir
if ($LASTEXITCODE -ne 0 -or -not $pybindDir) {
    throw 'Unable to locate pybind11 CMake files.'
}

New-Item -ItemType Directory -Path $buildDir -Force | Out-Null
$buildCommand = 'call "{0}" -arch=x64 && cmake -S "{1}" -B "{2}" -G "{3}" -DCMAKE_BUILD_TYPE={4} -DPython_EXECUTABLE="{5}" -DPYBIND11_FINDPYTHON=ON -Dpybind11_DIR="{6}" && cmake --build "{2}" --config {4}' -f `
    $VsDevCmd, $sourceDir, $buildDir, $Generator, $BuildType, $Python, $pybindDir
& $env:ComSpec /c $buildCommand
if ($LASTEXITCODE -ne 0) { throw 'C++ extension build failed.' }

Write-Host "Build completed. The extension is in cricket_vision\cpp."
