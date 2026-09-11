<#
.SYNOPSIS
  Clone, pin and build llama.cpp (llama-server + llama-bench) with CUDA for Blackwell (sm_120).

.PARAMETER Dir        Where to clone/build llama.cpp           (default B:\llama\llama.cpp)
.PARAMETER Pin        Git tag or commit to build               (default: latest tag on master; must be >= b10450 for Qwen3.8)
.PARAMETER CudaPath   CUDA toolkit root                        (default: best available non-13.2 toolkit)
.PARAMETER CudaArch   CMAKE_CUDA_ARCHITECTURES                 (default 120 = RTX 50xx)
.PARAMETER AllowCuda132  Permit building with the flagged 13.2 toolkit
.PARAMETER Clean      Wipe the build directory first

  Usage:  powershell -ExecutionPolicy Bypass -File scripts\build-llama.ps1 -Pin b10480
#>
[CmdletBinding()]
param(
    [string]$Dir = "B:\llama\llama.cpp",
    [string]$Pin = "",
    [string]$CudaPath = "",
    [string]$CudaArch = "120",
    [switch]$AllowCuda132,
    [switch]$AllowUnsupportedCompiler,
    [switch]$Clean,
    [int]$Jobs = 20
)

$ErrorActionPreference = "Stop"
$MinBuild = 10450

function Resolve-Cuda {
    param([string]$Requested)
    if ($Requested) { return $Requested }
    $root = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    $cands = Get-ChildItem $root -Directory | Where-Object { $_.Name -match '^v(\d+)\.(\d+)$' } | ForEach-Object {
        [pscustomobject]@{ Path = $_.FullName; Ver = [version]"$([int]$Matches[1]).$([int]$Matches[2])" }
    } | Where-Object { $_.Ver -ge [version]"12.8" -and (Test-Path (Join-Path $_.Path "bin\nvcc.exe")) }
    $preferred = $cands | Where-Object { $_.Ver -ne [version]"13.2" } | Sort-Object Ver -Descending | Select-Object -First 1
    if ($preferred) { return $preferred.Path }
    $v132 = $cands | Where-Object { $_.Ver -eq [version]"13.2" } | Select-Object -First 1
    if ($v132 -and $AllowCuda132) {
        Write-Warning "Using CUDA 13.2 (flagged by Unsloth). Validate with 'buzzcode engine doctor' (coherence test)."
        return $v132.Path
    }
    throw "No suitable CUDA toolkit (>= 12.8, not 13.2). Install CUDA 13.1 or 13.3+, or pass -AllowCuda132."
}

function Find-VcVars {
    $paths = @(
        "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
        "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat",
        "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat",
        "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat",
        "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
    )
    foreach ($p in $paths) { if (Test-Path $p) { return $p } }
    throw "vcvars64.bat not found - install MSVC C++ build tools."
}

# Locate tools without bloating PATH (vcvars64.bat + a long user PATH overflows cmd's 8191-char limit).
function Find-Tool($name, $extraDirs) {
    $c = Get-Command $name -ErrorAction SilentlyContinue
    if ($c) { return $c.Source }
    foreach ($d in $extraDirs) { $p = Join-Path $d "$name.exe"; if (Test-Path $p) { return $p } }
    $wg = Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Packages"
    if (Test-Path $wg) { $hit = Get-ChildItem $wg -Recurse -Filter "$name.exe" -ErrorAction SilentlyContinue | Select-Object -First 1; if ($hit) { return $hit.FullName } }
    throw "$name not found - run scripts\install-deps.ps1"
}
$cmakeExe = Find-Tool "cmake" @("C:\Program Files\CMake\bin")
$ninjaExe = Find-Tool "ninja" @()
$gitExe   = Find-Tool "git" @("C:\Program Files\Git\cmd")
$cmakeDir = Split-Path -Parent $cmakeExe
$ninjaDir = Split-Path -Parent $ninjaExe
$gitDir   = Split-Path -Parent $gitExe

$CudaPath = Resolve-Cuda $CudaPath
$vcvars = Find-VcVars
Write-Host "== llama.cpp build ==" -ForegroundColor Cyan
Write-Host "  dir     : $Dir"
Write-Host "  cuda    : $CudaPath"
Write-Host "  arch    : $CudaArch"
Write-Host "  vcvars  : $vcvars"

# --- clone / pin --------------------------------------------------------
if (-not (Test-Path (Join-Path $Dir ".git"))) {
    New-Item -ItemType Directory -Force -Path (Split-Path $Dir) | Out-Null
    git clone https://github.com/ggml-org/llama.cpp $Dir
}
git -C $Dir fetch --tags origin
if (-not $Pin) {
    $Pin = (git -C $Dir tag --list "b*" --sort=-v:refname | Select-Object -First 1)
    Write-Host "  pin     : $Pin (latest tag)"
} else { Write-Host "  pin     : $Pin" }
if ($Pin -match '^b(\d+)$' -and [int]$Matches[1] -lt $MinBuild) {
    throw "Pin $Pin is older than b$MinBuild; Qwen3.8 DeltaNet CUDA kernels were broken before that."
}
git -C $Dir checkout --quiet $Pin
$commit = git -C $Dir rev-parse --short HEAD
Write-Host "  commit  : $commit"

# --- configure + build (inside vcvars env) -------------------------------
$build = Join-Path $Dir "build"
if ($Clean -and (Test-Path $build)) { Remove-Item -Recurse -Force $build }

$cudaFlags = ""
if ($AllowUnsupportedCompiler) { $cudaFlags = "-allow-unsupported-compiler" }

$cmakeArgs = @(
    "-S", "`"$Dir`"", "-B", "`"$build`"", "-G", "Ninja",
    "-DCMAKE_BUILD_TYPE=Release",
    "-DGGML_CUDA=ON",
    "-DCMAKE_CUDA_ARCHITECTURES=$CudaArch",
    "-DCMAKE_CUDA_COMPILER=`"$CudaPath\bin\nvcc.exe`"",
    "-DGGML_CUDA_FA_ALL_QUANTS=ON",
    "-DGGML_NATIVE=ON",
    "-DLLAMA_BUILD_TESTS=OFF",
    "-DLLAMA_BUILD_EXAMPLES=OFF",
    "-DLLAMA_BUILD_TOOLS=ON",
    "-DLLAMA_BUILD_SERVER=ON",
    "-DLLAMA_CURL=OFF"
)
if ($cudaFlags) { $cmakeArgs += "-DCMAKE_CUDA_FLAGS=`"$cudaFlags`"" }

# cmd.exe has an 8191-char line limit, so drive the build from a small batch file with a MINIMAL PATH.
$bat = Join-Path $env:TEMP "buzzcode-build-llama.cmd"
$minPath = "$CudaPath\bin;$cmakeDir;$ninjaDir;$gitDir;$env:SystemRoot\system32;$env:SystemRoot;$env:SystemRoot\System32\Wbem;$env:SystemRoot\System32\WindowsPowerShell\v1.0"
@"
@echo off
set "PATH=$minPath"
call "$vcvars"
if errorlevel 1 exit /b 1
set "CUDA_PATH=$CudaPath"
cmake $($cmakeArgs -join ' ')
if errorlevel 1 exit /b 1
cmake --build "$build" --config Release --target llama-server llama-bench llama-cli -j $Jobs
"@ | Out-File -Encoding ascii $bat
Write-Host "  running : $bat" -ForegroundColor DarkGray
Get-Content $bat | ForEach-Object { Write-Host "    $_" -ForegroundColor DarkGray }
& cmd /c $bat
if ($LASTEXITCODE -ne 0) {
    if (-not $AllowUnsupportedCompiler) {
        Write-Warning "Build failed. If nvcc complained about the MSVC host compiler version, re-run with -AllowUnsupportedCompiler."
    }
    throw "llama.cpp build failed (exit $LASTEXITCODE)"
}

# --- locate + record ------------------------------------------------------
$server = Get-ChildItem $build -Recurse -Filter "llama-server.exe" | Select-Object -First 1
if (-not $server) { throw "llama-server.exe not produced" }
$info = @"
# written by scripts/build-llama.ps1
dir = "$($Dir -replace '\\','/')"
pin = "$Pin"
commit = "$commit"
cuda_path = "$($CudaPath -replace '\\','/')"
cuda_arch = "$CudaArch"
allow_unsupported_compiler = $($AllowUnsupportedCompiler.IsPresent.ToString().ToLower())
server = "$($server.FullName -replace '\\','/')"
built_at = "$(Get-Date -Format o)"
"@
$infoDir = Join-Path $env:USERPROFILE ".buzzcode\engine"
New-Item -ItemType Directory -Force -Path $infoDir | Out-Null
$info | Out-File -Encoding utf8 (Join-Path $infoDir "build-info.toml")
Write-Host "Built: $($server.FullName)" -ForegroundColor Green
Write-Host "Recorded: $infoDir\build-info.toml" -ForegroundColor Green
& $server.FullName --version
