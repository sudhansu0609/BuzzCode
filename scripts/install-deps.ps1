<#
.SYNOPSIS
  Installs / verifies the toolchain needed to build llama.cpp with CUDA for buzzcode.

  - cmake + ninja (via winget)
  - ripgrep (via winget) - used by the grep tool
  - verifies a usable CUDA toolkit (13.1 or >= 13.3; 13.2 is flagged as buggy by Unsloth for Qwen3.x)
  - verifies MSVC build tools (VS 2022 / VS 18 BuildTools)

  Run from an elevated or normal PowerShell:  powershell -ExecutionPolicy Bypass -File scripts\install-deps.ps1
#>
[CmdletBinding()]
param(
    [switch]$SkipWinget
)

$ErrorActionPreference = "Stop"

function Test-Cmd($name) { $null -ne (Get-Command $name -ErrorAction SilentlyContinue) }

Write-Host "== buzzcode: dependency check ==" -ForegroundColor Cyan

if (-not $SkipWinget) {
    if (-not (Test-Cmd cmake)) { Write-Host "Installing CMake..."; winget install --id Kitware.CMake -e --accept-source-agreements --accept-package-agreements }
    if (-not (Test-Cmd ninja)) { Write-Host "Installing Ninja..."; winget install --id Ninja-build.Ninja -e --accept-source-agreements --accept-package-agreements }
    if (-not (Test-Cmd rg))    { Write-Host "Installing ripgrep..."; winget install --id BurntSushi.ripgrep.MSVC -e --accept-source-agreements --accept-package-agreements }
}

# Refresh PATH for this session (winget adds to machine/user PATH but not the current process)
$env:PATH = [System.Environment]::GetEnvironmentVariable("PATH", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("PATH", "User")
if (Test-Path "C:\Program Files\CMake\bin") { $env:PATH = "C:\Program Files\CMake\bin;" + $env:PATH }

$ok = $true
foreach ($t in @("cmake", "ninja", "git", "rg")) {
    if (Test-Cmd $t) { Write-Host ("  [ok]   {0,-8} {1}" -f $t, (Get-Command $t).Source) -ForegroundColor Green }
    else { Write-Host ("  [MISSING] {0}" -f $t) -ForegroundColor Red; $ok = $false }
}

# --- CUDA ---------------------------------------------------------------
$cudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
$good = @()
$bad = @()
if (Test-Path $cudaRoot) {
    foreach ($d in Get-ChildItem $cudaRoot -Directory) {
        if ($d.Name -match '^v(\d+)\.(\d+)$') {
            $maj = [int]$Matches[1]; $min = [int]$Matches[2]
            $ver = [version]"$maj.$min"
            $nvcc = Join-Path $d.FullName "bin\nvcc.exe"
            if (-not (Test-Path $nvcc)) { continue }
            # Blackwell sm_120 requires CUDA >= 12.8. 13.2 is flagged (Unsloth: "avoid 13.2; use below 13.2 or 13.3+").
            if ($ver -lt [version]"12.8") { $bad += "$($d.Name) (too old for sm_120)"; continue }
            if ($ver -eq [version]"13.2") { $bad += "$($d.Name) (flagged buggy for Qwen3.x - prefer 13.1 or 13.3+)"; continue }
            $good += $d.FullName
        }
    }
}
if ($good.Count -gt 0) {
    Write-Host "  [ok]   CUDA toolkit(s): $($good -join ', ')" -ForegroundColor Green
} else {
    Write-Host "  [WARN] No preferred CUDA toolkit found. Present: $($bad -join ', ')" -ForegroundColor Yellow
    Write-Host "         Install CUDA 13.1 or 13.3+ side-by-side from https://developer.nvidia.com/cuda-toolkit-archive" -ForegroundColor Yellow
    Write-Host "         (build-llama.ps1 will fall back to 13.2 with -AllowCuda132; run 'buzzcode engine doctor' to validate output coherence)" -ForegroundColor Yellow
}

# --- MSVC ---------------------------------------------------------------
$vcvars = @(@(
    "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
    "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat",
    "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat",
    "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
) | Where-Object { Test-Path $_ })
if ($vcvars.Count -gt 0) { Write-Host "  [ok]   MSVC: $($vcvars[0])" -ForegroundColor Green }
else { Write-Host "  [MISSING] MSVC vcvars64.bat (install 'Desktop development with C++' via Visual Studio Installer)" -ForegroundColor Red; $ok = $false }

# --- GPU ----------------------------------------------------------------
if (Test-Cmd nvidia-smi) {
    $gpu = nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv,noheader
    Write-Host "  [ok]   GPU: $gpu" -ForegroundColor Green
} else { Write-Host "  [WARN] nvidia-smi not found" -ForegroundColor Yellow }

if ($ok) { Write-Host "All required tools present. Next: scripts\build-llama.ps1" -ForegroundColor Cyan; exit 0 }
else { Write-Host "Some tools are missing - fix the above and re-run." -ForegroundColor Red; exit 1 }
