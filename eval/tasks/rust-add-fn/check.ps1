# exit 0 = pass. Runs inside the task's working copy (repo-tmp).
param([string]$Dir = (Join-Path $PSScriptRoot "repo-tmp"))
Set-Location $Dir
if (-not (Select-String -Path src\lib.rs -Pattern 'pub fn clamp' -Quiet)) { Write-Host 'no clamp fn'; exit 1 }
cargo test -q 2>&1 | Out-Null
exit $LASTEXITCODE
