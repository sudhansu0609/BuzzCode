param([string]$Dir = (Join-Path $PSScriptRoot "repo-tmp"))
Set-Location $Dir
if (-not (Select-String -Path src\lib.rs -Pattern 'pub fn in_stock\(&self\) -> Vec<&str>' -Quiet)) { Write-Host 'signature changed'; exit 1 }
cargo test -q 2>&1 | Out-Null
exit $LASTEXITCODE
