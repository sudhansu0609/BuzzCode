param([string]$Dir = (Join-Path $PSScriptRoot "repo-tmp"))
Set-Location $Dir
# tests must be untouched
$orig = Get-Content (Join-Path $PSScriptRoot "repo\test_textstats.py") -Raw
$now = Get-Content "test_textstats.py" -Raw
if ($orig -ne $now) { Write-Host "tests were modified"; exit 1 }
python -m pytest -q 2>&1 | Out-Null
exit $LASTEXITCODE
