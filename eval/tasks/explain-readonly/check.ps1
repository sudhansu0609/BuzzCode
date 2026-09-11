param([string]$Dir = (Join-Path $PSScriptRoot "repo-tmp"))
# pass = no file changed + final answer mentions the test name and "removed"/"available"/"stock"
$orig = Get-Content (Join-Path $PSScriptRoot "repo\src\lib.rs") -Raw
$now = Get-Content (Join-Path $Dir "src\lib.rs") -Raw
if ($orig -ne $now) { Write-Host "file was modified"; exit 1 }
$log = Get-Content (Join-Path $PSScriptRoot "last-run.jsonl") -Raw -ErrorAction SilentlyContinue
if (-not $log) { exit 1 }
$final = ($log -split "`n" | Where-Object { $_ -match '"type":"assistant_message"' } | Select-Object -Last 1)
if ($final -match 'add_remove_total' -and ($final -match 'actually removed|amount removed|available|in stock|min')) { exit 0 }
Write-Host "final answer missing expected content: $final"
exit 1
