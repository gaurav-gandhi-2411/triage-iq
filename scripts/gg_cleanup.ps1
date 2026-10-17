# gg_cleanup.ps1 -- GG runs this himself. Nothing here runs unless you answer Y.
#
# Step 1: remove the merged triage-iq worktrees (git worktree remove, NO --force).
# Step 2: docker builder prune -f   (build cache only)
# Step 3: docker image prune -f     (dangling images only; tagged images are never touched)
#
# Before each step it prints what it will remove and the space it will reclaim, then asks Y/N.
#
# Usage (PowerShell):  .\scripts\gg_cleanup.ps1
#
# Worktree list: the 29 merged worktrees queued in docs/QUEUE_FOR_GG.md (#199). The 30th name on that list,
# "integ", is deliberately NOT here: it is the worktree the vscode expansion recorder is running in
# (eval/expansions/vscode_2026-10-12/cassettes holds its checkpoint). Remove it only after the recording is
# finished and committed.

$ErrorActionPreference = 'Stop'
$Main = 'C:\Users\gaura\ml-projects\triage-iq'
$Names = @(
    'analysis', 'budget', 'cached', 'ckptval', 'decay', 'degrade', 'docs5', 'docs6', 'docs7', 'docs8',
    'docs9', 'embfix', 'emblog', 'evalb', 'guard', 'ladapt', 'lcore', 'lock', 'log', 'lrec',
    'manifest', 'sklearn', 'smoke', 'startup', 'startup2', 'startup3', 'vscodenaive', 'm140', 'p3ro'
)

function Confirm-Step([string]$Question) {
    while ($true) {
        $a = Read-Host "$Question [Y/N]"
        if ($a -match '^(y|yes)$') { return $true }
        if ($a -match '^(n|no)$') { return $false }
    }
}

function Get-DirMB([string]$Path) {
    $s = (Get-ChildItem -LiteralPath $Path -Recurse -File -Force -ErrorAction SilentlyContinue |
        Measure-Object Length -Sum).Sum
    if ($null -eq $s) { return 0 }
    return [math]::Round($s / 1MB, 1)
}

Set-Location $Main
git fetch -q origin

# ---------------------------------------------------------------- step 1: worktrees
Write-Host "`n=== STEP 1: remove merged worktrees (git worktree remove, no --force) ===" -ForegroundColor Cyan
$plan = @()
foreach ($n in $Names) {
    $p = "C:\Users\gaura\ml-projects\triage-iq-wt-$n"
    if (-not (Test-Path -LiteralPath $p)) { Write-Host ("  {0,-12} not on disk, nothing to do" -f $n); continue }
    $dirty = @(git -C $p status --porcelain 2>&1)
    if ($dirty.Count -gt 0) {
        Write-Host ("  {0,-12} SKIP: {1} uncommitted/untracked path(s)" -f $n, $dirty.Count) -ForegroundColor Yellow
        continue
    }
    git -C $p merge-base --is-ancestor HEAD origin/main 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host ("  {0,-12} SKIP: HEAD is not contained in origin/main" -f $n) -ForegroundColor Yellow
        continue
    }
    $mb = Get-DirMB $p
    Write-Host ("  {0,-12} clean, merged, {1,8} MB  {2}" -f $n, $mb, $p)
    $plan += [pscustomobject]@{ Name = $n; Path = $p; MB = $mb }
}
$total = [math]::Round(($plan | Measure-Object MB -Sum).Sum, 1)
Write-Host "`nWill remove $($plan.Count) worktree(s); reclaims about $total MB (the branches themselves are kept)."
if ($plan.Count -gt 0 -and (Confirm-Step 'Remove these worktrees?')) {
    foreach ($w in $plan) {
        git worktree remove $w.Path
        if ($LASTEXITCODE -eq 0) { Write-Host "  removed $($w.Name)" } else { Write-Host "  FAILED  $($w.Name) (left in place)" -ForegroundColor Red }
    }
} else {
    Write-Host 'Step 1 skipped.'
}

# ---------------------------------------------------------------- step 2: docker builder prune
Write-Host "`n=== STEP 2: docker builder prune -f (build cache) ===" -ForegroundColor Cyan
docker system df
Write-Host "`nThe 'Build Cache' row above is what this step removes (the RECLAIMABLE column is the space it frees)."
if (Confirm-Step 'Run docker builder prune -f?') { docker builder prune -f } else { Write-Host 'Step 2 skipped.' }

# ---------------------------------------------------------------- step 3: docker image prune (dangling only)
Write-Host "`n=== STEP 3: docker image prune -f (dangling images only; tagged images are never removed) ===" -ForegroundColor Cyan
$dangling = @(docker images --filter 'dangling=true' --format '{{.ID}}  {{.CreatedSince}}  {{.Size}}')
if ($dangling.Count -eq 0) {
    Write-Host 'No dangling images. Nothing to remove.'
} else {
    Write-Host "Dangling images that will be removed ($($dangling.Count)):"
    $dangling | ForEach-Object { Write-Host "  $_" }
    Write-Host "`nThe sizes above add up to the space reclaimed (shared layers may free less)."
    if (Confirm-Step 'Run docker image prune -f?') { docker image prune -f } else { Write-Host 'Step 3 skipped.' }
}

Write-Host "`nDone. Disk use now:"
docker system df
