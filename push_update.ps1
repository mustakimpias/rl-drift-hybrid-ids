<#
    push_update.ps1

    Safe day-to-day update helper. Shows current git status, asks for a
    commit message, stages only already-tracked modified/deleted files
    (git add -u), commits, and pushes to origin main.

    Deliberately does NOT auto-add untracked files (git add . / git add -A)
    — this prevents an accidental dataset or large-file upload. To add a
    genuinely new safe file, do it explicitly first: git add path/to/file
#>

Write-Host "=== git status ===" -ForegroundColor Cyan
git status
if ($LASTEXITCODE -ne 0) {
    Write-Host "git status failed - is this a git repository?" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Review the changes above." -ForegroundColor Yellow
Write-Host "Untracked files listed under 'Untracked files:' will NOT be staged by this script." -ForegroundColor Yellow
$proceed = Read-Host "Proceed with 'git add -u' (stage tracked changes only)? [y/N]"
if ($proceed -ne "y" -and $proceed -ne "Y") {
    Write-Host "Aborted. Nothing was staged." -ForegroundColor Yellow
    exit 0
}

git add -u

Write-Host ""
Write-Host "=== Staged changes ===" -ForegroundColor Cyan
git diff --cached --name-only

$staged = git diff --cached --name-only
if (-not $staged) {
    Write-Host "Nothing staged (no tracked changes to commit). Exiting." -ForegroundColor Yellow
    exit 0
}

$message = Read-Host "Commit message"
if ([string]::IsNullOrWhiteSpace($message)) {
    Write-Host "Empty commit message - aborting." -ForegroundColor Red
    exit 1
}

git commit -m "$message"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Commit failed." -ForegroundColor Red
    exit 1
}

$push = Read-Host "Push to origin main now? [y/N]"
if ($push -eq "y" -or $push -eq "Y") {
    git push origin main
} else {
    Write-Host "Committed locally. Run 'git push origin main' when ready." -ForegroundColor Yellow
}
