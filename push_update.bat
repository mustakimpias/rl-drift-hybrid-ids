@echo off
REM push_update.bat
REM
REM Safe day-to-day update helper. Shows current git status, asks for a
REM commit message, stages only already-tracked modified/deleted files
REM (git add -u), commits, and pushes to origin main.
REM
REM Deliberately does NOT auto-add untracked files (git add . / git add -A)
REM - this prevents an accidental dataset or large-file upload. To add a
REM genuinely new safe file, do it explicitly first: git add path\to\file

echo === git status ===
git status
if errorlevel 1 (
    echo git status failed - is this a git repository?
    exit /b 1
)

echo.
echo Review the changes above.
echo Untracked files listed under "Untracked files:" will NOT be staged by this script.
set /p PROCEED="Proceed with 'git add -u' (stage tracked changes only)? [y/N]: "
if /i not "%PROCEED%"=="y" (
    echo Aborted. Nothing was staged.
    exit /b 0
)

git add -u

echo.
echo === Staged changes ===
git diff --cached --name-only

git diff --cached --name-only > "%TEMP%\push_update_staged.txt"
for %%A in ("%TEMP%\push_update_staged.txt") do set STAGED_SIZE=%%~zA
if "%STAGED_SIZE%"=="0" (
    echo Nothing staged (no tracked changes to commit^). Exiting.
    del "%TEMP%\push_update_staged.txt"
    exit /b 0
)
del "%TEMP%\push_update_staged.txt"

set /p MESSAGE="Commit message: "
if "%MESSAGE%"=="" (
    echo Empty commit message - aborting.
    exit /b 1
)

git commit -m "%MESSAGE%"
if errorlevel 1 (
    echo Commit failed.
    exit /b 1
)

set /p DOPUSH="Push to origin main now? [y/N]: "
if /i "%DOPUSH%"=="y" (
    git push origin main
) else (
    echo Committed locally. Run "git push origin main" when ready.
)
