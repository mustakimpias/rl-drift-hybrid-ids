# GitHub Update Commands

Safe day-to-day workflow for pushing future changes to this repository
without risking an accidental raw-dataset or large-file upload.

## 1. Check changed files

```bash
git status
```

## 2. See exact changes

```bash
git diff
```

## 3. Add only changed safe files manually

```bash
git add path/to/changed_file.py
git add configs/default.yaml
git add README.md
```

## 4. Or add all safe *tracked* changes at once

```bash
git add -u
```

`git add -u` (update) stages modifications and deletions to files **already
tracked** by git — it never picks up new/untracked files. This is the safe
default for routine updates: a freshly downloaded dataset or a new large
output file sitting in the working directory will never get staged by this
command, even if `.gitignore` doesn't happen to cover it.

## 5. Commit

```bash
git commit -m "Update thesis codebase"
```

## 6. Push

```bash
git push origin main
```

## 7. If adding a genuinely new safe file

Check `git status` first, confirm the new file is something you actually
want tracked (not a dataset, not a result folder, not a binary), then:

```bash
git add path/to/new_safe_file
git commit -m "Add new safe file"
git push origin main
```

## 8. Do not use `git add .` blindly

Only use `git add .` (or `git add -A`) if you have just reviewed
`.gitignore` and confirmed `git status` shows exactly the files you expect
— an untracked raw dataset, a stray `.venv/`, or a large result folder that
isn't yet covered by `.gitignore` would otherwise get staged silently.

## Emergency: unstage a file

If something gets staged that shouldn't be:

```bash
git restore --staged path/to/file
```

## Emergency: remove an accidentally tracked raw dataset

If `data/raw/` (or any other dataset) was ever committed before
`.gitignore` caught it, remove it from tracking (this keeps the local file
on disk, it just stops git from tracking it going forward):

```bash
git rm -r --cached data/raw
git commit -m "Stop tracking data/raw (should never have been committed)"
git push origin main
```

For a *fully* clean history (removing the data from past commits too, not
just going forward), you'd need `git filter-repo` or the BFG Repo-Cleaner
and a force-push — that's a much more invasive operation with real
consequences for anyone who already cloned the repo, so don't do it without
deliberately deciding it's necessary first.

## Scripted versions

`push_update.ps1` (PowerShell) and `push_update.bat` (Command Prompt) wrap
steps 1, 4, 5, and 6 above into one interactive script: they show
`git status`, ask you for a commit message, run `git add -u`, commit, and
push — without ever auto-adding untracked files.
