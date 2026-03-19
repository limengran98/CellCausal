#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REMOTE_URL="${GIT_REMOTE_URL:-git@github.com:limengran98/CellCausal.git}"
DEFAULT_BRANCH="${GIT_UPLOAD_BRANCH:-main}"
COMMIT_MESSAGE="${1:-chore: sync code $(TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M:%S CST')}"


CODE_PATHS=(
  ".gitignore"
  "README.md"
  "LOGGING_SYSTEM.md"
  "run_interactive.py"
  "run_pipeline.py"
  "push_code_only.sh"
  "cellscientist"
  "configs"
  "docs"
  "prompts"
  "tests"
)

UNWANTED_PATHS=(
  "data"
  "results"
  ".pytest_cache"
)

cd "$ROOT_DIR"

if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  CURRENT_BRANCH="$(git symbolic-ref --quiet --short HEAD 2>/dev/null || true)"
  TARGET_BRANCH="${CURRENT_BRANCH:-$DEFAULT_BRANCH}"
else
  TARGET_BRANCH="$DEFAULT_BRANCH"
  git init
  git symbolic-ref HEAD "refs/heads/$TARGET_BRANCH"
fi

if ! git remote get-url origin >/dev/null 2>&1; then
  git remote add origin "$REMOTE_URL"
else
  git remote set-url origin "$REMOTE_URL"
fi

GIT_USER_NAME="$(git config user.name || true)"
GIT_USER_EMAIL="$(git config user.email || true)"

if [[ -z "$GIT_USER_NAME" || -z "$GIT_USER_EMAIL" ]]; then
  echo "Git identity is not configured yet."
  echo "Run these once, then rerun ./push_code_only.sh:"
  echo '  git config --global user.name "limengran98"'
  echo '  git config --global user.email "1127147088@qq.com"'
  echo
  echo "If you prefer repo-only config, remove --global."
  exit 1
fi

git rm -r --cached --ignore-unmatch -- "${UNWANTED_PATHS[@]}" >/dev/null 2>&1 || true
find . -type d -name "__pycache__" -prune -exec git rm -r --cached --ignore-unmatch {} + >/dev/null 2>&1 || true
find . -type f \( -name "*.pyc" -o -name "*.pyo" \) -exec git rm --cached --ignore-unmatch {} + >/dev/null 2>&1 || true

git add --all -- "${CODE_PATHS[@]}"

if git diff --cached --quiet; then
  echo "No code changes to commit."
  exit 0
fi

git commit -m "$COMMIT_MESSAGE"
git push -u origin "$TARGET_BRANCH"

echo "Code-only push finished on branch: $TARGET_BRANCH"
