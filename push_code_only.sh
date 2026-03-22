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
  "evals"
  "prompts"
  "records/README.md"
  "records/paper"
  "skills"
  "tests"
  "references/enzyme_mining/README.md"
  "references/enzyme_mining/EBI.ipynb"
  "references/enzyme_mining/JGI.ipynb"
  "references/enzyme_mining/挖酶.ipynb"
  "references/enzyme_mining/CataPro-master/CataPro-master/LICENSE"
  "references/enzyme_mining/CataPro-master/CataPro-master/README.md"
  "references/enzyme_mining/CataPro-master/CataPro-master/inference"
  "references/enzyme_mining/CataPro-master/CataPro-master/training"
)

UNWANTED_PATHS=(
  "data"
  "results"
  "evals/results"
  "records/system"
  "records/tasks"
  "records/data"
  "records/evals"
  "records/notebooks"
  "records/ablations"
  "references/enzyme_mining/output_sequences.zip"
  "references/enzyme_mining/CataPro-master/CataPro-master/models"
  "references/enzyme_mining/CataPro-master/CataPro-master/datasets"
  "references/enzyme_mining/CataPro-master/CataPro-master/samples"
  "references/enzyme_mining/CataPro-master/CataPro-master/inference/catapro_test-pred.csv"
  ".pytest_cache"
)

DISALLOWED_FILE_PATTERNS=(
  "*.pth"
  "*.pt"
  "*.ckpt"
  "*.bin"
  "*.safetensors"
  "*.zip"
  "*.h5"
  "*.hdf5"
  "*.pkl"
  "*.pickle"
  "*.npy"
  "*.npz"
)

MAX_PUSH_FILE_BYTES=$((50 * 1024 * 1024))

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

clean_unwanted_index_entries() {
  git rm -r --cached --ignore-unmatch -- "${UNWANTED_PATHS[@]}" >/dev/null 2>&1 || true
  find . -type d -name "__pycache__" -prune -exec git rm -r --cached --ignore-unmatch {} + >/dev/null 2>&1 || true
  find . -type f \( -name "*.pyc" -o -name "*.pyo" \) -exec git rm --cached --ignore-unmatch {} + >/dev/null 2>&1 || true
  find . -type f \( \
    -name "*.pth" -o \
    -name "*.pt" -o \
    -name "*.ckpt" -o \
    -name "*.bin" -o \
    -name "*.safetensors" -o \
    -name "*.zip" -o \
    -name "*.h5" -o \
    -name "*.hdf5" -o \
    -name "*.pkl" -o \
    -name "*.pickle" -o \
    -name "*.npy" -o \
    -name "*.npz" \
  \) -exec git rm --cached --ignore-unmatch {} + >/dev/null 2>&1 || true
}

path_is_disallowed() {
  local path="$1"
  local unwanted
  local pattern

  for unwanted in "${UNWANTED_PATHS[@]}"; do
    if [[ "$path" == "$unwanted" || "$path" == "$unwanted/"* ]]; then
      return 0
    fi
  done

  for pattern in "${DISALLOWED_FILE_PATTERNS[@]}"; do
    if [[ "$path" == $pattern ]]; then
      return 0
    fi
  done

  return 1
}

history_has_disallowed_push_content() {
  local remote_ref="origin/$TARGET_BRANCH"
  local path
  local size

  git rev-parse --verify "$remote_ref" >/dev/null 2>&1 || return 1

  while IFS= read -r path; do
    [[ -z "$path" ]] && continue

    if path_is_disallowed "$path"; then
      echo "$path"
      return 0
    fi

    size="$(git cat-file -s "HEAD:$path" 2>/dev/null || echo 0)"
    if [[ "$size" =~ ^[0-9]+$ ]] && (( size > MAX_PUSH_FILE_BYTES )); then
      echo "$path"
      return 0
    fi
  done < <(git diff --name-only "$remote_ref..HEAD")

  return 1
}

clean_unwanted_index_entries
git add --all -- "${CODE_PATHS[@]}"
clean_unwanted_index_entries

if git diff --cached --quiet; then
  echo "No code changes to commit."
  exit 0
fi

git commit -m "$COMMIT_MESSAGE"

BLOCKED_PATH="$(history_has_disallowed_push_content || true)"
if [[ -n "${BLOCKED_PATH:-}" ]]; then
  echo "Push blocked before upload."
  echo "A disallowed or oversized file is already present in local commits ahead of origin:"
  echo "  $BLOCKED_PATH"
  echo
  echo "This script now excludes these files, but your local history still contains them."
  echo "Recommended one-time cleanup:"
  echo "  git fetch origin main"
  echo "  git branch backup_pre_clean_push_$(date +%Y%m%d_%H%M%S)"
  echo "  git reset --soft origin/main"
  echo "  ./push_code_only.sh"
  exit 1
fi

git push -u origin "$TARGET_BRANCH"

echo "Code-only push finished on branch: $TARGET_BRANCH"
