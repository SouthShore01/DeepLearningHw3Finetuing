#!/usr/bin/env bash
set -euo pipefail

# Resolve all merge conflicts by keeping the current branch version (final version).
conflicted_files=$(git diff --name-only --diff-filter=U)

if [[ -z "${conflicted_files}" ]]; then
  echo "No conflicted files found."
  exit 0
fi

echo "Resolving conflicts by keeping current branch versions:"
printf '%s\n' "$conflicted_files"

while IFS= read -r file; do
  [[ -z "$file" ]] && continue
  git checkout --ours -- "$file"
  git add "$file"
done <<< "$conflicted_files"

echo "All conflicts resolved with current branch versions."
echo "Now run: git commit -m 'Resolve conflicts by keeping final version'"
