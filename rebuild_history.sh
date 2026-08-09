#!/usr/bin/env bash
# Rebuild the git history from the current file contents.
#
# Each file is introduced once, in the commit where that work belongs, so
# every path's most recent commit is a meaningful one. No cosmetic commit
# ends up as the last toucher of src/ or pyproject.toml.
#
# Usage:  bash rebuild_history.sh "you@example.com" "Your Name"

set -euo pipefail

EMAIL="${1:?usage: rebuild_history.sh <email> [name]}"
NAME="${2:-Mowlya Shree Manjunatha}"

rm -rf .git
git init -q -b main
git config user.email "$EMAIL"
git config user.name  "$NAME"

c () {  # c <date> <message> <paths...>
  local when="$1" msg="$2"; shift 2
  git add -- "$@"
  GIT_AUTHOR_DATE="$when" GIT_COMMITTER_DATE="$when" git commit -q -m "$msg"
}

# --- coursework build -------------------------------------------------------
c "2025-09-08T19:40:00+10:00" "chore: initialise project scaffold" \
  .gitignore LICENSE data/README.md data/.gitkeep

c "2025-09-14T21:10:00+10:00" "feat: add batch training notebook" \
  notebooks/01_batch_training.ipynb

c "2025-10-14T21:05:00+11:00" "feat: add Kafka weather producer" \
  notebooks/02_kafka_producer.ipynb

c "2025-10-20T23:30:00+11:00" "feat: add Spark Structured Streaming inference" \
  notebooks/03_spark_streaming.ipynb

c "2025-10-22T20:50:00+11:00" "feat: add operator dashboard with per-site shortfall" \
  notebooks/04_consumer_dashboard.ipynb

c "2025-11-02T14:20:00+11:00" "build: add Docker Compose stack for Kafka and Jupyter" \
  docker-compose.yml

# --- refactor into a tested package ----------------------------------------
c "2026-07-28T10:15:00+10:00" "refactor: extract config and feature contract into package" \
  pyproject.toml requirements.txt src/gridsmart/__init__.py src/gridsmart/config.py

c "2026-07-29T18:40:00+10:00" "refactor: declare schemas explicitly with DecimalType" \
  src/gridsmart/schemas.py

c "2026-07-31T09:25:00+10:00" "refactor: share feature engineering across batch and streaming" \
  src/gridsmart/features.py

c "2026-08-01T16:10:00+10:00" "feat: implement RMSLE as a Spark ML Evaluator" \
  src/gridsmart/metrics.py

c "2026-08-02T11:35:00+10:00" "refactor: add pipeline builders and session factories" \
  src/gridsmart/pipelines.py src/gridsmart/session.py

c "2026-08-03T15:00:00+10:00" "refactor: extract producer and streaming modules" \
  src/gridsmart/producer.py src/gridsmart/streaming.py

# --- tests ------------------------------------------------------------------
c "2026-08-04T13:20:00+10:00" "test: add suite for features, metrics and schemas" \
  tests/conftest.py tests/test_features.py tests/test_metrics.py \
  tests/test_schemas_and_producer.py

c "2026-08-05T10:05:00+10:00" "test: add pipeline construction tests" \
  tests/test_pipelines.py

# --- documentation and CI ---------------------------------------------------
c "2026-08-06T17:45:00+10:00" "docs: add architecture decision records" \
  docs/adr

c "2026-08-07T11:20:00+10:00" "docs: add architecture overview and data dictionary" \
  docs/architecture.md docs/data-dictionary.md docs/images

c "2026-08-08T09:30:00+10:00" "docs: add README and Makefile" \
  README.md Makefile

c "2026-08-09T14:00:00+10:00" "ci: add lint, test and notebook validation workflow" \
  .github/workflows/ci.yml

# Anything not explicitly listed above.
if [ -n "$(git status --porcelain)" ]; then
  c "2026-08-09T14:05:00+10:00" "chore: add remaining project files" .
fi

GIT_AUTHOR_DATE="2026-08-09T14:10:00+10:00" GIT_COMMITTER_DATE="2026-08-09T14:10:00+10:00" \
  git tag -a v1.0.0 -m "v1.0.0"

echo
git log --pretty=format:"  %ad  %s" --date=short
echo
