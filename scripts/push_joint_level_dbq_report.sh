#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/Users/courtney.grow/Projects/jira_issue_agent"
cd "$PROJECT_DIR"

source .venv/bin/activate
set -a
source .env
set +a

PYTHON_BIN="$PROJECT_DIR/.venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing venv python at $PYTHON_BIN"
  exit 1
fi

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig}"
mkdir -p "$MPLCONFIGDIR"

export JIRA_EMAIL="${JIRA_EMAIL:-${ATLASSIAN_EMAIL:-}}"
export JIRA_API_TOKEN="${JIRA_API_TOKEN:-${ATLASSIAN_API_TOKEN:-}}"
if [[ -z "${JIRA_EMAIL:-}" || -z "${JIRA_API_TOKEN:-}" ]]; then
  echo "Missing credentials. Set JIRA_EMAIL/JIRA_API_TOKEN or ATLASSIAN_EMAIL/ATLASSIAN_API_TOKEN in .env"
  exit 2
fi

JIRA_URL_USE="${JIRA_URL:-https://halodi.atlassian.net/jira/software/projects/DQB/list}"
CONFLUENCE_BASE_USE="${CONFLUENCE_BASE_URL:-https://halodi.atlassian.net/wiki}"
DBQ_PAGE_ID_USE="${JOINT_LEVEL_DBQ_PAGE_ID:-5691277356}"
TIMEOUT_SEC="${JOINT_LEVEL_DBQ_TIMEOUT_SEC:-180}"
FETCH_RETRIES="${JOINT_LEVEL_DBQ_FETCH_RETRIES:-8}"
BACKOFF_SEC="${JOINT_LEVEL_DBQ_RETRY_BACKOFF_SEC:-3}"

HOST_CHECK="halodi.atlassian.net"
LAST_SUCCESS_FILE="/tmp/joint_level_dbq_last_success.txt"
TODAY_LOCAL="$(date +%Y-%m-%d)"

if [[ "${FORCE_RUN:-0}" != "1" ]] && [[ -f "$LAST_SUCCESS_FILE" ]] && [[ "$(cat "$LAST_SUCCESS_FILE" 2>/dev/null || true)" == "$TODAY_LOCAL" ]]; then
  echo "Joint Level DBQ report already published today ($TODAY_LOCAL). Skipping."
  exit 0
fi

wait_for_dns() {
  local attempts="${1:-8}"
  local sleep_s="${2:-15}"
  local i=1
  while [[ $i -le $attempts ]]; do
    local resolved
    resolved="$(/usr/bin/dig +short "$HOST_CHECK" | head -n 1 || true)"
    if [[ -n "$resolved" ]]; then
      echo "DNS check OK: $HOST_CHECK -> $resolved"
      return 0
    fi
    echo "DNS not ready for $HOST_CHECK (attempt $i/$attempts). Retrying in ${sleep_s}s..."
    sleep "$sleep_s"
    i=$((i + 1))
  done
  echo "DNS check failed for $HOST_CHECK after $attempts attempts."
  return 1
}

retry_cmd() {
  local max_attempts="${1:-3}"
  local sleep_s="${2:-20}"
  shift 2
  local n=1
  while true; do
    if "$@"; then
      return 0
    fi
    if [[ $n -ge $max_attempts ]]; then
      echo "Command failed after $n attempts: $*"
      return 1
    fi
    echo "Command failed (attempt $n/$max_attempts). Retrying in ${sleep_s}s..."
    sleep "$sleep_s"
    n=$((n + 1))
  done
}

wait_for_dns 8 15

echo "Publishing Joint Level DBQ report to Confluence page ${DBQ_PAGE_ID_USE}..."
retry_cmd 3 30 "$PYTHON_BIN" run_joint_level_dbq_report.py \
  --jira-url "$JIRA_URL_USE" \
  --publish-confluence \
  --confluence-base "$CONFLUENCE_BASE_USE" \
  --page-id "$DBQ_PAGE_ID_USE" \
  --timeout-sec "$TIMEOUT_SEC" \
  --fetch-retries "$FETCH_RETRIES" \
  --retry-backoff-sec "$BACKOFF_SEC" \
  "$@"

echo "$TODAY_LOCAL" > "$LAST_SUCCESS_FILE"
echo "Joint Level DBQ report publish complete."

