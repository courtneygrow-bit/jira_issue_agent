#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/Users/courtney.grow/Projects/jira_issue_agent"
LAUNCH_AGENTS_DIR="$HOME/Library/LaunchAgents"
PLIST_PATH="$LAUNCH_AGENTS_DIR/com.halodi.joint_level_dbq.report.plist"
LOG_OUT="/tmp/joint_level_dbq_report.log"
LOG_ERR="/tmp/joint_level_dbq_report.err"

HOUR="${1:-7}"
MINUTE="${2:-0}"

mkdir -p "$LAUNCH_AGENTS_DIR"
chmod +x "$PROJECT_DIR/scripts/push_joint_level_dbq_report.sh"

cat > "$PLIST_PATH" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.halodi.joint_level_dbq.report</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>$PROJECT_DIR/scripts/push_joint_level_dbq_report.sh</string>
  </array>
  <key>WorkingDirectory</key>
  <string>$PROJECT_DIR</string>
  <key>StartCalendarInterval</key>
  <dict>
    <key>Hour</key>
    <integer>$HOUR</integer>
    <key>Minute</key>
    <integer>$MINUTE</integer>
  </dict>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <dict>
    <key>NetworkState</key>
    <true/>
  </dict>
  <key>StandardOutPath</key>
  <string>$LOG_OUT</string>
  <key>StandardErrorPath</key>
  <string>$LOG_ERR</string>
</dict>
</plist>
EOF

launchctl unload "$PLIST_PATH" >/dev/null 2>&1 || true
launchctl load "$PLIST_PATH"

echo "Installed launchd auto-update for Joint Level DBQ report:"
echo "  plist: $PLIST_PATH"
echo "  schedule: daily at $(printf '%02d:%02d' "$HOUR" "$MINUTE") local time"
echo "  page id env override: JOINT_LEVEL_DBQ_PAGE_ID (default: 5691277356)"
echo "  logs: $LOG_OUT / $LOG_ERR"
echo
echo "Use these commands:"
echo "  launchctl list | grep com.halodi.joint_level_dbq.report"
echo "  launchctl kickstart -k gui/\$(id -u)/com.halodi.joint_level_dbq.report"
echo "  tail -n 100 $LOG_OUT"
echo "  tail -n 100 $LOG_ERR"

