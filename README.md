# Jira Similarity Agent

This agent:
1. Pulls Jira issues from a project/JQL query
2. Categorizes issues into high-level bins (Rope, PCBA, Harness, Fastener, Test, Bearing, Drum, Structural, plus additional generalized categories when needed)
3. Uses deterministic hybrid similarity matching to build granular sub-groups inside each high-level category
4. Produces one Excel dashboard sheet with:
   - overall issue trend
   - overall Pareto by high-level category
   - per-category trend and per-category granular Pareto
5. Produces chart PNG images and can publish them to a Confluence page as images (no embedded Excel)

## Setup

```bash
cd jira_issue_agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set credentials:

```bash
export JIRA_EMAIL="your-email@company.com"
export JIRA_API_TOKEN="your-jira-api-token"
```

You can generate Jira API token here:
https://id.atlassian.com/manage-profile/security/api-tokens

## Run

```bash
python agent.py \
  --jira-url "https://halodi.atlassian.net/jira/software/projects/DQB/list?sortBy=status&direction=ASC" \
  --threshold 0.84 \
  --trend-freq W \
  --pareto-top-n 20 \
  --output-dir output
```

Optional stricter matching:

```bash
python agent.py --jira-url "https://halodi.atlassian.net/jira/software/projects/DQB/list?sortBy=status&direction=ASC" --threshold 0.90
```

## Outputs

- `output/issue_analysis_dashboard.xlsx`: single-sheet dashboard with overall + per-category charts/tables
- `output/issues_clustered.csv`: each issue with high-level category and granular subcluster
- `output/category_subcluster_summary.csv`: subcluster counts within each high-level category
- `output/overall_trend.csv`: underlying trend data for all issues
- `output/charts/*.png`: chart images for overall and each category
- `output/charts_manifest.json`: image manifest used for Confluence publishing

## Publish PNG Charts To Confluence

This updates page `5657494052` with a managed dashboard section and uploads images as attachments.

```bash
python publish_confluence.py \
  --confluence-base "https://halodi.atlassian.net/wiki" \
  --page-id "5657494052" \
  --output-dir output
```

The publisher keeps all existing page content and only replaces the managed block:
- `<!-- DQB_DASHBOARD_START --> ... <!-- DQB_DASHBOARD_END -->`

## Daily Auto-Update (macOS launchd)

1. Test the dashboard push script once:

```bash
bash /Users/courtney.grow/Projects/jira_issue_agent/scripts/push_dqb_dashboard.sh
```

2. Install launchd auto-update (default: daily 07:00 local):

```bash
bash /Users/courtney.grow/Projects/jira_issue_agent/scripts/install_dqb_autoupdate.sh
```

Optional custom time (hour minute):

```bash
bash /Users/courtney.grow/Projects/jira_issue_agent/scripts/install_dqb_autoupdate.sh 8 30
```

3. Verify and run immediately:

```bash
launchctl list | grep com.halodi.dqb.dashboard
launchctl kickstart -k gui/$(id -u)/com.halodi.dqb.dashboard
tail -n 100 /tmp/dqb_dashboard.log
tail -n 100 /tmp/dqb_dashboard.err
```

## Notes on matching accuracy

- Matching is deterministic (no stochastic temperature effects)
- Hybrid score combines:
  - title TF-IDF cosine similarity
  - description TF-IDF cosine similarity
  - title token-set fuzzy similarity
- Increase `--threshold` for higher precision and fewer false positives

## Weekly Rope Failure Report To Slack

This flow fetches DQB Jira issues and filters to issues where `rope` appears in:
- summary
- description
- labels

It then posts a weekly summary to Slack, uploads a weekly trend chart (opened vs resolved), uploads a Pareto chart, and includes a table of new issues created in the last week.

### Configure env vars

Set these in `.env` (or export in your shell):

```bash
JIRA_EMAIL="your-email@company.com"
JIRA_API_TOKEN="your-jira-api-token"

# Choose one Slack publish method:
SLACK_WEBHOOK_URL="https://hooks.slack.com/services/..."
# or
SLACK_BOT_TOKEN="xoxb-..."
SLACK_CHANNEL_ID="C0AAH7QPF6F"

REPORT_TIMEZONE="America/Los_Angeles"
```

### Dry run (no Slack post)

```bash
python run_weekly_rope_report.py --dry-run --output-dir output
```

Preview output:
- `output/weekly_rope_report_preview.txt`

### Live run (posts to Slack message + 2 charts)

```bash
python run_weekly_rope_report.py --live --output-dir output
```

Or use wrapper script:

```bash
bash /Users/courtney.grow/Projects/jira_issue_agent/scripts/run_weekly_rope_report.sh
```

### Weekly schedule on macOS launchd

Install default schedule (Monday at 08:00 local):

```bash
bash /Users/courtney.grow/Projects/jira_issue_agent/scripts/install_weekly_rope_report.sh
```

Optional custom schedule:
- args: `weekday hour minute`
- weekday: `0=Sunday ... 6=Saturday`

```bash
bash /Users/courtney.grow/Projects/jira_issue_agent/scripts/install_weekly_rope_report.sh 1 9 30
```

## Weekly Design Issues Report (Email)

This flow fetches DQB Jira issues and scopes to issues assigned to the engineering roster. It generates:
- Weekly trend chart (opened vs resolved)
- Pareto chart (dashboard categories; generalized tags removed in labels)
- New issues table for last week (includes impacted robot system field data)
- Ageing issues table (`>30` days without workflow status change, includes impacted robot system field data)

### Generate report artifacts (dry run)

```bash
python run_weekly_design_report.py --output-dir output
```

Outputs:
- `output/weekly_design_report_preview.txt`
- `output/charts/design_opened_resolved_trend.png`
- `output/charts/design_issue_type_pareto.png`
- `output/charts/design_new_issues_table.png`
- `output/charts/design_ageing_issues_table.png`

Optional:
- `--ageing-days 30` (default)
- `--jql "project = DQB ORDER BY created DESC"` to override default filter

### Build email preview (dry run)

```bash
python send_design_report_email.py --dry-run
```

Output:
- `output/design_report_email_preview.eml`

### Create Gmail draft (not sent)

```bash
python send_design_report_email.py --draft
```

### Send live email

```bash
python send_design_report_email.py --live
```

Optional recipient controls:

## Quality Data Agent (Odoo via BigQuery)

Use this to generate:
- Station defect-per-unit (DPU): failures at QC points divided by units completed at each station.
- MRB inventory snapshot: total units in MRB plus part/serial breakdown.

Setup and usage:

```bash
python3 -m pip install -r requirements.txt
python3 run_quality_data_report.py --config quality_data_agent/config/quality_report_config.yaml
```

See details in:
- `quality_data_agent/README.md`
- `--recipients "person1@1x.tech,person2@1x.tech"`
- `--no-default-recipients` to disable built-in engineering roster recipients

## Import OKR Initiatives To Jira (QTP)

Create one Jira issue per initiative listed in column C of a Google Sheet export (or live Google Sheet).

### Dry run from Excel export

```bash
python import_okr_initiatives_to_jira.py \
  --jira-base-url "https://halodi.atlassian.net" \
  --project-key "QTP" \
  --issue-type "Task" \
  --xlsx-path "/Users/courtney.grow/Downloads/OKR Sandbox (1).xlsx" \
  --sheet-name "Updated" \
  --column "C" \
  --dry-run
```

### Live create in Jira

```bash
python import_okr_initiatives_to_jira.py \
  --jira-base-url "https://halodi.atlassian.net" \
  --project-key "QTP" \
  --issue-type "Task" \
  --xlsx-path "/Users/courtney.grow/Downloads/OKR Sandbox (1).xlsx" \
  --sheet-name "Updated" \
  --column "C" \
  --live
```

Notes:
- Requires `JIRA_EMAIL` + `JIRA_API_TOKEN` (or `ATLASSIAN_EMAIL` + `ATLASSIAN_API_TOKEN`).
- Header value `Initiatives` is ignored.
- Duplicate initiative names in column C are de-duplicated.
- Existing issues with matching summary are skipped by default (`--no-skip-existing` to disable).
- Google Sheets API mode is also supported using `--google-sheet-id` and `--google-service-account-json`.

## Joint Level DBQ Report by Impacted Robot System (Confluence)

This flow is independent from `run_weekly_design_system_report.py` and builds a section per impacted robot system with:
- full open-issues Jira sheet (editable via Jira Issues macro)
- weekly trend chart (opened, resolved, cumulative open)
- issue category Pareto chart using DQB dashboard categories

Default publish target page:
- `https://halodi.atlassian.net/wiki/spaces/QUAL/pages/5691277356/Joint+Level+DBQ+Report`

### Dry run (generate artifacts only)

```bash
python run_joint_level_dbq_report.py --output-dir output
```

### Publish to Confluence page

```bash
python run_joint_level_dbq_report.py \
  --publish-confluence \
  --confluence-base "https://halodi.atlassian.net/wiki" \
  --page-id "5691277356"
```

Optional for Jira macro compatibility in some Confluence instances:
- `--jira-server-name "<linked Jira name>"`
- `--jira-server-id "<linked Jira server id>"`

Managed section markers used on the page:
- `<!-- DQB_JOINT_LEVEL_REPORT_START -->`
- `<!-- DQB_JOINT_LEVEL_REPORT_END -->`

### Daily automation on macOS (launchd)

Install a daily morning publish schedule (default `07:00` local):

```bash
bash /Users/courtney.grow/Projects/jira_issue_agent/scripts/install_joint_level_dbq_autoupdate.sh
```

Custom time (hour minute):

```bash
bash /Users/courtney.grow/Projects/jira_issue_agent/scripts/install_joint_level_dbq_autoupdate.sh 8 15
```

Manual run:

```bash
bash /Users/courtney.grow/Projects/jira_issue_agent/scripts/push_joint_level_dbq_report.sh
```

The publisher script includes:
- DNS readiness check before publish
- retry around report publish command
- once-per-day success guard (`/tmp/joint_level_dbq_last_success.txt`) to avoid duplicates

Useful checks:

```bash
launchctl list | grep com.halodi.joint_level_dbq.report
launchctl kickstart -k gui/$(id -u)/com.halodi.joint_level_dbq.report
tail -n 100 /tmp/joint_level_dbq_report.log
tail -n 100 /tmp/joint_level_dbq_report.err
```

### Reliability options (recommended order)

1. Run on an always-on hosted runner (GitHub Actions scheduled workflow, AWS/GCP VM, or similar).
2. Run on a dedicated always-on office machine or mini-server (not a laptop).
3. Keep local launchd on laptop as fallback only (`RunAtLoad` catches up when you log in).

For production consistency, move this job off your laptop and add failure alerting (Slack/email) plus a heartbeat check on successful publish.

### GitHub Actions scheduled publish (off-laptop)

Workflow file:
- `.github/workflows/joint_level_dbq_report.yml`

Schedule:
- Daily around `07:10` America/Los_Angeles (DST-safe).
- Also supports manual trigger (`workflow_dispatch`).

Required GitHub repository secrets:
- `ATLASSIAN_EMAIL`
- `ATLASSIAN_API_TOKEN`

Optional GitHub repository variables:
- `CONFLUENCE_BASE_URL` (default: `https://halodi.atlassian.net/wiki`)
- `JOINT_LEVEL_DBQ_PAGE_ID` (default: `5691277356`)
- `JIRA_URL` (default: DQB project list URL)
- `CONFLUENCE_JIRA_SERVER_NAME`
- `CONFLUENCE_JIRA_SERVER_ID`

Validation checklist:
1. Push this workflow file to the default branch.
2. In GitHub: `Settings -> Secrets and variables -> Actions`, add required secrets.
3. Run `Actions -> Joint Level DBQ Report Publisher -> Run workflow` once.
4. Confirm page update and check workflow logs for any Jira/Confluence auth errors.
