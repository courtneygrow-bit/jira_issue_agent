#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import textwrap
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
from urllib.parse import quote
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import requests


DEFAULT_JIRA_URL = "https://halodi.atlassian.net/jira/software/projects/DQB/list?sortBy=status&direction=ASC"

ENGINEERING_ASSIGNEES: List[Tuple[str, str]] = [
    ("Shawn Gao", "shawn.gao@1x.tech"),
    ("Ryan Reich", "ryan.reich@1x.tech"),
    ("Luke Mariak", "luke.mariak@1x.tech"),
    ("Ethan Lietch", "ethan.lietch@1x.tech"),
    ("Grant Gabrielson", "grant.gabrielson@1x.tech"),
    ("John Strong", "john.strong@1x.tech"),
    ("Jiayao Yan", "jiayao.yan@1x.tech"),
    ("Xiang Li", "xiang.li@1x.tech"),
    ("Aditya Velivelli", "aditya.velivelli@1x.tech"),
    ("Miguel Espinal", "miguel.espinal@1x.tech"),
    ("Jeong Lyu", "jeongho.lyu@1x.tech"),
    ("Emily Song", "emily.song@1x.tech"),
    ("Elling Diesen", "elling@1x.tech"),
    ("Jonathan Terfurth", "jonathan.terfurth@1x.tech"),
    ("Cristofaro Pompermaier", "cristofaro@1x.tech"),
    ("Sabith Nakva", "sabith.nakva@1x.tech"),
    ("Bryn Cameron", "bryn.cameron@1x.tech"),
    ("Joel Filho", "joel@1x.tech"),
    ("Sjur Wroldsen", "sjur@1x.tech"),
    ("Sicheng Zou", "sicheng.zou@1x.tech"),
    ("Alex Granieri", "alex.granieri@1x.tech"),
    ("Rohan Karunaratne", "rohan.karunaratne@1x.tech"),
    ("Gabriel Bojer", "gabriel.bojer@1x.tech"),
    ("Pablo Ramírez", "pablo@1x.tech"),
    ("Sebastian Sterr", "sebastian.sterr@1x.tech"),
    ("Karun Balachandran", "karun.balachandran@1x.tech"),
    ("Sam Carmel", "samuel.carmel@1x.tech"),
    ("Chris Piekarski", "christopher.piekarski@1x.tech"),
    ("Karthik Bollam", "karthik.bollam@1x.tech"),
    ("Yeting (Nick) Liu", "nick.liu@1x.tech"),
    ("Conrad Ku", "conrad.ku@1x.tech"),
    ("Rishu Mohanka", "rishabh.mohanka@1x.tech"),
    ("Brandy Cao", "yue.cao@1x.tech"),
    ("Michael Webber", "michael.webber@1x.tech"),
    ("Emilie Boras", "emilie.boras@1x.tech"),
    ("Chris Goul", "chris.goul@1x.tech"),
    ("Meghan Heil", "meghan.heil@1x.tech"),
    ("Per Øyvind Valen", "per.valen@1x.tech"),
    ("Chris Jiang", "christian.jiang@1x.tech"),
    ("BinBin Chi", "binbin.chi@1x.tech"),
    ("Taylor Penn", "taylor.penn@1x.tech"),
    ("Kartik Iyer", "kartik.iyer@1x.tech"),
    ("Bhaumik Vashi", "bhaumik.vashi@1x.tech"),
    ("Nilesh Ashok Kharat", "nilesh.kharat@1x.tech"),
    ("Anjan Bose", "anjan.bose@1x.tech"),
    ("Andrew Dizon", "melandro.dizon@1x.tech"),
]

TERMINAL_STATUSES = {
    "done",
    "resolved",
    "closed",
    "canceled",
    "cancelled",
    "won't do",
    "wont do",
}


TERMINAL_STATUS_KEYWORDS = (
    "resolved",
    "closed",
    "done",
    "rejected",
    "cancelled",
    "canceled",
    "won't do",
    "wont do",
)


def normalize_status_value(status: str) -> str:
    return re.sub(r"\s+", " ", str(status or "").strip().lower())


def is_terminal_status(status: str) -> bool:
    norm = normalize_status_value(status)
    if not norm:
        return False
    if norm in TERMINAL_STATUSES:
        return True
    return any(tok in norm for tok in TERMINAL_STATUS_KEYWORDS)


HIGH_LEVEL_CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "Rope": [
        "rope",
        "line",
        "spool",
        "reeving",
        "fray",
        "fraying",
        "twist",
        "knot",
        "tension rope",
    ],
    "PCBA": [
        "pcba",
        "pcb",
        "board",
        "dad board",
        "encoder",
        "bms",
        "pdb",
        "resistor",
        "capacitor",
        "mosfet",
        "ic",
        "solder",
    ],
    "Harness": [
        "harness",
        "wire",
        "wiring",
        "connector",
        "ethercat",
        "cable",
        "pinout",
        "terminal",
        "crimp",
        "loom",
    ],
    "Fastener": [
        "bolt",
        "screw",
        "thread",
        "loctite",
        "cross thread",
        "stripped",
        "torque",
        "nut",
        "washer",
        "stud",
        "helicoil",
    ],
    "Test": [
        "test",
        "kt",
        "cogging",
        "ict",
        "board test",
        "validation",
        "verification",
        "qa",
        "screening",
    ],
    "Bearing": [
        "bearing",
        "race",
        "ball bearing",
        "roller bearing",
        "seized",
        "play",
        "radial",
        "axial",
    ],
    "Drum": [
        "drum",
        "sheave",
        "groove",
        "winch drum",
        "wrap",
    ],
    "Structural": [
        "structural",
        "frame",
        "bracket",
        "chassis",
        "housing",
        "mount",
        "deformation",
        "crack",
        "weld",
        "plate",
    ],
    "Software/Controls": [
        "firmware",
        "software",
        "bug",
        "control",
        "logic",
        "algorithm",
        "pid",
        "can",
        "fault code",
    ],
    "Process/Documentation": [
        "procedure",
        "process",
        "instruction",
        "work instruction",
        "doc",
        "documentation",
        "checklist",
        "training",
        "release note",
    ],
}

THEME_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
    "issue",
    "error",
    "failure",
    "problem",
    "during",
    "after",
    "before",
    "when",
    "have",
    "has",
    "into",
    "onto",
    "board",
    "test",
    "check",
    "task",
}


@dataclass
class JiraIssue:
    key: str
    summary: str
    description: str
    labels: List[str]
    issue_type: str
    status: str
    reporter: str
    assignee: str
    assignee_email: str
    impacted_robot_system: str
    due_date: str
    created: datetime
    updated: datetime
    resolved: Optional[datetime]
    status_changed: datetime
    url: str


def adf_to_text(node: Any) -> str:
    if node is None:
        return ""
    if isinstance(node, str):
        return node
    if isinstance(node, dict):
        out: List[str] = []
        if node.get("type") == "text":
            out.append(node.get("text", ""))
        for child in node.get("content", []):
            out.append(adf_to_text(child))
        if node.get("type") in {"paragraph", "heading", "listItem", "bulletList", "orderedList"}:
            out.append("\n")
        return "".join(out)
    if isinstance(node, list):
        return "".join(adf_to_text(item) for item in node)
    return ""


def normalize_name(value: str) -> str:
    s = unicodedata.normalize("NFKD", value or "")
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = re.sub(r"\([^)]*\)", " ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    return re.sub(r"\s+", " ", s)




def normalize_text(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def keyword_score(text: str, keywords: List[str]) -> int:
    score = 0
    for kw in keywords:
        pattern = re.escape(kw)
        if " " not in kw and "-" not in kw and "_" not in kw:
            pattern = rf"\b{pattern}\b"
        if re.search(pattern, text):
            score += 2 if " " in kw else 1
    return score


def infer_generalized_other_label(summary: str, description: str) -> str:
    text = normalize_text(f"{summary} {description}")
    tokens = re.findall(r"[a-z][a-z0-9\-]{2,}", text)
    filtered = [t for t in tokens if t not in THEME_STOPWORDS]
    if not filtered:
        return "General/Other"
    top_tokens = [t for t, _ in Counter(filtered).most_common(2)]
    return f"Generalized: {'/'.join(top_tokens)}"


def categorize_issue_dashboard(summary: str, description: str) -> str:
    text = normalize_text(f"{summary} {description}")
    scores = {
        category: keyword_score(text, keywords)
        for category, keywords in HIGH_LEVEL_CATEGORY_KEYWORDS.items()
    }
    best_category = max(scores, key=scores.get)
    if scores[best_category] > 0:
        return best_category
    return infer_generalized_other_label(summary, description)


def pareto_display_category(raw_category: str) -> str:
    raw = str(raw_category or "").strip()
    ll = raw.lower()
    if ll.startswith("generalized:"):
        stripped = raw.split(":", 1)[1].strip()
        return stripped or "General/Other"
    return raw or "General/Other"


def issue_pareto_category(issue: JiraIssue) -> str:
    raw = categorize_issue_dashboard(issue.summary, issue.description)
    return pareto_display_category(raw)

def impacted_robot_system_pareto_label(issue: JiraIssue) -> str:
    raw = str(issue.impacted_robot_system or "").strip()
    return raw if raw and raw != "-" else "Unspecified"


def is_reported_issue_status(status: str) -> bool:
    norm = re.sub(r"\s+", " ", str(status or "").strip().lower())
    return norm in {"reported issue", "reported"}

def parse_jira_url(jira_url: str) -> Tuple[str, Optional[str]]:
    base = jira_url.split("/jira/")[0].rstrip("/")
    match = re.search(r"/projects/([A-Z0-9_\-]+)/", jira_url)
    return base, (match.group(1) if match else None)


def parse_jira_datetime(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%f%z")


def build_auth_headers(email: str, token: str) -> Dict[str, str]:
    raw = f"{email}:{token}".encode("utf-8")
    auth = base64.b64encode(raw).decode("utf-8")
    return {
        "Authorization": f"Basic {auth}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


def extract_last_status_change(issue_item: Dict[str, Any], created: datetime) -> datetime:
    changelog = issue_item.get("changelog") or {}
    histories = changelog.get("histories") or []
    latest = created
    for history in histories:
        when_raw = (history or {}).get("created")
        if not when_raw:
            continue
        try:
            when = parse_jira_datetime(when_raw)
        except Exception:
            continue
        for entry in (history or {}).get("items") or []:
            if str((entry or {}).get("field", "")).strip().lower() == "status" and when > latest:
                latest = when
    return latest

def _normalize_status(status: str) -> str:
    return normalize_status_value(status)


def extract_terminal_resolution_change(issue_item: Dict[str, Any]) -> Optional[datetime]:
    changelog = issue_item.get("changelog") or {}
    histories = changelog.get("histories") or []
    latest_terminal_change: Optional[datetime] = None
    for history in histories:
        when_raw = (history or {}).get("created")
        if not when_raw:
            continue
        try:
            when = parse_jira_datetime(when_raw)
        except Exception:
            continue
        for entry in (history or {}).get("items") or []:
            if str((entry or {}).get("field", "")).strip().lower() != "status":
                continue
            to_status = _normalize_status((entry or {}).get("toString", ""))
            if is_terminal_status(to_status):
                if latest_terminal_change is None or when > latest_terminal_change:
                    latest_terminal_change = when
    return latest_terminal_change


def _flatten_field_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, dict):
        for key in ("value", "name", "displayName"):
            v = value.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return ""
    if isinstance(value, list):
        parts = [_flatten_field_value(v) for v in value]
        parts = [x for x in parts if x]
        return ", ".join(parts)
    return ""


def _detect_impacted_robot_system_field_id(names_map: Dict[str, Any]) -> Optional[str]:
    for field_id, field_name in (names_map or {}).items():
        name = str(field_name or "").strip().lower()
        if "impacted" in name and "robot" in name and "system" in name:
            return str(field_id)
    return None

def discover_impacted_robot_system_field_id(
    base_url: str,
    headers: Dict[str, str],
    timeout_sec: int,
) -> Optional[str]:
    url = f"{base_url}/rest/api/3/field"
    resp = requests.get(url, headers=headers, timeout=timeout_sec)
    resp.raise_for_status()
    fields = resp.json() or []
    for field in fields:
        field_id = str((field or {}).get("id", "")).strip()
        field_name = str((field or {}).get("name", "")).strip().lower()
        if field_id and "impacted" in field_name and "robot" in field_name and "system" in field_name:
            return field_id
    return None


def fetch_issues(
    base_url: str,
    jira_email: str,
    jira_api_token: str,
    jql: str,
    max_issues: int,
    timeout_sec: int,
    impacted_system_field_id_override: Optional[str] = None,
) -> List[JiraIssue]:
    headers = build_auth_headers(jira_email, jira_api_token)
    search_url_new = f"{base_url}/rest/api/3/search/jql"
    search_url_old = f"{base_url}/rest/api/3/search"

    page_size = min(100, max_issues)
    start_at = 0
    next_page_token: Optional[str] = None
    issues: List[JiraIssue] = []

    impacted_system_field_id: Optional[str] = (
        str(impacted_system_field_id_override or "").strip() or None
    )
    if not impacted_system_field_id:
        try:
            impacted_system_field_id = discover_impacted_robot_system_field_id(base_url, headers, timeout_sec)
        except Exception:
            impacted_system_field_id = None

    base_fields = [
        "summary",
        "description",
        "labels",
        "components",
        "issuetype",
        "created",
        "updated",
        "resolutiondate",
        "status",
        "reporter",
        "assignee",
        "duedate",
    ]
    if impacted_system_field_id and impacted_system_field_id not in base_fields:
        base_fields.append(impacted_system_field_id)

    fields_csv = ",".join(base_fields)

    while len(issues) < max_issues:
        params: Dict[str, Any] = {
            "jql": jql,
            "maxResults": page_size,
            "fields": fields_csv,
            "expand": "changelog",
        }
        payload: Dict[str, Any] = {
            "jql": jql,
            "startAt": start_at,
            "maxResults": page_size,
            "fields": base_fields,
            "expand": ["changelog"],
        }

        if next_page_token:
            params["nextPageToken"] = next_page_token
        else:
            params["startAt"] = start_at

        resp = requests.get(search_url_new, headers=headers, params=params, timeout=timeout_sec)
        if resp.status_code in {404, 405, 410}:
            resp = requests.get(search_url_old, headers=headers, params=params, timeout=timeout_sec)
        if resp.status_code in {404, 405, 410}:
            resp = requests.post(search_url_old, headers=headers, data=json.dumps(payload), timeout=timeout_sec)
        resp.raise_for_status()

        data = resp.json()
        page_items = data.get("issues", [])

        for item in page_items:
            fields = item.get("fields", {})
            status = fields.get("status") or {}
            assignee = fields.get("assignee") or {}
            reporter = fields.get("reporter") or {}
            issue_type = fields.get("issuetype") or {}

            created_raw = fields.get("created", "")
            updated_raw = fields.get("updated", "")
            resolved_raw = fields.get("resolutiondate") or ""
            if not created_raw or not updated_raw:
                continue

            created = parse_jira_datetime(created_raw)
            updated = parse_jira_datetime(updated_raw)
            status_name = (status.get("name") or "").strip()
            resolved = parse_jira_datetime(resolved_raw) if resolved_raw else None
            if resolved is None:
                resolved = extract_terminal_resolution_change(item)
            # Final fallback: if issue is currently terminal but resolution timestamp is missing,
            # use last update timestamp so resolved-series is not artificially undercounted.
            if resolved is None and is_terminal_status(status_name):
                resolved = updated
            issue_key = item.get("key", "")
            status_changed = extract_last_status_change(item, created)
            assignee_email = str(assignee.get("emailAddress") or "").strip().lower()

            impacted_robot_system = ""
            if impacted_system_field_id:
                impacted_robot_system = _flatten_field_value(fields.get(impacted_system_field_id))
            if not impacted_robot_system:
                impacted_robot_system = _flatten_field_value(fields.get("components"))

            issues.append(
                JiraIssue(
                    key=issue_key,
                    summary=(fields.get("summary") or "").strip(),
                    description=adf_to_text(fields.get("description")).strip(),
                    labels=[str(x).strip() for x in (fields.get("labels") or []) if str(x).strip()],
                    issue_type=(issue_type.get("name") or "Unknown").strip(),
                    status=(status.get("name") or "").strip(),
                    reporter=(reporter.get("displayName") or reporter.get("emailAddress") or "Unknown").strip(),
                    assignee=(assignee.get("displayName") or assignee_email or "Unassigned").strip(),
                    assignee_email=assignee_email,
                    impacted_robot_system=impacted_robot_system or "-",
                    due_date=(fields.get("duedate") or "").strip(),
                    created=created,
                    updated=updated,
                    resolved=resolved,
                    status_changed=status_changed,
                    url=f"{base_url}/browse/{issue_key}",
                )
            )
            if len(issues) >= max_issues:
                break

        next_page_token = data.get("nextPageToken")
        start_at += len(page_items)
        total = data.get("total")
        is_last = data.get("isLast")

        if len(page_items) == 0:
            break
        if is_last is True:
            break
        if total is not None and start_at >= int(total):
            break
        if total is None and not next_page_token and len(page_items) < page_size:
            break

    return issues


def compute_previous_week_window(tz_name: str) -> Tuple[datetime, datetime]:
    tz = ZoneInfo(tz_name)
    now = datetime.now(tz)
    start_of_this_week = (now - timedelta(days=now.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
    start_of_previous_week = start_of_this_week - timedelta(days=7)
    return start_of_previous_week, start_of_this_week


def week_bucket(dt: datetime, tz: ZoneInfo) -> datetime:
    local = dt.astimezone(tz)
    return (local - timedelta(days=local.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)


def engineering_roster_sets() -> Tuple[Set[str], Set[str], List[str]]:
    names: Set[str] = set()
    emails: Set[str] = set()
    ordered_emails: List[str] = []
    for name, email in ENGINEERING_ASSIGNEES:
        n = normalize_name(name)
        e = str(email or "").strip().lower()
        if n:
            names.add(n)
        if e:
            emails.add(e)
            if e not in ordered_emails:
                ordered_emails.append(e)
    return names, emails, ordered_emails


def filter_engineering_design_issues(issues: Sequence[JiraIssue]) -> List[JiraIssue]:
    names, emails, _ = engineering_roster_sets()
    out: List[JiraIssue] = []
    for issue in issues:
        assignee_email = (issue.assignee_email or "").strip().lower()
        assignee_name = normalize_name(issue.assignee)
        if assignee_email and assignee_email in emails:
            out.append(issue)
            continue
        if assignee_name and assignee_name in names:
            out.append(issue)
    return out


def generate_trend_chart(design_issues: List[JiraIssue], week_end: datetime, tz_name: str, output_path: Path) -> None:
    tz = ZoneInfo(tz_name)
    start = (week_end - timedelta(weeks=15)).replace(hour=0, minute=0, second=0, microsecond=0)
    opened = defaultdict(int)
    resolved = defaultdict(int)
    baseline_open = 0

    for issue in design_issues:
        created_local = issue.created.astimezone(tz)
        resolved_local = issue.resolved.astimezone(tz) if issue.resolved else None

        # Open backlog at chart-window start.
        if created_local < start and (resolved_local is None or resolved_local >= start):
            baseline_open += 1

        wb = week_bucket(issue.created, tz)
        if wb >= start:
            opened[wb] += 1
        if issue.resolved:
            rb = week_bucket(issue.resolved, tz)
            if rb >= start:
                resolved[rb] += 1

    weeks = [start + timedelta(weeks=i) for i in range(16)]
    open_series = [opened.get(w, 0) for w in weeks]
    resolved_series = [resolved.get(w, 0) for w in weeks]

    cumulative_open_series: List[int] = []
    running_open = baseline_open
    for created_count, resolved_count in zip(open_series, resolved_series):
        running_open += created_count - resolved_count
        cumulative_open_series.append(running_open)

    fig, ax = plt.subplots(figsize=(12, 5))
    labels = [w.strftime("%Y-%m-%d") for w in weeks]
    ax.plot(labels, open_series, marker="o", color="#1f77b4", label="Opened")
    ax.plot(labels, resolved_series, marker="o", color="#ff7f0e", label="Resolved")
    ax.plot(labels, cumulative_open_series, marker="o", color="#2ca02c", label="Total Open Issues Count")
    ax.set_title("Design Issues Weekly Trend (Opened, Resolved, Total Open Issues Count)")
    ax.set_xlabel("Week")
    ax.set_ylabel("Issue Count")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
        tick.set_ha("right")
    plt.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def generate_pareto_chart(category_counts: Counter[str], output_path: Path) -> None:
    items = category_counts.most_common()
    labels = [k for k, _ in items]
    counts = [v for _, v in items]
    total = sum(counts) or 1
    cum_pct: List[float] = []
    running = 0
    for count in counts:
        running += count
        cum_pct.append((running / total) * 100)

    fig, ax1 = plt.subplots(figsize=(12, 5))
    x = list(range(len(labels)))
    ax1.bar(x, counts, color="#2ca02c")
    ax1.set_title("Design Issue Pareto by Impacted Robot System")
    ax1.set_ylabel("Issue Count")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=35, ha="right")

    ax2 = ax1.twinx()
    ax2.plot(x, cum_pct, color="#d62728", marker="o")
    ax2.set_ylabel("Cumulative %")
    ax2.set_ylim(0, 105)

    plt.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def generate_new_issues_table_image(new_issues: List[JiraIssue], output_path: Path) -> None:
    headers = ["Key", "Summary", "Status", "Reporter", "Assignee", "Impacted Robot System", "Due Date"]
    rows: List[List[str]] = []
    for issue in new_issues:
        wrapped_summary = "\n".join(textwrap.wrap(issue.summary.replace("|", "/"), width=58)) or "-"
        rows.append(
            [
                issue.key or "-",
                wrapped_summary,
                issue.status or "-",
                issue.reporter or "-",
                issue.assignee or "-",
                issue.impacted_robot_system or "-",
                issue.due_date or "-",
            ]
        )

    if not rows:
        rows = [["-", "No new design issues in the last week.", "-", "-", "-", "-", "-"]]

    fig_height = min(max(2.5 + 0.45 * len(rows), 5.0), 20.0)
    fig, ax = plt.subplots(figsize=(17, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        colWidths=[0.08, 0.35, 0.11, 0.11, 0.12, 0.15, 0.08],
        cellLoc="left",
        colLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.25)

    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#000000")
        cell.set_linewidth(1.0)
        if row_idx == 0:
            cell.set_facecolor("#e8eef7")
            cell.get_text().set_weight("bold")
            cell.get_text().set_ha("center")
            cell.get_text().set_va("center")
        else:
            if col_idx in {0, 6}:
                cell.get_text().set_ha("center")
            else:
                cell.get_text().set_ha("left")
            cell.get_text().set_va("center")

    ax.set_title("New Design Issues Created in Last Week", fontsize=12, fontweight="bold", pad=14)
    plt.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def generate_ageing_issues_table_image(
    ageing_issues: List[Tuple[JiraIssue, int]],
    output_path: Path,
) -> None:
    headers = ["Key", "Summary", "Status", "Assignee", "Impacted Robot System", "Last Status Change", "Days"]
    rows: List[List[str]] = []
    for issue, age_days in ageing_issues:
        wrapped_summary = "\n".join(textwrap.wrap(issue.summary.replace("|", "/"), width=52)) or "-"
        rows.append(
            [
                issue.key or "-",
                wrapped_summary,
                issue.status or "-",
                issue.assignee or "-",
                issue.impacted_robot_system or "-",
                issue.status_changed.strftime("%Y-%m-%d"),
                str(age_days),
            ]
        )

    if not rows:
        rows = [["-", "No issues currently in Reported Issue status.", "-", "-", "-", "-", "-"]]

    fig_height = min(max(2.5 + 0.45 * len(rows), 5.0), 22.0)
    fig, ax = plt.subplots(figsize=(17, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        colWidths=[0.07, 0.34, 0.10, 0.11, 0.16, 0.13, 0.09],
        cellLoc="left",
        colLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.25)

    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#000000")
        cell.set_linewidth(1.0)
        if row_idx == 0:
            cell.set_facecolor("#fde68a")
            cell.get_text().set_weight("bold")
            cell.get_text().set_ha("center")
            cell.get_text().set_va("center")
        else:
            if col_idx in {0, 5, 6}:
                cell.get_text().set_ha("center")
            else:
                cell.get_text().set_ha("left")
            cell.get_text().set_va("center")

    ax.set_title("Top 10 Longest in Reported Issue Status", fontsize=12, fontweight="bold", pad=14)
    plt.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_ageing_issues(
    design_issues: Sequence[JiraIssue],
    now: datetime,
    top_n: int = 10,
) -> List[Tuple[JiraIssue, int]]:
    ageing: List[Tuple[JiraIssue, int]] = []
    for issue in design_issues:
        if not is_reported_issue_status(issue.status):
            continue
        age_days = (now - issue.status_changed.astimezone(now.tzinfo)).days
        ageing.append((issue, age_days))
    ageing.sort(key=lambda pair: pair[1], reverse=True)
    return ageing[:top_n]


def build_assignee_filter_jql(project_key: str, assignee_names: Sequence[str]) -> str:
    escaped = [f'"{n}"' for n in assignee_names if str(n).strip()]
    if not escaped:
        return f"project = {project_key} ORDER BY created DESC"
    return f"project = {project_key} AND assignee in ({', '.join(escaped)}) AND statusCategory != Done ORDER BY created DESC"


def build_report(
    design_issues: List[JiraIssue],
    week_start: datetime,
    week_end: datetime,
    base_url: str,
    project_key: str,
    assignee_names: Sequence[str],
    ageing_issues: Sequence[Tuple[JiraIssue, int]],
    threshold_days: int,
) -> str:
    created_this_week = [i for i in design_issues if week_start <= i.created.astimezone(week_start.tzinfo) < week_end]
    resolved_this_week = [
        i for i in design_issues if i.resolved and week_start <= i.resolved.astimezone(week_start.tzinfo) < week_end
    ]
    open_issues = [i for i in design_issues if not is_terminal_status(i.status)]
    category_counts = Counter(impacted_robot_system_pareto_label(i) for i in design_issues)

    top_types = [f"- {name}: {count}" for name, count in category_counts.most_common(10)]

    jql = build_assignee_filter_jql(project_key, assignee_names)
    jql_url = f"{base_url}/issues/?jql={quote(jql)}"

    lines = [
        f"*DQB Design Issues Weekly Report* ({week_start.strftime('%Y-%m-%d')} to {week_end.strftime('%Y-%m-%d')})",
        "",
        "- Scope: DQB issues assigned to engineering roster members.",
        f"- Total scoped design issues: {len(design_issues)}",
        f"- New scoped issues this week: {len(created_this_week)}",
        f"- Scoped issues resolved this week: {len(resolved_this_week)}",
        f"- Open scoped issues: {len(open_issues)}",
        f"- Ageing table: top {len(ageing_issues)} issues with longest time in Reported Issue status.",
        "",
        "*Pareto categories (Impacted Robot System)*",
        *(top_types or ["- None"]),
        "",
        "New-issues and ageing-issues tables include impacted robot system field data.",
        "",
        f"Filtered Jira view: [Design Open Issues]({jql_url})",
        "Charts uploaded in thread/email: trend (opened/resolved), Pareto, weekly new-issues table, ageing-issues table.",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate weekly DQB design-issues report scoped to engineering assignees")
    parser.add_argument("--jira-url", default=os.getenv("JIRA_URL", DEFAULT_JIRA_URL))
    parser.add_argument("--jql", default="")
    parser.add_argument("--max-issues", type=int, default=5000)
    parser.add_argument("--timeout-sec", type=int, default=45)
    parser.add_argument("--timezone", default=os.getenv("REPORT_TIMEZONE", "America/Los_Angeles"))
    parser.add_argument("--ageing-days", type=int, default=30)
    parser.add_argument("--output-dir", default="output")
    args = parser.parse_args()

    jira_email = os.getenv("JIRA_EMAIL", "").strip() or os.getenv("ATLASSIAN_EMAIL", "").strip()
    jira_api_token = os.getenv("JIRA_API_TOKEN", "").strip() or os.getenv("ATLASSIAN_API_TOKEN", "").strip()
    if not jira_email or not jira_api_token:
        raise ValueError("Missing Jira credentials: set JIRA_EMAIL/JIRA_API_TOKEN or ATLASSIAN_EMAIL/ATLASSIAN_API_TOKEN")

    base_url, project_key = parse_jira_url(args.jira_url)
    if not project_key:
        project_key = "DQB"

    _, _, assignee_emails = engineering_roster_sets()
    assignee_names = list(dict.fromkeys([name for name, _ in ENGINEERING_ASSIGNEES if str(name).strip()]))
    jql = args.jql.strip() or f"project = {project_key} ORDER BY created DESC"

    all_issues = fetch_issues(
        base_url=base_url,
        jira_email=jira_email,
        jira_api_token=jira_api_token,
        jql=jql,
        max_issues=args.max_issues,
        timeout_sec=args.timeout_sec,
    )

    design_issues = filter_engineering_design_issues(all_issues)
    impacted_counts = Counter(impacted_robot_system_pareto_label(i) for i in design_issues)
    week_start, week_end = compute_previous_week_window(args.timezone)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    charts_dir = out_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    trend_chart = charts_dir / "design_opened_resolved_trend.png"
    pareto_chart = charts_dir / "design_issue_type_pareto.png"
    new_issues_table_chart = charts_dir / "design_new_issues_table.png"
    ageing_issues_table_chart = charts_dir / "design_ageing_issues_table.png"

    generate_trend_chart(design_issues, week_end, args.timezone, trend_chart)
    generate_pareto_chart(Counter(impacted_robot_system_pareto_label(i) for i in design_issues), pareto_chart)
    created_this_week = sorted(
        [i for i in design_issues if week_start <= i.created.astimezone(week_start.tzinfo) < week_end],
        key=lambda i: i.created,
        reverse=True,
    )
    generate_new_issues_table_image(created_this_week, new_issues_table_chart)

    now_local = datetime.now(ZoneInfo(args.timezone))
    ageing_issues = build_ageing_issues(design_issues, now_local, top_n=10)
    generate_ageing_issues_table_image(ageing_issues, ageing_issues_table_chart)

    report_text = build_report(
        design_issues=design_issues,
        week_start=week_start,
        week_end=week_end,
        base_url=base_url,
        project_key=project_key,
        assignee_names=assignee_names,
        ageing_issues=ageing_issues,
        threshold_days=args.ageing_days,
    )

    preview_path = out_dir / "weekly_design_report_preview.txt"
    preview_path.write_text(report_text + "\n", encoding="utf-8")

    print(f"Fetched issues (JQL result): {len(all_issues)}")
    print(f"Scoped design issues (engineering assignees): {len(design_issues)}")
    print(f"Resolved timestamps available: {sum(1 for i in design_issues if i.resolved is not None)}")
    print(f"Impacted robot system values: {len(impacted_counts)} distinct; Unspecified={impacted_counts.get('Unspecified', 0)}")
    print(f"Report window: {week_start.strftime('%Y-%m-%d')} to {week_end.strftime('%Y-%m-%d')}")
    print("Ageing table basis: top 10 issues by time in Reported Issue status")
    print(f"Ageing issue count in table: {len(ageing_issues)}")
    print(f"Preview written: {preview_path}")
    print(f"Trend chart: {trend_chart}")
    print(f"Pareto chart: {pareto_chart}")
    print(f"New issues table chart: {new_issues_table_chart}")
    print(f"Ageing issues table chart: {ageing_issues_table_chart}")


if __name__ == "__main__":
    main()
