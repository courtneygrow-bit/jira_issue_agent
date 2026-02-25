#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import html
import json
import os
import re
import time
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import quote
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import requests

from run_weekly_design_report import (
    JiraIssue,
    build_auth_headers,
    categorize_issue_dashboard,
    compute_previous_week_window,
    discover_impacted_robot_system_field_id,
    fetch_issues,
    is_terminal_status,
    parse_jira_url,
)

DEFAULT_JIRA_URL = "https://halodi.atlassian.net/jira/software/projects/DQB/list?sortBy=status&direction=ASC"
DEFAULT_CONFLUENCE_BASE = "https://halodi.atlassian.net/wiki"
DEFAULT_CONFLUENCE_PAGE_ID = "5691277356"
MANAGED_START = "<!-- DQB_JOINT_LEVEL_REPORT_START -->"
MANAGED_END = "<!-- DQB_JOINT_LEVEL_REPORT_END -->"

RETRYABLE_FETCH_ERRORS = (
    requests.exceptions.ChunkedEncodingError,
    requests.exceptions.ConnectionError,
    requests.exceptions.ReadTimeout,
    requests.exceptions.Timeout,
)

SYSTEM_CANONICAL_MAP: Dict[str, str] = {
    "head": "Head",
    "neck": "Neck",
    "scapula": "Scapula",
    "spine": "Spine",
    "shoulder": "Shoulders",
    "shoulders": "Shoulders",
    "elbow": "Elbows",
    "elbows": "Elbows",
    "lower arm": "Lower Arm",
    "lower arms": "Lower Arm",
    "hand": "Hands",
    "hands": "Hands",
    "battery and pdb": "Battery and PDB",
    "hips": "Hips",
    "hip": "Hips",
    "upper leg": "Upper Leg",
    "lower leg": "Lower Leg",
    "feet": "Feet",
    "foot": "Feet",
}

UPPER_BODY_ORDER = [
    "Head",
    "Neck",
    "Scapula",
    "Spine",
    "Shoulders",
    "Elbows",
    "Lower Arm",
    "Hands",
    "Battery and PDB",
]

LOWER_BODY_ORDER = [
    "Hips",
    "Upper Leg",
    "Lower Leg",
    "Feet",
]

# Fallback system mapping if Impacted Robot System field is empty.
ASSIGNEE_SYSTEM_FALLBACK: List[Tuple[str, str]] = [
    ("Shawn Gao", "Body"),
    ("Xiang Li", "Shoulder"),
    ("Ethan Lietch", "Upper Arm"),
    ("Ethan Leitch", "Upper Arm"),
    ("Grant Gabrielson", "Scapula"),
    ("Chris Goul", "Hip"),
    ("Ryan Reich", "Upper Leg"),
    ("John Strong", "Lower Leg"),
    ("John Strong", "Foot"),
    ("Max Harris", "Covers"),
    ("BinBin Chi", "Battery"),
    ("Nilesh Ashok Kharat", "Charger"),
    ("Bhaumik Vashi", "BMS"),
    ("Bhoumik Vashi", "BMS"),
    ("Bhaumik Vashi", "PDB"),
    ("Bhoumik Vashi", "PDB"),
    ("Miguel Espinal", "Head"),
    ("Bryn Cameron", "Dual Axis Drive FFA (Dad Board)"),
    ("Bryn Cameron", "Encoder"),
    ("Megan Hall", "Harnesses"),
    ("Meghan Hall", "Harnesses"),
    ("Alex Granieri", "Hands"),
    ("Alex Grenieri", "Hands"),
]


def normalize_name(value: str) -> str:
    s = unicodedata.normalize("NFKD", value or "")
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = re.sub(r"\([^)]*\)", " ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    return re.sub(r"\s+", " ", s)


def slugify(text: str) -> str:
    t = normalize_name(text).replace(" ", "_")
    return t or "unspecified"


def clean_category_label(raw: str) -> str:
    txt = str(raw or "").strip()
    ll = txt.lower()
    if ll.startswith("generalized:"):
        return txt.split(":", 1)[1].strip() or "General/Other"
    return txt or "General/Other"


def assignee_to_system_map() -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = defaultdict(list)
    for name, system in ASSIGNEE_SYSTEM_FALLBACK:
        key = normalize_name(name)
        val = str(system or "").strip()
        if key and val and val not in out[key]:
            out[key].append(val)
    return dict(out)


def split_impacted_systems(value: str) -> List[str]:
    raw = str(value or "").strip()
    if not raw or raw == "-":
        return []
    parts = [x.strip() for x in re.split(r"[,;\n]+", raw) if x.strip()]
    return parts


def resolve_issue_systems(issue: JiraIssue) -> List[str]:
    primary = split_impacted_systems(issue.impacted_robot_system)
    if primary:
        return [canonical_system_name(x) for x in primary]
    return ["Unspecified"]


def canonical_system_name(system: str) -> str:
    raw = str(system or "").strip()
    key = normalize_name(raw)
    return SYSTEM_CANONICAL_MAP.get(key, raw or "Unspecified")


def week_bucket(dt: datetime, tz: ZoneInfo) -> datetime:
    local = dt.astimezone(tz)
    return (local - timedelta(days=local.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)


def generate_trend_chart(system: str, issues: Sequence[JiraIssue], week_end: datetime, tz_name: str, out_path: Path) -> None:
    tz = ZoneInfo(tz_name)
    start = (week_end - timedelta(weeks=15)).replace(hour=0, minute=0, second=0, microsecond=0)
    opened = defaultdict(int)
    resolved = defaultdict(int)
    baseline_open = 0

    for issue in issues:
        created_local = issue.created.astimezone(tz)
        resolved_local = issue.resolved.astimezone(tz) if issue.resolved else None
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
    opened_series = [opened.get(w, 0) for w in weeks]
    resolved_series = [resolved.get(w, 0) for w in weeks]

    total_open_series: List[int] = []
    running = baseline_open
    for o, r in zip(opened_series, resolved_series):
        running += o - r
        total_open_series.append(running)

    labels = [w.strftime("%Y-%m-%d") for w in weeks]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(labels, opened_series, marker="o", color="#1f77b4", label="Opened")
    ax.plot(labels, resolved_series, marker="o", color="#ff7f0e", label="Resolved")
    ax.plot(labels, total_open_series, marker="o", color="#2ca02c", label="Total Open Issues Count")
    ax.set_title(f"{system} Weekly Trend")
    ax.set_xlabel("Week")
    ax.set_ylabel("Issue Count")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
        tick.set_ha("right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def generate_category_pareto_chart(system: str, issues: Sequence[JiraIssue], out_path: Path) -> None:
    counts = Counter(clean_category_label(categorize_issue_dashboard(i.summary, i.description)) for i in issues)
    items = counts.most_common()
    labels = [k for k, _ in items]
    vals = [v for _, v in items]
    total = sum(vals) or 1
    cum = []
    running = 0
    for v in vals:
        running += v
        cum.append((running / total) * 100)

    fig, ax1 = plt.subplots(figsize=(12, 5))
    x = list(range(len(labels)))
    ax1.bar(x, vals, color="#2ca02c")
    ax1.set_title(f"{system} Pareto by Issue Category")
    ax1.set_ylabel("Issue Count")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=35, ha="right")

    ax2 = ax1.twinx()
    ax2.plot(x, cum, color="#d62728", marker="o")
    ax2.set_ylabel("Cumulative %")
    ax2.set_ylim(0, 105)

    plt.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def ensure_env(name: str) -> str:
    v = os.getenv(name, "").strip()
    if not v:
        raise ValueError(f"Missing required environment variable: {name}")
    return v


def auth_headers(email: str, token: str, include_json: bool = True) -> Dict[str, str]:
    auth = base64.b64encode(f"{email}:{token}".encode("utf-8")).decode("utf-8")
    h = {"Authorization": f"Basic {auth}", "Accept": "application/json"}
    if include_json:
        h["Content-Type"] = "application/json"
    return h


def get_page(confluence_base: str, page_id: str, email: str, token: str) -> Dict[str, Any]:
    url = f"{confluence_base}/rest/api/content/{page_id}"
    params = {"expand": "body.storage,version,title,type"}
    r = requests.get(url, headers=auth_headers(email, token), params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def get_attachment_id(confluence_base: str, page_id: str, filename: str, email: str, token: str) -> Optional[str]:
    url = f"{confluence_base}/rest/api/content/{page_id}/child/attachment"
    r = requests.get(url, headers=auth_headers(email, token), params={"filename": filename}, timeout=30)
    r.raise_for_status()
    rows = r.json().get("results", [])
    if not rows:
        return None
    return rows[0].get("id")


def upload_attachment(confluence_base: str, page_id: str, file_path: Path, email: str, token: str) -> None:
    existing = get_attachment_id(confluence_base, page_id, file_path.name, email, token)
    headers = auth_headers(email, token, include_json=False)
    headers["X-Atlassian-Token"] = "nocheck"

    def _post_with_retry(url: str) -> None:
        for attempt in range(1, 4):
            with file_path.open("rb") as f:
                files = {"file": (file_path.name, f, "image/png")}
                r = requests.post(url, headers=headers, files=files, timeout=120)
            if r.status_code >= 500 and attempt < 3:
                time.sleep(attempt)
                continue
            r.raise_for_status()
            return

    if existing:
        try:
            update_url = f"{confluence_base}/rest/api/content/{page_id}/child/attachment/{existing}/data"
            _post_with_retry(update_url)
            return
        except requests.RequestException:
            pass

    create_url = f"{confluence_base}/rest/api/content/{page_id}/child/attachment"
    _post_with_retry(create_url)


def image_macro(filename: str, width: int = 520) -> str:
    fn = html.escape(filename, quote=True)
    return f'<ac:image ac:width="{width}"><ri:attachment ri:filename="{fn}" /></ac:image>'


def two_chart_row(left_filename: str, right_filename: str) -> str:
    left = image_macro(left_filename)
    right = image_macro(right_filename)
    return (
        '<ac:structured-macro ac:name="section"><ac:rich-text-body>'
        '<ac:structured-macro ac:name="column"><ac:parameter ac:name="width">50%</ac:parameter>'
        f"<ac:rich-text-body>{left}</ac:rich-text-body></ac:structured-macro>"
        '<ac:structured-macro ac:name="column"><ac:parameter ac:name="width">50%</ac:parameter>'
        f"<ac:rich-text-body>{right}</ac:rich-text-body></ac:structured-macro>"
        "</ac:rich-text-body></ac:structured-macro>"
    )


def jql_escape(value: str) -> str:
    return value.replace('"', '\\"')


def open_issue_jql(project_key: str, system: str) -> str:
    if system == "Unspecified":
        return (
            f"project = {project_key} AND statusCategory != Done "
            'AND "Impacted Robot System" is EMPTY ORDER BY priority DESC, created DESC'
        )
    return (
        f"project = {project_key} AND statusCategory != Done "
        f'AND "Impacted Robot System" = "{jql_escape(system)}" ORDER BY priority DESC, created DESC'
    )


def jira_sheet_macro(
    jql: str,
    *,
    columns: str,
    max_issues: int,
    server_name: Optional[str],
    server_id: Optional[str],
) -> str:
    parts = [
        '<ac:structured-macro ac:name="jira">',
        f"<ac:parameter ac:name=\"jqlQuery\">{html.escape(jql)}</ac:parameter>",
        f"<ac:parameter ac:name=\"columns\">{html.escape(columns)}</ac:parameter>",
        f"<ac:parameter ac:name=\"maximumIssues\">{int(max_issues)}</ac:parameter>",
    ]
    if server_name:
        parts.append(f"<ac:parameter ac:name=\"server\">{html.escape(server_name)}</ac:parameter>")
    if server_id:
        parts.append(f"<ac:parameter ac:name=\"serverId\">{html.escape(server_id)}</ac:parameter>")
    parts.append("</ac:structured-macro>")
    return "".join(parts)


def build_managed_block(meta: Dict[str, Any], page_id: str, jira_server_name: Optional[str], jira_server_id: Optional[str]) -> str:
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        MANAGED_START,
        "<h2>Joint Level DBQ Report (Automated)</h2>",
        f"<p>Last updated: {generated_at}</p>",
        f"<p>Confluence page: {page_id}</p>",
    ]

    rows_by_name = {str(r.get("system")): r for r in meta.get("systems", [])}
    upper_rows = [rows_by_name[s] for s in UPPER_BODY_ORDER if s in rows_by_name]
    lower_rows = [rows_by_name[s] for s in LOWER_BODY_ORDER if s in rows_by_name]
    ordered = upper_rows + lower_rows
    seen = {r["system"] for r in ordered}
    other_rows = [r for r in meta.get("systems", []) if r.get("system") not in seen]

    if upper_rows:
        lines.append("<h3>Upper Body</h3>")
    for system_row in upper_rows:
        system = html.escape(system_row["system"], quote=False)
        lines.append(f"<h4>{system}</h4>")
        lines.append("<ul>")
        lines.append(f"<li>Total issues (history used for trend): {int(system_row.get('total_issues', 0))}</li>")
        lines.append(f"<li>Open issues: {int(system_row.get('open_issues', 0))}</li>")
        lines.append("</ul>")
        lines.append(two_chart_row(system_row["trend_chart"], system_row["pareto_chart"]))
        lines.append("<h5>Open Issues (Editable Jira Sheet)</h5>")
        lines.append(
            jira_sheet_macro(
                system_row["open_jql"],
                columns="key,summary,status,assignee,reporter,priority,created,updated,duedate",
                max_issues=2000,
                server_name=jira_server_name,
                server_id=jira_server_id,
            )
        )

    if lower_rows:
        lines.append("<h3>Lower Body</h3>")
    for system_row in lower_rows:
        system = html.escape(system_row["system"], quote=False)
        lines.append(f"<h4>{system}</h4>")
        lines.append("<ul>")
        lines.append(f"<li>Total issues (history used for trend): {int(system_row.get('total_issues', 0))}</li>")
        lines.append(f"<li>Open issues: {int(system_row.get('open_issues', 0))}</li>")
        lines.append("</ul>")
        lines.append(two_chart_row(system_row["trend_chart"], system_row["pareto_chart"]))
        lines.append("<h5>Open Issues (Editable Jira Sheet)</h5>")
        lines.append(
            jira_sheet_macro(
                system_row["open_jql"],
                columns="key,summary,status,assignee,reporter,priority,created,updated,duedate",
                max_issues=2000,
                server_name=jira_server_name,
                server_id=jira_server_id,
            )
        )

    if other_rows:
        lines.append("<h3>Other Systems</h3>")
    for system_row in other_rows:
        system = html.escape(system_row["system"], quote=False)
        lines.append(f"<h4>{system}</h4>")
        lines.append("<ul>")
        lines.append(f"<li>Total issues (history used for trend): {int(system_row.get('total_issues', 0))}</li>")
        lines.append(f"<li>Open issues: {int(system_row.get('open_issues', 0))}</li>")
        lines.append("</ul>")
        lines.append(two_chart_row(system_row["trend_chart"], system_row["pareto_chart"]))
        lines.append("<h5>Open Issues (Editable Jira Sheet)</h5>")
        lines.append(
            jira_sheet_macro(
                system_row["open_jql"],
                columns="key,summary,status,assignee,reporter,priority,created,updated,duedate",
                max_issues=2000,
                server_name=jira_server_name,
                server_id=jira_server_id,
            )
        )

    lines.append(MANAGED_END)
    return "\n".join(lines)


def upsert_managed_section(existing_body: str, managed_block: str) -> str:
    # Remove previously managed sections from current and legacy marker names.
    body = existing_body
    marker_patterns = [
        re.compile(r"<!-- DQB_JOINT_LEVEL_REPORT_START -->.*?<!-- DQB_JOINT_LEVEL_REPORT_END -->", re.DOTALL),
        re.compile(r"<!-- DQB_DESIGN_SYSTEM_REPORT_START -->.*?<!-- DQB_DESIGN_SYSTEM_REPORT_END -->", re.DOTALL),
    ]
    for pattern in marker_patterns:
        body = pattern.sub("", body)

    # Defensive cleanup for old runs where marker comments were not preserved.
    body = re.sub(
        r"<h2>Joint Level DBQ Report \(Automated\)</h2>.*?(?=<h2>|$)",
        "",
        body,
        flags=re.DOTALL,
    )
    body = body.strip()
    suffix = "\n" if body.endswith("\n") else "\n\n"
    return f"{body}{suffix}{managed_block}\n"


def update_page_body(confluence_base: str, page_id: str, page: Dict[str, Any], new_body_storage: str, email: str, token: str) -> None:
    url = f"{confluence_base}/rest/api/content/{page_id}"
    payload = {
        "id": page_id,
        "type": page.get("type", "page"),
        "title": page["title"],
        "version": {"number": int(page["version"]["number"]) + 1},
        "body": {"storage": {"value": new_body_storage, "representation": "storage"}},
    }
    r = requests.put(url, headers=auth_headers(email, token), data=json.dumps(payload), timeout=30)
    r.raise_for_status()


def fetch_issues_with_retry(
    base_url: str,
    jira_email: str,
    jira_api_token: str,
    jql: str,
    max_issues: int,
    timeout_sec: int,
    fetch_retries: int,
    retry_backoff_sec: float,
    impacted_system_field_id: Optional[str],
) -> List[JiraIssue]:
    retries = max(1, int(fetch_retries))
    for attempt in range(1, retries + 1):
        try:
            return fetch_issues(
                base_url=base_url,
                jira_email=jira_email,
                jira_api_token=jira_api_token,
                jql=jql,
                max_issues=max_issues,
                timeout_sec=timeout_sec,
                impacted_system_field_id_override=impacted_system_field_id,
            )
        except RETRYABLE_FETCH_ERRORS as exc:
            if attempt >= retries:
                raise
            delay = float(retry_backoff_sec) * attempt
            print(
                f"Jira fetch failed (attempt {attempt}/{retries}) with {type(exc).__name__}; "
                f"retrying in {delay:.1f}s..."
            )
            time.sleep(delay)


def publish_to_confluence(
    confluence_base: str,
    page_id: str,
    out_dir: Path,
    meta: Dict[str, Any],
    dry_run: bool,
    replace_page_body: bool,
    jira_server_name: Optional[str],
    jira_server_id: Optional[str],
) -> None:
    email = os.getenv("JIRA_EMAIL", "").strip() or ensure_env("ATLASSIAN_EMAIL")
    token = os.getenv("JIRA_API_TOKEN", "").strip() or ensure_env("ATLASSIAN_API_TOKEN")

    if dry_run:
        print(f"[DRY-RUN] Would publish {len(meta.get('systems', []))} system sections to Confluence page {page_id}")
        return

    chart_dir = out_dir / "charts" / "systems"
    for row in meta.get("systems", []):
        for key in ["trend_chart", "pareto_chart"]:
            p = chart_dir / row[key]
            upload_attachment(confluence_base, page_id, p, email, token)

    page = get_page(confluence_base, page_id, email, token)
    managed = build_managed_block(meta, page_id, jira_server_name, jira_server_id)
    if replace_page_body:
        new_body = managed
    else:
        old_body = page["body"]["storage"]["value"]
        new_body = upsert_managed_section(old_body, managed)
    update_page_body(confluence_base, page_id, page, new_body, email, token)
    print(f"Published joint-level DBQ report to Confluence page: {page_id}")


def run(args: argparse.Namespace) -> None:
    jira_email = os.getenv("JIRA_EMAIL", "").strip() or os.getenv("ATLASSIAN_EMAIL", "").strip()
    jira_api_token = os.getenv("JIRA_API_TOKEN", "").strip() or os.getenv("ATLASSIAN_API_TOKEN", "").strip()
    if not jira_email or not jira_api_token:
        raise ValueError("Missing Jira credentials: set JIRA_EMAIL/JIRA_API_TOKEN or ATLASSIAN_EMAIL/ATLASSIAN_API_TOKEN")

    base_url, project_key = parse_jira_url(args.jira_url)
    if not project_key:
        project_key = "DQB"

    impacted_field_id = str(args.impacted_system_field_id or "").strip()
    if not impacted_field_id:
        headers = build_auth_headers(jira_email, jira_api_token)
        for attempt in range(1, args.fetch_retries + 1):
            try:
                impacted_field_id = (
                    discover_impacted_robot_system_field_id(base_url, headers, args.timeout_sec) or ""
                ).strip()
                if impacted_field_id:
                    break
            except RETRYABLE_FETCH_ERRORS:
                pass
            if attempt < args.fetch_retries:
                delay = float(args.retry_backoff_sec) * attempt
                print(
                    f"Impacted Robot System field discovery failed (attempt {attempt}/{args.fetch_retries}); "
                    f"retrying in {delay:.1f}s..."
                )
                time.sleep(delay)
    if not impacted_field_id:
        raise ValueError(
            "Unable to discover Jira custom field id for 'Impacted Robot System'. "
            "Set --impacted-system-field-id (or env JIRA_IMPACTED_SYSTEM_FIELD_ID) and rerun."
        )

    jql = args.jql.strip() or f"project = {project_key} ORDER BY created DESC"
    all_issues = fetch_issues_with_retry(
        base_url=base_url,
        jira_email=jira_email,
        jira_api_token=jira_api_token,
        jql=jql,
        max_issues=args.max_issues,
        timeout_sec=args.timeout_sec,
        fetch_retries=args.fetch_retries,
        retry_backoff_sec=args.retry_backoff_sec,
        impacted_system_field_id=impacted_field_id,
    )

    system_to_all_issues: Dict[str, List[JiraIssue]] = defaultdict(list)
    system_to_open_issues: Dict[str, List[JiraIssue]] = defaultdict(list)
    for issue in all_issues:
        systems = resolve_issue_systems(issue)
        for system in systems:
            system_to_all_issues[system].append(issue)
            if not is_terminal_status(issue.status):
                system_to_open_issues[system].append(issue)

    week_start, week_end = compute_previous_week_window(args.timezone)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    chart_dir = out_dir / "charts" / "systems"
    chart_dir.mkdir(parents=True, exist_ok=True)

    systems = sorted(system_to_open_issues.keys(), key=lambda x: x.lower())
    order_index = {name: i for i, name in enumerate(UPPER_BODY_ORDER + LOWER_BODY_ORDER)}
    systems.sort(key=lambda s: (0, order_index[s]) if s in order_index else (1, s.lower()))
    meta: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "week_start": week_start.strftime("%Y-%m-%d"),
        "week_end": week_end.strftime("%Y-%m-%d"),
        "total_issues": len(all_issues),
        "systems": [],
    }

    md_lines = [
        f"# Joint Level DBQ Report ({week_start.strftime('%Y-%m-%d')} to {week_end.strftime('%Y-%m-%d')})",
        "",
        f"- Total DQB issues in query scope: {len(all_issues)}",
        f"- Systems with open issues: {len(systems)}",
        "",
    ]

    upper_set = set(UPPER_BODY_ORDER)
    lower_set = set(LOWER_BODY_ORDER)
    emitted_upper = False
    emitted_lower = False
    emitted_other = False

    for system in systems:
        if system in upper_set and not emitted_upper:
            md_lines.extend(["## Upper Body", ""])
            emitted_upper = True
        if system in lower_set and not emitted_lower:
            md_lines.extend(["## Lower Body", ""])
            emitted_lower = True
        if system not in upper_set and system not in lower_set and not emitted_other:
            md_lines.extend(["## Other Systems", ""])
            emitted_other = True
        all_system_issues = system_to_all_issues.get(system, [])
        open_system_issues = system_to_open_issues.get(system, [])
        slug = slugify(system)
        trend_fn = f"{slug}_trend.png"
        pareto_fn = f"{slug}_pareto.png"

        trend_path = chart_dir / trend_fn
        pareto_path = chart_dir / pareto_fn

        generate_trend_chart(system, all_system_issues, week_end, args.timezone, trend_path)
        generate_category_pareto_chart(system, open_system_issues, pareto_path)

        system_jql = open_issue_jql(project_key, system)
        jql_url = f"{base_url}/issues/?jql={quote(system_jql)}"

        md_lines.extend(
            [
                f"## {system}",
                f"- Total issues (history): {len(all_system_issues)}",
                f"- Open issues: {len(open_system_issues)}",
                f"- Open issues Jira sheet query: `{system_jql}`",
                f"- Open issues Jira view: [{system}]({jql_url})",
                "",
            ]
        )

        meta["systems"].append(
            {
                "system": system,
                "slug": slug,
                "total_issues": len(all_system_issues),
                "open_issues": len(open_system_issues),
                "trend_chart": trend_fn,
                "pareto_chart": pareto_fn,
                "open_jql": system_jql,
                "jql_url": jql_url,
            }
        )

    preview_path = out_dir / "joint_level_dbq_report_preview.md"
    preview_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    meta_path = out_dir / "joint_level_dbq_report_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Fetched issues (JQL result): {len(all_issues)}")
    print(f"Systems with open issues: {len(systems)}")
    print(f"Preview written: {preview_path}")
    print(f"Meta written: {meta_path}")

    if args.publish_confluence:
        publish_to_confluence(
            args.confluence_base.rstrip("/"),
            args.page_id,
            out_dir,
            meta,
            dry_run=args.dry_run,
            replace_page_body=args.replace_page_body,
            jira_server_name=args.jira_server_name,
            jira_server_id=args.jira_server_id,
        )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Generate joint-level DQB report by impacted robot system with per-system trend, "
            "category pareto, and editable Jira Issues table, then optionally publish to Confluence"
        )
    )
    p.add_argument("--jira-url", default=os.getenv("JIRA_URL", DEFAULT_JIRA_URL))
    p.add_argument("--jql", default="")
    p.add_argument("--max-issues", type=int, default=8000)
    p.add_argument("--timeout-sec", type=int, default=60)
    p.add_argument("--fetch-retries", type=int, default=5)
    p.add_argument("--retry-backoff-sec", type=float, default=2.0)
    p.add_argument("--impacted-system-field-id", default=os.getenv("JIRA_IMPACTED_SYSTEM_FIELD_ID", ""))
    p.add_argument("--timezone", default=os.getenv("REPORT_TIMEZONE", "America/Los_Angeles"))
    p.add_argument("--output-dir", default="output")
    p.add_argument("--publish-confluence", action="store_true")
    p.add_argument("--confluence-base", default=DEFAULT_CONFLUENCE_BASE)
    p.add_argument("--page-id", default=DEFAULT_CONFLUENCE_PAGE_ID)
    p.add_argument("--jira-server-name", default=os.getenv("CONFLUENCE_JIRA_SERVER_NAME", ""))
    p.add_argument("--jira-server-id", default=os.getenv("CONFLUENCE_JIRA_SERVER_ID", ""))
    p.add_argument("--dry-run", action="store_true", help="When used with --publish-confluence, skip live Confluence update")
    p.add_argument(
        "--replace-page-body",
        action="store_true",
        help="Replace full Confluence page body with this report content (not just managed section).",
    )
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
