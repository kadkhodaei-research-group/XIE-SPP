"""
Collects the repository traffic (clones and views) from the GitHub API and appends it to CSV files.
GitHub only keeps the last 14 days, so this runs weekly (.github/workflows/traffic.yml) and keeps the full history.

Environment variables: REPO (owner/name), TOKEN (token with read access to the repository administration).
Writes to the current directory: clones.csv, views.csv and README.md (summary).
"""
import csv
import datetime
import json
import os
import urllib.request
from pathlib import Path

KINDS = ('clones', 'views')


def fetch(kind):
    request = urllib.request.Request(
        f"https://api.github.com/repos/{os.environ['REPO']}/traffic/{kind}?per=day",
        headers={'Authorization': f"Bearer {os.environ['TOKEN']}", 'Accept': 'application/vnd.github+json'},
    )
    with urllib.request.urlopen(request) as response:
        return json.load(response)[kind]


def update(kind, days, out_dir=Path('.')):
    """Merges the daily counts into <kind>.csv. Days already in the file are overwritten with the newer values."""
    path = out_dir / f'{kind}.csv'
    rows = {}
    if path.exists():
        with open(path) as f:
            rows = {r['date']: r for r in csv.DictReader(f)}
    for day in days:
        date = day['timestamp'][:10]
        rows[date] = {'date': date, 'count': day['count'], 'uniques': day['uniques']}
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['date', 'count', 'uniques'])
        writer.writeheader()
        writer.writerows(rows[d] for d in sorted(rows))
    return [rows[d] for d in sorted(rows)]


def write_summary(data, out_dir=Path('.')):
    lines = [
        f"# Repository traffic: {os.environ.get('REPO', '')}",
        '',
        f'Updated: {datetime.date.today()}',
        '',
        '| | Total | Since |',
        '|---|---|---|',
    ]
    for kind, rows in data.items():
        total = sum(int(r['count']) for r in rows)
        since = rows[0]['date'] if rows else '-'
        lines.append(f'| {kind.capitalize()} | {total:,} | {since} |')
    lines += [
        '',
        'Daily counts (total and unique visitors/cloners per day): [clones.csv](clones.csv), [views.csv](views.csv).',
        'Daily unique counts cannot be summed: the same person on different days is counted on each day.',
        '',
    ]
    (out_dir / 'README.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    data = {kind: update(kind, fetch(kind)) for kind in KINDS}
    write_summary(data)
    print({kind: sum(int(r['count']) for r in rows) for kind, rows in data.items()})
