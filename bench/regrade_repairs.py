#!/usr/bin/env python3
"""Recheck archived repairs locally; retain original grades and append evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

from task_qualification import CASES, SANDBOX, json_answer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    cases = {c['id']: c for c in CASES if c['kind'] == 'repair'}
    for run in sorted(args.runs.iterdir()):
        events = run / 'events.jsonl'
        if not events.is_file():
            continue
        rows = [json.loads(line) for line in events.read_text().splitlines()]
        if not any(r.get('event') == 'run_finish' and r.get('status') == 'complete' for r in rows):
            continue
        changes = []
        for row in rows:
            if row.get('event') != 'task_result' or row['case'] not in cases:
                continue
            try:
                code = json_answer(row['requests'][-1]['output'])['code']
                check = subprocess.run([sys.executable, '-I', '-c', SANDBOX],
                    input=json.dumps({'code': code, 'tests': cases[row['case']]['tests']}),
                    capture_output=True, text=True, timeout=3)
                passed = check.returncode == 0
            except Exception:
                continue
            if passed != row['passed']:
                changes.append({'case': row['case'], 'repeat': row['repeat'],
                    'concurrency': row['concurrency'], 'original_passed': row['passed'],
                    'rechecked_passed': passed, 'code_sha256': hashlib.sha256(code.encode()).hexdigest(),
                    'original_check_error': row.get('check_error'),
                    'recheck_error': check.stderr[-300:] if check.returncode else None})
        if changes:
            directory = run / 'adjudications' / uuid.uuid4().hex
            directory.mkdir(parents=True)
            for filename in ('regrade_repairs.py', 'task_qualification.py'):
                (directory / filename).write_bytes(Path(__file__).with_name(filename).read_bytes())
            report = {'utc': time.time(), 'kind': 'offline-repair-checker-recheck',
                'reason': 'Checker permits standard type guards and restricted access to the two usage-count attributes; no model requests were rerun.',
                'original_events_sha256': hashlib.sha256(events.read_bytes()).hexdigest(),
                'checker_sha256': hashlib.sha256(SANDBOX.encode()).hexdigest(), 'changes': changes}
            (directory / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps({'run': run.name, 'adjudication': directory.name, 'changes': len(changes)}))


if __name__ == '__main__':
    main()
