#!/usr/bin/env python3
"""Retain allowlisted request metadata across inference container restarts."""
import argparse
import json
from pathlib import Path
import re
import sqlite3
import statistics
import subprocess
import time

FIELDS = {'event','id','started_at','client','status','max_tokens','thinking',
          'reasoning_effort','prompt_tokens','completion_tokens','cached_tokens',
          'time_to_first_token_ms','generation_time_ms','queue_time_ms','mean_itl_ms',
          'tokens_per_second','elapsed_ms','first_progress_ms','finish_reasons',
          'tool_call','stream_error','interrupted','disconnected',
          'request_metadata_omitted','response_metadata_omitted'}
RETENTION = 7 * 86400


def ingest(db, text, now):
    count = 0
    for line in text.splitlines():
        match = re.search(r'QWEN_REQUEST(_START)? (\{.*\})$', line)
        if not match:
            continue
        try:
            row = json.loads(match[2])
            if not isinstance(row, dict) or not re.fullmatch(r'[a-f0-9]{32}', str(row.get('id',''))):
                continue
            if type(row.get('started_at')) not in (int,float) or not now-RETENTION <= row['started_at'] <= now+60:
                continue
            row = {k:v for k,v in row.items() if k in FIELDS}
            phase = 'start' if match[1] else 'finish'
            db.execute('INSERT OR REPLACE INTO requests VALUES (?,?,?,?)',
                       (row['id'],phase,row['started_at'],json.dumps(row,separators=(',',':'))))
            count += 1
        except (ValueError,TypeError):
            continue
    db.execute('DELETE FROM requests WHERE started_at < ?', (now-RETENTION,))
    return count


def connect(path):
    path.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
    db = sqlite3.connect(path)
    path.chmod(0o600)
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE TABLE IF NOT EXISTS requests (id TEXT, phase TEXT, started_at REAL, data TEXT, PRIMARY KEY(id,phase))')
    db.execute('CREATE TABLE IF NOT EXISTS cursor (id INTEGER PRIMARY KEY, since REAL)')
    return db


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database',type=Path,default=Path.home()/'.local/state/qwen-cluster/requests.sqlite3')
    parser.add_argument('--once',action='store_true')
    parser.add_argument('--report-hours',type=float)
    args=parser.parse_args()
    db=connect(args.database)
    if args.report_hours is not None:
        if not 0 < args.report_hours <= 168:
            parser.error('report hours must be in (0, 168]')
        cutoff=time.time()-args.report_hours*3600
        rows=[json.loads(row[0]) for row in db.execute("SELECT data FROM requests WHERE phase='finish'")]
        rows=[row for row in rows if row['started_at']+(row.get('elapsed_ms') or 0)/1000 >= cutoff]
        missing=db.execute("SELECT COUNT(*) FROM requests s WHERE s.phase='start' AND s.started_at>=? AND NOT EXISTS (SELECT 1 FROM requests f WHERE f.id=s.id AND f.phase='finish')",(cutoff,)).fetchone()[0]
        report={'hours':args.report_hours,'finished_records':len(rows),'unfinished_records':missing,'clients':{}}
        for client in sorted({row.get('client','unknown') for row in rows}):
            group=[row for row in rows if row.get('client','unknown')==client]
            entry={'requests':len(group),'interrupted':sum(bool(r.get('interrupted') or r.get('disconnected')) for r in group)}
            for key in ('prompt_tokens','completion_tokens','cached_tokens','queue_time_ms','first_progress_ms','tokens_per_second'):
                values=[r[key] for r in group if type(r.get(key)) in (float,int)]
                entry[key]={'samples':len(values),'median':statistics.median(values) if values else None}
            report['clients'][client]=entry
        print(json.dumps(report,indent=2))
        return
    while True:
        now=time.time()
        with db:
            db.execute("DELETE FROM requests WHERE started_at < ?",(now-RETENTION,))
        cursor=db.execute('SELECT since FROM cursor WHERE id=1').fetchone()
        since=cursor[0] if cursor else now-RETENTION
        try:
            result=subprocess.run(['docker','logs','--since',str(int(since)-2),'vllm-fn'],capture_output=True,text=True,timeout=15)
            if result.returncode == 0:
                # Cursor uses the start of the read, so records arriving during
                # collection remain eligible. Primary keys remove overlap.
                with db:
                    ingest(db,result.stdout+'\n'+result.stderr,now)
                    db.execute('INSERT OR REPLACE INTO cursor VALUES (1,?)',(now,))
            else:
                print('Request history: container logs unavailable; retrying',flush=True)
        except subprocess.TimeoutExpired:
            print('Request history: log read timed out; retrying',flush=True)
        if args.once:
            return
        time.sleep(10)


if __name__=='__main__':
    main()
