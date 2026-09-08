import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
spec=importlib.util.spec_from_file_location('history',Path(__file__).resolve().parents[1]/'deploy/request_history.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class HistoryTests(unittest.TestCase):
    def test_restart_overlap_retention_and_field_allowlist(self):
        with tempfile.TemporaryDirectory() as directory:
            db=m.connect(Path(directory)/'history.sqlite3')
            row={'id':'a'*32,'started_at':1000000,'client':'abc','prompt':'PRIVATE'}
            line='INFO QWEN_REQUEST '+json.dumps(row)
            with db:
                m.ingest(db,line,1000001)
                m.ingest(db,line,1000001)
            records=db.execute('SELECT data FROM requests').fetchall()
            self.assertEqual(len(records),1)
            self.assertNotIn('PRIVATE',records[0][0])
            with db: m.ingest(db,'',1000001+m.RETENTION)
            self.assertEqual(db.execute('SELECT COUNT(*) FROM requests').fetchone()[0],0)
            db.close()
