import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'bench'))
import qualification


class InterruptedRequestAuditTests(unittest.TestCase):
    def test_identity_survives_interruption_without_retaining_prompt_contents(self):
        class Response:
            def __enter__(self):return self
            def __exit__(self,*args):return False
            def __iter__(self):
                yield b'data: {"id":"response-1","choices":[]}\n'
                raise KeyboardInterrupt('simulated interrupted test process')
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'audit.jsonl'
            with patch.object(qualification,'AUDIT_PATH',path),patch.object(qualification,'fetch',return_value=Response()):
                with self.assertRaises(KeyboardInterrupt):
                    qualification.request('unused',{'messages':[{'role':'user','content':'private fixture text'}]})
            text=path.read_text();rows=[json.loads(line) for line in text.splitlines()]
            self.assertEqual([r['event'] for r in rows],['request_start','response_identity'])
            self.assertEqual(rows[0]['audit_id'],rows[1]['audit_id'])
            self.assertEqual(rows[1]['response_id'],'response-1')
            self.assertNotIn('private fixture text',text)
