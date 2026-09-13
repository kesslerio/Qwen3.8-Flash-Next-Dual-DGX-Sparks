import importlib.util
import json
from pathlib import Path
import sys
import subprocess
import unittest
from unittest.mock import patch

BENCH = Path(__file__).resolve().parents[1] / 'bench'
sys.path.insert(0, str(BENCH))
spec = importlib.util.spec_from_file_location('task_qualification', BENCH / 'task_qualification.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def read_response():
    return {'output': '', 'tool_calls': {0: {
        'id': 'read-1', 'type': 'function',
        'function': {'name': 'read_file', 'arguments': '{"path":"cache.py"}'}}}}


class ToolLoopTests(unittest.TestCase):
    def test_valid_type_guard_is_executable(self):
        code = "def candidate(usage):\n    if not isinstance(usage, dict): return None\n    a, b = usage.get('inputTokens'), usage.get('cacheReadTokens')\n    return None if a is None or b is None else a + b"
        result = subprocess.run([sys.executable, '-I', '-c', runner.SANDBOX],
            input=json.dumps({'code':code, 'tests':runner.CASES[4]['tests']}),
            capture_output=True, text=True, timeout=3)
        self.assertEqual(result.returncode, 0, result.stderr)

    def run_case(self, responses, budget):
        payloads = []

        def request(base, body, **kwargs):
            payloads.append(json.loads(json.dumps(body)))
            return responses.pop(0)

        with patch.object(runner, 'request', side_effect=request):
            result = runner.task('unused', runner.CASES[0], tool_budget=budget)
        return result, payloads

    def test_repeated_read_completes_within_fixed_budget(self):
        result, payloads = self.run_case([
            read_response(), read_response(),
            {'output': '{"field":"cacheReadTokens"}', 'tool_calls': {}}], 6)
        self.assertTrue(result['passed'])
        self.assertEqual(result['model_calls'], 3)
        self.assertEqual([p['tool_choice'] for p in payloads], ['required', 'auto', 'auto'])
        self.assertEqual(len(payloads[-1]['messages']), len(payloads[0]['messages']) + 4)

    def test_repeated_read_exhaustion_is_a_failed_task(self):
        result, payloads = self.run_case([read_response() for _ in range(3)], 3)
        self.assertFalse(result['passed'])
        self.assertEqual(result['error'], 'tool budget exhausted')
        self.assertEqual(len(payloads), 3)

    def test_original_two_call_empty_followup_remains_failure(self):
        result, payloads = self.run_case([
            read_response(), {'output': '', 'tool_calls': {}}], 2)
        self.assertFalse(result['passed'])
        self.assertEqual(payloads[-1]['tool_choice'], 'none')


if __name__ == '__main__':
    unittest.main()
