import importlib.util
import io
from pathlib import Path
import unittest
from unittest.mock import patch

source = Path(__file__).resolve().parents[1] / 'bench/sparkdash_compare.py'
spec = importlib.util.spec_from_file_location('sparkdash_compare', source)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def response(count, running=0, waiting=0):
    return io.BytesIO(('vllm:generation_tokens_total{engine="0"} %s\n'
                       'vllm:num_requests_running{engine="0"} %s\n'
                       'vllm:num_requests_waiting{engine="0"} %s\n' %
                       (count, running, waiting)).encode())


class ExclusiveBenchmarkTests(unittest.TestCase):
    def test_requires_idle_and_settled_output_counter(self):
        samples = [response(10, running=1), response(11), response(12), response(12)]
        with patch.object(benchmark.urllib.request, 'urlopen', side_effect=samples), \
             patch.object(benchmark.time, 'sleep') as sleep:
            self.assertEqual(benchmark.idle_generation_count('http://example'), 12)
        self.assertEqual(sleep.call_count, 3)

    def test_missing_metrics_cannot_pass_as_an_idle_server(self):
        with patch.object(benchmark.urllib.request, 'urlopen', return_value=io.BytesIO(b'')):
            with self.assertRaisesRegex(ValueError, 'Required metric absent'):
                benchmark.idle_generation_count('http://example')
