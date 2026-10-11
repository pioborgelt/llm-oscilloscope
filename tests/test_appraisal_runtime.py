import unittest
from unittest.mock import patch

from llm_oscilloscope.appraisal_runtime import AppraisalRuntime, _check_runtime_versions, _check_torch_build


class AppraisalRuntimeTests(unittest.TestCase):
    expected = {'torch': '2.11.0+cu130', 'transformers': '4.57.6'}

    def test_tested_versions_are_accepted(self):
        installed = dict(self.expected, torch='2.11.0', bitsandbytes='0.49.2')
        with patch('llm_oscilloscope.appraisal_runtime.version', side_effect=installed.__getitem__):
            _check_runtime_versions(self.expected)
        _check_torch_build('2.11.0+cu130', self.expected['torch'])

    def test_fresh_newer_runtime_fails_before_model_load_with_remedy(self):
        installed = {'torch': '2.14.0+cu130', 'transformers': '4.57.6', 'bitsandbytes': '0.50.2'}
        with patch('llm_oscilloscope.appraisal_runtime.version', side_effect=installed.__getitem__), \
             patch('llm_oscilloscope.appraisal_runtime.QwenInstrumentedRuntime.__init__') as load:
            with self.assertRaises(ValueError) as error:
                AppraisalRuntime()
        load.assert_not_called()
        self.assertIn('installed 2.14.0+cu130, required 2.11.0', str(error.exception))
        self.assertIn('bitsandbytes: installed 0.50.2, required 0.49.2', str(error.exception))
        self.assertIn('pip install -e ".[runtime]"', str(error.exception))

    def test_other_torch_build_is_not_silently_accepted(self):
        with self.assertRaisesRegex(ValueError, 'Torch build mismatch: installed'):
            _check_torch_build('2.11.0+cpu', self.expected['torch'])
