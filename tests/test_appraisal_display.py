import json
import math
from importlib.resources import files
import unittest

from llm_oscilloscope.appraisal_display import appraisal_band, appraisal_color_scale, recorded_appraisal_row


class AppraisalDisplayTests(unittest.TestCase):
    def test_recorded_prefix_uses_full_window_without_changing_source(self):
        rows = [dict(appraisal_mean4=value, appraisal_window_ready=i >= 3)
                for i, value in enumerate((1., 2., 3., 4., 5.))]
        for i in range(3):
            display = recorded_appraisal_row(rows, i)
            self.assertEqual(display['appraisal_mean4'], 4.)
            self.assertTrue(display['appraisal_window_ready'])
            self.assertEqual(rows[i]['appraisal_mean4'], float(i + 1))
            self.assertFalse(rows[i]['appraisal_window_ready'])
        self.assertIs(recorded_appraisal_row(rows, 4), rows[4])
        self.assertIs(recorded_appraisal_row(rows[:2], 0), rows[0])

    def test_limits_match_equal_answer_weighted_reference_thirds(self):
        scale = appraisal_color_scale()
        samples = json.loads(files('llm_oscilloscope').joinpath(scale['reference']['file']).read_text())['samples']
        observations = []
        for sample in samples:
            values = sample['means4']
            observations.extend((value, 1 / len(values)) for value in values)
        self.assertEqual(len(samples), scale['reference']['answers'])
        self.assertEqual(len(observations), scale['reference']['complete_windows'])
        for fraction, key in ((1 / 3, 'yellow_from'), (2 / 3, 'red_from')):
            accumulated = 0
            for value, weight in sorted(observations):
                accumulated += weight
                if accumulated >= len(samples) * fraction:
                    self.assertAlmostEqual(value, scale[key])
                    break

    def test_boundaries_and_unavailable_values(self):
        scale = appraisal_color_scale()
        low, high = scale['yellow_from'], scale['red_from']
        self.assertEqual(appraisal_band(math.nextafter(low, -math.inf))[0], 'LOW')
        self.assertEqual(appraisal_band(low)[0], 'MEDIUM')
        self.assertEqual(appraisal_band(math.nextafter(high, -math.inf))[0], 'MEDIUM')
        self.assertEqual(appraisal_band(high)[0], 'HIGH')
        self.assertEqual(appraisal_band(100, ready=False)[0], 'PENDING')
        for value in (None, math.nan, math.inf):
            self.assertEqual(appraisal_band(value)[0], 'UNAVAILABLE')
