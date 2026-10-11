from io import StringIO
import unittest
from rich.console import Console
from llm_oscilloscope.recordings import list_recordings
from llm_oscilloscope.tui import recording_view
from llm_oscilloscope.cli import build_parser

class AppraisalTests(unittest.TestCase):
    def test_all_samples_render_in_existing_tui(self):
        for sample in list_recordings():
            if sample['kind'] != 'appraisal_native_generation':
                continue
            out=StringIO();console=Console(file=out,width=120,color_system=None)
            console.print(recording_view(sample,len(sample['tokens'])-1))
            text=out.getvalue()
            self.assertIn('POSITIVE APPRAISAL',text)
            self.assertNotIn('\nSCOPE ',text)
            self.assertIn('4-token mean',text)
            self.assertNotIn('Answer peak',text)
            self.assertNotIn('Raw token',text)
            self.assertNotIn('Green: low',text)
            self.assertFalse(any(r['combined_alert'] for r in sample['tokens']))
    def test_opt_in_parser(self):
        self.assertTrue(build_parser().parse_args(['generate','hello','--appraisal']).appraisal)
        self.assertFalse(build_parser().parse_args(['generate','hello']).appraisal)

    def test_native_samples_are_unedited_and_identified(self):
        from llm_oscilloscope.recordings import list_recordings
        samples=[s for s in list_recordings() if s['kind']=='appraisal_native_generation']
        self.assertEqual(len(samples),4)
        for sample in samples:
            self.assertEqual(sample['generated_text'],''.join(r['text'] for r in sample['tokens']))
            self.assertIn('no completion supplied',sample['provenance']['origin'])
            self.assertEqual(sample['provenance']['decoding'],'greedy')
            self.assertFalse(sample['provenance']['token_limit_reached'])
            self.assertNotIn('\ufffd',sample['generated_text'])
            output=StringIO()
            Console(file=output,width=120,color_system=None).print(recording_view(sample,len(sample['tokens'])-1))
            self.assertIn('POSITIVE APPRAISAL',output.getvalue())
